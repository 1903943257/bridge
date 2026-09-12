"""Puncture 4: complete Qwen3.5 THD chain at CP=1 and CP=2.

Run this file twice from the verl repository root::

    torchrun --standalone --nproc_per_node=1 -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_thd_cp_npu.py
    torchrun --standalone --nproc_per_node=2 -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_thd_cp_npu.py

The BSHD and THD passes use separate models with identical random parameters.
This exercises verl preprocessing and postprocessing, all 24 hybrid layers,
target-logprob loss, and complete parameter gradients without enabling TPR.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from ._qwen35_baseline_utils import (
    AllToAllProbe,
    VOCAB_SIZE,
    allreduce_parameter_gradients,
    assert_gradients_close,
    assert_hybrid_architecture,
    bind_stage1_gdn_primitives,
    broadcast_module_state,
    destroy_npu_runtime,
    initialize_npu_runtime,
    make_qwen35_model,
    packed_boundary_metadata,
    packed_seq_params_metadata,
)
from verl.utils.device import is_torch_npu_available


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size not in (1, 2):
        pytest.skip("puncture 4 supports CP=1 or CP=2")
    value = initialize_npu_runtime(world_size=world_size)
    yield value
    destroy_npu_runtime(value)


def _nested_tokens(device):
    pieces = (
        (torch.arange(29, device=device, dtype=torch.long) * 13 + 7) % VOCAB_SIZE,
        (torch.arange(35, device=device, dtype=torch.long) * 19 + 101) % VOCAB_SIZE,
    )
    return torch.nested.as_nested_tensor(pieces, layout=torch.jagged)


def _logits_processor(logits, label):
    if logits.dim() != 3:
        raise AssertionError(f"expected 3-D logits, got {tuple(logits.shape)}")
    if tuple(logits.shape[:2]) == tuple(label.shape):
        pass
    elif logits.shape[0] == label.shape[1] and logits.shape[1] == label.shape[0]:
        logits = logits.transpose(0, 1)
    else:
        raise AssertionError(
            "cannot align model logits with labels: "
            f"logits={tuple(logits.shape)}, label={tuple(label.shape)}"
        )
    target_logprob = torch.gather(
        F.log_softmax(logits.float(), dim=-1),
        dim=-1,
        index=label.long().unsqueeze(-1),
    ).squeeze(-1)
    probe_ids = torch.tensor([0, 17, 1024, VOCAB_SIZE - 1], device=logits.device)
    return {
        "target_logprob": target_logprob,
        "output_probe": logits.index_select(-1, probe_ids),
    }


@contextmanager
def _packed_layer_probe(model):
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    records = {"gdn": [], "fa": [], "metadata": []}
    handles = []
    for layer in model.decoder.layers:
        kind = "gdn" if isinstance(layer.self_attention, mindspeed_gdn.GatedDeltaNet) else "fa"

        def record(_module, args, kwargs, *, layer_number=layer.layer_number, layer_kind=kind):
            packed = kwargs.get("packed_seq_params")
            if packed is None:
                packed = next(
                    (value for value in reversed(args) if hasattr(value, "qkv_format")),
                    None,
                )
            if packed is None or packed.qkv_format != "thd":
                raise AssertionError(
                    f"layer {layer_number} ({layer_kind}) did not receive THD PackedSeqParams"
                )
            records[layer_kind].append(layer_number)
            records["metadata"].append(
                (layer_number, layer_kind, packed_seq_params_metadata(packed))
            )

        handles.append(layer.self_attention.register_forward_pre_hook(record, with_kwargs=True))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _ring_probe():
    import mindspeed.core.context_parallel.dot_product_attention as dpa

    original = dpa.ringattn_context_parallel
    calls = []

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    dpa.ringattn_context_parallel = traced
    try:
        yield calls
    finally:
        dpa.ringattn_context_parallel = original


@contextmanager
def _verl_data_path_probe(input_ids):
    import verl.models.mcore.model_forward as model_forward

    originals = {
        "thd_pre": model_forward.preprocess_thd_engine,
        "thd_post": model_forward.postprocess_thd_engine,
        "bshd_pre": model_forward.preprocess_bshd_engine,
        "bshd_post": model_forward.postprocess_bshd_engine,
    }
    records = {
        "thd_pre": 0,
        "thd_post": 0,
        "bshd_pre": 0,
        "bshd_post": 0,
        "model_packed_metadata": None,
        "metadata": None,
    }

    def thd_pre(*args, **kwargs):
        result = originals["thd_pre"](*args, **kwargs)
        records["thd_pre"] += 1
        if records["model_packed_metadata"] is None:
            records["model_packed_metadata"] = packed_seq_params_metadata(result[1])
            records["metadata"] = packed_boundary_metadata(input_ids, result[1])
        return result

    def thd_post(*args, **kwargs):
        records["thd_post"] += 1
        return originals["thd_post"](*args, **kwargs)

    def bshd_pre(*args, **kwargs):
        records["bshd_pre"] += 1
        return originals["bshd_pre"](*args, **kwargs)

    def bshd_post(*args, **kwargs):
        records["bshd_post"] += 1
        return originals["bshd_post"](*args, **kwargs)

    model_forward.preprocess_thd_engine = thd_pre
    model_forward.postprocess_thd_engine = thd_post
    model_forward.preprocess_bshd_engine = bshd_pre
    model_forward.postprocess_bshd_engine = bshd_post
    try:
        yield records
    finally:
        model_forward.preprocess_thd_engine = originals["thd_pre"]
        model_forward.postprocess_thd_engine = originals["thd_post"]
        model_forward.preprocess_bshd_engine = originals["bshd_pre"]
        model_forward.postprocess_bshd_engine = originals["bshd_post"]


def _run_engine_path(model, input_ids, *, use_remove_padding, runtime):
    import verl.models.mcore.model_forward as model_forward

    data_format = "thd" if use_remove_padding else "bshd"
    model.zero_grad(set_to_none=True)
    with _verl_data_path_probe(input_ids) as path_probe:
        output = model_forward.gptmodel_forward_model_engine(
            model,
            input_ids=input_ids,
            multi_modal_inputs={},
            logits_processor=_logits_processor,
            logits_processor_args={"label": input_ids},
            value_model=False,
            vision_model=False,
            pad_token_id=0,
            data_format=data_format,
        )
    expected_prefix = "thd" if use_remove_padding else "bshd"
    other_prefix = "bshd" if use_remove_padding else "thd"
    if path_probe[f"{expected_prefix}_pre"] == 0 or path_probe[f"{expected_prefix}_post"] == 0:
        raise AssertionError(f"verl did not execute its {expected_prefix.upper()} preprocess/postprocess")
    if path_probe[f"{other_prefix}_pre"] or path_probe[f"{other_prefix}_post"]:
        raise AssertionError(f"verl unexpectedly entered its {other_prefix.upper()} data path")
    if use_remove_padding and path_probe["metadata"] is None:
        raise AssertionError("THD preprocess did not expose packed boundary metadata")
    target_logprob = output["target_logprob"]
    output_probe = output["output_probe"]
    if not target_logprob.is_nested or not output_probe.is_nested:
        raise AssertionError(f"{data_format} postprocess did not return jagged NestedTensors")
    logical_lengths = tuple(int(value) for value in target_logprob.offsets().diff().tolist())
    probe_lengths = tuple(int(value) for value in output_probe.offsets().diff().tolist())
    expected_lengths = tuple(int(value) for value in input_ids.offsets().diff().tolist())
    if logical_lengths != expected_lengths or probe_lengths != expected_lengths:
        raise AssertionError(
            f"{data_format} postprocess restored target/probe lengths "
            f"{logical_lengths}/{probe_lengths}, expected {expected_lengths}"
        )
    # The preprocess rolls labels within each sequence. Exclude each final query,
    # whose rolled label points back to the first token rather than a next token.
    lengths = input_ids.offsets().diff().tolist()
    keep = torch.ones(target_logprob.values().numel(), dtype=torch.bool, device=runtime.device)
    offset = 0
    for length in lengths:
        keep[offset + length - 1] = False
        offset += length
    loss = -target_logprob.values()[keep].sum() / float(sum(lengths) - len(lengths))
    loss.backward()
    if runtime.world_size > 1:
        allreduce_parameter_gradients(model, runtime.cp_group)
    return {
        "target_logprob": target_logprob.values().detach().clone(),
        "output_probe": output_probe.values().detach().clone(),
        "loss": loss.detach().clone(),
        "path_probe": path_probe,
        "logical_lengths": logical_lengths,
    }


def test_complete_qwen35_thd_matches_bshd(runtime):
    torch.manual_seed(354001)
    bshd_model = make_qwen35_model(runtime, cp_size=runtime.world_size)
    broadcast_module_state(bshd_model, src=0)
    assert_hybrid_architecture(bshd_model)
    torch.manual_seed(354002)
    thd_model = make_qwen35_model(runtime, cp_size=runtime.world_size)
    thd_model.load_state_dict(bshd_model.state_dict(), strict=True)
    gdn_layers, fa_layers = assert_hybrid_architecture(thd_model)

    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    input_ids = _nested_tokens(runtime.device)
    with bind_stage1_gdn_primitives(mindspeed_gdn, bshd_model, thd_model) as binding:
        bshd = _run_engine_path(
            bshd_model,
            input_ids,
            use_remove_padding=False,
            runtime=runtime,
        )
        with (
            _packed_layer_probe(thd_model) as packed_calls,
            AllToAllProbe(mindspeed_gdn) as a2a_probe,
            _ring_probe() as ring_calls,
        ):
            thd = _run_engine_path(
                thd_model,
                input_ids,
                use_remove_padding=True,
                runtime=runtime,
            )

    if sorted(packed_calls["gdn"]) != [layer.layer_number for layer in gdn_layers]:
        raise AssertionError(f"not every GDN received PackedSeqParams: {packed_calls['gdn']}")
    if sorted(packed_calls["fa"]) != [layer.layer_number for layer in fa_layers]:
        raise AssertionError(f"not every Full Attention received PackedSeqParams: {packed_calls['fa']}")
    expected_packed_metadata = thd["path_probe"]["model_packed_metadata"]
    mismatched_metadata = [
        (layer_number, layer_kind, layer_metadata)
        for layer_number, layer_kind, layer_metadata in packed_calls["metadata"]
        if layer_metadata != expected_packed_metadata
    ]
    if not packed_calls["metadata"] or mismatched_metadata:
        raise AssertionError(
            "GDN/FA PackedSeqParams metadata differs from verl preprocess: "
            f"expected={expected_packed_metadata}, mismatches={mismatched_metadata}"
        )
    metadata = thd["path_probe"]["metadata"]
    if metadata.actual_cu_seqlens != (0, 29, 64):
        raise AssertionError(f"lost logical THD boundaries: {metadata.actual_cu_seqlens}")
    if metadata.segment_ids != (0, 1):
        raise AssertionError(f"lost packed segment identity: {metadata.segment_ids}")
    if runtime.world_size == 2:
        if a2a_probe.count("cp2hp") != 18 * 2 * 6 or a2a_probe.count("hp2cp") != 18 * 2:
            raise AssertionError(
                "THD CP=2 did not execute two packed GDN A2A segments per layer: "
                f"cp2hp={a2a_probe.count('cp2hp')}, hp2cp={a2a_probe.count('hp2cp')}"
            )
        if len(ring_calls) != 6:
            raise AssertionError(f"THD CP=2 expected six Full-Attention Ring calls, got {len(ring_calls)}")
    elif a2a_probe.calls or ring_calls:
        raise AssertionError("CP=1 unexpectedly entered a CP communication kernel")

    torch.testing.assert_close(thd["output_probe"], bshd["output_probe"], atol=8e-2, rtol=2e-2)
    torch.testing.assert_close(
        thd["target_logprob"], bshd["target_logprob"], atol=8e-2, rtol=2e-2
    )
    torch.testing.assert_close(thd["loss"], bshd["loss"], atol=2e-2, rtol=2e-2)
    assert_gradients_close(bshd_model, thd_model, rtol=1e-1, cosine_min=0.99)

    if runtime.rank == 0:
        print(
            "PUNCTURE-4 PASS"
            f"\n  CP: {runtime.world_size}"
            "\n  use_remove_padding: True (THD)"
            f"\n  architecture: {len(gdn_layers)} GDN + {len(fa_layers)} Full Attention"
            f"\n  causal-conv backend: {binding['causal_conv']}"
            f"\n  gated-delta backend: {binding['gated_delta_rule']}"
            f"\n  PackedSeqParams calls: GDN={len(packed_calls['gdn'])}, FA={len(packed_calls['fa'])}"
            f"\n  GDN A2A calls: cp2hp={a2a_probe.count('cp2hp')}, hp2cp={a2a_probe.count('hp2cp')}"
            f"\n  Full-Attention Ring calls: {len(ring_calls)}"
            f"\n  actual/padded cu_seqlens: {metadata.actual_cu_seqlens}/{metadata.padded_cu_seqlens}"
            f"\n  logical lengths after postprocess: {thd['logical_lengths']}"
            f"\n  loss BSHD/THD: {bshd['loss'].item():.8f}/{thd['loss'].item():.8f}"
            f"\n  verl THD preprocess/postprocess calls: "
            f"{thd['path_probe']['thd_pre']}/{thd['path_probe']['thd_post']}"
        )
