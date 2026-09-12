"""Puncture 4: complete Qwen3.5 THD chain at CP=1 and CP=2.

Run this file twice from the verl repository root::

    torchrun --master_addr=127.0.0.1 --master_port=29554 --nproc_per_node=1 \
      -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_thd_cp_npu.py
    torchrun --master_addr=127.0.0.1 --master_port=29555 --nproc_per_node=2 \
      -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_thd_cp_npu.py

The BSHD CP=1 reference and THD target use separate models with identical
random parameters. This exercises verl preprocessing and postprocessing, all
24 hybrid layers, target-logprob loss, and complete parameter gradients
without enabling TPR.
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
def _parallel_state_cp_override(runtime, *, cp_size):
    """Run a verl preprocessing helper with a model-local CP topology."""
    if cp_size == runtime.world_size:
        yield
        return
    if cp_size != 1:
        raise ValueError(f"unsupported CP override: {cp_size}")

    from megatron.core import parallel_state

    originals = {
        "world_size": parallel_state.get_context_parallel_world_size,
        "rank": parallel_state.get_context_parallel_rank,
        "group": parallel_state.get_context_parallel_group,
    }
    parallel_state.get_context_parallel_world_size = lambda: 1
    parallel_state.get_context_parallel_rank = lambda: 0
    parallel_state.get_context_parallel_group = lambda: runtime.tp_group
    try:
        yield
    finally:
        parallel_state.get_context_parallel_world_size = originals["world_size"]
        parallel_state.get_context_parallel_rank = originals["rank"]
        parallel_state.get_context_parallel_group = originals["group"]


@contextmanager
def _bshd_dense_attention_mask_mode(model, *, enabled):
    """Use general-mask FA mode for verl's explicit B1SS padding mask."""
    if not enabled:
        yield None
        return

    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    targets = {}
    for layer in model.decoder.layers:
        if isinstance(layer.self_attention, mindspeed_gdn.GatedDeltaNet):
            continue
        for module in layer.self_attention.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "sparse_mode"):
                targets.setdefault(
                    id(config),
                    (config, config.sparse_mode, getattr(config, "attention_mask_type", None)),
                )
    # core_v0.16.1 uses the config-bound MindSpeed TE attention path, whose
    # backend recomputes sparse_mode from attention_mask_type on every call.
    # An older patched entry reads both options from MindSpeed's global args.
    # Set both runtime objects so this test remains exact for the server stack.
    from mindspeed.args_utils import get_full_args

    mindspeed_args = get_full_args()
    if hasattr(mindspeed_args, "sparse_mode"):
        targets.setdefault(
            id(mindspeed_args),
            (
                mindspeed_args,
                mindspeed_args.sparse_mode,
                getattr(mindspeed_args, "attention_mask_type", None),
            ),
        )
    if not targets:
        raise AssertionError("could not locate Full-Attention sparse_mode configuration")

    for target, _, original_mask_type in targets.values():
        target.sparse_mode = 0
        if original_mask_type is not None:
            target.attention_mask_type = "no_mask"
    try:
        if any(target.sparse_mode != 0 for target, _, _ in targets.values()):
            raise AssertionError("failed to select sparse_mode=0 for the BSHD dense mask")
        if any(
            original_mask_type is not None and target.attention_mask_type != "no_mask"
            for target, _, original_mask_type in targets.values()
        ):
            raise AssertionError("failed to select no_mask mode for the explicit BSHD dense mask")
        yield {
            "target_count": len(targets),
            "sparse_mode": 0,
            "attention_mask_type": "no_mask",
        }
    finally:
        for target, original_sparse_mode, original_mask_type in targets.values():
            target.sparse_mode = original_sparse_mode
            if original_mask_type is not None:
                target.attention_mask_type = original_mask_type


@contextmanager
def _verl_data_path_probe(input_ids, *, runtime, bshd_cp_size):
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
        "bshd_attention_mask_shape": None,
        "bshd_fa_sparse_mode": None,
        "bshd_fa_attention_mask_type": None,
        "model_packed_metadata": None,
        "metadata": None,
        "thd_post_dtypes": [],
        "bshd_post_dtypes": [],
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
        input_value = args[0] if args else kwargs["output"]
        result = originals["thd_post"](*args, **kwargs)
        records["thd_post_dtypes"].append((str(input_value.dtype), str(result.dtype)))
        return result

    def bshd_pre(*args, **kwargs):
        with _parallel_state_cp_override(runtime, cp_size=bshd_cp_size):
            result = originals["bshd_pre"](*args, **kwargs)
        records["bshd_pre"] += 1
        if records["bshd_attention_mask_shape"] is None:
            records["bshd_attention_mask_shape"] = tuple(result[1].shape)
        return result

    def bshd_post(*args, **kwargs):
        records["bshd_post"] += 1
        input_value = args[0] if args else kwargs["output"]
        with _parallel_state_cp_override(runtime, cp_size=bshd_cp_size):
            result = originals["bshd_post"](*args, **kwargs)
        records["bshd_post_dtypes"].append((str(input_value.dtype), str(result.dtype)))
        return result

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


def _run_engine_path(model, input_ids, *, use_remove_padding, model_cp_size, runtime):
    import verl.models.mcore.model_forward as model_forward

    data_format = "thd" if use_remove_padding else "bshd"
    model.zero_grad(set_to_none=True)
    with _bshd_dense_attention_mask_mode(
        model,
        enabled=not use_remove_padding,
    ) as mask_mode:
        with _verl_data_path_probe(
            input_ids,
            runtime=runtime,
            bshd_cp_size=model_cp_size,
        ) as path_probe:
            if mask_mode is not None:
                path_probe["bshd_fa_sparse_mode"] = mask_mode["sparse_mode"]
                path_probe["bshd_fa_attention_mask_type"] = mask_mode["attention_mask_type"]
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
    if not use_remove_padding:
        mask_shape = path_probe["bshd_attention_mask_shape"]
        if mask_shape is None or len(mask_shape) != 4 or mask_shape[:2] != (input_ids.shape[0], 1):
            raise AssertionError(f"verl did not build the expected B1SS BSHD mask: {mask_shape}")
        if path_probe["bshd_fa_sparse_mode"] != 0:
            raise AssertionError("BSHD dense attention mask did not execute with sparse_mode=0")
        if path_probe["bshd_fa_attention_mask_type"] != "no_mask":
            raise AssertionError("BSHD dense attention mask did not execute with no_mask mode")
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
    # Each sequence's final query has no next-token target. THD and padded BSHD
    # preprocessing may use different wrap/padding sentinels there, so exclude
    # exactly the same logical boundary positions from both loss and logprob
    # equivalence. Model output probes still cover every logical token.
    lengths = input_ids.offsets().diff().tolist()
    keep = torch.ones(target_logprob.values().numel(), dtype=torch.bool, device=runtime.device)
    offset = 0
    excluded_target_indices = []
    for length in lengths:
        excluded_index = offset + length - 1
        keep[excluded_index] = False
        excluded_target_indices.append(excluded_index)
        offset += length
    valid_target_logprob = target_logprob.values()[keep]
    loss = -valid_target_logprob.sum() / float(sum(lengths) - len(lengths))
    # Keep the same explicit-mask mode active for custom FA backward wrappers
    # that consult runtime config in addition to their saved autograd context.
    with _bshd_dense_attention_mask_mode(model, enabled=not use_remove_padding):
        loss.backward()
    if model_cp_size > 1:
        allreduce_parameter_gradients(model, runtime.cp_group)
    return {
        "target_logprob": valid_target_logprob.detach().clone(),
        "excluded_target_indices": tuple(excluded_target_indices),
        "output_probe": output_probe.values().detach().clone(),
        "loss": loss.detach().clone(),
        "path_probe": path_probe,
        "logical_lengths": logical_lengths,
    }


def test_complete_qwen35_thd_matches_bshd(runtime):
    torch.manual_seed(354001)
    bshd_model = make_qwen35_model(runtime, cp_size=1)
    broadcast_module_state(bshd_model, src=0)
    assert_hybrid_architecture(bshd_model)
    torch.manual_seed(354002)
    thd_model = make_qwen35_model(runtime, cp_size=runtime.world_size)
    thd_model.load_state_dict(bshd_model.state_dict(), strict=True)
    gdn_layers, fa_layers = assert_hybrid_architecture(thd_model)

    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    input_ids = _nested_tokens(runtime.device)
    with bind_stage1_gdn_primitives(mindspeed_gdn, bshd_model, thd_model) as binding:
        with (
            AllToAllProbe(mindspeed_gdn) as reference_a2a_probe,
            _ring_probe() as reference_ring_calls,
        ):
            bshd = _run_engine_path(
                bshd_model,
                input_ids,
                use_remove_padding=False,
                model_cp_size=1,
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
                model_cp_size=runtime.world_size,
                runtime=runtime,
            )

    if reference_a2a_probe.calls or reference_ring_calls:
        raise AssertionError(
            "CP=1 BSHD reference unexpectedly entered a CP communication kernel: "
            f"A2A={reference_a2a_probe.calls}, Ring={len(reference_ring_calls)}"
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

    expected_excluded_targets = (28, 63)
    if bshd["excluded_target_indices"] != expected_excluded_targets:
        raise AssertionError(
            f"BSHD excluded wrong next-token boundaries: {bshd['excluded_target_indices']}"
        )
    if thd["excluded_target_indices"] != expected_excluded_targets:
        raise AssertionError(
            f"THD excluded wrong next-token boundaries: {thd['excluded_target_indices']}"
        )

    # verl 0.16's CP>1 THD postprocessor reconstructs zigzag shards in a
    # default-dtype temporary, promoting a BF16 probe to FP32. Compare model
    # values at a common precision and retain the dtype transition in the
    # path probe so this postprocess behavior remains visible.
    torch.testing.assert_close(
        thd["output_probe"].float(),
        bshd["output_probe"].float(),
        atol=8e-2,
        rtol=2e-2,
    )
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
            "\n  CP=1 BSHD reference communication: A2A=0, Ring=0"
            f"\n  GDN A2A calls: cp2hp={a2a_probe.count('cp2hp')}, hp2cp={a2a_probe.count('hp2cp')}"
            f"\n  Full-Attention Ring calls: {len(ring_calls)}"
            f"\n  actual/padded cu_seqlens: {metadata.actual_cu_seqlens}/{metadata.padded_cu_seqlens}"
            f"\n  logical lengths after postprocess: {thd['logical_lengths']}"
            f"\n  excluded next-token boundaries: {thd['excluded_target_indices']}"
            f"\n  loss BSHD/THD: {bshd['loss'].item():.8f}/{thd['loss'].item():.8f}"
            f"\n  BSHD mask/sparse-mode/mask-type: "
            f"{bshd['path_probe']['bshd_attention_mask_shape']}/"
            f"{bshd['path_probe']['bshd_fa_sparse_mode']}/"
            f"{bshd['path_probe']['bshd_fa_attention_mask_type']}"
            f"\n  postprocess input/output dtypes: "
            f"BSHD={bshd['path_probe']['bshd_post_dtypes']}, "
            f"THD={thd['path_probe']['thd_post_dtypes']}"
            f"\n  verl THD preprocess/postprocess calls: "
            f"{thd['path_probe']['thd_pre']}/{thd['path_probe']['thd_post']}"
        )
