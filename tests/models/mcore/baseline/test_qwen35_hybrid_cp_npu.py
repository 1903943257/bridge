"""Puncture 3: random-init Qwen3.5-0.8B Hybrid CP=1 versus CP=2.

Run from the verl repository root with two NPUs::

    torchrun --master_addr=127.0.0.1 --master_port=29553 --nproc_per_node=2 \
      -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_hybrid_cp_npu.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available

from ._qwen35_baseline_utils import (
    AllToAllProbe,
    SEQUENCE_LENGTH,
    allreduce_parameter_gradients,
    assert_gradient_maps_close,
    assert_hybrid_architecture,
    assert_tensor_close_by_norm,
    assert_tensor_gradient_close,
    broadcast_module_state,
    clone_parameter_gradients,
    destroy_npu_runtime,
    full_tokens,
    gather_native_zigzag,
    gradient_map_diagnostics,
    initialize_npu_runtime,
    make_qwen35_model,
    selected_output_and_loss,
    zigzag_indices,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="run puncture 3 with torchrun --nproc_per_node=2",
)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=2)
    yield value
    destroy_npu_runtime(value)


@contextmanager
def _ring_probe():
    import mindspeed.core.context_parallel.dot_product_attention as dpa

    original = dpa.ringattn_context_parallel
    calls = []

    def traced(*args, **kwargs):
        calls.append((tuple(args[0].shape), tuple(args[1].shape)))
        return original(*args, **kwargs)

    dpa.ringattn_context_parallel = traced
    try:
        yield calls
    finally:
        dpa.ringattn_context_parallel = original


@contextmanager
def _capture_embedding_gradient(model):
    captured = {}

    def retain(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            raise AssertionError("embedding output is not a differentiable tensor")
        tensor.retain_grad()
        captured["tensor"] = tensor

    handle = model.embedding.register_forward_hook(retain)
    try:
        yield captured
    finally:
        handle.remove()


def _singleton_sequence(tensor):
    if tensor.ndim != 3:
        raise AssertionError(f"expected a 3-D embedding gradient, got {tuple(tensor.shape)}")
    if tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.shape[0] == 1:
        return tensor[0]
    raise AssertionError(f"cannot identify singleton batch dimension: {tuple(tensor.shape)}")


def _captured_embedding_gradient(captured, *, model_name):
    tensor = captured.get("tensor")
    if tensor is None:
        raise AssertionError(f"{model_name} embedding hook did not observe a forward output")
    if tensor.grad is None:
        raise AssertionError(f"{model_name} embedding output has no input gradient")
    return _singleton_sequence(tensor.grad)


def _backend_name(backend, *, fallback):
    if backend is None:
        return fallback
    return f"{backend.__module__}.{backend.__name__}"


def _parameter_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise AssertionError(f"missing parameter gradient: {name}")
        if parameter.grad is not None:
            gradients[name] = parameter.grad
    return gradients


def _gradient_categories(model, reference, actual):
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    gdn_prefixes = []
    fa_prefixes = []
    for index, layer in enumerate(model.decoder.layers):
        prefix = f"decoder.layers.{index}.self_attention."
        if isinstance(layer.self_attention, mindspeed_gdn.GatedDeltaNet):
            gdn_prefixes.append(prefix)
        else:
            fa_prefixes.append(prefix)

    predicates = {
        "GDN": lambda name: name.startswith(tuple(gdn_prefixes)),
        "Full-Attention": lambda name: name.startswith(tuple(fa_prefixes)),
        "MLP": lambda name: ".mlp." in name,
        "embedding/output": lambda name: name.startswith(("embedding.", "output_layer.")),
        "norms": lambda name: "norm" in name.lower(),
    }
    diagnostics = {}
    for category, predicate in predicates.items():
        names = [name for name in reference if predicate(name)]
        if not names:
            raise AssertionError(f"gradient category {category!r} is empty")
        category_reference = {name: reference[name] for name in names}
        category_actual = {name: actual[name] for name in names}
        diagnostics[category] = (
            len(names),
            gradient_map_diagnostics(category_reference, category_actual),
        )
    return diagnostics


def _format_gradient_metrics(metrics, *, actual_label="cp_norm"):
    return (
        f"ref_norm={metrics.reference_norm:.6e}, {actual_label}={metrics.actual_norm:.6e}, "
        f"norm_ratio={metrics.norm_ratio:.9f}, abs_l2={metrics.absolute_l2:.6e}, "
        f"rel_l2={metrics.relative_l2:.6e}, cosine={metrics.cosine:.9f}"
    )


def test_random_init_qwen35_hybrid_cp2_matches_cp1(runtime):
    if SEQUENCE_LENGTH % 4:
        raise AssertionError("STAGE34_SEQUENCE_LENGTH must be divisible by 4 for CP=2")
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    torch.manual_seed(353001)
    reference = make_qwen35_model(runtime, cp_size=1)
    broadcast_module_state(reference, src=0)
    reference_gdn, reference_fa = assert_hybrid_architecture(reference)
    assert all(type(layer.self_attention) is mindspeed_gdn.GatedDeltaNet for layer in reference_gdn)

    torch.manual_seed(353002)
    actual = make_qwen35_model(runtime, cp_size=2)
    actual.load_state_dict(reference.state_dict(), strict=True)
    actual_gdn, actual_fa = assert_hybrid_architecture(actual)
    assert len(actual_gdn) == 18 and len(actual_fa) == 6

    tokens, positions, labels, valid = full_tokens(runtime.device)
    reference.zero_grad(set_to_none=True)
    with (
        AllToAllProbe(mindspeed_gdn) as reference_a2a_probe,
        _ring_probe() as reference_ring_calls,
    ):
        with _capture_embedding_gradient(reference) as reference_embedding:
            reference_logits = reference(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=None,
            )
            reference_probe, reference_loss = selected_output_and_loss(
                reference_logits,
                labels,
                valid,
            )
            reference_loss.backward()
        reference_input_gradient = (
            _captured_embedding_gradient(reference_embedding, model_name="CP=1 reference")
            .detach()
            .clone()
        )
        reference_parameter_gradients = clone_parameter_gradients(reference)

        # Establish the full-model native-BF16 repeatability floor using the
        # exact same CP=1 model and input. No CP communication may occur here.
        reference.zero_grad(set_to_none=True)
        repeat_logits = reference(
            input_ids=tokens,
            position_ids=positions,
            attention_mask=None,
        )
        _, repeat_loss = selected_output_and_loss(repeat_logits, labels, valid)
        repeat_loss.backward()
        repeat_parameter_gradients = _parameter_gradients(reference)

    indices = zigzag_indices(
        SEQUENCE_LENGTH,
        cp_rank=runtime.rank,
        cp_size=2,
        device=runtime.device,
    )
    local_tokens = tokens.index_select(1, indices)
    local_positions = positions.index_select(1, indices)
    local_labels = labels.index_select(0, indices)
    local_valid = valid.index_select(0, indices)
    actual.zero_grad(set_to_none=True)
    with (
        AllToAllProbe(mindspeed_gdn) as a2a_probe,
        _ring_probe() as ring_calls,
        _capture_embedding_gradient(actual) as actual_embedding,
    ):
        actual_logits = actual(
            input_ids=local_tokens,
            position_ids=local_positions,
            attention_mask=None,
        )
        local_probe, local_loss = selected_output_and_loss(actual_logits, local_labels, local_valid)
        local_loss.backward()
    local_input_gradient = _captured_embedding_gradient(
        actual_embedding,
        model_name="CP=2 model",
    ).detach()

    if reference_a2a_probe.calls or reference_ring_calls:
        raise AssertionError(
            "CP=1 reference unexpectedly entered a CP communication kernel: "
            f"A2A={reference_a2a_probe.calls}, Ring={len(reference_ring_calls)}"
        )

    cp_loss = local_loss.detach().clone()
    dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    allreduce_parameter_gradients(actual, runtime.cp_group)
    actual_parameter_gradients = _parameter_gradients(actual)
    global_probe = gather_native_zigzag(local_probe.detach(), runtime.cp_group, seq_dim=0)
    global_input_gradient = gather_native_zigzag(
        local_input_gradient,
        runtime.cp_group,
        seq_dim=0,
    )

    cp2hp = a2a_probe.count("cp2hp")
    hp2cp = a2a_probe.count("hp2cp")
    if cp2hp != 18 * 6 or hp2cp != 18:
        raise AssertionError(
            f"expected 18 GDN A2A paths (108 cp2hp low-level calls, 18 hp2cp), "
            f"got cp2hp={cp2hp}, hp2cp={hp2cp}"
        )
    if len(ring_calls) != 6:
        raise AssertionError(f"expected six Full-Attention Ring calls, got {len(ring_calls)}")

    parameter_diagnostics = gradient_map_diagnostics(
        reference_parameter_gradients,
        actual_parameter_gradients,
    )
    repeat_diagnostics = gradient_map_diagnostics(
        reference_parameter_gradients,
        repeat_parameter_gradients,
    )
    category_diagnostics = _gradient_categories(
        reference,
        reference_parameter_gradients,
        actual_parameter_gradients,
    )
    if runtime.rank == 0:
        category_lines = "".join(
            f"\n  {category} ({count} tensors): "
            f"{_format_gradient_metrics(diagnostics.aggregate)}"
            for category, (count, diagnostics) in category_diagnostics.items()
        )
        print(
            "PUNCTURE-3 GRADIENT DIAGNOSTICS"
            "\n  communication: CP1 A2A=0/Ring=0, "
            f"CP2 cp2hp={cp2hp}/hp2cp={hp2cp}/Ring={len(ring_calls)}"
            f"\n  global: {_format_gradient_metrics(parameter_diagnostics.aggregate)}"
            f"\n  worst tensor: {parameter_diagnostics.worst_name}: "
            f"{_format_gradient_metrics(parameter_diagnostics.worst)}"
            f"\n  CP1 repeat: "
            f"{_format_gradient_metrics(repeat_diagnostics.aggregate, actual_label='repeat_norm')}"
            f"\n  CP1 loss/repeat loss: {reference_loss.item():.8f}/{repeat_loss.item():.8f}"
            f"{category_lines}"
        )

    # A per-element relative tolerance is unstable for sampled logits near zero.
    # Use aggregate direction/magnitude checks, while retaining an absolute guard
    # against a localized token/layout error. Loss and complete gradients below
    # remain independent end-to-end equivalence checks.
    output_metrics = assert_tensor_close_by_norm(
        reference_probe,
        global_probe,
        rtol=8e-2,
        cosine_min=0.995,
        max_abs=2e-1,
        label="output probe",
    )
    torch.testing.assert_close(cp_loss, reference_loss.detach(), atol=2e-2, rtol=2e-2)
    input_gradient_metrics = assert_tensor_gradient_close(
        reference_input_gradient,
        global_input_gradient,
        rtol=1e-1,
        cosine_min=0.99,
    )
    # The 10% full-model BF16 bound is calibrated by the diagnostics above:
    # CP1 repeatability is non-zero, norm ratio remains near one, and no
    # parameter category shows a localized divergence.
    parameter_gradient_metrics = assert_gradient_maps_close(
        reference_parameter_gradients,
        actual_parameter_gradients,
        rtol=1e-1,
        cosine_min=0.995,
    )

    causal_conv_backend = _backend_name(
        mindspeed_gdn.causal_conv1d,
        fallback="torch.nn.functional.conv1d",
    )
    gated_delta_backend = _backend_name(
        actual_gdn[0].gated_delta_rule,
        fallback="<missing>",
    )

    if runtime.rank == 0:
        print(
            "PUNCTURE-3 PASS"
            "\n  use_remove_padding: False (BSHD)"
            "\n  architecture: 18 GDN + 6 Full Attention"
            f"\n  native causal-conv backend: {causal_conv_backend}"
            f"\n  native gated-delta backend: {gated_delta_backend}"
            "\n  CP=1 communication calls: A2A=0, Ring=0"
            f"\n  GDN A2A low-level calls: cp2hp={cp2hp}, hp2cp={hp2cp}"
            f"\n  Full-Attention Ring calls: {len(ring_calls)}"
            f"\n  output rel/cos/max-abs: "
            f"{output_metrics[0]:.6e}/{output_metrics[1]:.9f}/{output_metrics[2]:.6e}"
            f"\n  loss CP1/CP2: {reference_loss.item():.8f}/{cp_loss.item():.8f}"
            f"\n  input-grad rel/cos: {input_gradient_metrics[0]:.6e}/{input_gradient_metrics[1]:.9f}"
            f"\n  parameter-grad rel/cos: "
            f"{parameter_gradient_metrics[0]:.6e}/{parameter_gradient_metrics[1]:.9f}"
            f"\n  worst parameter gradient: {parameter_gradient_metrics[2][1]}:"
            f"{parameter_gradient_metrics[2][0]:.6e}"
        )
