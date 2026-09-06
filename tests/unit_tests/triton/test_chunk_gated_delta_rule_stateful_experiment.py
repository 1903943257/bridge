# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from mindspeed_ops.api.triton.chunk_gated_delta_rule import chunk_gated_delta_rule
from mindspeed_ops.api.triton.utils import get_available_device
from mindspeed_ops.utils import is_arch35
from tests.unit_tests.triton.test_chunk_gated_delta_rule import ref_torch_chunk_gated_delta_rule
from tests.utils import assert_close


CHUNK_SIZE = 64
INPUT_NAMES = ("q", "k", "v", "g", "beta")


@pytest.fixture(autouse=True)
def _require_arch32_npu():
    if get_available_device() != "npu":
        pytest.skip("This stateful GDR experiment requires an NPU Triton backend")
    if is_arch35():
        pytest.skip("This experiment targets the arch32 GDR implementation")


def _make_inputs(sequence_length, *, seed=42):
    torch.manual_seed(seed)
    batch_size, num_heads, key_dim, value_dim = 1, 2, 64, 64
    shape = (batch_size, sequence_length, num_heads)

    q = F.normalize(
        torch.randn(*shape, key_dim, dtype=torch.bfloat16, device="npu"), dim=-1
    ).requires_grad_(True)
    k = F.normalize(
        torch.randn(*shape, key_dim, dtype=torch.bfloat16, device="npu"), dim=-1
    ).requires_grad_(True)
    v = torch.randn(*shape, value_dim, dtype=torch.bfloat16, device="npu", requires_grad=True)
    g = F.logsigmoid(torch.randn(*shape, dtype=torch.bfloat16, device="npu")).detach().requires_grad_(True)
    beta = torch.sigmoid(torch.randn(*shape, dtype=torch.bfloat16, device="npu")).detach().requires_grad_(True)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta}


def _make_initial_state(*, seed=1729, requires_grad=False):
    torch.manual_seed(seed)
    state = torch.randn(1, 2, 64, 64, dtype=torch.float32, device="npu") * 0.1
    return state.requires_grad_(requires_grad)


def _clone_inputs(inputs):
    return {name: tensor.detach().clone().requires_grad_(True) for name, tensor in inputs.items()}


def _slice_inputs(inputs, start, end):
    return {
        name: tensor[:, start:end].detach().clone().requires_grad_(True)
        for name, tensor in inputs.items()
    }


def _run(inputs, *, initial_state=None, output_final_state=False):
    return chunk_gated_delta_rule(
        **inputs,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=CHUNK_SIZE,
        head_first=False,
    )


def _run_reference(inputs, *, initial_state=None, output_final_state=False):
    return ref_torch_chunk_gated_delta_rule(
        **inputs,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=CHUNK_SIZE,
    )


def _assert_bf16_close(name, reference, actual):
    assert_close(name, reference, actual, ratio=1e-2, err_atol=1e-3)


def _assert_finite_nonzero(name, tensor):
    assert tensor is not None, f"{name} was not produced"
    assert torch.isfinite(tensor).all(), f"{name} contains a non-finite value"
    assert torch.count_nonzero(tensor).item() > 0, f"{name} is unexpectedly all zero"


def test_stateful_forward_smoke():
    inputs = _make_inputs(CHUNK_SIZE)
    initial_state = _make_initial_state()

    output, final_state = _run(inputs, initial_state=initial_state, output_final_state=True)

    assert output.shape == (1, CHUNK_SIZE, 2, 64)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert final_state is not None
    assert final_state.shape == initial_state.shape
    assert final_state.dtype == torch.float32
    assert torch.isfinite(final_state).all()


@pytest.mark.parametrize(("prefix_length", "suffix_length"), [(64, 64), (128, 64)])
def test_forward_continuation_equivalence(prefix_length, suffix_length):
    full_inputs = _make_inputs(prefix_length + suffix_length)
    initial_state = _make_initial_state()

    with torch.no_grad():
        full_output, full_final_state = _run(
            full_inputs,
            initial_state=initial_state,
            output_final_state=True,
        )
        prefix_output, prefix_state = _run(
            _slice_inputs(full_inputs, 0, prefix_length),
            initial_state=initial_state,
            output_final_state=True,
        )
        del prefix_output
        suffix_output, segmented_final_state = _run(
            _slice_inputs(full_inputs, prefix_length, prefix_length + suffix_length),
            initial_state=prefix_state,
            output_final_state=True,
        )
        reference_output, reference_final_state = _run_reference(
            full_inputs,
            initial_state=initial_state,
            output_final_state=True,
        )

    _assert_bf16_close("segmented suffix output", full_output[:, prefix_length:], suffix_output)
    _assert_bf16_close("segmented final state", full_final_state, segmented_final_state)
    _assert_bf16_close("full output vs torch reference", reference_output, full_output)
    _assert_bf16_close("full final state vs torch reference", reference_final_state, full_final_state)


def test_initial_state_gradient_matches_torch_reference():
    inputs = _make_inputs(CHUNK_SIZE)
    reference_inputs = _clone_inputs(inputs)
    initial_state = _make_initial_state(requires_grad=True)
    reference_initial_state = initial_state.detach().clone().requires_grad_(True)

    output, _ = _run(inputs, initial_state=initial_state)
    reference_output, _ = _run_reference(reference_inputs, initial_state=reference_initial_state)
    output_gradient = torch.randn_like(output)

    output.backward(output_gradient)
    reference_output.backward(output_gradient)

    _assert_finite_nonzero("dh0", initial_state.grad)
    _assert_bf16_close("dh0 vs torch reference", reference_initial_state.grad, initial_state.grad)


def test_final_state_gradient_matches_torch_reference():
    inputs = _make_inputs(CHUNK_SIZE)
    reference_inputs = _clone_inputs(inputs)
    initial_state = _make_initial_state(requires_grad=True)
    reference_initial_state = initial_state.detach().clone().requires_grad_(True)

    output, final_state = _run(inputs, initial_state=initial_state, output_final_state=True)
    reference_output, reference_final_state = _run_reference(
        reference_inputs,
        initial_state=reference_initial_state,
        output_final_state=True,
    )
    output_gradient = torch.randn_like(output)
    final_state_gradient = torch.randn_like(final_state)

    torch.autograd.backward((output, final_state), (output_gradient, final_state_gradient))
    torch.autograd.backward(
        (reference_output, reference_final_state),
        (output_gradient, final_state_gradient),
    )

    _assert_finite_nonzero("dh0 with dht", initial_state.grad)
    _assert_bf16_close("dh0 with dht vs torch reference", reference_initial_state.grad, initial_state.grad)
    for name in INPUT_NAMES:
        _assert_finite_nonzero(f"d{name}", inputs[name].grad)
        _assert_bf16_close(
            f"d{name} with dht vs torch reference",
            reference_inputs[name].grad,
            inputs[name].grad,
        )


def test_tpr_prefix_suffix_backward_equivalence():
    prefix_length, suffix_length = 64, 64
    full_inputs = _make_inputs(prefix_length + suffix_length)
    segmented_source = _clone_inputs(full_inputs)
    output_gradient = torch.randn(1, suffix_length, 2, 64, dtype=torch.bfloat16, device="npu")

    full_output, _ = _run(full_inputs)
    full_output[:, prefix_length:].backward(output_gradient)

    prefix_push_inputs = _slice_inputs(segmented_source, 0, prefix_length)
    with torch.no_grad():
        _, detached_prefix_state = _run(prefix_push_inputs, output_final_state=True)

    suffix_inputs = _slice_inputs(segmented_source, prefix_length, prefix_length + suffix_length)
    suffix_initial_state = detached_prefix_state.detach().requires_grad_(True)
    suffix_output, _ = _run(suffix_inputs, initial_state=suffix_initial_state)
    suffix_output.backward(output_gradient)
    _assert_finite_nonzero("suffix dPrefixState", suffix_initial_state.grad)

    prefix_recompute_inputs = _slice_inputs(segmented_source, 0, prefix_length)
    _, recomputed_prefix_state = _run(prefix_recompute_inputs, output_final_state=True)
    recomputed_prefix_state.backward(suffix_initial_state.grad)

    for name in INPUT_NAMES:
        segmented_gradient = torch.cat(
            (prefix_recompute_inputs[name].grad, suffix_inputs[name].grad),
            dim=1,
        )
        _assert_bf16_close(f"TPR d{name} vs full", full_inputs[name].grad, segmented_gradient)
