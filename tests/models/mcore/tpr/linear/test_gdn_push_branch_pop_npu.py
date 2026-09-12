# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage 3.2: single-GDN Push/Branch/Pop state-gradient relay on one NPU.

Run from the verl repository root::

    torchrun --master_addr=127.0.0.1 --master_port=29561 --nproc_per_node=1 \
      -m pytest -s -v tests/models/mcore/tpr/linear/test_gdn_push_branch_pop_npu.py

The materialized reference executes ``P+S1`` and ``P+S2`` independently. The
TPR path saves graph-free state for P, visits both siblings from independent
anchors, sums dConv/dRecurrent, then recomputes P and applies the summed VJP.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from verl.utils.device import is_torch_npu_available

from ...baseline._qwen35_baseline_utils import (
    DTYPE,
    HIDDEN_SIZE,
    assert_gradient_maps_close,
    destroy_npu_runtime,
    initialize_npu_runtime,
    process_groups,
    qwen35_config,
)


_PREFIX_LENGTH = 64
_SUFFIX_LENGTH = 64
_GDN_LAYER_NUMBER = 1
_OUTPUT_ATOL = 2e-2
_OUTPUT_RTOL = 2e-2
_GRAD_RELATIVE_L2_TOL = 8e-2
_GRAD_COSINE_MIN = 0.995


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=1)
    yield value
    destroy_npu_runtime(value)


def _make_gdn(runtime) -> nn.Module:
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules

    from verl.models.mcore.tpr.gated_delta_net import TPRGatedDeltaNet

    config = qwen35_config(cp_size=1)
    backend = LocalSpecProvider()
    submodules = GatedDeltaNetSubmodules(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )
    model = TPRGatedDeltaNet(
        config,
        submodules=submodules,
        layer_number=_GDN_LAYER_NUMBER,
        bias=False,
        conv_bias=False,
        conv_init=0.1,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=process_groups(runtime, cp_size=1),
    )
    model = model.to(device=runtime.device, dtype=DTYPE)
    model.train()
    return model


def _random_tensor(runtime, shape, *, seed, requires_grad=False) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value = torch.randn(shape, generator=generator, dtype=torch.float32)
    value = value.to(device=runtime.device, dtype=DTYPE)
    return value.requires_grad_(requires_grad)


def _run_segment(model, hidden_states, *, prefix_length, initial_states=None):
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context

    context = TPRAttentionContext(
        prefix_length=prefix_length,
        suffix_length=hidden_states.shape[0],
        initial_gdn_states={} if initial_states is None else initial_states,
    )
    with use_tpr_attention_context(context):
        output, output_bias = model(hidden_states, attention_mask=None)
    if output_bias is not None:
        raise AssertionError("bias-free GDN unexpectedly returned an output bias")
    context.assert_new_gdn_layers((_GDN_LAYER_NUMBER,))
    return output, context.new_gdn_states[_GDN_LAYER_NUMBER]


def _loss_part(output: Tensor, target: Tensor, *, denominator: float) -> Tensor:
    return (output.float() - target.float()).square().sum() / denominator


def _parameter_gradients(model) -> dict[str, Tensor]:
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing parameter gradient: {name}")
        if not torch.isfinite(parameter.grad).all().item():
            raise AssertionError(f"non-finite parameter gradient: {name}")
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _assert_gradient_tensor(reference: Tensor, actual: Tensor, *, label: str):
    reference_float = reference.float()
    actual_float = actual.float()
    difference = actual_float - reference_float
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        reference_float.reshape(1, -1),
        actual_float.reshape(1, -1),
    ).item()
    if relative_l2 > _GRAD_RELATIVE_L2_TOL or cosine < _GRAD_COSINE_MIN:
        raise AssertionError(
            f"{label} gradient mismatch: relative_l2={relative_l2:.6e}, cosine={cosine:.9f}"
        )
    return relative_l2, cosine


def _assert_finite_nonzero(tensor: Tensor, *, label: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise AssertionError(f"{label} contains NaN/Inf")
    if torch.count_nonzero(tensor).item() == 0:
        raise AssertionError(f"{label} is unexpectedly all zero")


def test_single_gdn_push_branch_pop_matches_materialized_paths(runtime):
    from verl.models.mcore.tpr import GDNLayerState, GDNPrefixState

    torch.manual_seed(32001)
    reference_model = _make_gdn(runtime)
    torch.manual_seed(32002)
    tpr_model = _make_gdn(runtime)
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)

    prefix_reference = _random_tensor(
        runtime,
        (_PREFIX_LENGTH, 1, HIDDEN_SIZE),
        seed=32101,
        requires_grad=True,
    )
    suffix_references = tuple(
        _random_tensor(
            runtime,
            (_SUFFIX_LENGTH, 1, HIDDEN_SIZE),
            seed=32102 + index,
            requires_grad=True,
        )
        for index in range(2)
    )
    prefix_tpr = prefix_reference.detach().clone().requires_grad_(True)
    suffix_tpr = tuple(value.detach().clone().requires_grad_(True) for value in suffix_references)
    prefix_target = _random_tensor(runtime, prefix_reference.shape, seed=32201)
    suffix_targets = tuple(
        _random_tensor(runtime, value.shape, seed=32202 + index)
        for index, value in enumerate(suffix_references)
    )
    denominator = float(2 * (_PREFIX_LENGTH + _SUFFIX_LENGTH) * HIDDEN_SIZE)

    reference_outputs = []
    reference_loss = torch.zeros((), dtype=torch.float32, device=runtime.device)
    for suffix, suffix_target in zip(suffix_references, suffix_targets, strict=True):
        full_hidden = torch.cat((prefix_reference, suffix), dim=0)
        full_output, _ = _run_segment(
            reference_model,
            full_hidden,
            prefix_length=0,
        )
        branch_loss = _loss_part(
            full_output[:_PREFIX_LENGTH],
            prefix_target,
            denominator=denominator,
        ) + _loss_part(
            full_output[_PREFIX_LENGTH:],
            suffix_target,
            denominator=denominator,
        )
        branch_loss.backward()
        reference_loss = reference_loss + branch_loss.detach()
        reference_outputs.append(full_output.detach().clone())
    reference_gradients = _parameter_gradients(reference_model)

    # Push(P): save graph-free final state and discard the activation graph.
    with torch.no_grad():
        pushed_output, pushed_layer_state = _run_segment(
            tpr_model,
            prefix_tpr,
            prefix_length=0,
        )
    prefix_state = GDNPrefixState.save(
        segment_id=0,
        sequence_length=_PREFIX_LENGTH,
        layer_states={_GDN_LAYER_NUMBER: pushed_layer_state},
    )
    if pushed_layer_state.conv_state.requires_grad or pushed_layer_state.recurrent_state.requires_grad:
        raise AssertionError("Push(P) unexpectedly retained a state autograd graph")
    saved_layer_state = prefix_state.layer_states[_GDN_LAYER_NUMBER]
    saved_conv_snapshot = saved_layer_state.conv_state.clone()
    saved_recurrent_snapshot = saved_layer_state.recurrent_state.clone()

    # Visit(S1/S2): each sibling restores independent anchors from the same P state.
    branch_state_gradients = []
    tpr_suffix_outputs = []
    tpr_suffix_loss = torch.zeros((), dtype=torch.float32, device=runtime.device)
    for suffix, suffix_target in zip(suffix_tpr, suffix_targets, strict=True):
        anchors = prefix_state.make_anchors()
        suffix_output, _ = _run_segment(
            tpr_model,
            suffix,
            prefix_length=_PREFIX_LENGTH,
            initial_states=anchors.layer_states,
        )
        suffix_loss = _loss_part(suffix_output, suffix_target, denominator=denominator)
        suffix_loss.backward()
        anchor_state = anchors.layer_states[_GDN_LAYER_NUMBER]
        if anchor_state.conv_state.grad is None or anchor_state.recurrent_state.grad is None:
            raise AssertionError("suffix backward did not produce both GDN state gradients")
        branch_gradient = GDNLayerState(
            anchor_state.conv_state.grad.detach().clone(),
            anchor_state.recurrent_state.grad.detach().clone(),
        )
        branch_state_gradients.append(branch_gradient)
        prefix_state.accumulate_anchor_gradients(anchors)
        tpr_suffix_outputs.append(suffix_output.detach().clone())
        tpr_suffix_loss = tpr_suffix_loss + suffix_loss.detach()

    torch.testing.assert_close(
        prefix_state.layer_states[_GDN_LAYER_NUMBER].conv_state,
        saved_conv_snapshot,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        prefix_state.layer_states[_GDN_LAYER_NUMBER].recurrent_state,
        saved_recurrent_snapshot,
        atol=0.0,
        rtol=0.0,
    )

    accumulated = prefix_state.gradients[_GDN_LAYER_NUMBER]
    expected_conv_gradient = sum(
        (gradient.conv_state for gradient in branch_state_gradients[1:]),
        start=branch_state_gradients[0].conv_state,
    )
    expected_recurrent_gradient = sum(
        (gradient.recurrent_state for gradient in branch_state_gradients[1:]),
        start=branch_state_gradients[0].recurrent_state,
    )
    torch.testing.assert_close(accumulated.conv_state, expected_conv_gradient, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        accumulated.recurrent_state,
        expected_recurrent_gradient,
        atol=0.0,
        rtol=0.0,
    )
    _assert_finite_nonzero(accumulated.conv_state, label="summed dConvState")
    _assert_finite_nonzero(accumulated.recurrent_state, label="summed dRecurrentState")

    # Pop(P): recompute P and apply its own loss plus the summed sibling-state VJP.
    relayed_gradient = prefix_state.consume_gradients()[_GDN_LAYER_NUMBER]
    recomputed_prefix_output, recomputed_prefix_state = _run_segment(
        tpr_model,
        prefix_tpr,
        prefix_length=0,
    )
    prefix_loss = 2.0 * _loss_part(
        recomputed_prefix_output,
        prefix_target,
        denominator=denominator,
    )
    torch.autograd.backward(
        (
            prefix_loss,
            recomputed_prefix_state.conv_state,
            recomputed_prefix_state.recurrent_state,
        ),
        grad_tensors=(
            None,
            relayed_gradient.conv_state,
            relayed_gradient.recurrent_state,
        ),
    )
    tpr_loss = prefix_loss.detach() + tpr_suffix_loss
    prefix_state.release()
    if not prefix_state.released:
        raise AssertionError("Pop(P) did not release its saved GDN prefix state")

    for reference_output, suffix_output in zip(
        reference_outputs,
        tpr_suffix_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            suffix_output.float(),
            reference_output[_PREFIX_LENGTH:].float(),
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
    torch.testing.assert_close(
        recomputed_prefix_output.float(),
        reference_outputs[0][:_PREFIX_LENGTH].float(),
        atol=_OUTPUT_ATOL,
        rtol=_OUTPUT_RTOL,
    )
    torch.testing.assert_close(
        pushed_output.float(),
        recomputed_prefix_output.detach().float(),
        atol=_OUTPUT_ATOL,
        rtol=_OUTPUT_RTOL,
    )
    torch.testing.assert_close(tpr_loss, reference_loss, atol=2e-3, rtol=2e-3)

    prefix_input_metrics = _assert_gradient_tensor(
        prefix_reference.grad,
        prefix_tpr.grad,
        label="prefix input",
    )
    suffix_input_metrics = tuple(
        _assert_gradient_tensor(reference.grad, actual.grad, label=f"S{index} input")
        for index, (reference, actual) in enumerate(
            zip(suffix_references, suffix_tpr, strict=True),
            start=1,
        )
    )
    parameter_metrics = assert_gradient_maps_close(
        reference_gradients,
        _parameter_gradients(tpr_model),
        rtol=_GRAD_RELATIVE_L2_TOL,
        cosine_min=_GRAD_COSINE_MIN,
    )

    if runtime.rank == 0:
        print(
            "STAGE-3.2 SINGLE-GDN PUSH/BRANCH/POP PASS"
            "\n  topology: P -> {S1, S2}"
            f"\n  loss materialized/TPR: {reference_loss.item():.9f}/{tpr_loss.item():.9f}"
            f"\n  prefix-input grad rel/cos: {prefix_input_metrics[0]:.6e}/"
            f"{prefix_input_metrics[1]:.9f}"
            f"\n  S1-input grad rel/cos: {suffix_input_metrics[0][0]:.6e}/"
            f"{suffix_input_metrics[0][1]:.9f}"
            f"\n  S2-input grad rel/cos: {suffix_input_metrics[1][0]:.6e}/"
            f"{suffix_input_metrics[1][1]:.9f}"
            f"\n  parameter grad rel/cos: {parameter_metrics[0]:.6e}/{parameter_metrics[1]:.9f}"
            f"\n  summed dConvState norm: {relayed_gradient.conv_state.float().norm().item():.6e}"
            f"\n  summed dRecurrentState norm: "
            f"{relayed_gradient.recurrent_state.float().norm().item():.6e}"
        )
