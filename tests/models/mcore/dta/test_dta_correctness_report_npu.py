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

"""Opt-in 0.566B Engine-level DTA correctness report."""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

import test_dta_engine_profile_npu as profile
from test_dta_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_plan,
    _reference_data,
)
from test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.dta import DTA_REQUEST_KEY, DTAForwardBackwardRequest, SegmentExecutor
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_CORRECTNESS_REPORT") != "1",
    reason="Set DTA_RUN_CORRECTNESS_REPORT=1 for the 0.566B correctness report",
)

_PREFIX_LENGTH = 1024
_SUFFIX_LENGTH = 512
_SIBLING_COUNT = 2
_LOGPROB_ATOL = 3e-2
_LOGPROB_RTOL = 3e-2
_LOSS_ATOL = 2e-2
_LOSS_RTOL = 2e-2
_GLOBAL_GRAD_RELATIVE_L2_TOL = 2e-2
_PER_PARAMETER_GRAD_RELATIVE_L2_TOL = 5e-2
_GLOBAL_GRAD_COSINE_MIN = 0.999


def _tokens(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device) % profile._PROFILE_VOCAB_SIZE


def _target_logprobs(logits, targets):
    return F.log_softmax(logits.detach().float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _logprob_metrics(actual, expected):
    difference = actual - expected
    expected_norm = torch.linalg.vector_norm(expected).clamp_min(1e-12)
    actual_norm = torch.linalg.vector_norm(actual).clamp_min(1e-12)
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / expected_norm).item(),
        "cosine": (torch.dot(actual.flatten(), expected.flatten()) / (actual_norm * expected_norm)).item(),
    }


def _gradient_snapshot(model):
    snapshot = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}"
        snapshot[name] = parameter.grad.detach().cpu().clone()
    return snapshot


def _gradient_metrics(actual_gradients, expected_gradients):
    assert actual_gradients.keys() == expected_gradients.keys()

    difference_sq = 0.0
    actual_sq = 0.0
    expected_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    worst_name = None
    worst_relative_l2 = -1.0

    for name, expected_gradient in expected_gradients.items():
        actual_gradient = actual_gradients[name]
        assert torch.isfinite(actual_gradient).all().item(), f"actual non-finite gradient: {name}"
        assert torch.isfinite(expected_gradient).all().item(), f"expected non-finite gradient: {name}"

        actual_float = actual_gradient.float()
        expected_float = expected_gradient.float()
        difference = actual_float - expected_float
        parameter_difference_sq = torch.sum(difference * difference).item()
        parameter_expected_sq = torch.sum(expected_float * expected_float).item()
        parameter_relative_l2 = math.sqrt(parameter_difference_sq) / max(
            math.sqrt(parameter_expected_sq), 1e-12
        )
        assert parameter_relative_l2 <= _PER_PARAMETER_GRAD_RELATIVE_L2_TOL, (
            f"gradient {name}: relative_l2={parameter_relative_l2:.6g}"
        )
        if parameter_relative_l2 > worst_relative_l2:
            worst_name = name
            worst_relative_l2 = parameter_relative_l2

        difference_sq += parameter_difference_sq
        actual_sq += torch.sum(actual_float * actual_float).item()
        expected_sq += parameter_expected_sq
        dot += torch.sum(actual_float * expected_float).item()
        max_abs = max(max_abs, difference.abs().max().item())

    relative_l2 = math.sqrt(difference_sq) / max(math.sqrt(expected_sq), 1e-12)
    cosine = dot / max(math.sqrt(actual_sq * expected_sq), 1e-12)
    return {
        "parameter_count": len(actual_gradients),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs": max_abs,
        "worst_name": worst_name,
        "worst_relative_l2": worst_relative_l2,
    }


def test_dta_correctness_report(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    full_length = _PREFIX_LENGTH + _SUFFIX_LENGTH

    reference_model = profile._make_model(
        device,
        dta=False,
        max_sequence_length=full_length,
        core_attention_module=profile._ProfileFusedCausalAttention,
        model_shape=profile._PROFILE_MODEL_SHAPE,
    )
    dta_model = profile._make_model(
        device,
        dta=True,
        max_sequence_length=full_length,
        core_attention_module=profile._ProfileFusedCausalAttention,
        model_shape=profile._PROFILE_MODEL_SHAPE,
    )
    dta_model.load_state_dict(reference_model.state_dict(), strict=True)
    parameter_count = profile._assert_profile_model_scale(reference_model)
    assert profile._assert_profile_model_scale(dta_model) == parameter_count
    _configure_model_runtime(reference_model)
    _configure_model_runtime(dta_model)

    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffixes = tuple(
        _tokens(700 + sibling * 1300, _SUFFIX_LENGTH, device)
        for sibling in range(_SIBLING_COUNT)
    )
    trajectories = tuple(torch.cat((prefix, suffix)) for suffix in suffixes)
    plan = _make_plan(prefix, *suffixes)

    reference_engine = _make_engine(reference_model, dta_enabled=False, monkeypatch=monkeypatch)
    observed_microbatches = []
    _install_reference_forward(reference_engine, observed_microbatches, monkeypatch)
    reference_logprobs = []

    def reference_loss_function(*, model_output, data, dp_group):
        del dp_group
        logits = model_output["logits"]
        tokens = data["input_ids"]
        reference_logprobs.append(_target_logprobs(logits[:, :-1, :], tokens[:, 1:]).cpu())
        loss_sum = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, profile._PROFILE_VOCAB_SIZE),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        return loss_sum / data["batch_num_tokens"], {}

    reference_output_a = reference_engine.forward_backward_batch(
        _reference_data(*trajectories),
        loss_function=reference_loss_function,
        forward_only=False,
    )
    reference_logprobs_a = torch.cat(reference_logprobs, dim=0)
    reference_loss_a = sum(reference_output_a["loss"])
    reference_gradients_a = _gradient_snapshot(reference_model)

    reference_model.zero_grad(set_to_none=True)
    reference_logprobs.clear()
    reference_output_b = reference_engine.forward_backward_batch(
        _reference_data(*trajectories),
        loss_function=reference_loss_function,
        forward_only=False,
    )
    assert observed_microbatches == [1] * (_SIBLING_COUNT * 2)
    reference_logprobs_b = torch.cat(reference_logprobs, dim=0)
    reference_loss_b = sum(reference_output_b["loss"])
    reference_gradients_b = _gradient_snapshot(reference_model)

    torch.testing.assert_close(
        reference_logprobs_b,
        reference_logprobs_a,
        atol=_LOGPROB_ATOL,
        rtol=_LOGPROB_RTOL,
    )
    torch.testing.assert_close(
        torch.tensor(reference_loss_b),
        torch.tensor(reference_loss_a),
        atol=_LOSS_ATOL,
        rtol=_LOSS_RTOL,
    )
    reference_repeat_logprob = _logprob_metrics(reference_logprobs_b, reference_logprobs_a)
    reference_repeat_gradients = _gradient_metrics(reference_gradients_b, reference_gradients_a)
    assert reference_repeat_gradients["relative_l2"] <= _GLOBAL_GRAD_RELATIVE_L2_TOL
    assert reference_repeat_gradients["cosine"] >= _GLOBAL_GRAD_COSINE_MIN
    del reference_gradients_b

    dta_logprobs = torch.full(
        (_SIBLING_COUNT, full_length - 1),
        float("nan"),
        dtype=torch.float32,
    )
    original_compute_loss = SegmentExecutor._compute_loss

    def capture_dta_logprobs(self, segment, logits):
        if segment.loss_terms:
            query_offsets = torch.tensor(
                [term.query_offset for term in segment.loss_terms],
                dtype=torch.long,
                device=logits.device,
            )
            targets = torch.tensor(
                [term.target_token_id for term in segment.loss_terms],
                dtype=torch.long,
                device=logits.device,
            )
            selected = logits[0].index_select(0, query_offsets)
            values = _target_logprobs(selected, targets).cpu()
            for term, value in zip(segment.loss_terms, values, strict=True):
                global_query = segment.position_start + term.query_offset
                if term.sample_id is None:
                    dta_logprobs[:, global_query] = value
                else:
                    dta_logprobs[term.sample_id - 1, global_query] = value
        return original_compute_loss(self, segment, logits)

    monkeypatch.setattr(SegmentExecutor, "_compute_loss", capture_dta_logprobs)
    dta_engine = _make_engine(dta_model, dta_enabled=True, monkeypatch=monkeypatch)
    dta_data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(dta_data, **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(plan)})
    dta_output = dta_engine.forward_backward_batch(dta_data, loss_function=None, forward_only=False)
    dta_gradients = _gradient_snapshot(dta_model)

    assert not torch.isnan(dta_logprobs).any().item(), "DTA logprob reconstruction is incomplete"
    torch.testing.assert_close(
        dta_logprobs,
        reference_logprobs_a,
        atol=_LOGPROB_ATOL,
        rtol=_LOGPROB_RTOL,
    )
    torch.testing.assert_close(
        torch.tensor(dta_output["loss"]),
        torch.tensor(reference_loss_a),
        atol=_LOSS_ATOL,
        rtol=_LOSS_RTOL,
    )
    dta_logprob = _logprob_metrics(dta_logprobs, reference_logprobs_a)
    dta_gradients = _gradient_metrics(dta_gradients, reference_gradients_a)
    assert dta_gradients["relative_l2"] <= _GLOBAL_GRAD_RELATIVE_L2_TOL
    assert dta_gradients["cosine"] >= _GLOBAL_GRAD_COSINE_MIN

    reference_repeat_loss_abs = abs(reference_loss_b - reference_loss_a)
    reference_repeat_loss_relative = reference_repeat_loss_abs / max(abs(reference_loss_a), 1e-12)
    dta_loss_abs = abs(dta_output["loss"] - reference_loss_a)
    dta_loss_relative = dta_loss_abs / max(abs(reference_loss_a), 1e-12)
    print(f"Model parameters: {parameter_count / 1e9:.3f}B")
    print(f"Case: P={_PREFIX_LENGTH}, S={_SUFFIX_LENGTH}, N={_SIBLING_COUNT}")
    print("Noise-floor comparison:")
    print("                                      Ref A vs Ref B      DTA vs Ref A")
    print(
        "  Logprob relative L2              "
        f"{reference_repeat_logprob['relative_l2']:>14.6e}  {dta_logprob['relative_l2']:>14.6e}"
    )
    print(
        "  Logprob cosine                   "
        f"{reference_repeat_logprob['cosine']:>14.9f}  {dta_logprob['cosine']:>14.9f}"
    )
    print(
        "  Logprob max abs diff             "
        f"{reference_repeat_logprob['max_abs']:>14.6e}  {dta_logprob['max_abs']:>14.6e}"
    )
    print(
        "  Loss relative diff               "
        f"{reference_repeat_loss_relative:>14.6e}  {dta_loss_relative:>14.6e}"
    )
    print(
        "  Gradient relative L2             "
        f"{reference_repeat_gradients['relative_l2']:>14.6e}  {dta_gradients['relative_l2']:>14.6e}"
    )
    print(
        "  Gradient cosine                  "
        f"{reference_repeat_gradients['cosine']:>14.9f}  {dta_gradients['cosine']:>14.9f}"
    )
    print(
        "  Gradient max abs diff            "
        f"{reference_repeat_gradients['max_abs']:>14.6e}  {dta_gradients['max_abs']:>14.6e}"
    )
    print("Loss values:")
    print(f"  Reference A: {reference_loss_a:.9f}")
    print(f"  Reference B: {reference_loss_b:.9f}")
    print(f"  DTA:         {dta_output['loss']:.9f}")
    print(f"Gradient tensors compared: {dta_gradients['parameter_count']}")
    print(
        "Worst Ref/Ref gradient tensor: "
        f"{reference_repeat_gradients['worst_name']} "
        f"({reference_repeat_gradients['worst_relative_l2']:.6e})"
    )
    print(
        "Worst DTA/Ref gradient tensor: "
        f"{dta_gradients['worst_name']} ({dta_gradients['worst_relative_l2']:.6e})"
    )
