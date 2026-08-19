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

"""Opt-in 20-step Reference/TPR optimizer-path correctness experiment."""

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
    os.getenv("DTA_RUN_20_STEP_CORRECTNESS") != "1",
    reason="Set DTA_RUN_20_STEP_CORRECTNESS=1 for the opt-in 20-step experiment",
)

_PREFIX_LENGTH = 1024
_SUFFIX_LENGTH = 512
_SIBLING_COUNT = 2
_STEPS = 20
_LEARNING_RATE = 1e-5
_BETAS = (0.9, 0.95)
_EPS = 1e-8
_WEIGHT_DECAY = 0.1
_MAX_GRAD_NORM = 1.0
_MAX_LOGPROB_RELATIVE_L2 = 5e-3
_MAX_LOSS_RELATIVE_DIFF = 1e-3
_MAX_GRADIENT_RELATIVE_L2 = 5e-2
_MIN_GRADIENT_COSINE = 0.998


def _tokens(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device) % profile._PROFILE_VOCAB_SIZE


def _target_logprobs(logits, targets):
    return F.log_softmax(logits.detach().float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _mapping_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    device = next(iter(actual.values())).device
    difference_sq = torch.zeros((), dtype=torch.float32, device=device)
    actual_sq = torch.zeros((), dtype=torch.float32, device=device)
    expected_sq = torch.zeros((), dtype=torch.float32, device=device)
    dot = torch.zeros((), dtype=torch.float32, device=device)
    max_abs = torch.zeros((), dtype=torch.float32, device=device)
    all_finite = torch.ones((), dtype=torch.bool, device=device)
    relative_by_tensor = []
    names = []
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        all_finite = all_finite & torch.isfinite(actual_tensor).all()
        all_finite = all_finite & torch.isfinite(expected_tensor).all()
        actual_float = actual_tensor.float()
        expected_float = expected_tensor.float()
        difference = actual_float - expected_float
        tensor_difference_sq = torch.sum(difference.square())
        tensor_expected_sq = torch.sum(expected_float.square())
        difference_sq = difference_sq + tensor_difference_sq
        actual_sq = actual_sq + torch.sum(actual_float.square())
        expected_sq = expected_sq + tensor_expected_sq
        dot = dot + torch.sum(actual_float * expected_float)
        max_abs = torch.maximum(max_abs, difference.abs().max())
        relative_by_tensor.append(torch.sqrt(tensor_difference_sq) / torch.sqrt(tensor_expected_sq).clamp_min(1e-12))
        names.append(name)
    relatives = torch.stack(relative_by_tensor)
    assert all_finite.item(), "mapping contains a non-finite tensor"
    worst_index = int(torch.argmax(relatives).item())
    return {
        "tensor_count": len(names),
        "relative_l2": (torch.sqrt(difference_sq) / torch.sqrt(expected_sq).clamp_min(1e-12)).item(),
        "cosine": (dot / torch.sqrt(actual_sq * expected_sq).clamp_min(1e-12)).item(),
        "max_abs": max_abs.item(),
        "worst_name": names[worst_index],
        "worst_relative_l2": relatives[worst_index].item(),
    }


def _model_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        gradients[name] = parameter.grad
    return gradients


def _make_fp32_optimizer(model):
    masters = {
        name: torch.nn.Parameter(parameter.detach().float().clone())
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        masters.values(),
        lr=_LEARNING_RATE,
        betas=_BETAS,
        eps=_EPS,
        weight_decay=_WEIGHT_DECAY,
        foreach=False,
    )
    return masters, optimizer


def _optimizer_step(model, masters, optimizer):
    model_parameters = dict(model.named_parameters())
    assert model_parameters.keys() == masters.keys()
    for name, master in masters.items():
        master.grad = model_parameters[name].grad.detach().float().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, parameter in model_parameters.items():
            parameter.copy_(masters[name].to(dtype=parameter.dtype))


def _optimizer_state(optimizer, masters, key):
    result = {}
    for name, master in masters.items():
        state = optimizer.state[master]
        assert key in state, f"optimizer state {key} is missing for {name}"
        result[name] = state[key]
    return result


def _logprob_metrics(actual, expected):
    difference = actual - expected
    absolute = difference.abs().flatten()
    expected_norm = torch.linalg.vector_norm(expected).clamp_min(1e-12)
    actual_norm = torch.linalg.vector_norm(actual).clamp_min(1e-12)
    ratio_error = torch.expm1(difference).abs().flatten()
    return {
        "relative_l2": (torch.linalg.vector_norm(difference) / expected_norm).item(),
        "cosine": (torch.dot(actual.flatten(), expected.flatten()) / (actual_norm * expected_norm)).item(),
        "mean_abs": absolute.mean().item(),
        "p95_abs": torch.quantile(absolute, 0.95).item(),
        "p99_abs": torch.quantile(absolute, 0.99).item(),
        "max_abs": absolute.max().item(),
        "p99_ratio_error": torch.quantile(ratio_error, 0.99).item(),
        "max_ratio_error": ratio_error.max().item(),
    }


def test_tpr_matches_reference_for_twenty_adamw_steps(monkeypatch):
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
    tpr_model = profile._make_model(
        device,
        dta=True,
        max_sequence_length=full_length,
        core_attention_module=profile._ProfileFusedCausalAttention,
        model_shape=profile._PROFILE_MODEL_SHAPE,
    )
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)
    parameter_count = profile._assert_profile_model_scale(reference_model)
    assert profile._assert_profile_model_scale(tpr_model) == parameter_count
    _configure_model_runtime(reference_model)
    _configure_model_runtime(tpr_model)
    reference_engine = _make_engine(reference_model, dta_enabled=False, monkeypatch=monkeypatch)
    tpr_engine = _make_engine(tpr_model, dta_enabled=True, monkeypatch=monkeypatch)
    observed_microbatches = []
    _install_reference_forward(reference_engine, observed_microbatches, monkeypatch)
    reference_masters, reference_optimizer = _make_fp32_optimizer(reference_model)
    tpr_masters, tpr_optimizer = _make_fp32_optimizer(tpr_model)

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

    capture = {"logprobs": None}
    original_compute_loss = SegmentExecutor._compute_loss

    def capture_tpr_logprobs(self, segment, logits):
        output = capture["logprobs"]
        if output is not None and segment.loss_terms:
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
            values = _target_logprobs(logits[0].index_select(0, query_offsets), targets).cpu()
            for term, value in zip(segment.loss_terms, values, strict=True):
                global_query = segment.position_start + term.query_offset
                if term.sample_id is None:
                    output[:, global_query] = value
                else:
                    output[term.sample_id - 1, global_query] = value
        return original_compute_loss(self, segment, logits)

    monkeypatch.setattr(SegmentExecutor, "_compute_loss", capture_tpr_logprobs)
    rows = []
    for step in range(1, _STEPS + 1):
        reference_model.zero_grad(set_to_none=True)
        tpr_model.zero_grad(set_to_none=True)
        reference_logprobs.clear()
        prefix = _tokens(17 + step * 97, _PREFIX_LENGTH, device)
        suffixes = tuple(
            _tokens(700 + step * 193 + sibling * 1300, _SUFFIX_LENGTH, device)
            for sibling in range(_SIBLING_COUNT)
        )
        trajectories = tuple(torch.cat((prefix, suffix)) for suffix in suffixes)
        plan = _make_plan(prefix, *suffixes)

        reference_output = reference_engine.forward_backward_batch(
            _reference_data(*trajectories),
            loss_function=reference_loss_function,
            forward_only=False,
        )
        reference_step_logprobs = torch.cat(reference_logprobs, dim=0)
        reference_loss = sum(reference_output["loss"])

        tpr_step_logprobs = torch.full(
            (_SIBLING_COUNT, full_length - 1),
            float("nan"),
            dtype=torch.float32,
        )
        capture["logprobs"] = tpr_step_logprobs
        tpr_data = TensorDict({}, batch_size=[])
        tu.assign_non_tensor(tpr_data, **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(plan)})
        tpr_output = tpr_engine.forward_backward_batch(tpr_data, loss_function=None, forward_only=False)
        capture["logprobs"] = None
        assert not torch.isnan(tpr_step_logprobs).any().item()

        logprob = _logprob_metrics(tpr_step_logprobs, reference_step_logprobs)
        loss_relative = abs(tpr_output["loss"] - reference_loss) / max(abs(reference_loss), 1e-12)
        gradients = _mapping_metrics(_model_gradients(tpr_model), _model_gradients(reference_model))
        assert logprob["relative_l2"] <= _MAX_LOGPROB_RELATIVE_L2
        assert loss_relative <= _MAX_LOSS_RELATIVE_DIFF
        assert gradients["relative_l2"] <= _MAX_GRADIENT_RELATIVE_L2
        assert gradients["cosine"] >= _MIN_GRADIENT_COSINE

        reference_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(reference_model.parameters(), _MAX_GRAD_NORM, foreach=False)
        )
        tpr_grad_norm = float(
            torch.nn.utils.clip_grad_norm_(tpr_model.parameters(), _MAX_GRAD_NORM, foreach=False)
        )
        reference_clipped = reference_grad_norm > _MAX_GRAD_NORM
        tpr_clipped = tpr_grad_norm > _MAX_GRAD_NORM
        _optimizer_step(reference_model, reference_masters, reference_optimizer)
        _optimizer_step(tpr_model, tpr_masters, tpr_optimizer)
        parameters = _mapping_metrics(tpr_masters, reference_masters)
        rows.append(
            {
                "step": step,
                "reference_loss": reference_loss,
                "tpr_loss": tpr_output["loss"],
                "loss_relative": loss_relative,
                "logprob": logprob,
                "gradients": gradients,
                "parameters": parameters,
                "reference_grad_norm": reference_grad_norm,
                "tpr_grad_norm": tpr_grad_norm,
                "reference_clipped": reference_clipped,
                "tpr_clipped": tpr_clipped,
            }
        )
        print(
            f"step={step:02d} loss_rel={loss_relative:.3e} "
            f"logp_rel={logprob['relative_l2']:.3e} "
            f"grad_rel={gradients['relative_l2']:.3e} "
            f"grad_cos={gradients['cosine']:.7f} "
            f"param_rel={parameters['relative_l2']:.3e}"
        )

    assert observed_microbatches == [1] * (_SIBLING_COUNT * _STEPS)
    first_moment = _mapping_metrics(
        _optimizer_state(tpr_optimizer, tpr_masters, "exp_avg"),
        _optimizer_state(reference_optimizer, reference_masters, "exp_avg"),
    )
    second_moment = _mapping_metrics(
        _optimizer_state(tpr_optimizer, tpr_masters, "exp_avg_sq"),
        _optimizer_state(reference_optimizer, reference_masters, "exp_avg_sq"),
    )
    final = rows[-1]
    max_loss_relative = max(row["loss_relative"] for row in rows)
    max_logprob_relative = max(row["logprob"]["relative_l2"] for row in rows)
    max_gradient_relative = max(row["gradients"]["relative_l2"] for row in rows)
    min_gradient_cosine = min(row["gradients"]["cosine"] for row in rows)
    clipping_disagreements = sum(
        row["reference_clipped"] != row["tpr_clipped"] for row in rows
    )
    assert clipping_disagreements == 0
    assert math.isfinite(final["parameters"]["relative_l2"])
    assert final["parameters"]["cosine"] >= 0.999

    print("\n20-step TPR correctness summary")
    print(f"Model parameters: {parameter_count / 1e9:.3f}B")
    print(f"Case per step: P={_PREFIX_LENGTH}, S={_SUFFIX_LENGTH}, N={_SIBLING_COUNT}")
    print(f"AdamW: lr={_LEARNING_RATE}, betas={_BETAS}, weight_decay={_WEIGHT_DECAY}")
    print(f"Max loss relative diff:       {max_loss_relative:.6e}")
    print(f"Max logprob relative L2:      {max_logprob_relative:.6e}")
    print(f"Final logprob p99 abs diff:   {final['logprob']['p99_abs']:.6e}")
    print(f"Final logprob max abs diff:   {final['logprob']['max_abs']:.6e}")
    print(f"Final ratio p99 error:        {final['logprob']['p99_ratio_error']:.6e}")
    print(f"Final ratio max error:        {final['logprob']['max_ratio_error']:.6e}")
    print(f"Max gradient relative L2:     {max_gradient_relative:.6e}")
    print(f"Min gradient cosine:          {min_gradient_cosine:.9f}")
    print(f"Final parameter relative L2:  {final['parameters']['relative_l2']:.6e}")
    print(f"Final parameter cosine:       {final['parameters']['cosine']:.9f}")
    print(f"Adam first-moment rel L2:     {first_moment['relative_l2']:.6e}")
    print(f"Adam first-moment cosine:     {first_moment['cosine']:.9f}")
    print(f"Adam second-moment rel L2:    {second_moment['relative_l2']:.6e}")
    print(f"Adam second-moment cosine:    {second_moment['cosine']:.9f}")
    print(f"Gradient clipping mismatches: {clipping_disagreements}")
