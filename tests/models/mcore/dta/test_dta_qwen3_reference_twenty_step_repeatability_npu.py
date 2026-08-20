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

"""Opt-in 20-step real-Qwen Reference/Reference repeatability experiment."""

import math
import os

import pytest
import torch
import torch.nn.functional as F

import test_dta_twenty_step_correctness_npu as experiment
from test_dta_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _reference_data,
)
from test_dta_qwen3_compatibility_npu import (
    QWEN_VOCAB_SIZE,
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from test_segment_push_pop_npu import _install_single_rank_runtime
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_QWEN_REF_REPEATABILITY") != "1",
    reason="Set DTA_RUN_QWEN_REF_REPEATABILITY=1 for the real-Qwen Ref/Ref experiment",
)

_MAX_REFERENCE_GRADIENT_RELATIVE_L2 = 0.1
_MIN_REFERENCE_GRADIENT_COSINE = 0.995


def _make_loss_function(logprob_output):
    def loss_function(*, model_output, data, dp_group):
        del dp_group
        logits = model_output["logits"]
        tokens = data["input_ids"]
        logprob_output.append(
            experiment._target_logprobs(logits[:, :-1, :], tokens[:, 1:]).cpu()
        )
        loss_sum = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, QWEN_VOCAB_SIZE),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        return loss_sum / data["batch_num_tokens"], {}

    return loss_function


def test_qwen3_reference_twenty_step_repeatability(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _initialize_single_rank_megatron()
    _install_single_rank_runtime(monkeypatch, device)
    monkeypatch.setattr(experiment.profile, "_PROFILE_VOCAB_SIZE", QWEN_VOCAB_SIZE)
    full_length = experiment._PREFIX_LENGTH + experiment._SUFFIX_LENGTH

    reference_a = _make_qwen_model(
        device,
        dta=False,
        max_sequence_length=full_length,
        core_attention_module=experiment.profile._ProfileFusedCausalAttention,
        model_shape=experiment.profile._PROFILE_MODEL_SHAPE,
    )
    reference_b = _make_qwen_model(
        device,
        dta=False,
        max_sequence_length=full_length,
        core_attention_module=experiment.profile._ProfileFusedCausalAttention,
        model_shape=experiment.profile._PROFILE_MODEL_SHAPE,
    )
    reference_b.load_state_dict(reference_a.state_dict(), strict=True)
    parameter_count = experiment.profile._assert_profile_model_scale(reference_a)
    assert experiment.profile._assert_profile_model_scale(reference_b) == parameter_count
    _configure_model_runtime(reference_a)
    _configure_model_runtime(reference_b)

    engine_a = _make_engine(reference_a, dta_enabled=False, monkeypatch=monkeypatch)
    engine_b = _make_engine(reference_b, dta_enabled=False, monkeypatch=monkeypatch)
    observed_a = []
    observed_b = []
    _install_reference_forward(engine_a, observed_a, monkeypatch)
    _install_reference_forward(engine_b, observed_b, monkeypatch)
    masters_a, optimizer_a = experiment._make_fp32_optimizer(reference_a)
    masters_b, optimizer_b = experiment._make_fp32_optimizer(reference_b)
    logprobs_a = []
    logprobs_b = []
    loss_function_a = _make_loss_function(logprobs_a)
    loss_function_b = _make_loss_function(logprobs_b)
    rows = []

    for step in range(1, experiment._STEPS + 1):
        reference_a.zero_grad(set_to_none=True)
        reference_b.zero_grad(set_to_none=True)
        logprobs_a.clear()
        logprobs_b.clear()
        prefix = experiment._tokens(17 + step * 97, experiment._PREFIX_LENGTH, device)
        suffixes = tuple(
            experiment._tokens(
                700 + step * 193 + sibling * 1300,
                experiment._SUFFIX_LENGTH,
                device,
            )
            for sibling in range(experiment._SIBLING_COUNT)
        )
        trajectories = tuple(torch.cat((prefix, suffix)) for suffix in suffixes)

        output_a = engine_a.forward_backward_batch(
            _reference_data(*trajectories),
            loss_function=loss_function_a,
            forward_only=False,
        )
        output_b = engine_b.forward_backward_batch(
            _reference_data(*trajectories),
            loss_function=loss_function_b,
            forward_only=False,
        )
        step_logprobs_a = torch.cat(logprobs_a, dim=0)
        step_logprobs_b = torch.cat(logprobs_b, dim=0)
        loss_a = sum(output_a["loss"])
        loss_b = sum(output_b["loss"])
        logprob = experiment._logprob_metrics(step_logprobs_b, step_logprobs_a)
        loss_relative = abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)
        gradients = experiment._mapping_metrics(
            experiment._model_gradients(reference_b),
            experiment._model_gradients(reference_a),
        )
        assert math.isfinite(loss_relative)
        assert gradients["relative_l2"] <= _MAX_REFERENCE_GRADIENT_RELATIVE_L2
        assert gradients["cosine"] >= _MIN_REFERENCE_GRADIENT_COSINE

        grad_norm_a = float(
            torch.nn.utils.clip_grad_norm_(
                reference_a.parameters(), experiment._MAX_GRAD_NORM, foreach=False
            )
        )
        grad_norm_b = float(
            torch.nn.utils.clip_grad_norm_(
                reference_b.parameters(), experiment._MAX_GRAD_NORM, foreach=False
            )
        )
        clipped_a = grad_norm_a > experiment._MAX_GRAD_NORM
        clipped_b = grad_norm_b > experiment._MAX_GRAD_NORM
        experiment._optimizer_step(reference_a, masters_a, optimizer_a)
        experiment._optimizer_step(reference_b, masters_b, optimizer_b)
        parameters = experiment._mapping_metrics(masters_b, masters_a)
        rows.append(
            {
                "loss_relative": loss_relative,
                "logprob": logprob,
                "gradients": gradients,
                "parameters": parameters,
                "clipping_mismatch": clipped_a != clipped_b,
            }
        )
        print(
            f"step={step:02d} loss_rel={loss_relative:.3e} "
            f"logp_rel={logprob['relative_l2']:.3e} "
            f"grad_rel={gradients['relative_l2']:.3e} "
            f"grad_cos={gradients['cosine']:.7f} "
            f"param_rel={parameters['relative_l2']:.3e}"
        )

    expected_microbatches = [1] * (experiment._SIBLING_COUNT * experiment._STEPS)
    assert observed_a == expected_microbatches
    assert observed_b == expected_microbatches
    first_moment = experiment._mapping_metrics(
        experiment._optimizer_state(optimizer_b, masters_b, "exp_avg"),
        experiment._optimizer_state(optimizer_a, masters_a, "exp_avg"),
    )
    second_moment = experiment._mapping_metrics(
        experiment._optimizer_state(optimizer_b, masters_b, "exp_avg_sq"),
        experiment._optimizer_state(optimizer_a, masters_a, "exp_avg_sq"),
    )
    final = rows[-1]
    clipping_mismatches = sum(row["clipping_mismatch"] for row in rows)
    assert clipping_mismatches == 0
    assert final["parameters"]["cosine"] >= 0.999

    print("\n20-step real-Qwen Reference repeatability summary")
    print(f"Model parameters: {parameter_count / 1e9:.3f}B")
    print(
        f"Case per step: P={experiment._PREFIX_LENGTH}, "
        f"S={experiment._SUFFIX_LENGTH}, N={experiment._SIBLING_COUNT}"
    )
    print(f"Max loss relative diff:       {max(row['loss_relative'] for row in rows):.6e}")
    print(
        "Max logprob relative L2:      "
        f"{max(row['logprob']['relative_l2'] for row in rows):.6e}"
    )
    print(
        "Max gradient relative L2:     "
        f"{max(row['gradients']['relative_l2'] for row in rows):.6e}"
    )
    print(
        "Min gradient cosine:          "
        f"{min(row['gradients']['cosine'] for row in rows):.9f}"
    )
    print(f"Final parameter relative L2:  {final['parameters']['relative_l2']:.6e}")
    print(f"Final parameter cosine:       {final['parameters']['cosine']:.9f}")
    print(f"Adam first-moment rel L2:     {first_moment['relative_l2']:.6e}")
    print(f"Adam first-moment cosine:     {first_moment['cosine']:.9f}")
    print(f"Adam second-moment rel L2:    {second_moment['relative_l2']:.6e}")
    print(f"Adam second-moment cosine:    {second_moment['cosine']:.9f}")
    print(f"Gradient clipping mismatches: {clipping_mismatches}")
