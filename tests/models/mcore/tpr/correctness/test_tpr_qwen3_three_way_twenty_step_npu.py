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

"""Same-process Reference-A / Reference-B / TPR 20-step Qwen experiment."""

import math

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from . import test_tpr_twenty_step_correctness_npu as experiment
from ..equivalence.test_tpr_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_plan,
    _reference_data,
)
from .test_tpr_qwen3_compatibility_npu import (
    QWEN_VOCAB_SIZE,
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from ..equivalence.test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest, SegmentExecutor
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

_MAX_TPR_GRADIENT_RELATIVE_L2 = 1e-1


def _make_reference_loss_function(logprob_output):
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


def test_qwen3_reference_repeat_and_tpr_for_twenty_adamw_steps(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _initialize_single_rank_megatron()
    _install_single_rank_runtime(monkeypatch, device)
    monkeypatch.setattr(experiment.profile, "_PROFILE_VOCAB_SIZE", QWEN_VOCAB_SIZE)
    full_length = experiment._PREFIX_LENGTH + experiment._SUFFIX_LENGTH

    reference_a = _make_qwen_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=experiment.profile._ProfileFusedCausalAttention,
        model_shape=experiment.profile._PROFILE_MODEL_SHAPE,
    )
    reference_b = _make_qwen_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=experiment.profile._ProfileFusedCausalAttention,
        model_shape=experiment.profile._PROFILE_MODEL_SHAPE,
    )
    tpr_model = _make_qwen_model(
        device,
        tpr=True,
        max_sequence_length=full_length,
        core_attention_module=experiment.profile._ProfileFusedCausalAttention,
        model_shape=experiment.profile._PROFILE_MODEL_SHAPE,
    )
    initial_state = reference_a.state_dict()
    reference_b.load_state_dict(initial_state, strict=True)
    tpr_model.load_state_dict(initial_state, strict=True)

    parameter_count = experiment.profile._assert_profile_model_scale(reference_a)
    assert experiment.profile._assert_profile_model_scale(reference_b) == parameter_count
    assert experiment.profile._assert_profile_model_scale(tpr_model) == parameter_count
    for model in (reference_a, reference_b, tpr_model):
        _configure_model_runtime(model)

    engine_a = _make_engine(reference_a, tpr_enabled=False, monkeypatch=monkeypatch)
    engine_b = _make_engine(reference_b, tpr_enabled=False, monkeypatch=monkeypatch)
    tpr_engine = _make_engine(tpr_model, tpr_enabled=True, monkeypatch=monkeypatch)
    observed_a, observed_b = [], []
    _install_reference_forward(engine_a, observed_a, monkeypatch)
    _install_reference_forward(engine_b, observed_b, monkeypatch)

    masters_a, optimizer_a = experiment._make_fp32_optimizer(reference_a)
    masters_b, optimizer_b = experiment._make_fp32_optimizer(reference_b)
    masters_tpr, optimizer_tpr = experiment._make_fp32_optimizer(tpr_model)
    assert experiment._mapping_metrics(masters_b, masters_a)["relative_l2"] == 0.0
    assert experiment._mapping_metrics(masters_tpr, masters_a)["relative_l2"] == 0.0

    logprobs_a, logprobs_b = [], []
    loss_function_a = _make_reference_loss_function(logprobs_a)
    loss_function_b = _make_reference_loss_function(logprobs_b)
    capture = {"logprobs": None}
    original_compute_loss = SegmentExecutor._compute_loss

    def capture_tpr_logprobs(executor, segment, logits):
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
            values = experiment._target_logprobs(
                logits[0].index_select(0, query_offsets), targets
            ).cpu()
            for term, value in zip(segment.loss_terms, values, strict=True):
                global_query = segment.position_start + term.query_offset
                if term.sample_id is None:
                    output[:, global_query] = value
                else:
                    output[term.sample_id - 1, global_query] = value
        return original_compute_loss(executor, segment, logits)

    monkeypatch.setattr(SegmentExecutor, "_compute_loss", capture_tpr_logprobs)
    rows = []

    for step in range(1, experiment._STEPS + 1):
        for model in (reference_a, reference_b, tpr_model):
            model.zero_grad(set_to_none=True)
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
        plan = _make_plan(prefix, *suffixes)

        def run_reference(engine, loss_function):
            output = engine.forward_backward_batch(
                _reference_data(*trajectories),
                loss_function=loss_function,
                forward_only=False,
            )
            return sum(output["loss"])

        tpr_step_logprobs = torch.full(
            (experiment._SIBLING_COUNT, full_length - 1),
            float("nan"),
            dtype=torch.float32,
        )

        def run_tpr():
            capture["logprobs"] = tpr_step_logprobs
            data = TensorDict({}, batch_size=[])
            tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})
            try:
                return tpr_engine.forward_backward_batch(
                    data, loss_function=None, forward_only=False
                )["loss"]
            finally:
                capture["logprobs"] = None

        loss_a = run_reference(engine_a, loss_function_a)
        if step % 2:
            loss_b = run_reference(engine_b, loss_function_b)
            loss_tpr = run_tpr()
        else:
            loss_tpr = run_tpr()
            loss_b = run_reference(engine_b, loss_function_b)

        assert not torch.isnan(tpr_step_logprobs).any().item()
        step_logprobs_a = torch.cat(logprobs_a, dim=0)
        step_logprobs_b = torch.cat(logprobs_b, dim=0)
        ref_logprob = experiment._logprob_metrics(step_logprobs_b, step_logprobs_a)
        tpr_logprob = experiment._logprob_metrics(tpr_step_logprobs, step_logprobs_a)
        ref_loss_relative = abs(loss_b - loss_a) / max(abs(loss_a), 1e-12)
        tpr_loss_relative = abs(loss_tpr - loss_a) / max(abs(loss_a), 1e-12)
        ref_gradients = experiment._mapping_metrics(
            experiment._model_gradients(reference_b),
            experiment._model_gradients(reference_a),
        )
        tpr_gradients = experiment._mapping_metrics(
            experiment._model_gradients(tpr_model),
            experiment._model_gradients(reference_a),
        )
        for value in (
            ref_loss_relative,
            tpr_loss_relative,
            ref_logprob["relative_l2"],
            tpr_logprob["relative_l2"],
            ref_gradients["relative_l2"],
            tpr_gradients["relative_l2"],
        ):
            assert math.isfinite(value)
        assert tpr_logprob["relative_l2"] <= experiment._MAX_LOGPROB_RELATIVE_L2
        assert tpr_loss_relative <= experiment._MAX_LOSS_RELATIVE_DIFF
        assert tpr_gradients["relative_l2"] <= _MAX_TPR_GRADIENT_RELATIVE_L2
        assert tpr_gradients["cosine"] >= experiment._MIN_GRADIENT_COSINE

        norms = {
            "a": float(torch.nn.utils.clip_grad_norm_(reference_a.parameters(), experiment._MAX_GRAD_NORM, foreach=False)),
            "b": float(torch.nn.utils.clip_grad_norm_(reference_b.parameters(), experiment._MAX_GRAD_NORM, foreach=False)),
            "tpr": float(torch.nn.utils.clip_grad_norm_(tpr_model.parameters(), experiment._MAX_GRAD_NORM, foreach=False)),
        }
        experiment._optimizer_step(reference_a, masters_a, optimizer_a)
        experiment._optimizer_step(reference_b, masters_b, optimizer_b)
        experiment._optimizer_step(tpr_model, masters_tpr, optimizer_tpr)
        ref_parameters = experiment._mapping_metrics(masters_b, masters_a)
        tpr_parameters = experiment._mapping_metrics(masters_tpr, masters_a)
        rows.append(
            {
                "ref_loss": ref_loss_relative,
                "tpr_loss": tpr_loss_relative,
                "ref_logprob": ref_logprob,
                "tpr_logprob": tpr_logprob,
                "ref_gradients": ref_gradients,
                "tpr_gradients": tpr_gradients,
                "ref_parameters": ref_parameters,
                "tpr_parameters": tpr_parameters,
                "ref_clip_mismatch": (norms["a"] > experiment._MAX_GRAD_NORM) != (norms["b"] > experiment._MAX_GRAD_NORM),
                "tpr_clip_mismatch": (norms["a"] > experiment._MAX_GRAD_NORM) != (norms["tpr"] > experiment._MAX_GRAD_NORM),
            }
        )
        print(
            f"step={step:02d} "
            f"ref/ref: loss={ref_loss_relative:.3e} logp={ref_logprob['relative_l2']:.3e} "
            f"grad={ref_gradients['relative_l2']:.3e} cos={ref_gradients['cosine']:.7f} "
            f"param={ref_parameters['relative_l2']:.3e} | "
            f"tpr/ref: loss={tpr_loss_relative:.3e} logp={tpr_logprob['relative_l2']:.3e} "
            f"grad={tpr_gradients['relative_l2']:.3e} cos={tpr_gradients['cosine']:.7f} "
            f"param={tpr_parameters['relative_l2']:.3e}"
        )

    expected_microbatches = [1] * (experiment._SIBLING_COUNT * experiment._STEPS)
    assert observed_a == expected_microbatches
    assert observed_b == expected_microbatches
    assert sum(row["tpr_clip_mismatch"] for row in rows) == 0
    final = rows[-1]
    assert final["tpr_parameters"]["cosine"] >= 0.999
    moments = {
        "ref_first": experiment._mapping_metrics(
            experiment._optimizer_state(optimizer_b, masters_b, "exp_avg"),
            experiment._optimizer_state(optimizer_a, masters_a, "exp_avg"),
        ),
        "tpr_first": experiment._mapping_metrics(
            experiment._optimizer_state(optimizer_tpr, masters_tpr, "exp_avg"),
            experiment._optimizer_state(optimizer_a, masters_a, "exp_avg"),
        ),
        "ref_second": experiment._mapping_metrics(
            experiment._optimizer_state(optimizer_b, masters_b, "exp_avg_sq"),
            experiment._optimizer_state(optimizer_a, masters_a, "exp_avg_sq"),
        ),
        "tpr_second": experiment._mapping_metrics(
            experiment._optimizer_state(optimizer_tpr, masters_tpr, "exp_avg_sq"),
            experiment._optimizer_state(optimizer_a, masters_a, "exp_avg_sq"),
        ),
    }

    def print_pair(name, prefix_name):
        print(f"{name}:")
        print(f"  max loss relative diff:      {max(row[prefix_name + '_loss'] for row in rows):.6e}")
        print(f"  max logprob relative L2:     {max(row[prefix_name + '_logprob']['relative_l2'] for row in rows):.6e}")
        print(f"  max gradient relative L2:    {max(row[prefix_name + '_gradients']['relative_l2'] for row in rows):.6e}")
        print(f"  min gradient cosine:         {min(row[prefix_name + '_gradients']['cosine'] for row in rows):.9f}")
        print(f"  final parameter relative L2: {final[prefix_name + '_parameters']['relative_l2']:.6e}")
        print(f"  final parameter cosine:      {final[prefix_name + '_parameters']['cosine']:.9f}")
        print(f"  Adam first-moment rel L2:    {moments[prefix_name + '_first']['relative_l2']:.6e}")
        print(f"  Adam second-moment rel L2:   {moments[prefix_name + '_second']['relative_l2']:.6e}")
        print(f"  clipping mismatches:         {sum(row[prefix_name + '_clip_mismatch'] for row in rows)}")

    print("\n20-step same-process Qwen3 three-way summary")
    print(f"Model parameters: {parameter_count / 1e9:.3f}B")
    print(f"Case: P={experiment._PREFIX_LENGTH}, S={experiment._SUFFIX_LENGTH}, N={experiment._SIBLING_COUNT}")
    print_pair("Reference B vs Reference A", "ref")
    print_pair("TPR vs Reference A", "tpr")
