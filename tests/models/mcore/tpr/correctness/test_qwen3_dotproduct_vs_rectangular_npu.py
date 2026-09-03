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

"""One-step native DotProductAttention versus rectangular CANN diagnostic."""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.dot_product_attention import DotProductAttention
from . import test_tpr_qwen3_compatibility_npu as qwen_fixture
from . import test_tpr_twenty_step_correctness_npu as metrics
from ..equivalence.test_tpr_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _reference_data,
)
from ..profiling.test_tpr_engine_profile_npu import _ProfileFusedCausalAttention
from ..equivalence.test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.tpr import rectangular_attention as rectangular_attention_module
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


def _make_loss_function(logprob_output):
    def loss_function(*, model_output, data, dp_group):
        del dp_group
        logits = model_output["logits"]
        tokens = data["input_ids"]
        logprob_output.append(
            metrics._target_logprobs(logits[:, :-1, :], tokens[:, 1:]).cpu()
        )
        loss_sum = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, qwen_fixture.QWEN_VOCAB_SIZE),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        return loss_sum / data["batch_num_tokens"], {}

    return loss_function


def _assert_core_attention(model, expected_type):
    core_modules = tuple(layer.self_attention.core_attention for layer in model.decoder.layers)
    assert len(core_modules) == qwen_fixture.QWEN_NUM_LAYERS
    assert all(type(module) is expected_type for module in core_modules)


def _disable_cuda_fused_softmax(model):
    """Force native DotProductAttention onto its portable torch-softmax path."""

    model.config.masked_softmax_fusion = False
    for layer in model.decoder.layers:
        softmax = layer.self_attention.core_attention.scale_mask_softmax
        softmax.scaled_masked_softmax_fusion = False
        assert not softmax.scaled_masked_softmax_fusion


def test_qwen3_native_dotproduct_vs_rectangular_cann_one_step(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    qwen_fixture._initialize_single_rank_megatron()
    _install_single_rank_runtime(monkeypatch, device)
    monkeypatch.setattr(metrics.profile, "_PROFILE_VOCAB_SIZE", qwen_fixture.QWEN_VOCAB_SIZE)
    full_length = metrics._PREFIX_LENGTH + metrics._SUFFIX_LENGTH

    # The real-Qwen helper normally installs the controlled fused reference.
    # Disable only that test-time replacement while constructing the two native
    # Megatron DotProductAttention controls.
    with patch.object(qwen_fixture, "_replace_core_attention", lambda spec: spec):
        native_a = qwen_fixture._make_qwen_model(
            device,
            tpr=False,
            max_sequence_length=full_length,
            core_attention_module=None,
        )
        native_b = qwen_fixture._make_qwen_model(
            device,
            tpr=False,
            max_sequence_length=full_length,
            core_attention_module=None,
        )
    rectangular_a = qwen_fixture._make_qwen_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    rectangular_b = qwen_fixture._make_qwen_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    initial_state = native_a.state_dict()
    native_b.load_state_dict(initial_state, strict=True)
    rectangular_a.load_state_dict(initial_state, strict=True)
    rectangular_b.load_state_dict(initial_state, strict=True)
    _assert_core_attention(native_a, DotProductAttention)
    _assert_core_attention(native_b, DotProductAttention)
    _assert_core_attention(rectangular_a, _ProfileFusedCausalAttention)
    _assert_core_attention(rectangular_b, _ProfileFusedCausalAttention)
    _disable_cuda_fused_softmax(native_a)
    _disable_cuda_fused_softmax(native_b)

    for model in (native_a, native_b, rectangular_a, rectangular_b):
        _configure_model_runtime(model)
    engine_a = _make_engine(native_a, tpr_enabled=False, monkeypatch=monkeypatch)
    engine_b = _make_engine(native_b, tpr_enabled=False, monkeypatch=monkeypatch)
    engine_rect_a = _make_engine(rectangular_a, tpr_enabled=False, monkeypatch=monkeypatch)
    engine_rect_b = _make_engine(rectangular_b, tpr_enabled=False, monkeypatch=monkeypatch)
    observed_a, observed_b, observed_rect_a, observed_rect_b = [], [], [], []
    _install_reference_forward(engine_a, observed_a, monkeypatch)
    _install_reference_forward(engine_b, observed_b, monkeypatch)
    _install_reference_forward(engine_rect_a, observed_rect_a, monkeypatch)
    _install_reference_forward(engine_rect_b, observed_rect_b, monkeypatch)

    prefix = metrics._tokens(114, metrics._PREFIX_LENGTH, device)
    suffixes = tuple(
        metrics._tokens(893 + sibling * 1300, metrics._SUFFIX_LENGTH, device)
        for sibling in range(metrics._SIBLING_COUNT)
    )
    trajectories = tuple(torch.cat((prefix, suffix)) for suffix in suffixes)
    logprobs = {
        "native_a": [],
        "native_b": [],
        "rectangular_a": [],
        "rectangular_b": [],
    }
    losses = {}
    adapter_calls = {
        "native_a": 0,
        "native_b": 0,
        "rectangular_a": 0,
        "rectangular_b": 0,
    }
    active_path = [None]
    original_adapter = rectangular_attention_module.rectangular_causal_attention

    def counted_adapter(*args, **kwargs):
        path_name = active_path[0]
        assert path_name is not None
        adapter_calls[path_name] += 1
        return original_adapter(*args, **kwargs)

    def run_path(path_name, model, engine):
        model.zero_grad(set_to_none=True)
        active_path[0] = path_name
        try:
            output = engine.forward_backward_batch(
                _reference_data(*trajectories),
                loss_function=_make_loss_function(logprobs[path_name]),
                forward_only=False,
            )
        finally:
            active_path[0] = None
        losses[path_name] = sum(output["loss"])

    with patch.object(
        rectangular_attention_module,
        "rectangular_causal_attention",
        counted_adapter,
    ):
        run_path("native_a", native_a, engine_a)
        run_path("native_b", native_b, engine_b)
        run_path("rectangular_a", rectangular_a, engine_rect_a)
        run_path("rectangular_b", rectangular_b, engine_rect_b)

    assert adapter_calls == {
        "native_a": 0,
        "native_b": 0,
        "rectangular_a": qwen_fixture.QWEN_NUM_LAYERS * metrics._SIBLING_COUNT,
        "rectangular_b": qwen_fixture.QWEN_NUM_LAYERS * metrics._SIBLING_COUNT,
    }
    expected_microbatches = [1] * metrics._SIBLING_COUNT
    assert observed_a == expected_microbatches
    assert observed_b == expected_microbatches
    assert observed_rect_a == expected_microbatches
    assert observed_rect_b == expected_microbatches

    native_repeat = {
        "loss": abs(losses["native_b"] - losses["native_a"])
        / max(abs(losses["native_a"]), 1e-12),
        "logprob": metrics._logprob_metrics(
            torch.cat(logprobs["native_b"], dim=0),
            torch.cat(logprobs["native_a"], dim=0),
        ),
        "gradients": metrics._mapping_metrics(
            metrics._model_gradients(native_b), metrics._model_gradients(native_a)
        ),
    }
    rectangular_repeat = {
        "loss": abs(losses["rectangular_b"] - losses["rectangular_a"])
        / max(abs(losses["rectangular_a"]), 1e-12),
        "logprob": metrics._logprob_metrics(
            torch.cat(logprobs["rectangular_b"], dim=0),
            torch.cat(logprobs["rectangular_a"], dim=0),
        ),
        "gradients": metrics._mapping_metrics(
            metrics._model_gradients(rectangular_b),
            metrics._model_gradients(rectangular_a),
        ),
    }
    rectangular_vs_native = {
        "loss": abs(losses["rectangular_a"] - losses["native_a"])
        / max(abs(losses["native_a"]), 1e-12),
        "logprob": metrics._logprob_metrics(
            torch.cat(logprobs["rectangular_a"], dim=0),
            torch.cat(logprobs["native_a"], dim=0),
        ),
        "gradients": metrics._mapping_metrics(
            metrics._model_gradients(rectangular_a), metrics._model_gradients(native_a)
        ),
    }
    for comparison in (native_repeat, rectangular_repeat, rectangular_vs_native):
        assert math.isfinite(comparison["loss"])
        assert math.isfinite(comparison["logprob"]["relative_l2"])
        assert math.isfinite(comparison["gradients"]["relative_l2"])
        assert math.isfinite(comparison["gradients"]["cosine"])

    print("\nQwen3 one-step attention-kernel diagnostic")
    print(f"Case: P={metrics._PREFIX_LENGTH}, S={metrics._SUFFIX_LENGTH}, N={metrics._SIBLING_COUNT}")
    print(f"Adapter calls: {adapter_calls}")
    for name, comparison in (
        ("Native DotProduct B vs Native DotProduct A", native_repeat),
        ("Rectangular CANN B vs Rectangular CANN A", rectangular_repeat),
        ("Rectangular CANN A vs Native DotProduct A", rectangular_vs_native),
    ):
        print(f"{name}:")
        print(f"  loss relative diff:     {comparison['loss']:.6e}")
        print(f"  logprob relative L2:    {comparison['logprob']['relative_l2']:.6e}")
        print(f"  logprob max abs diff:   {comparison['logprob']['max_abs']:.6e}")
        print(f"  gradient relative L2:  {comparison['gradients']['relative_l2']:.6e}")
        print(f"  gradient cosine:       {comparison['gradients']['cosine']:.9f}")
        print(f"  gradient max abs diff: {comparison['gradients']['max_abs']:.6e}")
