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

"""Opt-in TPR-only capacity smoke for real Qwen3-1.7B/4B.

Defaults: Qwen3-1.7B from /workspace/hf_models/Qwen3-1.7B and P=S=8192.
Synthetic and Qwen3-0.6B capacity experiments are intentionally unsupported.
"""

import gc
import math
import os

import pytest
import torch
from tensordict import TensorDict

from ._qwen3_profile_target import resolve_qwen3_profile_target
from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention
from ..correctness import test_tpr_qwen3_compatibility_npu as qwen_fixture
from ..equivalence.test_tpr_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _make_engine,
    _make_plan,
)
from ..correctness.test_tpr_qwen3_compatibility_npu import (
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from ..equivalence.test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_TPR_CAPACITY") != "1",
    reason="Set TPR_RUN_QWEN_TPR_CAPACITY=1 for the real Qwen3 TPR capacity smoke",
)

_PREFIX_LENGTH = int(os.getenv("TPR_PREFIX", "8192"))
_SUFFIX_LENGTH = int(os.getenv("TPR_SUFFIX", "8192"))
_SIBLING_COUNT = int(os.getenv("TPR_SIBLINGS", "2"))


def _tokens(start: int, length: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long, device=device) % vocab_size


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def test_qwen3_tpr_capacity_engine_step(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    target = resolve_qwen3_profile_target()
    monkeypatch.setattr(qwen_fixture, "QWEN_MODEL_PATH", target.path)

    _initialize_single_rank_megatron()
    _install_single_rank_runtime(monkeypatch, device)

    model = _make_qwen_model(
        device,
        tpr=True,
        max_sequence_length=_PREFIX_LENGTH + _SUFFIX_LENGTH,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    parameter_count = target.assert_model_scale(model)
    _configure_model_runtime(model)
    engine = _make_engine(model, tpr_enabled=True, monkeypatch=monkeypatch)

    vocab_size = target.hf_config.vocab_size
    prefix = _tokens(17, _PREFIX_LENGTH, vocab_size, device)
    suffixes = tuple(
        _tokens(50000 + sibling * 20000, _SUFFIX_LENGTH, vocab_size, device)
        for sibling in range(_SIBLING_COUNT)
    )
    plan = _make_plan(prefix, *suffixes)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    baseline_allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()

    output = engine.forward_backward_batch(data, loss_function=None, forward_only=False)
    torch.npu.synchronize()
    peak_allocated = torch.npu.max_memory_allocated()

    loss = float(output["loss"])
    assert math.isfinite(loss)
    assert output["metrics"]["tpr_peak_path_tokens"] == _PREFIX_LENGTH + _SUFFIX_LENGTH
    assert output["metrics"]["tpr_segment_count"] == 1 + _SIBLING_COUNT
    assert output["metrics"]["tpr_direct_leaf_count"] == _SIBLING_COUNT

    gradient_tensor_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}"
        gradient_tensor_count += 1
    assert gradient_tensor_count > 0

    print(f"\n{target.label} TPR-only capacity smoke")
    print(f"Checkpoint: {target.path}")
    print(f"Parameters: {parameter_count / 1e9:.3f}B")
    print(f"Case: P={_PREFIX_LENGTH}, S={_SUFFIX_LENGTH}, N={_SIBLING_COUNT}")
    print(f"Loss: {loss:.9f}")
    print(f"Gradient tensors checked: {gradient_tensor_count}")
    print(f"Baseline allocated: {_gib(baseline_allocated):.3f} GiB")
    print(f"Peak allocated: {_gib(peak_allocated):.3f} GiB")
    print(f"Incremental peak: {_gib(peak_allocated - baseline_allocated):.3f} GiB")
