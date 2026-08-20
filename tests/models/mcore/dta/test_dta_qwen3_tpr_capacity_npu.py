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

"""Opt-in TPR-only capacity smoke for real Qwen3-0.6B at an 8K + 8K path."""

import gc
import math
import os

import pytest
import torch
from tensordict import TensorDict

from test_dta_engine_profile_npu import _ProfileFusedCausalAttention
from test_dta_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _make_engine,
    _make_plan,
)
from test_dta_qwen3_compatibility_npu import (
    QWEN_VOCAB_SIZE,
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.dta import DTA_REQUEST_KEY, DTAForwardBackwardRequest
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_QWEN_TPR_CAPACITY") != "1",
    reason="Set DTA_RUN_QWEN_TPR_CAPACITY=1 for the real Qwen3 TPR capacity smoke",
)

_PREFIX_LENGTH = 8192
_SUFFIX_LENGTH = 8192
_SIBLING_COUNT = 2


def _tokens(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long, device=device) % QWEN_VOCAB_SIZE


def _gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def test_qwen3_tpr_runs_8k_prefix_8k_suffix_engine_step(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")

    # Megatron must own the global memory buffer before the test-only runtime
    # stubs are installed; otherwise model construction attempts a second init.
    _initialize_single_rank_megatron()
    _install_single_rank_runtime(monkeypatch, device)

    model = _make_qwen_model(
        device,
        dta=True,
        max_sequence_length=_PREFIX_LENGTH + _SUFFIX_LENGTH,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    _configure_model_runtime(model)
    engine = _make_engine(model, dta_enabled=True, monkeypatch=monkeypatch)

    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffixes = tuple(
        _tokens(50000 + sibling * 20000, _SUFFIX_LENGTH, device)
        for sibling in range(_SIBLING_COUNT)
    )
    plan = _make_plan(prefix, *suffixes)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(plan)})

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
    assert output["metrics"]["dta_peak_path_tokens"] == _PREFIX_LENGTH + _SUFFIX_LENGTH
    assert output["metrics"]["dta_segment_count"] == 1 + _SIBLING_COUNT
    assert output["metrics"]["dta_direct_leaf_count"] == _SIBLING_COUNT

    gradient_tensor_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}"
        gradient_tensor_count += 1
    assert gradient_tensor_count > 0

    print("\nQwen3-0.6B TPR-only capacity smoke")
    print(f"Case: P={_PREFIX_LENGTH}, S={_SUFFIX_LENGTH}, N={_SIBLING_COUNT}")
    print(f"Loss: {loss:.9f}")
    print(f"Gradient tensors checked: {gradient_tensor_count}")
    print(f"Baseline allocated: {_gib(baseline_allocated):.3f} GiB")
    print(f"Peak allocated: {_gib(peak_allocated):.3f} GiB")
    print(f"Incremental peak: {_gib(peak_allocated - baseline_allocated):.3f} GiB")
