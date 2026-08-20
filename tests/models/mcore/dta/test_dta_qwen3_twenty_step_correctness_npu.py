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

"""Opt-in 20-step correctness experiment using real Qwen3-0.6B weights."""

import os

import pytest

import test_dta_twenty_step_correctness_npu as synthetic_correctness
from test_dta_qwen3_compatibility_npu import (
    QWEN_VOCAB_SIZE,
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_QWEN_20_STEP_CORRECTNESS") != "1",
    reason="Set DTA_RUN_QWEN_20_STEP_CORRECTNESS=1 for the real Qwen3 20-step experiment",
)


def test_qwen3_tpr_matches_reference_for_twenty_adamw_steps(monkeypatch):
    # Initialize Megatron before the shared fixture installs its test-only
    # global memory buffer; Qwen model construction must not initialize twice.
    _initialize_single_rank_megatron()
    monkeypatch.setattr(synthetic_correctness.profile, "_make_model", _make_qwen_model)
    monkeypatch.setattr(synthetic_correctness.profile, "_PROFILE_VOCAB_SIZE", QWEN_VOCAB_SIZE)
    synthetic_correctness.test_tpr_matches_reference_for_twenty_adamw_steps(monkeypatch)
