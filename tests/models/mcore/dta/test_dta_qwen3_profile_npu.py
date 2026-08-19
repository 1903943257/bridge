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

"""Opt-in controlled profile using the real Qwen3-0.6B architecture and weights."""

import os

import pytest

import test_dta_engine_profile_npu as synthetic_profile
from test_dta_qwen3_compatibility_npu import (
    QWEN_NUM_LAYERS,
    QWEN_VOCAB_SIZE,
    _make_qwen_model,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_QWEN_PROFILE") != "1",
    reason="Set DTA_RUN_QWEN_PROFILE=1 for the real Qwen3 profile",
)

_QWEN_PROFILE_CASES = (
    (16384, 512, 2),
    (16384, 512, 4),
    (16384, 512, 8),
    (8192, 1024, 2),
    (8192, 1024, 4),
    (8192, 1024, 8),
    (8192, 8192, 2),
    (8192, 8192, 4),
    (8192, 8192, 8),
    (1024, 8192, 2),
    (1024, 8192, 4),
    (1024, 8192, 8),
)


def test_real_qwen3_dta_engine_profile(monkeypatch):
    monkeypatch.setattr(synthetic_profile, "_make_model", _make_qwen_model)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_CASES", _QWEN_PROFILE_CASES)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_NUM_LAYERS", QWEN_NUM_LAYERS)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_VOCAB_SIZE", QWEN_VOCAB_SIZE)
    synthetic_profile.test_dta_engine_profile_report(monkeypatch)
