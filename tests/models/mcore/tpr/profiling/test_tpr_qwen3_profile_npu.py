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

"""Opt-in controlled profile using real Qwen3-1.7B or Qwen3-4B only.

TPR_QWEN_PROFILE_SIZE defaults to 1.7B. Checkpoints default to
/workspace/hf_models/Qwen3-<size>; TPR_QWEN_MODEL_PATH overrides that path.

Example::

    TPR_RUN_QWEN_PROFILE=1 TPR_QWEN_PROFILE_SIZE=1.7B \
        pytest -s tests/models/mcore/tpr/profiling/test_tpr_qwen3_profile_npu.py

Run each size in a separate process. Full-vocabulary logits and loss are part of
the measured training path. Synthetic-model profiling is intentionally disabled.
"""

import os

import pytest

from . import test_tpr_engine_profile_npu as profile_harness
from ._qwen3_profile_target import resolve_qwen3_profile_target
from ..correctness import test_tpr_qwen3_compatibility_npu as qwen_fixture
from ..correctness.test_tpr_qwen3_compatibility_npu import (
    _initialize_single_rank_megatron,
    _make_qwen_model,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_PROFILE") != "1",
    reason="Set TPR_RUN_QWEN_PROFILE=1 for the real Qwen3 profile",
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


def test_real_qwen3_tpr_engine_profile(monkeypatch):
    target = resolve_qwen3_profile_target()
    hf_config = target.hf_config

    print(f"\n{target.label} profile: checkpoint={target.path}")
    print(
        f"layers={hf_config.num_hidden_layers}, hidden={hf_config.hidden_size}, "
        f"vocab={hf_config.vocab_size}, tied_embeddings={hf_config.tie_word_embeddings}"
    )
    monkeypatch.setattr(qwen_fixture, "QWEN_MODEL_PATH", target.path)
    _initialize_single_rank_megatron()
    monkeypatch.setattr(profile_harness, "_make_model", _make_qwen_model)
    monkeypatch.setattr(profile_harness, "_PROFILE_CASES", _QWEN_PROFILE_CASES)
    monkeypatch.setattr(profile_harness, "_PROFILE_NUM_LAYERS", hf_config.num_hidden_layers)
    monkeypatch.setattr(profile_harness, "_PROFILE_VOCAB_SIZE", hf_config.vocab_size)
    monkeypatch.setattr(
        profile_harness,
        "_assert_profile_model_scale",
        target.assert_model_scale,
    )
    profile_harness._run_tpr_engine_profile_report(monkeypatch)
