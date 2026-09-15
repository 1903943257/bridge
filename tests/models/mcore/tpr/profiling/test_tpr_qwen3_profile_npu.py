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

"""Opt-in controlled profile using Qwen3-0.6B, Qwen3-1.7B or Qwen3-4B.

Select a size with TPR_QWEN_PROFILE_SIZE (default: 0.6B). Checkpoints default
to /workspace/hf_models/Qwen3-<size>; TPR_QWEN_MODEL_PATH overrides that path.
For example, on the NPU host::

    TPR_RUN_PROFILE=1 TPR_RUN_QWEN_PROFILE=1 TPR_QWEN_PROFILE_SIZE=1.7B \\
        pytest -s tests/models/mcore/tpr/profiling/test_tpr_qwen3_profile_npu.py

Run each size in a separate process. The profile includes full-vocabulary
logits and loss, with the same sequence cases as the 0.6B profile.
"""

import os
from pathlib import Path

import pytest
from transformers import AutoConfig

from . import test_tpr_engine_profile_npu as synthetic_profile
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

_QWEN_PARAMETER_RANGES = {
    "0.6B": (500_000_000, 700_000_000),
    "1.7B": (1_500_000_000, 1_900_000_000),
    "4B": (3_500_000_000, 4_500_000_000),
}


def test_real_qwen3_tpr_engine_profile(monkeypatch):
    size = os.getenv("TPR_QWEN_PROFILE_SIZE", "0.6B")
    if size not in _QWEN_PARAMETER_RANGES:
        raise ValueError(
            f"unsupported TPR_QWEN_PROFILE_SIZE={size!r}; "
            f"choose from {tuple(_QWEN_PARAMETER_RANGES)}"
        )
    model_path = Path(os.getenv("TPR_QWEN_MODEL_PATH", f"/workspace/hf_models/Qwen3-{size}"))
    qwen_fixture._validate_checkpoint_files(model_path)
    hf_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    if hf_config.model_type != "qwen3":
        raise ValueError(f"expected a dense Qwen3 checkpoint, got {hf_config.model_type!r}")
    min_parameters, max_parameters = _QWEN_PARAMETER_RANGES[size]

    def assert_model_scale(model):
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert min_parameters <= parameter_count <= max_parameters, (
            f"Qwen3-{size} profile expected {min_parameters / 1e9:.1f}B to "
            f"{max_parameters / 1e9:.1f}B parameters, got {parameter_count / 1e9:.3f}B; "
            f"check TPR_QWEN_PROFILE_SIZE and TPR_QWEN_MODEL_PATH={model_path}"
        )
        return parameter_count

    print(f"\nQwen3-{size} profile: checkpoint={model_path}")
    print(
        f"layers={hf_config.num_hidden_layers}, hidden={hf_config.hidden_size}, "
        f"vocab={hf_config.vocab_size}, tied_embeddings={hf_config.tie_word_embeddings}"
    )
    monkeypatch.setattr(qwen_fixture, "QWEN_MODEL_PATH", model_path)
    # Initialize Megatron before the synthetic profile installs its test-only
    # global memory buffer; otherwise model construction attempts a second init.
    _initialize_single_rank_megatron()
    monkeypatch.setattr(synthetic_profile, "_make_model", _make_qwen_model)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_CASES", _QWEN_PROFILE_CASES)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_NUM_LAYERS", hf_config.num_hidden_layers)
    monkeypatch.setattr(synthetic_profile, "_PROFILE_VOCAB_SIZE", hf_config.vocab_size)
    monkeypatch.setattr(synthetic_profile, "_assert_profile_model_scale", assert_model_scale)
    synthetic_profile.test_tpr_engine_profile_report(monkeypatch)
