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

"""CP=2 equivalence tests comparing TPR CP backends with independent AllGather CP.

Run from the verl repository root with two visible NPUs::

    torchrun --standalone --nproc_per_node=2 -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_tpr_cp_equivalence_npu.py
"""

from __future__ import annotations

import pytest

from verl.utils.device import is_torch_npu_available

from ._tpr_cp_test_utils import (
    _EQUIVALENCE_CASES,
    _run_controlled_equivalence,
    cp_runtime,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.mark.parametrize(
    ("prefix_length", "first_suffix_length", "second_suffix_length"),
    _EQUIVALENCE_CASES,
)
@pytest.mark.parametrize("tpr_backend", ("allgather", "ulysses", "ring"))
def test_cp2_tpr_matches_independent_allgather_cp(
    cp_runtime,
    prefix_length,
    first_suffix_length,
    second_suffix_length,
    tpr_backend,
):
    _run_controlled_equivalence(
        cp_runtime,
        prefix_length,
        first_suffix_length,
        second_suffix_length,
        tpr_backend,
    )
