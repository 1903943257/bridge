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

from types import SimpleNamespace

import pytest

from verl.models.mcore.tpr import (
    ALLGATHER_CP_BACKEND,
    AllGatherCPBackend,
    PrefixShard,
    RangeSequenceShard,
    TPRCPBackend,
)
from verl.models.mcore.tpr.parallel import resolve_tpr_cp_backend


class _CustomBackend:
    backend_name = "test-range"
    parallel_size = 2
    parallel_rank = 0

    def validate_segment_length(self, global_length):
        assert global_length > 0

    def make_sequence_shard(self, global_length):
        return RangeSequenceShard(global_length, ((0, 1),), cp_rank=0, cp_size=2)

    def make_attention_backend(self, kv_stack, **kwargs):
        del kv_stack, kwargs
        return SimpleNamespace()


def test_allgather_backend_owns_contiguous_shard_policy():
    backend = AllGatherCPBackend(object(), parallel_size=2, parallel_rank=1)

    assert isinstance(backend, TPRCPBackend)
    assert backend.backend_name == ALLGATHER_CP_BACKEND
    assert backend.make_sequence_shard(8) == PrefixShard.contiguous(8, cp_rank=1, cp_size=2)
    with pytest.raises(ValueError, match="divisible"):
        backend.validate_segment_length(7)


def test_resolver_accepts_a_structural_custom_backend():
    backend = _CustomBackend()

    assert resolve_tpr_cp_backend(
        backend,
        cp_group=object(),
        parallel_size=2,
        parallel_rank=0,
    ) is backend


@pytest.mark.parametrize("name", ["ulysses", "ring", "hybrid"])
def test_planned_backends_fail_explicitly_instead_of_falling_back(name):
    with pytest.raises(NotImplementedError, match=name):
        resolve_tpr_cp_backend(
            name,
            cp_group=object(),
            parallel_size=2,
            parallel_rank=0,
        )


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_tpr_cp_backend(
            "typo",
            cp_group=object(),
            parallel_size=2,
            parallel_rank=0,
        )
