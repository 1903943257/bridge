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

import verl.models.mcore.tpr.parallel.backend as backend_module
from verl.models.mcore.tpr import (
    ALLGATHER_CP_BACKEND,
    AllGatherCPBackend,
    HybridCPBackend,
    HybridCPTopology,
    PrefixShard,
    RangeSequenceShard,
    RingCPBackend,
    TPRCPBackend,
    UlyssesCPBackend,
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
    padded = backend.make_sequence_shard(7)
    assert (padded.global_length, padded.padded_length, padded.valid_local_length) == (7, 8, 3)


def test_resolver_accepts_the_mindspeed_allgather_algorithm_name():
    backend = resolve_tpr_cp_backend(
        "kvallgather_cp_algo",
        cp_group=object(),
        parallel_size=2,
        parallel_rank=0,
    )

    assert isinstance(backend, AllGatherCPBackend)


def test_resolver_accepts_a_structural_custom_backend():
    backend = _CustomBackend()

    assert resolve_tpr_cp_backend(
        backend,
        cp_group=object(),
        parallel_size=2,
        parallel_rank=0,
    ) is backend


def test_resolver_constructs_the_ulysses_adapter():
    backend = resolve_tpr_cp_backend(
        "ulysses",
        cp_group=object(),
        parallel_size=2,
        parallel_rank=1,
    )

    assert isinstance(backend, UlyssesCPBackend)
    assert backend.backend_name == "ulysses"
    assert backend.make_sequence_shard(8) == PrefixShard.contiguous(8, cp_rank=1, cp_size=2)


def test_resolver_accepts_the_mindspeed_ulysses_algorithm_name():
    backend = resolve_tpr_cp_backend(
        "ulysses_cp_algo",
        cp_group=object(),
        parallel_size=2,
        parallel_rank=0,
    )

    assert isinstance(backend, UlyssesCPBackend)


@pytest.mark.parametrize("name", ["ring", "megatron_cp_algo"])
def test_resolver_constructs_the_ring_adapter(name):
    backend = resolve_tpr_cp_backend(
        name,
        cp_group=object(),
        parallel_size=2,
        parallel_rank=0,
    )

    assert isinstance(backend, RingCPBackend)
    assert backend.backend_name == "ring"
    assert backend.make_sequence_shard(16) == RangeSequenceShard(
        16,
        ((0, 4), (12, 16)),
        cp_rank=0,
        cp_size=2,
    )
    padded = backend.make_sequence_shard(10)
    assert (padded.global_length, padded.padded_length) == (10, 12)


@pytest.mark.parametrize("name", ["hybrid", "hybrid_cp_algo"])
def test_resolver_constructs_the_hybrid_adapter(monkeypatch, name):
    cp_group = object()
    topology = HybridCPTopology(
        cp_group=cp_group,
        ulysses_group=object(),
        ring_group=object(),
        cp_size=4,
        cp_rank=0,
        ulysses_size=2,
        ulysses_rank=0,
        ring_size=2,
        ring_rank=0,
    )
    monkeypatch.setattr(
        backend_module,
        "resolve_mindspeed_hybrid_topology",
        lambda group, **kwargs: topology,
    )

    backend = resolve_tpr_cp_backend(
        name,
        cp_group=cp_group,
        parallel_size=4,
        parallel_rank=0,
    )

    assert isinstance(backend, HybridCPBackend)
    assert backend.backend_name == "hybrid"
    assert backend.make_sequence_shard(16) == RangeSequenceShard(
        16,
        ((0, 4),),
        cp_rank=0,
        cp_size=4,
    )


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_tpr_cp_backend(
            "typo",
            cp_group=object(),
            parallel_size=2,
            parallel_rank=0,
        )
