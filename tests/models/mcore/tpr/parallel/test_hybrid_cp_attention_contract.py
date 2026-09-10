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

import pytest
import torch

from verl.models.mcore.tpr import (
    HybridCPTopology,
    make_hybrid_sequence_shard,
    make_ring_sequence_shard,
)


def test_hybrid_shards_reconstruct_each_ring_rank_before_ulysses_head_split():
    source = torch.arange(32)
    shards = tuple(
        make_hybrid_sequence_shard(
            32,
            cp_rank=rank,
            cp_size=4,
            ulysses_degree=2,
        )
        for rank in range(4)
    )

    assert tuple(shard.global_ranges for shard in shards) == (
        ((0, 8),),
        ((24, 32),),
        ((8, 16),),
        ((16, 24),),
    )
    for ring_rank in range(2):
        first_rank = ring_rank * 2
        hybrid_ring_tokens = torch.cat(
            (
                shards[first_rank].select(source),
                shards[first_rank + 1].select(source),
            )
        )
        ring_shard = make_ring_sequence_shard(32, cp_rank=ring_rank, cp_size=2)
        torch.testing.assert_close(hybrid_ring_tokens, ring_shard.select(source))


def test_hybrid_topology_uses_mindspeed_linear_cp_coordinates():
    topology = HybridCPTopology(
        cp_group=object(),
        ulysses_group=object(),
        ring_group=object(),
        cp_size=8,
        cp_rank=6,
        ulysses_size=2,
        ulysses_rank=0,
        ring_size=4,
        ring_rank=3,
    )

    assert topology.cp_rank == topology.ring_rank * topology.ulysses_size


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_hybrid_sequence_shard(
            32,
            cp_rank=0,
            cp_size=4,
            ulysses_degree=1,
        ),
        lambda: make_hybrid_sequence_shard(
            32,
            cp_rank=0,
            cp_size=4,
            ulysses_degree=4,
        ),
        lambda: HybridCPTopology(
            cp_group=object(),
            ulysses_group=object(),
            ring_group=object(),
            cp_size=4,
            cp_rank=1,
            ulysses_size=2,
            ulysses_rank=0,
            ring_size=2,
            ring_rank=0,
        ),
    ],
)
def test_invalid_hybrid_topologies_and_shards_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
