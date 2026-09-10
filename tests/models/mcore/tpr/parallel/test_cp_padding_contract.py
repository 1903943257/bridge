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

import torch

from verl.models.mcore.tpr import (
    AllGatherCPBackend,
    KVPrefixState,
    PaddedSequenceShard,
    iter_valid_sequence_slices,
    make_hybrid_sequence_shard,
    make_ring_sequence_shard,
    round_up_sequence_length,
)


def test_round_up_sequence_length_uses_backend_alignment():
    assert round_up_sequence_length(7, 2) == 8
    assert round_up_sequence_length(14, 4) == 16
    assert round_up_sequence_length(16, 4) == 16


def test_contiguous_padding_keeps_logical_and_physical_lengths_separate():
    source = torch.arange(7)
    rank_zero = AllGatherCPBackend(object(), parallel_size=2, parallel_rank=0).make_sequence_shard(7)
    rank_one = AllGatherCPBackend(object(), parallel_size=2, parallel_rank=1).make_sequence_shard(7)

    assert isinstance(rank_zero, PaddedSequenceShard)
    assert isinstance(rank_one, PaddedSequenceShard)
    assert (rank_zero.global_length, rank_zero.padded_length) == (7, 8)
    assert (rank_one.local_length, rank_one.valid_local_length) == (4, 3)
    assert rank_zero.select(source).tolist() == [0, 1, 2, 3]
    assert rank_one.select(source).tolist() == [4, 5, 6, 0]
    assert rank_one.global_indices().tolist() == [4, 5, 6, 6]
    assert iter_valid_sequence_slices(rank_one) == (((4, 7), slice(0, 3)),)


def test_ring_padding_preserves_zigzag_physical_blocks_and_valid_ranges():
    source = torch.arange(14)
    rank_zero = make_ring_sequence_shard(14, cp_rank=0, cp_size=2)
    rank_one = make_ring_sequence_shard(14, cp_rank=1, cp_size=2)

    assert isinstance(rank_zero, PaddedSequenceShard)
    assert rank_zero.padded_length == rank_one.padded_length == 16
    assert rank_zero.physical_global_ranges == ((0, 4), (12, 16))
    assert rank_zero.global_ranges == ((0, 4), (12, 14))
    assert rank_zero.select(source).tolist() == [0, 1, 2, 3, 12, 13, 0, 0]
    assert rank_one.select(source).tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


def test_hybrid_padding_uses_one_shared_cp4_physical_length():
    source = torch.arange(30)
    shards = tuple(
        make_hybrid_sequence_shard(30, cp_rank=rank, cp_size=4, ulysses_degree=2)
        for rank in range(4)
    )

    assert all(isinstance(shard, PaddedSequenceShard) for shard in shards)
    assert {shard.padded_length for shard in shards} == {32}
    assert tuple(shard.physical_global_ranges for shard in shards) == (
        ((0, 8),),
        ((24, 32),),
        ((8, 16),),
        ((16, 24),),
    )
    assert shards[1].valid_local_length == 6
    assert shards[1].select(source).tolist() == [24, 25, 26, 27, 28, 29, 0, 0]


def test_padding_offsets_are_not_owned_by_the_logical_shard():
    shard = AllGatherCPBackend(object(), parallel_size=4, parallel_rank=3).make_sequence_shard(1)

    assert shard.local_length == 1
    assert shard.valid_local_length == 0
    assert shard.global_ranges == ()
    assert shard.select(torch.tensor([9])).tolist() == [0]


def test_kv_prefix_state_tracks_logical_length_with_physical_padding():
    shard = AllGatherCPBackend(object(), parallel_size=2, parallel_rank=1).make_sequence_shard(7)
    key = torch.zeros(shard.local_length, 1, 2, 4)
    value = torch.ones_like(key)

    state = KVPrefixState(3, 7, {1: (key, value)}, shard=shard)

    assert state.global_length == 7
    assert state.local_length == 4
    assert state.shard.valid_local_length == 3
    assert state.key_values[1][0].shape[0] == 4
