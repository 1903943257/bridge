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
    RingBlockKind,
    classify_ring_block,
    make_ring_sequence_shard,
)


def test_mindspeed_causal_ring_shard_uses_symmetric_ranges():
    rank_zero = make_ring_sequence_shard(16, cp_rank=0, cp_size=2)
    rank_one = make_ring_sequence_shard(16, cp_rank=1, cp_size=2)
    source = torch.arange(16)

    assert rank_zero.global_ranges == ((0, 4), (12, 16))
    assert rank_one.global_ranges == ((4, 8), (8, 12))
    assert rank_zero.select(source).tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert rank_one.select(source).tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


@pytest.mark.parametrize(
    ("query_range", "kv_range", "is_prefix", "expected"),
    [
        ((8, 12), (0, 4), False, RingBlockKind.FULL),
        ((8, 12), (8, 12), False, RingBlockKind.CAUSAL),
        ((8, 12), (12, 16), False, RingBlockKind.SKIP),
        ((0, 4), (12, 16), True, RingBlockKind.FULL),
    ],
)
def test_ring_block_visibility(query_range, kv_range, is_prefix, expected):
    assert classify_ring_block(query_range, kv_range, is_prefix=is_prefix) is expected


def test_ring_block_visibility_rejects_partial_overlap():
    with pytest.raises(ValueError, match="partially overlapping"):
        classify_ring_block((4, 8), (6, 10), is_prefix=False)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_ring_sequence_shard(14, cp_rank=0, cp_size=2),
        lambda: make_ring_sequence_shard(16, cp_rank=2, cp_size=2),
        lambda: make_ring_sequence_shard(16, cp_rank=0, cp_size=1),
    ],
)
def test_invalid_ring_shard_requests_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()
