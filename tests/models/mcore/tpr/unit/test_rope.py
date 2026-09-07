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
    RangeSequenceShard,
    build_sharded_rotary_pos_emb,
    build_suffix_rotary_pos_emb,
)


class _RecordingRotaryEmbedding:
    def __init__(self, rotary_dim=8):
        self.rotary_dim = rotary_dim
        self.calls = []

    def __call__(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        self.calls.append(
            {
                "max_seq_len": max_seq_len,
                "offset": offset,
                "packed_seq": packed_seq,
                "cp_group": cp_group,
            }
        )
        positions = torch.arange(offset, offset + max_seq_len, dtype=torch.float32)
        return positions[:, None, None, None].expand(-1, 1, 1, self.rotary_dim).clone()


@pytest.mark.parametrize(("prefix_length", "suffix_length"), [(0, 6), (6, 3), (32, 1)])
def test_suffix_rope_matches_full_rope_slice(prefix_length, suffix_length):
    rotary_embedding = _RecordingRotaryEmbedding()

    suffix_rope = build_suffix_rotary_pos_emb(
        rotary_embedding,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
    )
    full_rope = _RecordingRotaryEmbedding()(
        prefix_length + suffix_length,
        offset=0,
    )

    assert torch.equal(suffix_rope, full_rope[prefix_length : prefix_length + suffix_length])


def test_suffix_rope_calls_megatron_embedding_with_offset_contract():
    rotary_embedding = _RecordingRotaryEmbedding()

    result = build_suffix_rotary_pos_emb(
        rotary_embedding,
        prefix_length=6,
        suffix_length=3,
    )

    assert result.shape == (3, 1, 1, 8)
    assert rotary_embedding.calls == [
        {
            "max_seq_len": 3,
            "offset": 6,
            "packed_seq": False,
            "cp_group": None,
        }
    ]


def test_suffix_rope_can_disable_the_embedding_bound_cp_shard():
    rotary_embedding = _RecordingRotaryEmbedding()

    build_suffix_rotary_pos_emb(
        rotary_embedding,
        prefix_length=8,
        suffix_length=4,
        disable_context_parallel_sharding=True,
    )

    cp_group = rotary_embedding.calls[0]["cp_group"]
    assert cp_group is not None
    assert cp_group.size() == 1


def test_sharded_rope_follows_noncontiguous_local_token_order():
    rotary_embedding = _RecordingRotaryEmbedding()
    shard = RangeSequenceShard(8, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)

    result = build_sharded_rotary_pos_emb(
        rotary_embedding,
        position_start=16,
        shard=shard,
        disable_context_parallel_sharding=True,
    )

    assert result[:, 0, 0, 0].tolist() == [16.0, 17.0, 22.0, 23.0]
    assert [(call["max_seq_len"], call["offset"]) for call in rotary_embedding.calls] == [
        (2, 16),
        (2, 22),
    ]
    assert all(call["cp_group"].size() == 1 for call in rotary_embedding.calls)


@pytest.mark.parametrize(
    ("prefix_length", "suffix_length", "match"),
    [(-1, 3, "prefix_length"), (True, 3, "prefix_length"), (0, 0, "suffix_length")],
)
def test_suffix_rope_rejects_invalid_lengths(prefix_length, suffix_length, match):
    with pytest.raises(ValueError, match=match):
        build_suffix_rotary_pos_emb(
            _RecordingRotaryEmbedding(),
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )


def test_suffix_rope_rejects_nonstandard_return_type():
    def tuple_returning_rope(*args, **kwargs):
        tensor = torch.zeros(3, 1, 1, 8)
        return tensor, tensor

    with pytest.raises(TypeError, match="standard RotaryEmbedding"):
        build_suffix_rotary_pos_emb(
            tuple_returning_rope,
            prefix_length=6,
            suffix_length=3,
        )
