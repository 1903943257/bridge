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

import verl.models.mcore.tpr.parallel.allgather_attention as cp_attention
from verl.models.mcore.tpr import LocalKVBlock, PaddedSequenceShard, PrefixShard


def _tensor(length, *, heads=2, head_dim=4):
    return torch.randn(length, 1, heads, head_dim)


@pytest.mark.parametrize(("rank", "expected_kv_length"), [(0, 12), (1, 16)])
def test_each_rank_truncates_current_kv_at_its_local_end(monkeypatch, rank, expected_kv_length):
    monkeypatch.setattr(cp_attention, "_group_world_size_and_rank", lambda group: (2, rank))
    monkeypatch.setattr(
        cp_attention,
        "all_gather_sequence",
        lambda tensor, group: torch.cat((tensor, tensor), dim=0),
    )
    captured = {}

    def fake_attention(query, key, value, **kwargs):
        captured["key"] = key
        captured["value"] = value
        captured["kwargs"] = kwargs
        return query.reshape(query.shape[0], 1, -1)

    monkeypatch.setattr(cp_attention, "rectangular_causal_attention", fake_attention)
    shard = PrefixShard.contiguous(8, cp_rank=rank, cp_size=2)
    prefix_shard = PrefixShard.contiguous(8, cp_rank=rank, cp_size=2)
    query = _tensor(4)
    current_key = _tensor(4)
    current_value = _tensor(4)
    prefix = LocalKVBlock(0, prefix_shard, _tensor(4), _tensor(4))

    output = cp_attention.allgather_cp_rectangular_attention(
        query,
        current_key,
        current_value,
        prefix_blocks=(prefix,),
        current_shard=shard,
        cp_group=object(),
    )

    assert output.shape == (4, 1, 8)
    assert captured["key"].shape[0] == expected_kv_length
    assert captured["value"].shape[0] == expected_kv_length
    assert "attention_mask" not in captured["kwargs"]


def test_padding_reaches_attention_and_is_masked_from_valid_queries(monkeypatch):
    monkeypatch.setattr(cp_attention, "_group_world_size_and_rank", lambda group: (2, 1))
    monkeypatch.setattr(
        cp_attention,
        "all_gather_sequence",
        lambda tensor, group: torch.cat((tensor, tensor), dim=0),
    )
    captured = {}

    def fake_attention(query, key, value, **kwargs):
        captured["query"] = query
        captured["key"] = key
        captured["value"] = value
        captured["attention_mask"] = kwargs["attention_mask"]
        return torch.ones(query.shape[0], 1, query.shape[2] * query.shape[3])

    monkeypatch.setattr(cp_attention, "rectangular_causal_attention", fake_attention)
    physical_shard = PrefixShard.contiguous(8, cp_rank=1, cp_size=2)
    shard = PaddedSequenceShard(7, physical_shard)
    prefix = LocalKVBlock(0, shard, _tensor(4), _tensor(4))

    output = cp_attention.allgather_cp_rectangular_attention(
        _tensor(4),
        _tensor(4),
        _tensor(4),
        prefix_blocks=(prefix,),
        current_shard=shard,
        cp_group=object(),
    )

    assert captured["query"].shape[0] == 4
    assert captured["key"].shape[0] == 16
    assert captured["value"].shape[0] == 16
    attention_mask = captured["attention_mask"]
    assert attention_mask.dtype == torch.bool
    assert attention_mask.shape == (4, 16)
    assert torch.all(attention_mask[:, 7])
    assert torch.all(attention_mask[:, 15])
    assert not torch.any(torch.all(attention_mask, dim=1))
    assert tuple((~attention_mask).sum(dim=1).tolist()) == (12, 13, 14, 14)
    torch.testing.assert_close(output[:3], torch.ones_like(output[:3]))
    torch.testing.assert_close(output[3], torch.zeros_like(output[3]))


def test_multiple_prefix_blocks_are_gathered_in_path_order(monkeypatch):
    monkeypatch.setattr(cp_attention, "_group_world_size_and_rank", lambda group: (2, 0))
    monkeypatch.setattr(
        cp_attention,
        "all_gather_sequence",
        lambda tensor, group: torch.cat((tensor, tensor + 10), dim=0),
    )
    captured = {}

    def fake_attention(query, key, value, **kwargs):
        captured["key"] = key
        return query.reshape(query.shape[0], 1, -1)

    monkeypatch.setattr(cp_attention, "rectangular_causal_attention", fake_attention)
    shard = PrefixShard.contiguous(4, cp_rank=0, cp_size=2)
    first = LocalKVBlock(0, shard, torch.ones(2, 1, 2, 4), torch.ones(2, 1, 2, 4))
    second = LocalKVBlock(1, shard, torch.full((2, 1, 2, 4), 2.0), torch.ones(2, 1, 2, 4))

    cp_attention.allgather_cp_rectangular_attention(
        _tensor(2),
        _tensor(2),
        _tensor(2),
        prefix_blocks=(first, second),
        current_shard=shard,
        cp_group=object(),
    )

    key = captured["key"]
    torch.testing.assert_close(key[:2], torch.ones_like(key[:2]))
    torch.testing.assert_close(key[2:4], torch.full_like(key[2:4], 11.0))
    torch.testing.assert_close(key[4:6], torch.full_like(key[4:6], 2.0))
    torch.testing.assert_close(key[6:8], torch.full_like(key[6:8], 12.0))


@pytest.mark.parametrize(
    "build_call,match",
    [
        (
            lambda: (_tensor(3), _tensor(4), _tensor(4), (), PrefixShard.contiguous(8, cp_rank=0, cp_size=2)),
            "query local sequence length",
        ),
        (
            lambda: (
                _tensor(4),
                _tensor(4),
                _tensor(4),
                (),
                PrefixShard.contiguous(8, cp_rank=1, cp_size=2),
            ),
            "CP metadata",
        ),
        (
            lambda: (
                _tensor(4),
                _tensor(4),
                _tensor(4),
                (
                    LocalKVBlock(
                        0,
                        PrefixShard.contiguous(8, cp_rank=0, cp_size=2),
                        _tensor(3),
                        _tensor(3),
                    ),
                ),
                PrefixShard.contiguous(8, cp_rank=0, cp_size=2),
            ),
            "local sequence length",
        ),
    ],
)
def test_invalid_local_cp_contract_is_rejected(monkeypatch, build_call, match):
    monkeypatch.setattr(cp_attention, "_group_world_size_and_rank", lambda group: (2, 0))
    query, key, value, prefixes, shard = build_call()
    with pytest.raises(ValueError, match=match):
        cp_attention.allgather_cp_rectangular_attention(
            query,
            key,
            value,
            prefix_blocks=prefixes,
            current_shard=shard,
            cp_group=object(),
        )


def test_duplicate_prefix_segment_ids_are_rejected(monkeypatch):
    monkeypatch.setattr(cp_attention, "_group_world_size_and_rank", lambda group: (2, 0))
    shard = PrefixShard.contiguous(8, cp_rank=0, cp_size=2)
    block = LocalKVBlock(3, shard, _tensor(4), _tensor(4))
    with pytest.raises(ValueError, match="duplicate"):
        cp_attention.allgather_cp_rectangular_attention(
            _tensor(4),
            _tensor(4),
            _tensor(4),
            prefix_blocks=(block, block),
            current_shard=shard,
            cp_group=object(),
        )
