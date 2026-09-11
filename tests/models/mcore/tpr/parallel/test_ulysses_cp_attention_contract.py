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

import verl.models.mcore.tpr.parallel.ulysses_attention as ulysses
from verl.models.mcore.tpr import LocalKVBlock, PaddedSequenceShard, PrefixShard


def _tensor(length, heads, value=0.0):
    return torch.full((length, 1, heads, 8), value)


def _padded_contiguous_shard(logical_length, *, rank):
    padded_length = ((logical_length + 1) // 2) * 2
    return PaddedSequenceShard(
        logical_length,
        PrefixShard.contiguous(padded_length, cp_rank=rank, cp_size=2),
    )


def test_prefix_and_current_are_transformed_separately_before_concatenation(monkeypatch):
    calls = []
    captured = {}

    def fake_all_to_all(tensor, group, *, scatter_dim, gather_dim, gather_size):
        del group
        calls.append((tuple(tensor.shape), scatter_dim, gather_dim, gather_size))
        if (scatter_dim, gather_dim) == (2, 0):
            return tensor[:, :, : tensor.shape[2] // 2].repeat(2, 1, 1, 1)
        local = tensor[: tensor.shape[0] // 2]
        return torch.cat((local, local), dim=2)

    def fake_attention(query, key, value, *, softmax_scale):
        captured.update(query=query, key=key, value=value, scale=softmax_scale)
        return torch.zeros(query.shape[0], 1, query.shape[2] * query.shape[3])

    monkeypatch.setattr(ulysses, "_group_world_size_and_rank", lambda group: (2, 0))
    monkeypatch.setattr(ulysses, "_mindspeed_all_to_all", fake_all_to_all)
    monkeypatch.setattr(ulysses, "rectangular_causal_attention", fake_attention)
    current_shard = PrefixShard.contiguous(4, cp_rank=0, cp_size=2)
    prefix_shard = PrefixShard.contiguous(8, cp_rank=0, cp_size=2)

    output = ulysses.ulysses_cp_rectangular_attention(
        _tensor(2, 4),
        _tensor(2, 2, 2.0),
        _tensor(2, 2, 3.0),
        prefix_blocks=(
            LocalKVBlock(7, prefix_shard, _tensor(4, 2, 1.0), _tensor(4, 2, 4.0)),
        ),
        current_shard=current_shard,
        cp_group=object(),
        softmax_scale=0.125,
    )

    assert output.shape == (2, 1, 32)
    assert captured["query"].shape == (4, 1, 2, 8)
    assert captured["key"].shape == (12, 1, 1, 8)
    assert torch.all(captured["key"][:8] == 1)
    assert torch.all(captured["key"][8:] == 2)
    assert captured["scale"] == 0.125
    assert calls == [
        ((2, 1, 4, 8), 2, 0, 4),
        ((4, 1, 2, 8), 2, 0, 8),
        ((4, 1, 2, 8), 2, 0, 8),
        ((2, 1, 2, 8), 2, 0, 4),
        ((2, 1, 2, 8), 2, 0, 4),
        ((4, 1, 2, 8), 0, 2, 4),
    ]


def test_ulysses_requires_q_and_kv_heads_divisible_by_cp(monkeypatch):
    monkeypatch.setattr(ulysses, "_group_world_size_and_rank", lambda group: (2, 0))
    shard = PrefixShard.contiguous(4, cp_rank=0, cp_size=2)

    with pytest.raises(ValueError, match="query heads"):
        ulysses.ulysses_cp_rectangular_attention(
            _tensor(2, 3),
            _tensor(2, 2),
            _tensor(2, 2),
            current_shard=shard,
            cp_group=object(),
        )

    with pytest.raises(ValueError, match="KV heads"):
        ulysses.ulysses_cp_rectangular_attention(
            _tensor(2, 6),
            _tensor(2, 3),
            _tensor(2, 3),
            current_shard=shard,
            cp_group=object(),
        )


def test_non_divisible_keeps_physical_qkv_through_attention(monkeypatch):
    rank = 1
    calls = []
    captured = {}

    def fake_all_to_all(tensor, group, *, scatter_dim, gather_dim, gather_size):
        del group
        calls.append((tuple(tensor.shape), scatter_dim, gather_dim, gather_size))
        if (scatter_dim, gather_dim) == (2, 0):
            return tensor[:, :, : tensor.shape[2] // 2].repeat(2, 1, 1, 1)
        local = tensor[tensor.shape[0] // 2 :]
        return torch.cat((local, local), dim=2)

    def fake_attention(query, key, value, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            attention_mask=kwargs["attention_mask"],
        )
        return torch.ones(query.shape[0], 1, query.shape[2] * query.shape[3])

    monkeypatch.setattr(ulysses, "_group_world_size_and_rank", lambda group: (2, rank))
    monkeypatch.setattr(ulysses, "_mindspeed_all_to_all", fake_all_to_all)
    monkeypatch.setattr(ulysses, "rectangular_causal_attention", fake_attention)
    prefix_shard = _padded_contiguous_shard(3, rank=rank)
    current_shard = _padded_contiguous_shard(3, rank=rank)

    output = ulysses.ulysses_cp_rectangular_attention(
        _tensor(2, 4),
        _tensor(2, 2, 2.0),
        _tensor(2, 2, 3.0),
        prefix_blocks=(
            LocalKVBlock(
                0,
                prefix_shard,
                _tensor(2, 2, 1.0),
                _tensor(2, 2, 4.0),
            ),
        ),
        current_shard=current_shard,
        cp_group=object(),
        softmax_scale=0.125,
    )

    assert captured["query"].shape == (4, 1, 2, 8)
    assert captured["key"].shape == (8, 1, 1, 8)
    assert captured["value"].shape == (8, 1, 1, 8)
    attention_mask = captured["attention_mask"]
    assert attention_mask.dtype == torch.bool
    assert attention_mask.shape == (4, 8)
    assert torch.all(attention_mask[:, (3, 7)])
    assert not torch.any(torch.all(attention_mask, dim=1))
    assert tuple((~attention_mask).sum(dim=1).tolist()) == (4, 5, 6, 6)
    torch.testing.assert_close(output[0], torch.ones_like(output[0]))
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert calls[-1] == ((4, 1, 2, 8), 0, 2, 4)


def test_multiple_prefix_padding_holes_do_not_hide_later_valid_kv():
    rank = 1
    first_shard = _padded_contiguous_shard(3, rank=rank)
    second_shard = _padded_contiguous_shard(5, rank=rank)
    current_shard = _padded_contiguous_shard(3, rank=rank)
    blocks = (
        LocalKVBlock(0, first_shard, _tensor(2, 2), _tensor(2, 2)),
        LocalKVBlock(1, second_shard, _tensor(3, 2), _tensor(3, 2)),
    )

    attention_mask, valid_query = ulysses._physical_causal_padding_mask(
        current_shard,
        blocks,
        device=torch.device("cpu"),
        global_query=True,
    )

    assert attention_mask.shape == (4, 14)
    assert valid_query.tolist() == [True, True, True, False]
    assert torch.all(attention_mask[:, (3, 9, 13)])
    assert not attention_mask[0, 4]
    assert not attention_mask[0, 8]
    assert not attention_mask[0, 10]
    assert int((~attention_mask[0]).sum()) == 9
