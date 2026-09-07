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
from verl.models.mcore.tpr import LocalKVBlock, PrefixShard


def _tensor(length, heads, value=0.0):
    return torch.full((length, 1, heads, 8), value)


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
