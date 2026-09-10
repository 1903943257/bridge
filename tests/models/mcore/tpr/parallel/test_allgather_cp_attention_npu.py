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

"""CP=2 equivalence for TPR's differentiable AllGather attention backend.

Run with::

    torchrun --standalone --nproc_per_node=2 -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_allgather_cp_attention_npu.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import Tensor

import verl.models.mcore.tpr.parallel.allgather_attention as cp_attention
from verl.models.mcore.tpr import AllGatherCPBackend, LocalKVBlock, PrefixShard, SequenceShard
from verl.utils.device import is_torch_npu_available


_EXPECTED_WORLD_SIZE = 2
_DTYPE = torch.bfloat16
_OUTPUT_ATOL = 6e-3
_OUTPUT_RTOL = 1e-2
_GRAD_ATOL = 6e-3
_GRAD_RTOL = 2e-2


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != _EXPECTED_WORLD_SIZE,
    reason=(
        "Run with: torchrun --standalone --nproc_per_node=2 -m pytest -s -v "
        "tests/models/mcore/tpr/parallel/test_allgather_cp_attention_npu.py"
    ),
)


@dataclass
class _CollectiveCounts:
    all_gather: int = 0
    reduce_scatter: int = 0


class _CollectiveProbe:
    def __init__(self):
        self.counts = _CollectiveCounts()
        self._all_gather = cp_attention._all_gather_into_tensor
        self._reduce_scatter = cp_attention._reduce_scatter_tensor

    def __enter__(self):
        def all_gather(output, input_, group):
            self.counts.all_gather += 1
            return self._all_gather(output, input_, group)

        def reduce_scatter(output, input_, group):
            self.counts.reduce_scatter += 1
            return self._reduce_scatter(output, input_, group)

        cp_attention._all_gather_into_tensor = all_gather
        cp_attention._reduce_scatter_tensor = reduce_scatter
        return self.counts

    def __exit__(self, exc_type, exc_value, traceback):
        cp_attention._all_gather_into_tensor = self._all_gather
        cp_attention._reduce_scatter_tensor = self._reduce_scatter


@pytest.fixture(scope="module")
def cp_runtime():
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="hccl")
    if dist.get_world_size() != _EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"AllGather CP attention requires world_size=2, got {dist.get_world_size()}")
    yield torch.device("npu", local_rank), dist.group.WORLD
    dist.barrier()
    if owns_process_group:
        dist.destroy_process_group()


def _randn(shape, *, seed, device, requires_grad=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32).to(device=device, dtype=_DTYPE)
    return tensor.requires_grad_(requires_grad)


def _local_leaf(tensor: Tensor, shard: SequenceShard) -> Tensor:
    return shard.select(tensor).detach().clone().requires_grad_(True)


def _reference(query, prefix_keys, prefix_values, current_key, current_value, scale):
    query_fp32 = query.float()
    key = torch.cat((*prefix_keys, current_key), dim=0).float()
    value = torch.cat((*prefix_values, current_value), dim=0).float()
    repeats = query.shape[2] // key.shape[2]
    key = key.repeat_interleave(repeats, dim=2)
    value = value.repeat_interleave(repeats, dim=2)
    query_t = query_fp32.squeeze(1)
    key_t = key.squeeze(1)
    value_t = value.squeeze(1)
    scores = torch.einsum("qhd,khd->hqk", query_t, key_t) * scale
    prefix_length = sum(tensor.shape[0] for tensor in prefix_keys)
    query_positions = torch.arange(query.shape[0], device=query.device) + prefix_length
    key_positions = torch.arange(key.shape[0], device=query.device)
    scores.masked_fill_(key_positions[None, None, :] > query_positions[None, :, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value_t)
    return output.reshape(query.shape[0], 1, -1)


def _assert_close(actual, expected, *, gradient=False):
    torch.testing.assert_close(
        actual.detach().float(),
        expected.detach().float(),
        atol=_GRAD_ATOL if gradient else _OUTPUT_ATOL,
        rtol=_GRAD_RTOL if gradient else _OUTPUT_RTOL,
    )


@pytest.mark.parametrize(
    ("prefix_lengths", "current_length", "query_heads", "kv_heads"),
    [
        ((), 128, 2, 2),
        ((1024,), 512, 4, 2),
        ((512, 512), 256, 4, 2),
        ((127,), 63, 4, 2),
    ],
)
def test_allgather_cp_attention_matches_full_causal_forward_backward(
    cp_runtime,
    prefix_lengths,
    current_length,
    query_heads,
    kv_heads,
):
    device, cp_group = cp_runtime
    rank = dist.get_rank(cp_group)
    head_dim = 64
    scale = head_dim**-0.5
    shard_policy = AllGatherCPBackend(
        cp_group,
        parallel_size=_EXPECTED_WORLD_SIZE,
        parallel_rank=rank,
    )
    current_shard = shard_policy.make_sequence_shard(current_length)

    global_query = _randn((current_length, 1, query_heads, head_dim), seed=100, device=device)
    global_current_key = _randn((current_length, 1, kv_heads, head_dim), seed=101, device=device)
    global_current_value = _randn((current_length, 1, kv_heads, head_dim), seed=102, device=device)
    global_prefix_keys = [
        _randn((length, 1, kv_heads, head_dim), seed=200 + index * 2, device=device)
        for index, length in enumerate(prefix_lengths)
    ]
    global_prefix_values = [
        _randn((length, 1, kv_heads, head_dim), seed=201 + index * 2, device=device)
        for index, length in enumerate(prefix_lengths)
    ]

    local_query = _local_leaf(global_query, current_shard)
    local_current_key = _local_leaf(global_current_key, current_shard)
    local_current_value = _local_leaf(global_current_value, current_shard)
    local_prefix_keys = []
    local_prefix_values = []
    prefix_blocks = []
    for index, (length, global_key, global_value) in enumerate(
        zip(prefix_lengths, global_prefix_keys, global_prefix_values, strict=True)
    ):
        shard = shard_policy.make_sequence_shard(length)
        local_key = _local_leaf(global_key, shard)
        local_value = _local_leaf(global_value, shard)
        local_prefix_keys.append(local_key)
        local_prefix_values.append(local_value)
        prefix_blocks.append(LocalKVBlock(index, shard, local_key, local_value))

    global_gradient = _randn(
        (current_length, 1, query_heads * head_dim),
        seed=999,
        device=device,
    )
    local_gradient = current_shard.select(global_gradient)
    with _CollectiveProbe() as counts:
        actual = cp_attention.allgather_cp_rectangular_attention(
            local_query,
            local_current_key,
            local_current_value,
            prefix_blocks=tuple(prefix_blocks),
            current_shard=current_shard,
            cp_group=cp_group,
            softmax_scale=scale,
        )
        actual.backward(local_gradient)

    reference_query = global_query.detach().clone().requires_grad_(True)
    reference_current_key = global_current_key.detach().clone().requires_grad_(True)
    reference_current_value = global_current_value.detach().clone().requires_grad_(True)
    reference_prefix_keys = [tensor.detach().clone().requires_grad_(True) for tensor in global_prefix_keys]
    reference_prefix_values = [tensor.detach().clone().requires_grad_(True) for tensor in global_prefix_values]
    expected = _reference(
        reference_query,
        reference_prefix_keys,
        reference_prefix_values,
        reference_current_key,
        reference_current_value,
        scale,
    )
    expected.backward(global_gradient.float())

    _assert_close(actual, current_shard.select(expected))
    _assert_close(local_query.grad, current_shard.select(reference_query.grad), gradient=True)
    _assert_close(
        local_current_key.grad,
        current_shard.select(reference_current_key.grad),
        gradient=True,
    )
    _assert_close(
        local_current_value.grad,
        current_shard.select(reference_current_value.grad),
        gradient=True,
    )
    for index, length in enumerate(prefix_lengths):
        shard = shard_policy.make_sequence_shard(length)
        _assert_close(
            local_prefix_keys[index].grad,
            shard.select(reference_prefix_keys[index].grad),
            gradient=True,
        )
        _assert_close(
            local_prefix_values[index].grad,
            shard.select(reference_prefix_values[index].grad),
            gradient=True,
        )
        assert torch.count_nonzero(local_prefix_keys[index].grad).item() > 0
        assert torch.count_nonzero(local_prefix_values[index].grad).item() > 0

    expected_collectives = 2 * (len(prefix_lengths) + 1)
    assert counts.all_gather == expected_collectives
    assert counts.reduce_scatter == expected_collectives


def test_rank_local_query_uses_the_correct_right_down_causal_boundary(cp_runtime):
    device, cp_group = cp_runtime
    rank = dist.get_rank(cp_group)
    heads, head_dim = 1, 64
    prefix_length, current_length = 6, 4
    prefix_shard = PrefixShard.contiguous(prefix_length, cp_rank=rank, cp_size=_EXPECTED_WORLD_SIZE)
    current_shard = PrefixShard.contiguous(current_length, cp_rank=rank, cp_size=_EXPECTED_WORLD_SIZE)
    global_query = torch.zeros(current_length, 1, heads, head_dim, device=device, dtype=_DTYPE)
    global_key = torch.zeros_like(global_query)
    positions = torch.arange(prefix_length + current_length, device=device, dtype=torch.float32).to(_DTYPE)
    global_value = positions[:, None, None, None].expand(-1, 1, heads, head_dim).contiguous()
    prefix_key = torch.zeros(prefix_length, 1, heads, head_dim, device=device, dtype=_DTYPE)
    prefix_value = global_value[:prefix_length]
    current_key = global_key
    current_value = global_value[prefix_length:]

    actual = cp_attention.allgather_cp_rectangular_attention(
        global_query[current_shard.local_start : current_shard.local_end].contiguous(),
        current_key[current_shard.local_start : current_shard.local_end].contiguous(),
        current_value[current_shard.local_start : current_shard.local_end].contiguous(),
        prefix_blocks=(
            LocalKVBlock(
                0,
                prefix_shard,
                prefix_key[prefix_shard.local_start : prefix_shard.local_end].contiguous(),
                prefix_value[prefix_shard.local_start : prefix_shard.local_end].contiguous(),
            ),
        ),
        current_shard=current_shard,
        cp_group=cp_group,
    )

    absolute_query_positions = torch.arange(
        prefix_length + current_shard.local_start,
        prefix_length + current_shard.local_end,
        device=device,
        dtype=torch.float32,
    )
    expected = (absolute_query_positions / 2)[:, None, None].expand(-1, 1, heads * head_dim)
    torch.testing.assert_close(actual.float(), expected, atol=5e-3, rtol=5e-3)
