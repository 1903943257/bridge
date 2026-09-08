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

"""CP=2 correctness for the MindSpeed Ring TPR attention extension.

Run with::

    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29505 \
        -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist
from torch import Tensor

import verl.models.mcore.tpr.parallel.ring_attention as ring
from verl.models.mcore.tpr import RingLocalKVBlock, make_ring_sequence_shard
from verl.utils.device import is_torch_npu_available

_EXPECTED_WORLD_SIZE = 2
_DTYPE = torch.bfloat16
_OUTPUT_ATOL = 8e-3
_OUTPUT_RTOL = 2e-2
_GRAD_ATOL = 1e-2
_GRAD_RTOL = 3e-2


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != _EXPECTED_WORLD_SIZE,
    reason="Run this test with torchrun --nproc_per_node=2",
)


@pytest.fixture(scope="module")
def cp_runtime():
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="hccl")
    if dist.get_world_size() != _EXPECTED_WORLD_SIZE:
        raise RuntimeError(f"Ring CP test requires world_size=2, got {dist.get_world_size()}")
    yield torch.device("npu", local_rank), dist.group.WORLD
    dist.barrier()
    if owns_process_group:
        dist.destroy_process_group()


def _randn(shape, *, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=_DTYPE)


def _local_leaf(tensor: Tensor, shard) -> Tensor:
    return shard.select(tensor).detach().clone().requires_grad_(True)


def _reference(query, prefix_keys, prefix_values, current_key, current_value, scale):
    key = torch.cat((*prefix_keys, current_key), dim=0).float()
    value = torch.cat((*prefix_values, current_value), dim=0).float()
    repeats = query.shape[2] // key.shape[2]
    key = key.repeat_interleave(repeats, dim=2)
    value = value.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("qhd,khd->hqk", query.float().squeeze(1), key.squeeze(1)) * scale
    prefix_length = sum(tensor.shape[0] for tensor in prefix_keys)
    query_positions = torch.arange(query.shape[0], device=query.device) + prefix_length
    key_positions = torch.arange(key.shape[0], device=query.device)
    scores.masked_fill_(
        key_positions[None, None, :] > query_positions[None, :, None],
        float("-inf"),
    )
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value.squeeze(1))
    return output.reshape(query.shape[0], 1, -1)


def _assert_close(actual, expected, *, gradient=False):
    torch.testing.assert_close(
        actual.detach().float(),
        expected.detach().float(),
        atol=_GRAD_ATOL if gradient else _OUTPUT_ATOL,
        rtol=_GRAD_RTOL if gradient else _OUTPUT_RTOL,
    )


@contextmanager
def _kernel_probe():
    original_forward = ring._block_attention_forward
    original_backward = ring._block_attention_backward
    calls = {"forward": [], "backward": []}

    def traced_forward(*args, **kwargs):
        calls["forward"].append(kwargs["block_kind"])
        return original_forward(*args, **kwargs)

    def traced_backward(*args, **kwargs):
        calls["backward"].append(kwargs["block_kind"])
        return original_backward(*args, **kwargs)

    ring._block_attention_forward = traced_forward
    ring._block_attention_backward = traced_backward
    try:
        yield calls
    finally:
        ring._block_attention_forward = original_forward
        ring._block_attention_backward = original_backward


@pytest.mark.parametrize(
    ("prefix_lengths", "current_length", "query_heads", "kv_heads"),
    [
        ((), 128, 4, 2),
        ((1024,), 512, 4, 2),
        ((512, 512), 256, 4, 2),
    ],
)
def test_ring_cp_attention_matches_full_causal_forward_backward(
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
    current_shard = make_ring_sequence_shard(
        current_length,
        cp_rank=rank,
        cp_size=_EXPECTED_WORLD_SIZE,
    )
    global_query = _randn((current_length, 1, query_heads, head_dim), seed=100, device=device)
    global_current_key = _randn((current_length, 1, kv_heads, head_dim), seed=101, device=device)
    global_current_value = _randn((current_length, 1, kv_heads, head_dim), seed=102, device=device)
    global_prefix_keys = [
        _randn((length, 1, kv_heads, head_dim), seed=200 + 2 * index, device=device)
        for index, length in enumerate(prefix_lengths)
    ]
    global_prefix_values = [
        _randn((length, 1, kv_heads, head_dim), seed=201 + 2 * index, device=device)
        for index, length in enumerate(prefix_lengths)
    ]

    local_query = _local_leaf(global_query, current_shard)
    local_current_key = _local_leaf(global_current_key, current_shard)
    local_current_value = _local_leaf(global_current_value, current_shard)
    local_prefix_keys = []
    local_prefix_values = []
    prefix_blocks = []
    for index, (length, key, value) in enumerate(
        zip(prefix_lengths, global_prefix_keys, global_prefix_values, strict=True)
    ):
        shard = make_ring_sequence_shard(
            length,
            cp_rank=rank,
            cp_size=_EXPECTED_WORLD_SIZE,
        )
        local_key = _local_leaf(key, shard)
        local_value = _local_leaf(value, shard)
        local_prefix_keys.append(local_key)
        local_prefix_values.append(local_value)
        prefix_blocks.append(RingLocalKVBlock(index, shard, local_key, local_value))

    global_gradient = _randn(
        (current_length, 1, query_heads * head_dim),
        seed=999,
        device=device,
    )
    with _kernel_probe() as calls:
        actual = ring.ring_cp_attention(
            local_query,
            local_current_key,
            local_current_value,
            prefix_blocks=prefix_blocks,
            current_shard=current_shard,
            cp_group=cp_group,
            softmax_scale=scale,
        )
        actual.backward(current_shard.select(global_gradient))

    reference_query = global_query.detach().clone().requires_grad_(True)
    reference_current_key = global_current_key.detach().clone().requires_grad_(True)
    reference_current_value = global_current_value.detach().clone().requires_grad_(True)
    reference_prefix_keys = [item.detach().clone().requires_grad_(True) for item in global_prefix_keys]
    reference_prefix_values = [item.detach().clone().requires_grad_(True) for item in global_prefix_values]
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
        shard = make_ring_sequence_shard(
            length,
            cp_rank=rank,
            cp_size=_EXPECTED_WORLD_SIZE,
        )
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

    expected_calls = 5 + 8 * len(prefix_lengths)
    assert len(calls["forward"]) == expected_calls
    assert len(calls["backward"]) == expected_calls
    assert ring.RingBlockKind.CAUSAL in calls["forward"]
    RingP2P, _ = ring._load_mindspeed_ring_primitives()
    assert RingP2P.__module__.endswith("context_parallel.utils")

    if rank == 0:
        print(
            "\nRing TPR attention equivalence passed"
            f"\n  Prefix segments: {prefix_lengths or 'none'}"
            f"\n  Current: {current_length}, QH/KVH: {query_heads}/{kv_heads}"
            f"\n  Fused-attention blocks/rank: {expected_calls}"
        )
    dist.barrier(group=cp_group)
