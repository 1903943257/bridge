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

"""Differentiable contiguous AllGather CP for rectangular TPR attention."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ..shard import (
    PrefixShard,
    SequenceShard,
    iter_valid_sequence_slices,
    physical_sequence_shard,
)
from ..rectangular_attention import rectangular_causal_attention


@dataclass(frozen=True, slots=True)
class LocalKVBlock:
    """One segment's local post-RoPE KV shard for a single layer."""

    segment_id: int
    shard: SequenceShard
    key: Tensor
    value: Tensor


def _group_world_size_and_rank(process_group: Any) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        if process_group is not None:
            raise RuntimeError("a CP process group requires torch.distributed to be initialized")
        return 1, 0
    return dist.get_world_size(process_group), dist.get_rank(process_group)


def _all_gather_into_tensor(output: Tensor, input_: Tensor, process_group: Any) -> None:
    dist.all_gather_into_tensor(output, input_, group=process_group)


def _reduce_scatter_tensor(output: Tensor, input_: Tensor, process_group: Any) -> None:
    dist.reduce_scatter_tensor(output, input_, op=dist.ReduceOp.SUM, group=process_group)


class _AllGatherSequence(torch.autograd.Function):
    """AllGather on sequence in forward and SUM ReduceScatter in backward."""

    @staticmethod
    def forward(ctx, local_tensor: Tensor, process_group: Any) -> Tensor:
        world_size, _ = _group_world_size_and_rank(process_group)
        ctx.process_group = process_group
        ctx.world_size = world_size
        if world_size == 1:
            return local_tensor

        output_shape = list(local_tensor.shape)
        output_shape[0] *= world_size
        output = torch.empty(
            output_shape,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            memory_format=torch.contiguous_format,
        )
        _all_gather_into_tensor(output, local_tensor.contiguous(), process_group)
        return output

    @staticmethod
    def backward(ctx, global_gradient: Tensor) -> tuple[Tensor, None]:
        if ctx.world_size == 1:
            return global_gradient, None
        if global_gradient.shape[0] % ctx.world_size != 0:
            raise RuntimeError(
                f"global gradient sequence length {global_gradient.shape[0]} "
                f"is not divisible by CP size {ctx.world_size}"
            )

        output_shape = list(global_gradient.shape)
        output_shape[0] //= ctx.world_size
        local_gradient = torch.empty(
            output_shape,
            dtype=global_gradient.dtype,
            device=global_gradient.device,
            memory_format=torch.contiguous_format,
        )
        _reduce_scatter_tensor(local_gradient, global_gradient.contiguous(), ctx.process_group)
        return local_gradient, None


def all_gather_sequence(local_tensor: Tensor, process_group: Any) -> Tensor:
    """Gather uniform contiguous sequence shards with an autograd inverse."""

    if not isinstance(local_tensor, Tensor):
        raise TypeError(f"local_tensor must be a torch.Tensor, got {type(local_tensor).__name__}")
    if local_tensor.ndim == 0 or local_tensor.shape[0] <= 0:
        raise ValueError(f"local_tensor must have a non-empty sequence dimension, got {local_tensor.shape}")
    return _AllGatherSequence.apply(local_tensor, process_group)


def _validate_kv_tensor_pair(
    key: Tensor,
    value: Tensor,
    *,
    name: str,
    expected_local_length: int,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"{name} key/value must be torch tensors")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"{name} key/value must have shape [sequence, batch, heads, head_dim], "
            f"got K={tuple(key.shape)}, V={tuple(value.shape)}"
        )
    if key.shape != value.shape:
        raise ValueError(f"{name} key/value shapes must match, got K={key.shape}, V={value.shape}")
    if key.shape[0] != expected_local_length:
        raise ValueError(
            f"{name} local sequence length must be {expected_local_length}, got {key.shape[0]}"
        )
    if key.shape[1] != 1:
        raise ValueError(f"{name} batch size must be 1, got {key.shape[1]}")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError(f"{name} key/value dtype and device must match")
    if not key.is_floating_point():
        raise ValueError(f"{name} key/value must be floating point, got {key.dtype}")


def _validate_shard(shard: SequenceShard, *, cp_size: int, cp_rank: int, name: str) -> None:
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"{name} shard must implement SequenceShard, got {type(shard).__name__}")
    if not isinstance(physical_sequence_shard(shard), PrefixShard):
        raise TypeError(f"{name} shard must use contiguous physical placement")
    if shard.cp_size != cp_size or shard.cp_rank != cp_rank:
        raise ValueError(
            f"{name} shard CP metadata must match the process group: "
            f"expected rank {cp_rank}/{cp_size}, got {shard.cp_rank}/{shard.cp_size}"
        )


def _normalize_prefix_blocks(
    prefix_blocks: Sequence[LocalKVBlock],
    *,
    cp_size: int,
    cp_rank: int,
    current_key: Tensor,
) -> tuple[LocalKVBlock, ...]:
    if not isinstance(prefix_blocks, Sequence):
        raise TypeError(f"prefix_blocks must be a sequence, got {type(prefix_blocks).__name__}")
    normalized = tuple(prefix_blocks)
    seen_segment_ids: set[int] = set()
    for index, block in enumerate(normalized):
        if not isinstance(block, LocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be LocalKVBlock, got {type(block).__name__}")
        if not isinstance(block.segment_id, int) or isinstance(block.segment_id, bool) or block.segment_id < 0:
            raise ValueError(f"prefix_blocks[{index}] segment_id must be non-negative, got {block.segment_id!r}")
        if block.segment_id in seen_segment_ids:
            raise ValueError(f"duplicate prefix segment_id {block.segment_id}")
        seen_segment_ids.add(block.segment_id)
        _validate_shard(block.shard, cp_size=cp_size, cp_rank=cp_rank, name=f"prefix block {block.segment_id}")
        _validate_kv_tensor_pair(
            block.key,
            block.value,
            name=f"prefix block {block.segment_id}",
            expected_local_length=block.shard.local_length,
        )
        if block.key.shape[1:] != current_key.shape[1:]:
            raise ValueError(
                f"prefix block {block.segment_id} non-sequence shape must match current KV: "
                f"prefix={block.key.shape[1:]}, current={current_key.shape[1:]}"
            )
        if block.key.dtype != current_key.dtype or block.key.device != current_key.device:
            raise ValueError(f"prefix block {block.segment_id} dtype/device must match current KV")
    return normalized


def allgather_cp_rectangular_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[LocalKVBlock] = (),
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
) -> Tensor:
    """Run local-query causal attention over gathered prefix/current KV.

    Every prefix segment is gathered independently so a multi-level path stays
    in segment order. Current KV is truncated at this rank's local shard end;
    sparse mode 3 then right-aligns local Q to its correct global positions.
    """

    cp_size, cp_rank = _group_world_size_and_rank(cp_group)
    _validate_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
    _validate_kv_tensor_pair(
        current_key,
        current_value,
        name="current",
        expected_local_length=current_shard.local_length,
    )
    if not isinstance(query, Tensor) or query.ndim != 4:
        shape = getattr(query, "shape", None)
        raise ValueError(f"query must have shape [sequence, batch, heads, head_dim], got {shape}")
    if query.shape[0] != current_shard.local_length:
        raise ValueError(
            f"query local sequence length must be {current_shard.local_length}, got {query.shape[0]}"
        )
    if query.shape[1] != 1:
        raise ValueError(f"query batch size must be 1, got {query.shape[1]}")
    if query.device != current_key.device or query.dtype != current_key.dtype:
        raise ValueError("query dtype/device must match current KV")
    if query.shape[-1] != current_key.shape[-1]:
        raise ValueError(
            f"query and KV head dimensions must match, got Q={query.shape[-1]}, KV={current_key.shape[-1]}"
        )
    if query.shape[2] % current_key.shape[2] != 0:
        raise ValueError(
            f"query head count must be divisible by KV head count, got Q={query.shape[2]}, KV={current_key.shape[2]}"
        )
    if softmax_scale is not None and (not math.isfinite(softmax_scale) or softmax_scale <= 0):
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}")

    blocks = _normalize_prefix_blocks(
        prefix_blocks,
        cp_size=cp_size,
        cp_rank=cp_rank,
        current_key=current_key,
    )
    full_prefix_keys: list[Tensor] = []
    full_prefix_values: list[Tensor] = []
    for block in blocks:
        full_prefix_keys.append(all_gather_sequence(block.key, cp_group)[: block.shard.global_length])
        full_prefix_values.append(all_gather_sequence(block.value, cp_group)[: block.shard.global_length])

    full_current_key = all_gather_sequence(current_key, cp_group)[: current_shard.global_length]
    full_current_value = all_gather_sequence(current_value, cp_group)[: current_shard.global_length]
    output = query.reshape(query.shape[0], 1, -1) * 0.0
    valid_slices = iter_valid_sequence_slices(current_shard)
    if not valid_slices:
        dependency = full_current_key.sum() + full_current_value.sum()
        dependency = dependency + sum(item.sum() for item in (*full_prefix_keys, *full_prefix_values))
        return output + dependency * 0.0

    for query_range, local_slice in valid_slices:
        visible_current_key = full_current_key[: query_range[1]]
        visible_current_value = full_current_value[: query_range[1]]
        gathered_key = torch.cat((*full_prefix_keys, visible_current_key), dim=0)
        gathered_value = torch.cat((*full_prefix_values, visible_current_value), dim=0)
        output[local_slice] = rectangular_causal_attention(
            query[local_slice],
            gathered_key,
            gathered_value,
            softmax_scale=softmax_scale,
        )
    return output
