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

"""MindSpeed Ring CP extension for external-prefix TPR attention.

MindSpeed's native causal Ring Attention assumes that Q/K/V are equal-length
shards of one sequence. TPR instead supplies Current Q and an ordered sequence
of external Prefix KV blocks plus Current KV. This module keeps MindSpeed's
RingP2P transport and online-softmax semantics, but owns the TPR-specific block
schedule and gradient routing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ..kv_stack import KVStack
from ..rectangular_attention import _compressed_causal_mask
from ..shard import RangeSequenceShard, SequenceRange
from .allgather_attention import _group_world_size_and_rank
from .execution_context import ShardedPastKVAnchors

_TND_LAYOUT = "TND"
_RIGHT_DOWN_CAUSAL_MODE = 3
_FULL_ATTENTION_MODE = 0
_MAX_TOKENS = 2**31 - 1


class RingBlockKind(str, Enum):
    """Visibility of one Current query range against one KV range."""

    FULL = "full"
    CAUSAL = "causal"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class RingLocalKVBlock:
    """One Prefix segment's local zigzag KV shard for a single layer."""

    segment_id: int
    shard: RangeSequenceShard
    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_id, int)
            or isinstance(self.segment_id, bool)
            or self.segment_id < 0
        ):
            raise ValueError(
                f"segment_id must be a non-negative integer, got {self.segment_id!r}"
            )


@dataclass(frozen=True, slots=True)
class _RingAttentionConfig:
    cp_group: Any
    global_ranks: tuple[int, ...]
    cp_size: int
    cp_rank: int
    segment_lengths: tuple[int, ...]
    current_shard: RangeSequenceShard
    query_heads: int
    head_dim: int
    softmax_scale: float


def make_ring_sequence_shard(
    global_length: int,
    *,
    cp_rank: int,
    cp_size: int,
) -> RangeSequenceShard:
    """Return MindSpeed's symmetric two-range causal Ring shard."""

    if not isinstance(global_length, int) or isinstance(global_length, bool) or global_length <= 0:
        raise ValueError(f"global_length must be a positive integer, got {global_length!r}")
    if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1:
        raise ValueError(f"Ring CP requires cp_size greater than one, got {cp_size!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank!r}")
    chunk_count = 2 * cp_size
    if global_length % chunk_count != 0:
        raise ValueError(
            f"length {global_length} must be divisible by 2 * CP size ({chunk_count})"
        )
    chunk_length = global_length // chunk_count
    first_start = cp_rank * chunk_length
    second_start = (chunk_count - cp_rank - 1) * chunk_length
    return RangeSequenceShard(
        global_length,
        (
            (first_start, first_start + chunk_length),
            (second_start, second_start + chunk_length),
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
    )


def classify_ring_block(
    query_range: SequenceRange,
    kv_range: SequenceRange,
    *,
    is_prefix: bool,
) -> RingBlockKind:
    """Classify FULL/CAUSAL/SKIP using logical Segment-local positions."""

    if is_prefix:
        return RingBlockKind.FULL
    query_start, query_end = query_range
    kv_start, kv_end = kv_range
    if kv_end <= query_start:
        return RingBlockKind.FULL
    if kv_start >= query_end:
        return RingBlockKind.SKIP
    if query_range != kv_range:
        raise ValueError(
            "partially overlapping Current query/KV ranges cannot use the aligned Ring schedule: "
            f"Q={query_range}, KV={kv_range}"
        )
    return RingBlockKind.CAUSAL


def _load_mindspeed_ring_primitives():
    try:
        from mindspeed.core.context_parallel.utils import RingP2P, forward_update_without_fused
    except ImportError as exc:  # pragma: no cover - requires the server MindSpeed runtime
        raise RuntimeError("MindSpeed RingP2P and online-softmax utilities are required") from exc
    return RingP2P, forward_update_without_fused


def _process_group_global_ranks(process_group: Any) -> tuple[int, ...]:
    cp_size, cp_rank = _group_world_size_and_rank(process_group)
    if hasattr(dist, "get_process_group_ranks"):
        global_ranks = tuple(dist.get_process_group_ranks(process_group))
    elif hasattr(dist, "get_global_rank"):
        global_ranks = tuple(dist.get_global_rank(process_group, rank) for rank in range(cp_size))
    elif process_group is dist.group.WORLD:
        global_ranks = tuple(range(cp_size))
    else:  # pragma: no cover - only relevant to obsolete torch.distributed versions
        raise RuntimeError("cannot resolve global ranks for the CP process group")
    if len(global_ranks) != cp_size:
        raise RuntimeError(f"CP group exposes {len(global_ranks)} global ranks, expected {cp_size}")
    if dist.get_rank() != global_ranks[cp_rank]:
        raise RuntimeError("CP process-group rank order does not match the current global rank")
    return global_ranks


def _validate_ring_shard(
    shard: RangeSequenceShard,
    *,
    cp_size: int,
    cp_rank: int,
    name: str,
) -> None:
    if not isinstance(shard, RangeSequenceShard):
        raise TypeError(f"{name} shard must be RangeSequenceShard, got {type(shard).__name__}")
    expected = make_ring_sequence_shard(
        shard.global_length,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    if shard != expected:
        raise ValueError(f"{name} shard must use the MindSpeed causal Ring layout")


def _validate_kv_pair(
    key: Tensor,
    value: Tensor,
    *,
    shard: RangeSequenceShard,
    name: str,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"{name} K/V must be torch tensors")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{name} K/V must have shape [sequence, batch, heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError(f"{name} K/V shapes must match")
    if key.shape[0] != shard.local_length or key.shape[1] != 1:
        raise ValueError(
            f"{name} K/V must start with [{shard.local_length}, 1], got {tuple(key.shape)}"
        )
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError(f"{name} K/V dtype and device must match")
    if not key.is_floating_point():
        raise ValueError(f"{name} K/V must be floating point")


def _normalize_inputs(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock],
    current_shard: RangeSequenceShard,
    cp_group: Any,
    softmax_scale: float | None,
) -> tuple[tuple[RingLocalKVBlock, ...], _RingAttentionConfig]:
    cp_size, cp_rank = _group_world_size_and_rank(cp_group)
    _validate_ring_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
    _validate_kv_pair(current_key, current_value, shard=current_shard, name="current")
    if not isinstance(query, Tensor) or query.ndim != 4:
        raise ValueError("query must have shape [sequence, batch, heads, head_dim]")
    if query.shape[0] != current_shard.local_length or query.shape[1] != 1:
        raise ValueError(
            f"query must start with [{current_shard.local_length}, 1], got {tuple(query.shape)}"
        )
    if query.dtype != current_key.dtype or query.device != current_key.device:
        raise ValueError("query dtype/device must match Current KV")
    if query.shape[-1] != current_key.shape[-1]:
        raise ValueError("query and KV head dimensions must match")
    if query.shape[2] % current_key.shape[2] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    blocks = tuple(prefix_blocks)
    seen: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, RingLocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be RingLocalKVBlock")
        if block.segment_id in seen:
            raise ValueError(f"duplicate Prefix segment_id {block.segment_id}")
        seen.add(block.segment_id)
        _validate_ring_shard(
            block.shard,
            cp_size=cp_size,
            cp_rank=cp_rank,
            name=f"Prefix block {block.segment_id}",
        )
        _validate_kv_pair(
            block.key,
            block.value,
            shard=block.shard,
            name=f"Prefix block {block.segment_id}",
        )
        if block.key.shape[1:] != current_key.shape[1:]:
            raise ValueError(f"Prefix block {block.segment_id} shape must match Current KV")
        if block.key.dtype != current_key.dtype or block.key.device != current_key.device:
            raise ValueError(f"Prefix block {block.segment_id} dtype/device must match Current KV")

    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"softmax_scale must be finite and positive, got {scale!r}")
    config = _RingAttentionConfig(
        cp_group=cp_group,
        global_ranks=_process_group_global_ranks(cp_group),
        cp_size=cp_size,
        cp_rank=cp_rank,
        segment_lengths=tuple(block.shard.global_length for block in blocks)
        + (current_shard.global_length,),
        current_shard=current_shard,
        query_heads=query.shape[2],
        head_dim=query.shape[-1],
        softmax_scale=float(scale),
    )
    return blocks, config


def _circulate_kv(
    key: Tensor,
    value: Tensor,
    config: _RingAttentionConfig,
) -> tuple[tuple[Tensor, Tensor], ...]:
    RingP2P, _ = _load_mindspeed_ring_primitives()
    ring = RingP2P(config.global_ranks, config.cp_group)
    by_source: list[tuple[Tensor, Tensor] | None] = [None] * config.cp_size
    current = torch.stack((key, value), dim=0).contiguous()
    source_rank = config.cp_rank
    for step in range(config.cp_size):
        if step == 0:
            by_source[source_rank] = (key, value)
        else:
            by_source[source_rank] = (current[0], current[1])
        if step + 1 < config.cp_size:
            received = torch.empty_like(current)
            ring.async_send_recv(current, received)
            ring.wait()
            current = received
            source_rank = (source_rank - 1) % config.cp_size
    if any(block is None for block in by_source):
        raise RuntimeError("Ring KV circulation did not visit every source rank")
    return tuple(block for block in by_source if block is not None)


def _iter_range_slices(
    tensor: Tensor,
    shard: RangeSequenceShard,
) -> tuple[tuple[SequenceRange, slice, Tensor], ...]:
    result = []
    local_start = 0
    for global_range in shard.global_ranges:
        length = global_range[1] - global_range[0]
        local_slice = slice(local_start, local_start + length)
        result.append((global_range, local_slice, tensor[local_slice]))
        local_start += length
    return tuple(result)


def _block_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_heads: int,
    softmax_scale: float,
    block_kind: RingBlockKind,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention") from exc
    causal = block_kind is RingBlockKind.CAUSAL
    result = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=_compressed_causal_mask(query.device) if causal else None,
        scale=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal else _MAX_TOKENS,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=_RIGHT_DOWN_CAUSAL_MODE if causal else _FULL_ATTENTION_MODE,
        actual_seq_qlen=[query.shape[0]],
        actual_seq_kvlen=[key.shape[0]],
    )
    return result[0], result[1], result[2]


def _merge_attention(
    previous: tuple[Tensor, Tensor, Tensor] | None,
    current: tuple[Tensor, Tensor, Tensor],
    *,
    query_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if previous is None:
        return current
    previous_output, previous_max, previous_sum = previous
    current_output, current_max, current_sum = current
    actual_seq_qlen = (query_length,)

    # MindSpeed 376e9cc's TND unflatten_softmax uses view() immediately
    # after transpose(), which fails on current PyTorch because that tensor is
    # non-contiguous. Keep its exact layout semantics locally and use reshape
    # at the non-contiguous boundary instead of patching MindSpeed globally.
    previous_max = _flatten_tnd_softmax(previous_max, actual_seq_qlen)
    previous_sum = _flatten_tnd_softmax(previous_sum, actual_seq_qlen)
    current_max = _flatten_tnd_softmax(current_max, actual_seq_qlen)
    current_sum = _flatten_tnd_softmax(current_sum, actual_seq_qlen)

    merged_max = torch.maximum(previous_max, current_max)
    previous_scale = torch.exp(previous_max - merged_max)
    current_scale = torch.exp(current_max - merged_max)
    previous_sum = previous_sum * previous_scale
    current_sum = current_sum * current_scale
    merged_sum = previous_sum + current_sum

    head_dim = previous_output.shape[-1]
    previous_output_scale = (previous_sum / merged_sum)[..., 0].unsqueeze(2)
    current_output_scale = (current_sum / merged_sum)[..., 0].unsqueeze(2)
    previous_output_scale = previous_output_scale.expand(-1, -1, head_dim)
    current_output_scale = current_output_scale.expand(-1, -1, head_dim)
    merged_output = previous_output * previous_output_scale
    merged_output.add_(current_output * current_output_scale)
    merged_output = merged_output.to(previous_output.dtype)
    return (
        merged_output,
        _unflatten_tnd_softmax(merged_max, actual_seq_qlen),
        _unflatten_tnd_softmax(merged_sum, actual_seq_qlen),
    )


def _flatten_tnd_softmax(
    tensor: Tensor,
    actual_seq_qlen: Sequence[int],
) -> Tensor:
    original_shape = tensor.shape
    section_lengths = tuple(length * original_shape[1] for length in actual_seq_qlen)
    flattened = tensor.reshape(-1, original_shape[-1])
    if sum(section_lengths) != flattened.shape[0]:
        raise ValueError(
            "softmax statistics do not match actual_seq_qlen: "
            f"shape={tuple(original_shape)}, lengths={tuple(actual_seq_qlen)}"
        )
    sections = flattened.split(section_lengths, dim=0)
    reordered = tuple(
        section.reshape(original_shape[1], -1, original_shape[-1]).transpose(0, 1)
        for section in sections
    )
    return torch.cat(reordered, dim=0)


def _unflatten_tnd_softmax(
    tensor: Tensor,
    actual_seq_qlen: Sequence[int],
) -> Tensor:
    original_shape = tensor.shape
    section_lengths = tuple(length * original_shape[1] for length in actual_seq_qlen)
    flattened = tensor.reshape(-1, original_shape[-1])
    if sum(section_lengths) != flattened.shape[0]:
        raise ValueError(
            "softmax statistics do not match actual_seq_qlen: "
            f"shape={tuple(original_shape)}, lengths={tuple(actual_seq_qlen)}"
        )
    sections = flattened.split(section_lengths, dim=0)
    reordered = tuple(
        section.reshape(-1, original_shape[1], original_shape[-1])
        .transpose(0, 1)
        .reshape(-1, original_shape[-1])
        for section in sections
    )
    return torch.cat(reordered, dim=0).reshape(original_shape)


def _block_attention_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    *,
    attention_output: Tensor,
    softmax_max: Tensor,
    softmax_sum: Tensor,
    query_heads: int,
    softmax_scale: float,
    block_kind: RingBlockKind,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention backward") from exc
    causal = block_kind is RingBlockKind.CAUSAL
    result = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        grad_output,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=_compressed_causal_mask(query.device) if causal else None,
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_output,
        scale_value=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal else _MAX_TOKENS,
        keep_prob=1.0,
        sparse_mode=_RIGHT_DOWN_CAUSAL_MODE if causal else _FULL_ATTENTION_MODE,
        actual_seq_qlen=[query.shape[0]],
        actual_seq_kvlen=[key.shape[0]],
    )
    return result[0], result[1], result[2]


def _reduce_ring_gradients_to_owner(
    contributions: Sequence[tuple[Tensor, Tensor]],
    config: _RingAttentionConfig,
) -> tuple[Tensor, Tensor]:
    if len(contributions) != config.cp_size:
        raise RuntimeError("Ring gradient contribution count does not match CP size")
    RingP2P, _ = _load_mindspeed_ring_primitives()
    ring = RingP2P(config.global_ranks, config.cp_group)
    owner_rank = (config.cp_rank - 1) % config.cp_size
    key_grad, value_grad = contributions[owner_rank]
    accumulated = torch.stack((key_grad, value_grad), dim=0).contiguous()
    for step in range(config.cp_size - 1):
        received = torch.empty_like(accumulated)
        ring.async_send_recv(accumulated, received)
        ring.wait()
        owner_rank = (config.cp_rank - step - 2) % config.cp_size
        next_key_grad, next_value_grad = contributions[owner_rank]
        received[0].add_(next_key_grad)
        received[1].add_(next_value_grad)
        accumulated = received
    return accumulated[0], accumulated[1]


class _RingTPRAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, *tensor_args: Any) -> Tensor:
        config = tensor_args[-1]
        local_kv = tensor_args[:-1]
        if not isinstance(config, _RingAttentionConfig):
            raise TypeError("last Ring attention argument must be _RingAttentionConfig")
        if len(local_kv) != 2 * len(config.segment_lengths):
            raise RuntimeError("Ring attention K/V argument count does not match Segment metadata")

        segment_blocks = []
        for index in range(len(config.segment_lengths)):
            segment_blocks.append(
                _circulate_kv(local_kv[2 * index], local_kv[2 * index + 1], config)
            )

        query_tnd = query.squeeze(1).contiguous()
        query_results = []
        query_slices = _iter_range_slices(query_tnd, config.current_shard)
        for query_range, _, query_part in query_slices:
            merged = None
            for segment_index, source_blocks in enumerate(segment_blocks):
                is_prefix = segment_index + 1 < len(segment_blocks)
                global_length = config.segment_lengths[segment_index]
                for source_rank, (source_key, source_value) in enumerate(source_blocks):
                    source_shard = make_ring_sequence_shard(
                        global_length,
                        cp_rank=source_rank,
                        cp_size=config.cp_size,
                    )
                    key_slices = _iter_range_slices(source_key.squeeze(1), source_shard)
                    value_slices = _iter_range_slices(source_value.squeeze(1), source_shard)
                    for key_item, value_item in zip(key_slices, value_slices, strict=True):
                        kv_range, _, key_part = key_item
                        _, _, value_part = value_item
                        block_kind = classify_ring_block(
                            query_range,
                            kv_range,
                            is_prefix=is_prefix,
                        )
                        if block_kind is RingBlockKind.SKIP:
                            continue
                        current = _block_attention_forward(
                            query_part,
                            key_part,
                            value_part,
                            query_heads=config.query_heads,
                            softmax_scale=config.softmax_scale,
                            block_kind=block_kind,
                        )
                        merged = _merge_attention(
                            merged,
                            current,
                            query_length=query_part.shape[0],
                        )
            if merged is None:
                raise RuntimeError(f"query range {query_range} has no visible KV block")
            query_results.append(merged)

        saved_blocks = []
        for source_blocks in segment_blocks:
            for key, value in source_blocks:
                saved_blocks.extend((key, value))
        saved_results = []
        for output, softmax_max, softmax_sum in query_results:
            saved_results.extend((output, softmax_max, softmax_sum))
        ctx.save_for_backward(query, *saved_blocks, *saved_results)
        ctx.config = config
        ctx.block_tensor_count = len(saved_blocks)
        output = torch.cat(tuple(result[0] for result in query_results), dim=0)
        return output.reshape(query.shape[0], 1, config.query_heads * config.head_dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        config = ctx.config
        saved = ctx.saved_tensors
        query = saved[0]
        block_tensors = saved[1 : 1 + ctx.block_tensor_count]
        result_tensors = saved[1 + ctx.block_tensor_count :]
        segment_blocks = []
        tensor_offset = 0
        for _ in config.segment_lengths:
            source_blocks = []
            for _ in range(config.cp_size):
                source_blocks.append(
                    (block_tensors[tensor_offset], block_tensors[tensor_offset + 1])
                )
                tensor_offset += 2
            segment_blocks.append(tuple(source_blocks))
        query_results = tuple(
            tuple(result_tensors[index : index + 3])
            for index in range(0, len(result_tensors), 3)
        )

        query_tnd = query.squeeze(1).contiguous()
        grad_output_tnd = grad_output.reshape(
            query.shape[0],
            config.query_heads,
            config.head_dim,
        ).contiguous()
        query_gradient = torch.zeros_like(query_tnd)
        segment_contributions = [
            [
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in source_blocks
            ]
            for source_blocks in segment_blocks
        ]

        query_slices = _iter_range_slices(query_tnd, config.current_shard)
        grad_slices = _iter_range_slices(grad_output_tnd, config.current_shard)
        for query_item, grad_item, final_result in zip(
            query_slices,
            grad_slices,
            query_results,
            strict=True,
        ):
            query_range, query_local_slice, query_part = query_item
            _, _, grad_part = grad_item
            final_output, final_max, final_sum = final_result
            for segment_index, source_blocks in enumerate(segment_blocks):
                is_prefix = segment_index + 1 < len(segment_blocks)
                global_length = config.segment_lengths[segment_index]
                for source_rank, (source_key, source_value) in enumerate(source_blocks):
                    source_shard = make_ring_sequence_shard(
                        global_length,
                        cp_rank=source_rank,
                        cp_size=config.cp_size,
                    )
                    key_slices = _iter_range_slices(source_key.squeeze(1), source_shard)
                    value_slices = _iter_range_slices(source_value.squeeze(1), source_shard)
                    for key_item, value_item in zip(key_slices, value_slices, strict=True):
                        kv_range, kv_local_slice, key_part = key_item
                        _, _, value_part = value_item
                        block_kind = classify_ring_block(
                            query_range,
                            kv_range,
                            is_prefix=is_prefix,
                        )
                        if block_kind is RingBlockKind.SKIP:
                            continue
                        query_grad, key_grad, value_grad = _block_attention_backward(
                            query_part,
                            key_part,
                            value_part,
                            grad_part,
                            attention_output=final_output,
                            softmax_max=final_max,
                            softmax_sum=final_sum,
                            query_heads=config.query_heads,
                            softmax_scale=config.softmax_scale,
                            block_kind=block_kind,
                        )
                        query_gradient[query_local_slice].add_(query_grad)
                        key_buffer, value_buffer = segment_contributions[segment_index][source_rank]
                        key_buffer[kv_local_slice].add_(key_grad.unsqueeze(1))
                        value_buffer[kv_local_slice].add_(value_grad.unsqueeze(1))

        local_gradients = []
        for contributions in segment_contributions:
            local_key_grad, local_value_grad = _reduce_ring_gradients_to_owner(
                contributions,
                config,
            )
            local_gradients.extend((local_key_grad, local_value_grad))
        return (query_gradient.unsqueeze(1), *local_gradients, None)


def ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock] = (),
    current_shard: RangeSequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
) -> Tensor:
    """Run blockwise TPR attention over MindSpeed causal Ring shards."""

    blocks, config = _normalize_inputs(
        query,
        current_key,
        current_value,
        prefix_blocks=prefix_blocks,
        current_shard=current_shard,
        cp_group=cp_group,
        softmax_scale=softmax_scale,
    )
    flat_kv = []
    for block in blocks:
        flat_kv.extend((block.key, block.value))
    flat_kv.extend((current_key, current_value))
    return _RingTPRAttention.apply(query, *flat_kv, config)


class RingCPAttentionBackend:
    """Bind one TPR Segment and ordered Prefix blocks to MindSpeed Ring CP."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: RangeSequenceShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[RingLocalKVBlock]],
        cp_group: Any,
    ) -> None:
        if (
            not isinstance(global_prefix_length, int)
            or isinstance(global_prefix_length, bool)
            or global_prefix_length < 0
        ):
            raise ValueError("global_prefix_length must be a non-negative integer")
        cp_size, cp_rank = _group_world_size_and_rank(cp_group)
        _validate_ring_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
        normalized = {
            layer: tuple(blocks)
            for layer, blocks in sorted(prefix_blocks_by_layer.items())
        }
        if not normalized:
            raise ValueError("prefix_blocks_by_layer must contain every TPR layer")
        path_signature = None
        for layer, blocks in normalized.items():
            if not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0:
                raise ValueError(f"layer numbers must be positive integers, got {layer!r}")
            signature = tuple(block.segment_id for block in blocks)
            if path_signature is None:
                path_signature = signature
            elif signature != path_signature:
                raise ValueError("Prefix path must be identical across Ring attention layers")
            if sum(block.shard.global_length for block in blocks) != global_prefix_length:
                raise ValueError(f"layer {layer} Prefix blocks do not cover the Prefix length")
        self._global_prefix_length = global_prefix_length
        self._current_shard = current_shard
        self._prefix_blocks_by_layer = MappingProxyType(normalized)
        self._cp_group = cp_group
        self._parallel_size = cp_size

    @property
    def global_prefix_length(self) -> int:
        return self._global_prefix_length

    @property
    def global_suffix_length(self) -> int:
        return self._current_shard.global_length

    @property
    def local_suffix_length(self) -> int:
        return self._current_shard.local_length

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    def attention(
        self,
        layer_number: int,
        query: Tensor,
        new_key: Tensor,
        new_value: Tensor,
        *,
        softmax_scale: float | None,
    ) -> Tensor:
        try:
            blocks = self._prefix_blocks_by_layer[layer_number]
        except KeyError as exc:
            raise KeyError(f"Ring CP context has no layer {layer_number}") from exc
        return ring_cp_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=blocks,
            current_shard=self._current_shard,
            cp_group=self._cp_group,
            softmax_scale=softmax_scale,
        )


def _cached_blocks(kv_stack: KVStack, layer_number: int) -> tuple[RingLocalKVBlock, ...]:
    blocks = []
    for segment_id in kv_stack.segment_ids:
        state = kv_stack.get(segment_id).kv
        if not isinstance(state.shard, RangeSequenceShard):
            raise TypeError("Ring CP requires RangeSequenceShard Prefix states")
        blocks.append(
            RingLocalKVBlock(
                segment_id,
                state.shard,
                *state.key_values[layer_number],
            )
        )
    return tuple(blocks)


def make_cached_ring_cp_backend(
    kv_stack: KVStack,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: RangeSequenceShard,
    cp_group: Any,
) -> RingCPAttentionBackend:
    return RingCPAttentionBackend(
        global_prefix_length=kv_stack.prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: _cached_blocks(kv_stack, layer) for layer in expected_layer_numbers
        },
        cp_group=cp_group,
    )


def make_anchored_ring_cp_backend(
    anchors: ShardedPastKVAnchors,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: RangeSequenceShard,
    cp_group: Any,
) -> RingCPAttentionBackend:
    for entry in anchors.entries:
        if not isinstance(entry.shard, RangeSequenceShard):
            raise TypeError("Ring CP requires RangeSequenceShard Prefix anchors")
    return RingCPAttentionBackend(
        global_prefix_length=anchors.global_prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: tuple(
                RingLocalKVBlock(
                    entry.segment_id,
                    entry.shard,
                    *entry.key_values[layer],
                )
                for entry in anchors.entries
            )
            for layer in expected_layer_numbers
        },
        cp_group=cp_group,
    )
