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
import os
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
from ..shard import (
    RangeSequenceShard,
    SequenceRange,
    SequenceShard,
    maybe_pad_sequence_shard,
    physical_sequence_shard,
    round_up_sequence_length,
)
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
    shard: SequenceShard
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
    segment_padded_lengths: tuple[int, ...]
    current_shard: SequenceShard
    query_heads: int
    head_dim: int
    softmax_scale: float
    coalesce_prefix_full: bool = False
    coalesce_prefix_query: bool = False


@dataclass(frozen=True, slots=True)
class _RingRangeSlice:
    """One physical Ring chunk plus its clipped logical validity range."""

    logical_range: SequenceRange
    physical_range: SequenceRange
    local_slice: slice
    tensor: Tensor

    @property
    def physical_length(self) -> int:
        return self.physical_range[1] - self.physical_range[0]

    @property
    def valid_length(self) -> int:
        return self.logical_range[1] - self.logical_range[0]


def make_ring_sequence_shard(
    global_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    padded_length: int | None = None,
) -> SequenceShard:
    """Return MindSpeed's symmetric two-range causal Ring shard."""

    if not isinstance(global_length, int) or isinstance(global_length, bool) or global_length <= 0:
        raise ValueError(f"global_length must be a positive integer, got {global_length!r}")
    if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1:
        raise ValueError(f"Ring CP requires cp_size greater than one, got {cp_size!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank!r}")
    chunk_count = 2 * cp_size
    if padded_length is None:
        padded_length = round_up_sequence_length(global_length, chunk_count)
    if padded_length < global_length or padded_length % chunk_count != 0:
        raise ValueError(
            f"padded_length must cover {global_length} and be divisible by {chunk_count}, "
            f"got {padded_length}"
        )
    chunk_length = padded_length // chunk_count
    first_start = cp_rank * chunk_length
    second_start = (chunk_count - cp_rank - 1) * chunk_length
    physical_shard = RangeSequenceShard(
        padded_length,
        (
            (first_start, first_start + chunk_length),
            (second_start, second_start + chunk_length),
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
    )
    return maybe_pad_sequence_shard(global_length, physical_shard)


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
    shard: SequenceShard,
    *,
    cp_size: int,
    cp_rank: int,
    name: str,
) -> None:
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"{name} shard must implement SequenceShard, got {type(shard).__name__}")
    if not isinstance(physical_sequence_shard(shard), RangeSequenceShard):
        raise TypeError(f"{name} shard must use Ring range placement")
    expected = make_ring_sequence_shard(
        shard.global_length,
        cp_rank=cp_rank,
        cp_size=cp_size,
        padded_length=shard.padded_length,
    )
    if shard != expected:
        raise ValueError(f"{name} shard must use the MindSpeed causal Ring layout")


def _validate_kv_pair(
    key: Tensor,
    value: Tensor,
    *,
    shard: SequenceShard,
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
    current_shard: SequenceShard,
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
        segment_padded_lengths=tuple(block.shard.padded_length for block in blocks)
        + (current_shard.padded_length,),
        current_shard=current_shard,
        query_heads=query.shape[2],
        head_dim=query.shape[-1],
        softmax_scale=float(scale),
        coalesce_prefix_full=os.getenv("TPR_RING_COALESCE_PREFIX_FULL", "0") == "1",
        coalesce_prefix_query=os.getenv("TPR_RING_COALESCE_PREFIX_QUERY", "0") == "1",
    )
    return blocks, config


def _circulate_kv(key, value, config, consume):
    """Consume each source immediately; two reusable KV buffers, no KV cache.

    Like native Ring, launch the next transfer before FA and wait before
    reusing either buffer. The callback must not retain remote tensor views.
    """
    RingP2P, _ = _load_mindspeed_ring_primitives()
    ring = RingP2P(config.global_ranks, config.cp_group)
    current = torch.stack((key, value), dim=0).contiguous()
    received = torch.empty_like(current)
    _observe_ring_storage("forward_buffers", (current, received))
    try:
        for step in range(config.cp_size):
            if step + 1 < config.cp_size:
                ring.async_send_recv(current, received)
            source = (config.cp_rank - step) % config.cp_size
            consume(source, key if step == 0 else current[0],
                    value if step == 0 else current[1])
            if step + 1 < config.cp_size:
                ring.wait()
                current, received = received, current
    finally:
        ring.wait()


def _iter_range_slices(
    tensor: Tensor,
    shard: SequenceShard,
) -> tuple[_RingRangeSlice, ...]:
    """Keep partially padded Ring chunks physical while exposing logical ranges."""

    if tensor.shape[0] != shard.local_length:
        raise ValueError(
            f"Ring tensor length must be {shard.local_length}, got {tensor.shape[0]}"
        )
    result = []
    local_start = 0
    for physical_start, physical_end in shard.physical_global_ranges:
        physical_length = physical_end - physical_start
        local_slice = slice(local_start, local_start + physical_length)
        logical_end = min(physical_end, shard.global_length)
        if physical_start < logical_end:
            result.append(
                _RingRangeSlice(
                    logical_range=(physical_start, logical_end),
                    physical_range=(physical_start, physical_end),
                    local_slice=local_slice,
                    tensor=tensor[local_slice],
                )
            )
        local_start += physical_length
    if local_start != shard.local_length:
        raise RuntimeError("Ring physical ranges do not cover the local tensor")
    return tuple(result)


def _prefix_full_slices(tensor, shard, *, enabled):
    """View one fully valid source buffer as a single FULL-attention KV block.

    Logical zigzag ranges are non-adjacent, but their physical storage is
    adjacent. FULL visibility is independent of their logical positions.
    Synthetic ranges below describe packed positions only; callers must never
    use them for causal classification. Padded shards keep the original path.
    The local slice also maps dK/dV directly into the original source buffer.
    """
    if enabled and shard.global_length == shard.padded_length:
        length = shard.local_length
        return (_RingRangeSlice((0, length), (0, length), slice(0, length), tensor),)
    return _iter_range_slices(tensor, shard)


# Patched only in untimed probes; no event dictionaries on the hot path.
_trace_ring_block = None
_trace_ring_storage = None


def _observe_ring_storage(phase, tensors):
    """Opt-in metadata-only probe; never export owning tensor references."""
    if _trace_ring_storage is None:
        return
    storages = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
                for tensor in tensors}
    _trace_ring_storage(phase=phase, tensor_count=len(tensors),
                        storage_count=len(storages), storage_bytes=sum(storages.values()))


def _can_coalesce_prefix_query(query: Tensor, config: _RingAttentionConfig) -> bool:
    # Check the original Q view, BEFORE the baseline contiguous conversion.
    # Do not add a packing allocation just to make the new path eligible.
    return (
        config.coalesce_prefix_full
        and config.coalesce_prefix_query
        and len(config.segment_lengths) > 1
        and config.segment_lengths == config.segment_padded_lengths
        and query.squeeze(1).is_contiguous()
    )


def _slice_tnd_result(result, local_slice):
    """Slice TND output and CANN's head-major softmax statistics.

    CANN exposes statistics with shape [T,H,8] but single-sequence storage
    order [H,T,8]. A query slice is therefore NOT stats[local_slice].
    Only the small statistics need a contiguous chunk representation.
    """
    output, maximum, total = result
    length, heads = output.shape[:2]
    chunk_length = local_slice.stop - local_slice.start
    stats = tuple(
        item.view(heads, length, item.shape[-1])[:, local_slice, :]
        .contiguous().view(chunk_length, heads, item.shape[-1])
        for item in (maximum, total)
    )
    return output[local_slice], *stats


def _trace_merged_query(config, query, key, segment_index, source_rank, phase):
    if _trace_ring_block is None:
        return
    source_shard = make_ring_sequence_shard(
        config.segment_lengths[segment_index], cp_rank=source_rank,
        cp_size=config.cp_size,
        padded_length=config.segment_padded_lengths[segment_index],
    )
    q_chunk = config.current_shard.padded_length // (2 * config.cp_size)
    k_chunk = source_shard.padded_length // (2 * config.cp_size)
    _trace_ring_block(
        phase=phase, rank=config.cp_rank,
        ring_step=(config.cp_rank - source_rank) % config.cp_size,
        source_rank=source_rank, segment_index=segment_index, segment_type="prefix",
        query_range=config.current_shard.global_ranges,
        query_chunk=tuple(start // q_chunk for start, _ in config.current_shard.physical_global_ranges),
        kv_ranges=source_shard.global_ranges,
        kv_chunks=tuple(start // k_chunk for start, _ in source_shard.physical_global_ranges),
        block_type=RingBlockKind.FULL.value,
        query_length=query.shape[0], kv_length=key.shape[0], fa_called=True,
        coalesced=True, query_coalesced=True,
        query_storage_ptr=query.untyped_storage().data_ptr(),
        query_storage_offset=query.storage_offset(), query_shape=tuple(query.shape),
        query_stride=tuple(query.stride()),
    )





def _range_validity(item: _RingRangeSlice, *, device: torch.device) -> Tensor:
    return torch.arange(item.physical_length, device=device) < item.valid_length


def _physical_block_attention_mask(
    query_item: _RingRangeSlice,
    kv_item: _RingRangeSlice,
    *,
    block_kind: RingBlockKind,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Return a physical block mask plus optional query/KV validity vectors."""

    if block_kind is RingBlockKind.SKIP:
        raise ValueError("SKIP blocks must not enter fused attention")
    query_has_padding = query_item.valid_length != query_item.physical_length
    kv_has_padding = kv_item.valid_length != kv_item.physical_length
    query_validity = (
        _range_validity(query_item, device=query_item.tensor.device)
        if query_has_padding
        else None
    )
    kv_validity = (
        _range_validity(kv_item, device=kv_item.tensor.device)
        if kv_has_padding
        else None
    )
    if block_kind is RingBlockKind.FULL:
        if kv_validity is None:
            return None, query_validity, None
        attention_mask = torch.logical_not(kv_validity).unsqueeze(0).expand(
            query_item.physical_length,
            -1,
        )
        return attention_mask.contiguous(), query_validity, kv_validity

    if not query_has_padding and not kv_has_padding:
        return None, None, None
    query_start, query_end = query_item.logical_range
    query_physical_end = query_item.physical_range[1]
    kv_start, kv_end = kv_item.logical_range
    kv_physical_end = kv_item.physical_range[1]
    query_positions = torch.arange(
        query_start,
        query_physical_end,
        device=query_item.tensor.device,
    ).clamp_max(query_end - 1)
    kv_positions = torch.arange(
        kv_start,
        kv_physical_end,
        device=kv_item.tensor.device,
    ).clamp_max(kv_end - 1)
    if kv_validity is None:
        kv_validity = torch.ones(
            kv_item.physical_length,
            dtype=torch.bool,
            device=kv_item.tensor.device,
        )
    attention_mask = torch.logical_or(
        torch.logical_not(kv_validity).unsqueeze(0),
        kv_positions.unsqueeze(0) > query_positions.unsqueeze(1),
    )
    return attention_mask.contiguous(), query_validity, (
        kv_validity if kv_has_padding else None
    )


def _block_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_heads: int,
    softmax_scale: float,
    block_kind: RingBlockKind,
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention") from exc
    causal_fast_path = block_kind is RingBlockKind.CAUSAL and attention_mask is None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("Ring physical attention mask must use torch.bool")
        if tuple(attention_mask.shape) != (query.shape[0], key.shape[0]):
            raise ValueError(
                "Ring physical attention mask shape must match Q/KV, got "
                f"{tuple(attention_mask.shape)} for Q={query.shape[0]}, KV={key.shape[0]}"
            )
        attention_mask = attention_mask.contiguous()
    result = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=(
            _compressed_causal_mask(query.device)
            if causal_fast_path
            else attention_mask
        ),
        scale=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal_fast_path else _MAX_TOKENS,
        keep_prob=1.0,
        inner_precise=0,
        sparse_mode=(
            _RIGHT_DOWN_CAUSAL_MODE
            if causal_fast_path
            else _FULL_ATTENTION_MODE
        ),
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
    # Keep the online-softmax accumulator FP32, including the first block.
    # Casting after every merge repeatedly rounds the partial context in BF16.
    current = tuple(t.float() for t in current)
    if previous is None:
        return current
    previous_output, previous_max, previous_sum = (t.float() for t in previous)
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
    return (
        merged_output,
        _unflatten_tnd_softmax(merged_max, actual_seq_qlen),
        _unflatten_tnd_softmax(merged_sum, actual_seq_qlen),
    )


def _finalize_attention_result(result, *, dtype):
    """One final output cast shared by the caller and fused backward save."""
    output, maximum, total = result
    return output.to(dtype).contiguous(), maximum, total


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
    attention_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - requires the server NPU runtime
        raise RuntimeError("torch_npu is required for Ring CP attention backward") from exc
    causal_fast_path = block_kind is RingBlockKind.CAUSAL and attention_mask is None
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("Ring physical attention mask must use torch.bool")
        if tuple(attention_mask.shape) != (query.shape[0], key.shape[0]):
            raise ValueError(
                "Ring physical attention mask shape must match Q/KV, got "
                f"{tuple(attention_mask.shape)} for Q={query.shape[0]}, KV={key.shape[0]}"
            )
        attention_mask = attention_mask.contiguous()
    result = torch_npu.npu_fusion_attention_grad(
        query,
        key,
        value,
        grad_output,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=(
            _compressed_causal_mask(query.device)
            if causal_fast_path
            else attention_mask
        ),
        softmax_max=softmax_max,
        softmax_sum=softmax_sum,
        attention_in=attention_output,
        scale_value=softmax_scale,
        pre_tockens=_MAX_TOKENS,
        next_tockens=0 if causal_fast_path else _MAX_TOKENS,
        keep_prob=1.0,
        sparse_mode=(
            _RIGHT_DOWN_CAUSAL_MODE
            if causal_fast_path
            else _FULL_ATTENTION_MODE
        ),
        actual_seq_qlen=[query.shape[0]],
        actual_seq_kvlen=[key.shape[0]],
    )
    return result[0], result[1], result[2]


def _reduce_ring_gradients_to_owner(key, value, config, consume):
    """Replay native reverse Ring with constant KV/dKV ping-pong storage.

    Native no-cache backward starts at the last forward source (rank+1).
    Since ctx retains LOCAL KV only, seed that source with one reverse hop.
    Replay rank+1,...,rank-1; the last step uses local KV without a transfer.
    dKV follows the same reverse Ring and ends at its owner after CP-1 hops.
    """
    RingP2P, _ = _load_mindspeed_ring_primitives()
    kv_ring = RingP2P(config.global_ranks, config.cp_group, is_backward=True)
    grad_ring = RingP2P(config.global_ranks, config.cp_group, is_backward=True)
    current = torch.stack((key, value), dim=0).contiguous()
    received = torch.empty_like(current)
    accumulated = torch.zeros_like(current)
    next_grad = torch.empty_like(current)
    _observe_ring_storage("backward_buffers", (current, received, accumulated, next_grad))
    try:
        if config.cp_size > 1:
            kv_ring.async_send_recv(current, received)
            kv_ring.wait()
            current, received = received, current
        for step in range(config.cp_size):
            source = (config.cp_rank + step + 1) % config.cp_size
            if step + 2 < config.cp_size:
                kv_ring.async_send_recv(current, received)
            local = step + 1 == config.cp_size
            consume(source, key if local else current[0], value if local else current[1],
                    accumulated[0], accumulated[1])
            if step + 2 < config.cp_size:
                kv_ring.wait()
                current, received = received, current
            if not local:
                grad_ring.async_send_recv(accumulated, next_grad)
                grad_ring.wait()
                accumulated, next_grad = next_grad, accumulated
    finally:
        kv_ring.wait()
        grad_ring.wait()
    return accumulated[0], accumulated[1]


def _source_schedule(query_slices, source_key, source_value, config,
                     segment_index, source_rank, phase):
    """Existing TPR block classification/padding, evaluated for ONE live source."""
    is_prefix = segment_index + 1 < len(config.segment_lengths)
    global_length = config.segment_lengths[segment_index]
    padded_length = config.segment_padded_lengths[segment_index]
    source_shard = make_ring_sequence_shard(
        global_length, cp_rank=source_rank, cp_size=config.cp_size,
        padded_length=padded_length,
    )
    coalesced = is_prefix and config.coalesce_prefix_full and global_length == padded_length
    key_slices = _prefix_full_slices(source_key.squeeze(1), source_shard, enabled=coalesced)
    value_slices = _prefix_full_slices(source_value.squeeze(1), source_shard, enabled=coalesced)
    for query_index, query_item in enumerate(query_slices):
        for key_item, value_item in zip(key_slices, value_slices, strict=True):
            if (key_item.logical_range != value_item.logical_range
                    or key_item.physical_range != value_item.physical_range
                    or key_item.local_slice != value_item.local_slice):
                raise RuntimeError("Ring K/V physical slices differ")
            block_kind = classify_ring_block(
                query_item.logical_range, key_item.logical_range, is_prefix=is_prefix,
            )
            if _trace_ring_block is not None:
                _trace_ring_block(
                    phase=phase, rank=config.cp_rank,
                    ring_step=(config.cp_rank - source_rank) % config.cp_size,
                    source_rank=source_rank, segment_index=segment_index,
                    segment_type="prefix" if is_prefix else "current",
                    query_range=query_item.logical_range,
                    query_chunk=query_item.physical_range[0] // (config.current_shard.padded_length // (2 * config.cp_size)),
                    kv_ranges=source_shard.global_ranges if coalesced else (key_item.logical_range,),
                    kv_chunks=tuple(start // (padded_length // (2 * config.cp_size)) for start, _ in
                                    (source_shard.physical_global_ranges if coalesced else (key_item.physical_range,))),
                    block_type=block_kind.value,
                    query_length=query_item.tensor.shape[0], kv_length=key_item.tensor.shape[0],
                    fa_called=block_kind is not RingBlockKind.SKIP, coalesced=coalesced,
                )
            if block_kind is not RingBlockKind.SKIP:
                yield query_index, query_item, key_item, value_item, block_kind


class _RingTPRAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, *tensor_args: Any) -> Tensor:
        config = tensor_args[-1]
        local_kv = tensor_args[:-1]
        if not isinstance(config, _RingAttentionConfig):
            raise TypeError("last Ring attention argument must be _RingAttentionConfig")
        if len(local_kv) != 2 * len(config.segment_lengths):
            raise RuntimeError("Ring attention K/V argument count does not match Segment metadata")
        ctx.query_coalesced = _can_coalesce_prefix_query(query, config)
        query_tnd = query.squeeze(1).contiguous()
        query_slices = _iter_range_slices(query_tnd, config.current_shard)
        query_results = [None] * len(query_slices)
        prefix_result = None
        for segment_index in range(len(config.segment_lengths)):
            merged_query = ctx.query_coalesced and segment_index + 1 < len(config.segment_lengths)
            if ctx.query_coalesced and not merged_query:
                query_results = [_slice_tnd_result(prefix_result, item.local_slice) for item in query_slices]
                prefix_result = None

            def consume(source_rank, key, value):
                nonlocal prefix_result
                if merged_query:
                    _trace_merged_query(config, query_tnd, key, segment_index, source_rank, "forward")
                    current = _block_attention_forward(
                        query_tnd, key.squeeze(1), value.squeeze(1),
                        query_heads=config.query_heads, softmax_scale=config.softmax_scale,
                        block_kind=RingBlockKind.FULL, attention_mask=None,
                    )
                    prefix_result = _merge_attention(prefix_result, current, query_length=query_tnd.shape[0])
                    return
                for index, q_item, k_item, v_item, kind in _source_schedule(
                    query_slices, key, value, config, segment_index, source_rank, "forward"
                ):
                    mask, _, _ = _physical_block_attention_mask(q_item, k_item, block_kind=kind)
                    current = _block_attention_forward(
                        q_item.tensor, k_item.tensor, v_item.tensor,
                        query_heads=config.query_heads, softmax_scale=config.softmax_scale,
                        block_kind=kind, attention_mask=mask,
                    )
                    query_results[index] = _merge_attention(
                        query_results[index], current, query_length=q_item.physical_length,
                    )

            _circulate_kv(local_kv[2 * segment_index], local_kv[2 * segment_index + 1], config, consume)
        if any(result is None for result in query_results):
            raise RuntimeError("query range has no visible KV block")
        query_results = [_finalize_attention_result(result, dtype=query.dtype) for result in query_results]
        ctx.config = config
        ctx.block_tensor_count = len(local_kv)
        _observe_ring_storage("saved_local_kv", local_kv)
        output = query.new_zeros((query.shape[0], config.query_heads, config.head_dim))
        for query_item, result in zip(query_slices, query_results, strict=True):
            result_output = result[0]
            if query_item.valid_length != query_item.physical_length:
                valid_query = _range_validity(query_item, device=query.device)
                result_output = result_output * valid_query[:, None, None].to(
                    result_output.dtype
                )
            output[query_item.local_slice] = result_output
        if ctx.query_coalesced:
            # Save the already-created returned output, not a concatenated copy.
            # One final stats pair replaces the per-chunk saved pairs. No packing
            # per Ring source and no retained extra Prefix results.
            length, heads = query.shape[0], config.query_heads
            full_stats = tuple(
                query_results[0][index].new_empty((length, heads, query_results[0][index].shape[-1]))
                for index in (1, 2)
            )
            for item, result in zip(query_slices, query_results, strict=True):
                for full, part in zip(full_stats, result[1:], strict=True):
                    full.view(heads, length, full.shape[-1])[:, item.local_slice, :].copy_(
                        part.view(heads, item.physical_length, part.shape[-1])
                    )
            ctx.save_for_backward(query, *local_kv, output, *full_stats)
        else:
            saved_results = [tensor for result in query_results for tensor in result]
            ctx.save_for_backward(query, *local_kv, *saved_results)
        return output.reshape(query.shape[0], 1, config.query_heads * config.head_dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        config = ctx.config
        saved = ctx.saved_tensors
        query = saved[0]
        local_kv = saved[1:1 + ctx.block_tensor_count]
        result_tensors = saved[1 + ctx.block_tensor_count:]
        query_tnd = query.squeeze(1).contiguous()
        grad_output_tnd = grad_output.reshape(
            query.shape[0], config.query_heads, config.head_dim,
        ).contiguous()
        query_gradient = torch.zeros_like(query_tnd)
        query_slices = _iter_range_slices(query_tnd, config.current_shard)
        if ctx.query_coalesced:
            query_results = tuple(_slice_tnd_result(result_tensors, item.local_slice) for item in query_slices)
        else:
            query_results = tuple(tuple(result_tensors[i:i + 3]) for i in range(0, len(result_tensors), 3))
        grad_slices = _iter_range_slices(grad_output_tnd, config.current_shard)
        for q_item, g_item in zip(query_slices, grad_slices, strict=True):
            if (q_item.logical_range != g_item.logical_range
                    or q_item.physical_range != g_item.physical_range
                    or q_item.local_slice != g_item.local_slice):
                raise RuntimeError("Ring query/gradient physical slices differ")
        local_gradients = []
        for segment_index in range(len(config.segment_lengths)):
            merged_query = ctx.query_coalesced and segment_index + 1 < len(config.segment_lengths)

            def consume(source_rank, key, value, key_buffer, value_buffer):
                if merged_query:
                    _trace_merged_query(config, query_tnd, key, segment_index, source_rank, "backward")
                    dq, dk, dv = _block_attention_backward(
                        query_tnd, key.squeeze(1), value.squeeze(1), grad_output_tnd,
                        attention_output=result_tensors[0], softmax_max=result_tensors[1],
                        softmax_sum=result_tensors[2], query_heads=config.query_heads,
                        softmax_scale=config.softmax_scale, block_kind=RingBlockKind.FULL,
                        attention_mask=None,
                    )
                    query_gradient.add_(dq)
                    key_buffer.add_(dk.unsqueeze(1))
                    value_buffer.add_(dv.unsqueeze(1))
                    return
                for index, q_item, k_item, v_item, kind in _source_schedule(
                    query_slices, key, value, config, segment_index, source_rank, "backward"
                ):
                    grad_part = grad_slices[index].tensor
                    mask, valid_query, valid_kv = _physical_block_attention_mask(q_item, k_item, block_kind=kind)
                    if valid_query is not None:
                        grad_part = grad_part * valid_query[:, None, None].to(grad_part.dtype)
                    output, maximum, total = query_results[index]
                    dq, dk, dv = _block_attention_backward(
                        q_item.tensor, k_item.tensor, v_item.tensor, grad_part,
                        attention_output=output, softmax_max=maximum, softmax_sum=total,
                        query_heads=config.query_heads, softmax_scale=config.softmax_scale,
                        block_kind=kind, attention_mask=mask,
                    )
                    if valid_query is not None:
                        dq = dq * valid_query[:, None, None].to(dq.dtype)
                    if valid_kv is not None:
                        dk = dk * valid_kv[:, None, None].to(dk.dtype)
                        dv = dv * valid_kv[:, None, None].to(dv.dtype)
                    query_gradient[q_item.local_slice].add_(dq)
                    key_buffer[k_item.local_slice].add_(dk.unsqueeze(1))
                    value_buffer[k_item.local_slice].add_(dv.unsqueeze(1))

            gradients = _reduce_ring_gradients_to_owner(
                local_kv[2 * segment_index], local_kv[2 * segment_index + 1], config, consume,
            )
            local_gradients.extend(gradients)
        return (query_gradient.unsqueeze(1), *local_gradients, None)


def ordinary_ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
) -> Tensor:
    """Whole, unpadded causal attention using paired MindSpeed's native Ring.

    Explicit entry point: an empty prefix alone does NOT identify whole mode
    (a segmented execution's root also has no prefix). Keep that root on the
    existing TPR schedule until the prefix extension is migrated separately.

    MindSpeed owns streaming KV, causal grouping, output/stat correction,
    reverse backward and owner dKV accumulation. No extra gradient SUM/cast.
    """
    _, config = _normalize_inputs(
        query, current_key, current_value, prefix_blocks=(),
        current_shard=current_shard, cp_group=cp_group,
        softmax_scale=softmax_scale,
    )
    if current_shard.global_length != current_shard.padded_length:
        raise ValueError("ordinary whole Ring requires an unpadded sequence")
    from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import (
        ringattn_context_parallel,
    )

    cp_para = {
        "causal": True,
        "cp_group": cp_group,
        "cp_size": config.cp_size,
        "rank": config.cp_rank,
        "cp_global_ranks": list(config.global_ranks),
        "cp_inner_ranks": [dist.get_rank()],
        "cp_outer_ranks": list(config.global_ranks),
        "cp_dkv_outer_ranks": list(config.global_ranks),
        "megatron_cp_in_bnsd": False,
        "cache_policy": None,
        "pse_type": 1,
    }
    # Native ordinary causal Ring consumes SBH, retaining GQA's smaller KV H.
    q, k, v = (x.flatten(2).contiguous() for x in (query, current_key, current_value))
    return ringattn_context_parallel(
        q, k, v, config.query_heads, cp_para,
        softmax_scale=config.softmax_scale, dropout_p=0.0,
    )


def ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock] = (),
    current_shard: SequenceShard,
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
        current_shard: SequenceShard,
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
        if not isinstance(physical_sequence_shard(state.shard), RangeSequenceShard):
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
    current_shard: SequenceShard,
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
    current_shard: SequenceShard,
    cp_group: Any,
) -> RingCPAttentionBackend:
    for entry in anchors.entries:
        if not isinstance(physical_sequence_shard(entry.shard), RangeSequenceShard):
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
