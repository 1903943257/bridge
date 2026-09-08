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

"""Hybrid Ulysses + Ring CP for external-prefix TPR attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch.distributed as dist
from torch import Tensor

from ..kv_stack import KVStack
from ..shard import RangeSequenceShard
from .allgather_attention import _group_world_size_and_rank
from .execution_context import ShardedPastKVAnchors
from .ring_attention import RingLocalKVBlock, make_ring_sequence_shard, ring_cp_attention
from .ulysses_attention import _head_to_sequence, _sequence_to_head


@dataclass(frozen=True, slots=True)
class HybridCPTopology:
    """MindSpeed-compatible coordinates for one Ulysses x Ring CP rank."""

    cp_group: Any
    ulysses_group: Any
    ring_group: Any
    cp_size: int
    cp_rank: int
    ulysses_size: int
    ulysses_rank: int
    ring_size: int
    ring_rank: int

    def __post_init__(self) -> None:
        for name, value in (
            ("cp_size", self.cp_size),
            ("cp_rank", self.cp_rank),
            ("ulysses_size", self.ulysses_size),
            ("ulysses_rank", self.ulysses_rank),
            ("ring_size", self.ring_size),
            ("ring_rank", self.ring_rank),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.ulysses_size <= 1 or self.ring_size <= 1:
            raise ValueError(
                "Hybrid CP requires both Ulysses and Ring sizes greater than one, "
                f"got U={self.ulysses_size}, R={self.ring_size}"
            )
        if self.cp_size != self.ulysses_size * self.ring_size:
            raise ValueError(
                f"CP size {self.cp_size} must equal Ulysses {self.ulysses_size} "
                f"times Ring {self.ring_size}"
            )
        if not 0 <= self.ulysses_rank < self.ulysses_size:
            raise ValueError("ulysses_rank is outside the Ulysses group")
        if not 0 <= self.ring_rank < self.ring_size:
            raise ValueError("ring_rank is outside the Ring group")
        expected_cp_rank = self.ring_rank * self.ulysses_size + self.ulysses_rank
        if self.cp_rank != expected_cp_rank:
            raise ValueError(
                "CP rank does not match MindSpeed Hybrid layout: "
                f"expected {expected_cp_rank}, got {self.cp_rank}"
            )


@dataclass(frozen=True, slots=True)
class HybridLocalKVBlock:
    """One Prefix segment's local Hybrid CP KV shard for a single layer."""

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


def _process_group_global_ranks(process_group: Any) -> tuple[int, ...]:
    size, _ = _group_world_size_and_rank(process_group)
    if hasattr(dist, "get_process_group_ranks"):
        ranks = tuple(dist.get_process_group_ranks(process_group))
    elif hasattr(dist, "get_global_rank"):
        ranks = tuple(dist.get_global_rank(process_group, rank) for rank in range(size))
    elif process_group is dist.group.WORLD:
        ranks = tuple(range(size))
    else:  # pragma: no cover - obsolete torch.distributed versions
        raise RuntimeError("cannot resolve process-group global ranks")
    if len(ranks) != size:
        raise RuntimeError(f"process group exposes {len(ranks)} ranks, expected {size}")
    return ranks


def resolve_mindspeed_hybrid_topology(
    cp_group: Any,
    *,
    cp_size: int | None = None,
    cp_rank: int | None = None,
) -> HybridCPTopology:
    """Read initialized Hybrid subgroups and verify MindSpeed's rank layout."""

    try:
        from mindspeed.core.context_parallel.model_parallel_utils import (
            get_context_parallel_group_for_hybrid_ring,
            get_context_parallel_group_for_hybrid_ulysses,
        )
    except ImportError as exc:  # pragma: no cover - requires server MindSpeed
        raise RuntimeError("MindSpeed Hybrid CP group accessors are required") from exc

    actual_cp_size, actual_cp_rank = _group_world_size_and_rank(cp_group)
    if cp_size is not None and cp_size != actual_cp_size:
        raise ValueError(f"configured CP size {cp_size} does not match group size {actual_cp_size}")
    if cp_rank is not None and cp_rank != actual_cp_rank:
        raise ValueError(f"configured CP rank {cp_rank} does not match group rank {actual_cp_rank}")

    ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
    ring_group = get_context_parallel_group_for_hybrid_ring()
    ulysses_size, ulysses_rank = _group_world_size_and_rank(ulysses_group)
    ring_size, ring_rank = _group_world_size_and_rank(ring_group)
    topology = HybridCPTopology(
        cp_group=cp_group,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        cp_size=actual_cp_size,
        cp_rank=actual_cp_rank,
        ulysses_size=ulysses_size,
        ulysses_rank=ulysses_rank,
        ring_size=ring_size,
        ring_rank=ring_rank,
    )

    cp_ranks = _process_group_global_ranks(cp_group)
    ulysses_ranks = _process_group_global_ranks(ulysses_group)
    ring_ranks = _process_group_global_ranks(ring_group)
    expected_ulysses = cp_ranks[
        topology.ring_rank * topology.ulysses_size :
        (topology.ring_rank + 1) * topology.ulysses_size
    ]
    expected_ring = cp_ranks[topology.ulysses_rank :: topology.ulysses_size]
    if ulysses_ranks != expected_ulysses:
        raise RuntimeError(
            f"MindSpeed Hybrid Ulysses group order mismatch: {ulysses_ranks} != {expected_ulysses}"
        )
    if ring_ranks != expected_ring:
        raise RuntimeError(
            f"MindSpeed Hybrid Ring group order mismatch: {ring_ranks} != {expected_ring}"
        )
    return topology


def _slice_ordered_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    local_start: int,
    local_end: int,
) -> tuple[tuple[int, int], ...]:
    selected = []
    cursor = 0
    for global_start, global_end in ranges:
        range_length = global_end - global_start
        overlap_start = max(local_start, cursor)
        overlap_end = min(local_end, cursor + range_length)
        if overlap_start < overlap_end:
            selected.append(
                (
                    global_start + overlap_start - cursor,
                    global_start + overlap_end - cursor,
                )
            )
        cursor += range_length
    if sum(end - start for start, end in selected) != local_end - local_start:
        raise RuntimeError("Hybrid shard interval was not fully mapped to Ring ranges")
    return tuple(selected)


def make_hybrid_sequence_shard(
    global_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    ulysses_degree: int,
) -> RangeSequenceShard:
    """Select one rank's pre-A2A shard using MindSpeed's Hybrid layout."""

    if (
        not isinstance(ulysses_degree, int)
        or isinstance(ulysses_degree, bool)
        or ulysses_degree <= 1
    ):
        raise ValueError(f"ulysses_degree must be greater than one, got {ulysses_degree!r}")
    if cp_size <= ulysses_degree or cp_size % ulysses_degree != 0:
        raise ValueError(
            f"CP size {cp_size} must be a multiple greater than Ulysses degree {ulysses_degree}"
        )
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank!r}")
    if global_length % cp_size != 0:
        raise ValueError(f"length {global_length} must be divisible by CP size {cp_size}")

    ring_size = cp_size // ulysses_degree
    ring_rank, ulysses_rank = divmod(cp_rank, ulysses_degree)
    ring_shard = make_ring_sequence_shard(
        global_length,
        cp_rank=ring_rank,
        cp_size=ring_size,
    )
    if ring_shard.local_length % ulysses_degree != 0:
        raise ValueError(
            f"Ring-local length {ring_shard.local_length} must be divisible by "
            f"Ulysses degree {ulysses_degree}"
        )
    local_length = ring_shard.local_length // ulysses_degree
    local_start = ulysses_rank * local_length
    ranges = _slice_ordered_ranges(
        ring_shard.global_ranges,
        local_start=local_start,
        local_end=local_start + local_length,
    )
    return RangeSequenceShard(
        global_length,
        ranges,
        cp_rank=cp_rank,
        cp_size=cp_size,
    )


def _validate_hybrid_shard(
    shard: RangeSequenceShard,
    *,
    topology: HybridCPTopology,
    name: str,
) -> None:
    if not isinstance(shard, RangeSequenceShard):
        raise TypeError(f"{name} shard must be RangeSequenceShard, got {type(shard).__name__}")
    expected = make_hybrid_sequence_shard(
        shard.global_length,
        cp_rank=topology.cp_rank,
        cp_size=topology.cp_size,
        ulysses_degree=topology.ulysses_size,
    )
    if shard != expected:
        raise ValueError(f"{name} shard must use the MindSpeed Hybrid Ulysses + Ring layout")


def _validate_local_kv(
    key: Tensor,
    value: Tensor,
    *,
    shard: RangeSequenceShard,
    name: str,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"{name} K/V must be torch tensors")
    if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError(f"{name} K/V must have matching [sequence, batch, heads, head_dim] shapes")
    if key.shape[0] != shard.local_length or key.shape[1] != 1:
        raise ValueError(
            f"{name} K/V must start with [{shard.local_length}, 1], got {tuple(key.shape)}"
        )
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError(f"{name} K/V dtype and device must match")


def hybrid_cp_rectangular_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[HybridLocalKVBlock] = (),
    current_shard: RangeSequenceShard,
    topology: HybridCPTopology,
    softmax_scale: float | None = None,
) -> Tensor:
    """Run Ulysses CP-to-HP, TPR Ring attention, then HP-to-CP."""

    _validate_hybrid_shard(current_shard, topology=topology, name="Current")
    _validate_local_kv(current_key, current_value, shard=current_shard, name="Current")
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
    query_heads, kv_heads = query.shape[2], current_key.shape[2]
    if query_heads % topology.ulysses_size != 0:
        raise ValueError("query heads must be divisible by the Hybrid Ulysses size")
    if kv_heads % topology.ulysses_size != 0:
        raise ValueError("KV heads must be divisible by the Hybrid Ulysses size")
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")

    blocks = tuple(prefix_blocks)
    seen = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, HybridLocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be HybridLocalKVBlock")
        if block.segment_id in seen:
            raise ValueError(f"duplicate Prefix segment_id {block.segment_id}")
        seen.add(block.segment_id)
        _validate_hybrid_shard(
            block.shard,
            topology=topology,
            name=f"Prefix block {block.segment_id}",
        )
        _validate_local_kv(
            block.key,
            block.value,
            shard=block.shard,
            name=f"Prefix block {block.segment_id}",
        )
        if block.key.shape[1:] != current_key.shape[1:]:
            raise ValueError(f"Prefix block {block.segment_id} shape must match Current KV")
        if block.key.dtype != current_key.dtype or block.key.device != current_key.device:
            raise ValueError(f"Prefix block {block.segment_id} dtype/device must match Current KV")

    current_ring_shard = make_ring_sequence_shard(
        current_shard.global_length,
        cp_rank=topology.ring_rank,
        cp_size=topology.ring_size,
    )
    head_query = _sequence_to_head(
        query,
        global_length=current_ring_shard.local_length,
        cp_group=topology.ulysses_group,
    )
    head_current_key = _sequence_to_head(
        current_key,
        global_length=current_ring_shard.local_length,
        cp_group=topology.ulysses_group,
    )
    head_current_value = _sequence_to_head(
        current_value,
        global_length=current_ring_shard.local_length,
        cp_group=topology.ulysses_group,
    )

    ring_prefix_blocks = []
    for block in blocks:
        ring_shard = make_ring_sequence_shard(
            block.shard.global_length,
            cp_rank=topology.ring_rank,
            cp_size=topology.ring_size,
        )
        ring_prefix_blocks.append(
            RingLocalKVBlock(
                block.segment_id,
                ring_shard,
                _sequence_to_head(
                    block.key,
                    global_length=ring_shard.local_length,
                    cp_group=topology.ulysses_group,
                ),
                _sequence_to_head(
                    block.value,
                    global_length=ring_shard.local_length,
                    cp_group=topology.ulysses_group,
                ),
            )
        )

    head_output = ring_cp_attention(
        head_query,
        head_current_key,
        head_current_value,
        prefix_blocks=ring_prefix_blocks,
        current_shard=current_ring_shard,
        cp_group=topology.ring_group,
        softmax_scale=softmax_scale,
    )
    return _head_to_sequence(
        head_output,
        local_length=current_shard.local_length,
        global_heads=query_heads,
        head_dim=query.shape[-1],
        cp_group=topology.ulysses_group,
    )


class HybridCPAttentionBackend:
    """Bind one TPR Segment to MindSpeed-compatible Hybrid CP groups."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: RangeSequenceShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[HybridLocalKVBlock]],
        topology: HybridCPTopology,
    ) -> None:
        if (
            not isinstance(global_prefix_length, int)
            or isinstance(global_prefix_length, bool)
            or global_prefix_length < 0
        ):
            raise ValueError("global_prefix_length must be a non-negative integer")
        _validate_hybrid_shard(current_shard, topology=topology, name="Current")
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
                raise ValueError("Prefix path must be identical across Hybrid attention layers")
            if sum(block.shard.global_length for block in blocks) != global_prefix_length:
                raise ValueError(f"layer {layer} Prefix blocks do not cover the Prefix length")
        self._global_prefix_length = global_prefix_length
        self._current_shard = current_shard
        self._prefix_blocks_by_layer = MappingProxyType(normalized)
        self._topology = topology

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
        return self._topology.cp_size

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
            raise KeyError(f"Hybrid CP context has no layer {layer_number}") from exc
        return hybrid_cp_rectangular_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=blocks,
            current_shard=self._current_shard,
            topology=self._topology,
            softmax_scale=softmax_scale,
        )


def _cached_blocks(kv_stack: KVStack, layer_number: int) -> tuple[HybridLocalKVBlock, ...]:
    blocks = []
    for segment_id in kv_stack.segment_ids:
        state = kv_stack.get(segment_id).kv
        if not isinstance(state.shard, RangeSequenceShard):
            raise TypeError("Hybrid CP requires RangeSequenceShard Prefix states")
        blocks.append(
            HybridLocalKVBlock(
                segment_id,
                state.shard,
                *state.key_values[layer_number],
            )
        )
    return tuple(blocks)


def make_cached_hybrid_cp_backend(
    kv_stack: KVStack,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: RangeSequenceShard,
    topology: HybridCPTopology,
) -> HybridCPAttentionBackend:
    return HybridCPAttentionBackend(
        global_prefix_length=kv_stack.prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: _cached_blocks(kv_stack, layer) for layer in expected_layer_numbers
        },
        topology=topology,
    )


def make_anchored_hybrid_cp_backend(
    anchors: ShardedPastKVAnchors,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: RangeSequenceShard,
    topology: HybridCPTopology,
) -> HybridCPAttentionBackend:
    for entry in anchors.entries:
        if not isinstance(entry.shard, RangeSequenceShard):
            raise TypeError("Hybrid CP requires RangeSequenceShard Prefix anchors")
    return HybridCPAttentionBackend(
        global_prefix_length=anchors.global_prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: tuple(
                HybridLocalKVBlock(
                    entry.segment_id,
                    entry.shard,
                    *entry.key_values[layer],
                )
                for entry in anchors.entries
            )
            for layer in expected_layer_numbers
        },
        topology=topology,
    )
