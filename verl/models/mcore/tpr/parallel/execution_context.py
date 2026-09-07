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

"""Per-forward AllGather CP state used by the TPR segment executor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from torch import Tensor

from ..kv_stack import KVStack
from ..prefix_state import KVPrefixAnchors, PrefixShard
from .allgather_attention import (
    LocalKVBlock,
    _group_world_size_and_rank,
    allgather_cp_rectangular_attention,
)


def resolve_cp_group(cp_group: Any) -> tuple[int, int]:
    """Return ``(size, rank)`` for a concrete CP process group."""

    return _group_world_size_and_rank(cp_group)


@dataclass(frozen=True, slots=True)
class ShardedPastKVAnchors:
    """Per-segment local KV leaves for one active root-to-parent path."""

    entries: tuple[KVPrefixAnchors, ...]
    stack_signature: tuple[int, ...]

    @property
    def global_prefix_length(self) -> int:
        return sum(entry.shard.global_length for entry in self.entries)


class AllGatherCPAttentionBackend:
    """Bind one local Segment shard and its Prefix path to AllGather CP."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: PrefixShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[LocalKVBlock]],
        cp_group: Any,
    ) -> None:
        if not isinstance(global_prefix_length, int) or isinstance(global_prefix_length, bool):
            raise ValueError(f"global_prefix_length must be an integer, got {global_prefix_length!r}")
        if global_prefix_length < 0:
            raise ValueError(f"global_prefix_length must be non-negative, got {global_prefix_length}")
        if not isinstance(current_shard, PrefixShard):
            raise TypeError(f"current_shard must be PrefixShard, got {type(current_shard).__name__}")

        cp_size, cp_rank = _group_world_size_and_rank(cp_group)
        if (current_shard.cp_size, current_shard.cp_rank) != (cp_size, cp_rank):
            raise ValueError(
                "current shard metadata must match the CP process group: "
                f"expected rank {cp_rank}/{cp_size}, "
                f"got {current_shard.cp_rank}/{current_shard.cp_size}"
            )
        if not isinstance(prefix_blocks_by_layer, Mapping) or not prefix_blocks_by_layer:
            raise ValueError("prefix_blocks_by_layer must contain every TPR layer")

        normalized: dict[int, tuple[LocalKVBlock, ...]] = {}
        path_signature: tuple[int, ...] | None = None
        for layer_number, blocks in sorted(prefix_blocks_by_layer.items()):
            if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
                raise ValueError(f"layer_number must be positive, got {layer_number!r}")
            layer_blocks = tuple(blocks)
            signature = tuple(block.segment_id for block in layer_blocks)
            if path_signature is None:
                path_signature = signature
            elif signature != path_signature:
                raise ValueError(
                    f"prefix path differs across layers: expected {path_signature}, got {signature}"
                )
            layer_prefix_length = sum(block.shard.global_length for block in layer_blocks)
            if layer_prefix_length != global_prefix_length:
                raise ValueError(
                    f"layer {layer_number} prefix blocks cover {layer_prefix_length} tokens, "
                    f"expected {global_prefix_length}"
                )
            normalized[layer_number] = layer_blocks

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

    @property
    def current_shard(self) -> PrefixShard:
        return self._current_shard

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return tuple(self._prefix_blocks_by_layer)

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
            prefix_blocks = self._prefix_blocks_by_layer[layer_number]
        except KeyError as exc:
            raise KeyError(f"AllGather CP context has no layer {layer_number}") from exc
        return allgather_cp_rectangular_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=prefix_blocks,
            current_shard=self.current_shard,
            cp_group=self._cp_group,
            softmax_scale=softmax_scale,
        )


def build_sharded_past_anchors(kv_stack: KVStack) -> ShardedPastKVAnchors:
    """Create one independent local anchor pair for every Prefix segment."""

    entries = tuple(kv_stack.get(segment_id).kv.make_anchors() for segment_id in kv_stack.segment_ids)
    return ShardedPastKVAnchors(entries=entries, stack_signature=kv_stack.segment_ids)


def accumulate_sharded_past_anchor_gradients(
    kv_stack: KVStack,
    anchors: ShardedPastKVAnchors,
) -> None:
    """Accumulate ReduceScatter-produced local gradients into Prefix states."""

    if not isinstance(anchors, ShardedPastKVAnchors):
        raise TypeError(f"anchors must be ShardedPastKVAnchors, got {type(anchors).__name__}")
    if anchors.stack_signature != kv_stack.segment_ids:
        raise RuntimeError(
            f"sharded past KV anchors are stale: built for {anchors.stack_signature}, "
            f"current stack is {kv_stack.segment_ids}"
        )
    for entry in anchors.entries:
        kv_stack.get(entry.segment_id).kv.accumulate_anchor_gradients(entry)


def make_cached_allgather_cp_backend(
    kv_stack: KVStack,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: PrefixShard,
    cp_group: Any,
) -> AllGatherCPAttentionBackend:
    """Build a no-grad backend over graph-free Prefix states."""

    blocks = {
        layer_number: tuple(
            LocalKVBlock(
                segment_id,
                kv_stack.get(segment_id).kv.shard,
                *kv_stack.get(segment_id).kv.key_values[layer_number],
            )
            for segment_id in kv_stack.segment_ids
        )
        for layer_number in expected_layer_numbers
    }
    return AllGatherCPAttentionBackend(
        global_prefix_length=kv_stack.prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer=blocks,
        cp_group=cp_group,
    )


def make_anchored_allgather_cp_backend(
    anchors: ShardedPastKVAnchors,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: PrefixShard,
    cp_group: Any,
) -> AllGatherCPAttentionBackend:
    """Build a grad-enabled backend over local Prefix anchor leaves."""

    blocks = {
        layer_number: tuple(
            LocalKVBlock(entry.segment_id, entry.shard, *entry.key_values[layer_number])
            for entry in anchors.entries
        )
        for layer_number in expected_layer_numbers
    }
    return AllGatherCPAttentionBackend(
        global_prefix_length=anchors.global_prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer=blocks,
        cp_group=cp_group,
    )
