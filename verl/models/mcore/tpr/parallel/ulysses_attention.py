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

"""MindSpeed Ulysses adapter for external-prefix TPR attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from ..kv_stack import KVStack
from ..rectangular_attention import rectangular_causal_attention
from ..shard import PrefixShard, SequenceShard, physical_sequence_shard
from .allgather_attention import LocalKVBlock, _group_world_size_and_rank
from .execution_context import ShardedPastKVAnchors


def _load_mindspeed_seq_all_to_all():
    try:
        from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import (
            _SeqAllToAll,
        )
    except ImportError as exc:  # pragma: no cover - requires the server MindSpeed runtime
        raise RuntimeError("MindSpeed Ulysses _SeqAllToAll is required") from exc
    return _SeqAllToAll


def _mindspeed_all_to_all(
    tensor: Tensor,
    process_group: Any,
    *,
    scatter_dim: int,
    gather_dim: int,
    gather_size: int,
) -> Tensor:
    del gather_size  # The aligned native helper derives both dimensions from the input.
    return _load_mindspeed_seq_all_to_all().apply(
        process_group,
        tensor,
        scatter_dim,
        gather_dim,
    )


def _validate_shard(shard: SequenceShard, *, cp_size: int, cp_rank: int, name: str) -> None:
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"{name} shard must implement SequenceShard, got {type(shard).__name__}")
    if not isinstance(physical_sequence_shard(shard), PrefixShard):
        raise TypeError(f"{name} shard must use contiguous physical placement")
    if (shard.cp_size, shard.cp_rank) != (cp_size, cp_rank):
        raise ValueError(
            f"{name} shard must match CP rank {cp_rank}/{cp_size}, "
            f"got {shard.cp_rank}/{shard.cp_size}"
        )


def _validate_qkv(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    current_shard: SequenceShard,
    cp_size: int,
) -> tuple[int, int]:
    for name, tensor in (
        ("query", query),
        ("current_key", current_key),
        ("current_value", current_value),
    ):
        if not isinstance(tensor, Tensor) or tensor.ndim != 4:
            shape = getattr(tensor, "shape", None)
            raise ValueError(f"{name} must have shape [sequence, batch, heads, head_dim], got {shape}")
        if tensor.shape[0] != current_shard.local_length or tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must start with [{current_shard.local_length}, 1], got {tuple(tensor.shape)}"
            )
    if current_key.shape != current_value.shape:
        raise ValueError("current K/V shapes must match")
    if query.device != current_key.device or query.device != current_value.device:
        raise ValueError("Q/K/V devices must match")
    if query.dtype != current_key.dtype or query.dtype != current_value.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if query.shape[-1] != current_key.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    query_heads, kv_heads = query.shape[2], current_key.shape[2]
    if query_heads % cp_size != 0:
        raise ValueError(f"query heads {query_heads} must be divisible by CP size {cp_size}")
    if kv_heads % cp_size != 0:
        raise ValueError(f"KV heads {kv_heads} must be divisible by CP size {cp_size}")
    if query_heads % kv_heads != 0:
        raise ValueError(f"query heads {query_heads} must be divisible by KV heads {kv_heads}")
    return query_heads, kv_heads


def _sequence_to_head(tensor: Tensor, *, global_length: int, cp_group: Any) -> Tensor:
    result = _mindspeed_all_to_all(
        tensor.contiguous(),
        cp_group,
        scatter_dim=2,
        gather_dim=0,
        gather_size=global_length,
    )
    expected_sequence = global_length
    if result.ndim != 4 or result.shape[0] != expected_sequence:
        raise RuntimeError(
            f"Ulysses CP-to-HP output must have global sequence length {expected_sequence}, "
            f"got {tuple(result.shape)}"
        )
    return result


def _head_to_sequence(
    tensor: Tensor,
    *,
    local_length: int,
    global_heads: int,
    head_dim: int,
    cp_group: Any,
) -> Tensor:
    head_layout = tensor.reshape(tensor.shape[0], 1, -1, head_dim)
    result = _mindspeed_all_to_all(
        head_layout.contiguous(),
        cp_group,
        scatter_dim=0,
        gather_dim=2,
        gather_size=global_heads,
    )
    expected_shape = (local_length, 1, global_heads, head_dim)
    if tuple(result.shape) != expected_shape:
        raise RuntimeError(
            f"Ulysses HP-to-CP output must have shape {expected_shape}, got {tuple(result.shape)}"
        )
    return result.reshape(local_length, 1, global_heads * head_dim)


def _normalize_prefix_blocks(
    prefix_blocks: Sequence[LocalKVBlock],
    *,
    cp_size: int,
    cp_rank: int,
    current_key: Tensor,
) -> tuple[LocalKVBlock, ...]:
    blocks = tuple(prefix_blocks)
    seen: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, LocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be LocalKVBlock")
        if block.segment_id in seen:
            raise ValueError(f"duplicate prefix segment_id {block.segment_id}")
        seen.add(block.segment_id)
        _validate_shard(
            block.shard,
            cp_size=cp_size,
            cp_rank=cp_rank,
            name=f"prefix block {block.segment_id}",
        )
        if not isinstance(block.key, Tensor) or not isinstance(block.value, Tensor):
            raise TypeError(f"prefix block {block.segment_id} K/V must be torch tensors")
        if block.key.ndim != 4 or block.value.ndim != 4:
            raise ValueError(f"prefix block {block.segment_id} K/V must be four-dimensional")
        if block.key.shape != block.value.shape:
            raise ValueError(f"prefix block {block.segment_id} K/V shapes must match")
        if block.key.shape[0] != block.shard.local_length or block.key.shape[1:] != current_key.shape[1:]:
            raise ValueError(
                f"prefix block {block.segment_id} shape must be "
                f"[{block.shard.local_length}, *{tuple(current_key.shape[1:])}], "
                f"got {tuple(block.key.shape)}"
            )
        if block.key.device != current_key.device or block.key.dtype != current_key.dtype:
            raise ValueError(f"prefix block {block.segment_id} dtype/device must match current KV")
    return blocks


def ulysses_cp_rectangular_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[LocalKVBlock] = (),
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
) -> Tensor:
    """Run TPR attention through MindSpeed's Ulysses All-to-All primitive."""

    cp_size, cp_rank = _group_world_size_and_rank(cp_group)
    _validate_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
    query_heads, _ = _validate_qkv(
        query,
        current_key,
        current_value,
        current_shard=current_shard,
        cp_size=cp_size,
    )
    blocks = _normalize_prefix_blocks(
        prefix_blocks,
        cp_size=cp_size,
        cp_rank=cp_rank,
        current_key=current_key,
    )

    head_query = _sequence_to_head(
        query,
        global_length=current_shard.padded_length,
        cp_group=cp_group,
    )[: current_shard.global_length]
    prefix_keys = [
        _sequence_to_head(block.key, global_length=block.shard.padded_length, cp_group=cp_group)[
            : block.shard.global_length
        ]
        for block in blocks
    ]
    prefix_values = [
        _sequence_to_head(block.value, global_length=block.shard.padded_length, cp_group=cp_group)[
            : block.shard.global_length
        ]
        for block in blocks
    ]
    head_current_key = _sequence_to_head(
        current_key,
        global_length=current_shard.padded_length,
        cp_group=cp_group,
    )[: current_shard.global_length]
    head_current_value = _sequence_to_head(
        current_value,
        global_length=current_shard.padded_length,
        cp_group=cp_group,
    )[: current_shard.global_length]
    head_key = torch.cat((*prefix_keys, head_current_key), dim=0)
    head_value = torch.cat((*prefix_values, head_current_value), dim=0)
    head_output = rectangular_causal_attention(
        head_query,
        head_key,
        head_value,
        softmax_scale=softmax_scale,
    )
    if current_shard.padded_length != current_shard.global_length:
        pad_shape = list(head_output.shape)
        pad_shape[0] = current_shard.padded_length - current_shard.global_length
        head_output = torch.cat((head_output, head_output.new_zeros(pad_shape)), dim=0)
    return _head_to_sequence(
        head_output,
        local_length=current_shard.local_length,
        global_heads=query_heads,
        head_dim=query.shape[-1],
        cp_group=cp_group,
    )


class UlyssesCPAttentionBackend:
    """Bind one TPR Segment and ordered Prefix blocks to Ulysses CP."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: SequenceShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[LocalKVBlock]],
        cp_group: Any,
    ) -> None:
        cp_size, cp_rank = _group_world_size_and_rank(cp_group)
        _validate_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
        if (
            not isinstance(global_prefix_length, int)
            or isinstance(global_prefix_length, bool)
            or global_prefix_length < 0
        ):
            raise ValueError(
                f"global_prefix_length must be a non-negative integer, got {global_prefix_length!r}"
            )
        normalized = {layer: tuple(blocks) for layer, blocks in sorted(prefix_blocks_by_layer.items())}
        if not normalized:
            raise ValueError("prefix_blocks_by_layer must contain every TPR layer")
        for layer, blocks in normalized.items():
            if not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0:
                raise ValueError(f"layer numbers must be positive integers, got {layer!r}")
            if sum(block.shard.global_length for block in blocks) != global_prefix_length:
                raise ValueError(f"layer {layer} Prefix blocks do not cover {global_prefix_length} tokens")
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
            raise KeyError(f"Ulysses CP context has no layer {layer_number}") from exc
        return ulysses_cp_rectangular_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=blocks,
            current_shard=self._current_shard,
            cp_group=self._cp_group,
            softmax_scale=softmax_scale,
        )


def _cached_blocks(kv_stack: KVStack, layer_number: int) -> tuple[LocalKVBlock, ...]:
    blocks = []
    for segment_id in kv_stack.segment_ids:
        state = kv_stack.get(segment_id).kv
        if not isinstance(physical_sequence_shard(state.shard), PrefixShard):
            raise TypeError("Ulysses CP requires contiguous Prefix shards")
        blocks.append(LocalKVBlock(segment_id, state.shard, *state.key_values[layer_number]))
    return tuple(blocks)


def make_cached_ulysses_cp_backend(
    kv_stack: KVStack,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: SequenceShard,
    cp_group: Any,
) -> UlyssesCPAttentionBackend:
    return UlyssesCPAttentionBackend(
        global_prefix_length=kv_stack.prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: _cached_blocks(kv_stack, layer) for layer in expected_layer_numbers
        },
        cp_group=cp_group,
    )


def make_anchored_ulysses_cp_backend(
    anchors: ShardedPastKVAnchors,
    *,
    expected_layer_numbers: tuple[int, ...],
    current_shard: SequenceShard,
    cp_group: Any,
) -> UlyssesCPAttentionBackend:
    for entry in anchors.entries:
        if not isinstance(physical_sequence_shard(entry.shard), PrefixShard):
            raise TypeError("Ulysses CP requires contiguous Prefix anchor shards")
    return UlyssesCPAttentionBackend(
        global_prefix_length=anchors.global_prefix_length,
        current_shard=current_shard,
        prefix_blocks_by_layer={
            layer: tuple(
                LocalKVBlock(entry.segment_id, entry.shard, *entry.key_values[layer])
                for entry in anchors.entries
            )
            for layer in expected_layer_numbers
        },
        cp_group=cp_group,
    )
