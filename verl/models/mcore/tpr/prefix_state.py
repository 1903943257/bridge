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

"""Prefix-state lifecycle primitives shared by TPR attention backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from .context import KVPair
from .segment_plan import SegmentId, SegmentSpec


def _validate_segment_id(segment_id: SegmentId) -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
        raise ValueError(f"segment_id must be a non-negative integer, got {segment_id!r}")


def _validate_layer_number(layer_number: int) -> None:
    if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer_number!r}")


@dataclass(frozen=True, slots=True)
class PrefixShard:
    """One uniform contiguous shard of a segment's sequence dimension."""

    global_length: int
    local_start: int
    local_end: int
    cp_rank: int = 0
    cp_size: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("global_length", self.global_length),
            ("local_start", self.local_start),
            ("local_end", self.local_end),
            ("cp_rank", self.cp_rank),
            ("cp_size", self.cp_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if self.global_length <= 0:
            raise ValueError(f"global_length must be positive, got {self.global_length}")
        if self.cp_size <= 0:
            raise ValueError(f"cp_size must be positive, got {self.cp_size}")
        if self.cp_rank < 0 or self.cp_rank >= self.cp_size:
            raise ValueError(f"cp_rank must be in [0, {self.cp_size}), got {self.cp_rank}")
        if self.global_length % self.cp_size != 0:
            raise ValueError(
                f"global_length {self.global_length} must be divisible by cp_size {self.cp_size}"
            )

        expected_local_length = self.global_length // self.cp_size
        expected_start = self.cp_rank * expected_local_length
        expected_end = expected_start + expected_local_length
        if (self.local_start, self.local_end) != (expected_start, expected_end):
            raise ValueError(
                "contiguous shard range mismatch: "
                f"rank {self.cp_rank}/{self.cp_size} must own [{expected_start}, {expected_end}), "
                f"got [{self.local_start}, {self.local_end})"
            )

    @classmethod
    def full(cls, global_length: int) -> PrefixShard:
        return cls(global_length, 0, global_length)

    @classmethod
    def contiguous(cls, global_length: int, *, cp_rank: int, cp_size: int) -> PrefixShard:
        if (
            not isinstance(global_length, int)
            or isinstance(global_length, bool)
            or global_length <= 0
        ):
            raise ValueError(f"global_length must be positive, got {global_length!r}")
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 0:
            raise ValueError(f"cp_size must be positive, got {cp_size!r}")
        if global_length % cp_size != 0:
            raise ValueError(f"global_length {global_length} must be divisible by cp_size {cp_size}")
        local_length = global_length // cp_size
        return cls(
            global_length,
            cp_rank * local_length,
            (cp_rank + 1) * local_length,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )

    @property
    def local_length(self) -> int:
        return self.local_end - self.local_start

    @property
    def is_full(self) -> bool:
        return self.cp_size == 1


@runtime_checkable
class PrefixState(Protocol):
    """Minimal state contract consumed by the TPR path stack.

    Concrete state types own local tensors and gradient buffers. Distributed
    communication deliberately belongs to a separate CP backend.
    """

    @property
    def segment_id(self) -> SegmentId: ...

    @property
    def state_kind(self) -> str: ...

    @property
    def shard(self) -> PrefixShard: ...

    @property
    def global_length(self) -> int: ...

    @property
    def local_length(self) -> int: ...

    @property
    def layer_numbers(self) -> tuple[int, ...]: ...

    @property
    def released(self) -> bool: ...

    def validate(self) -> None: ...

    def make_anchors(self) -> Any: ...

    def accumulate_anchor_gradients(self, anchors: Any) -> None: ...

    def consume_gradients(self) -> Any: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class KVPrefixAnchors:
    """Autograd leaves for one local KV prefix state."""

    segment_id: SegmentId
    shard: PrefixShard
    generation: int
    key_values: Mapping[int, KVPair]


def _compact_graph_free_tensor(tensor: Tensor) -> Tensor:
    required_storage_bytes = tensor.numel() * tensor.element_size()
    storage_bytes = tensor.untyped_storage().nbytes()
    if tensor.is_contiguous() and tensor.storage_offset() == 0 and storage_bytes == required_storage_bytes:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _normalize_local_kv(
    key_values: Mapping[int, KVPair],
    *,
    local_length: int,
) -> dict[int, KVPair]:
    if not isinstance(key_values, Mapping) or not key_values:
        raise ValueError("key_values must be a non-empty mapping")

    normalized: dict[int, KVPair] = {}
    reference_device: torch.device | None = None
    reference_dtype: torch.dtype | None = None
    for layer_number, pair in key_values.items():
        _validate_layer_number(layer_number)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"layer {layer_number} KV must be a (key, value) tuple")
        key, value = pair
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise TypeError(f"layer {layer_number} K/V must be torch tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"layer {layer_number} K/V must have shape [sequence, batch, heads, head_dim], "
                f"got K={tuple(key.shape)}, V={tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(f"layer {layer_number} K/V shapes must match, got K={key.shape}, V={value.shape}")
        if key.shape[0] != local_length:
            raise ValueError(
                f"layer {layer_number} KV local sequence length must be {local_length}, got {key.shape[0]}"
            )
        if key.shape[1] != 1:
            raise ValueError(f"layer {layer_number} KV batch size must be 1, got {key.shape[1]}")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError(
                f"layer {layer_number} K/V device and dtype must match, "
                f"got K=({key.device}, {key.dtype}), V=({value.device}, {value.dtype})"
            )
        if key.requires_grad or value.requires_grad or key.grad_fn or value.grad_fn:
            raise ValueError(f"layer {layer_number} cached K/V must be detached and graph-free")
        if reference_device is None:
            reference_device, reference_dtype = key.device, key.dtype
        elif key.device != reference_device or key.dtype != reference_dtype:
            raise ValueError("all KV layers in a prefix state must use the same device and dtype")
        normalized[layer_number] = (
            _compact_graph_free_tensor(key),
            _compact_graph_free_tensor(value),
        )
    return dict(sorted(normalized.items()))


class KVPrefixState:
    """Local post-RoPE KV shards and delayed local gradients for one segment."""

    state_kind = "kv"

    def __init__(
        self,
        segment_id: SegmentId,
        sequence_length: int,
        key_values: Mapping[int, KVPair],
        *,
        shard: PrefixShard | None = None,
    ) -> None:
        _validate_segment_id(segment_id)
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")
        if shard is None:
            shard = PrefixShard.full(sequence_length)
        elif not isinstance(shard, PrefixShard):
            raise TypeError(f"shard must be PrefixShard, got {type(shard).__name__}")
        elif shard.global_length != sequence_length:
            raise ValueError(
                f"sequence_length {sequence_length} must match shard global_length {shard.global_length}"
            )

        self._segment_id = segment_id
        self._shard = shard
        self._key_values = _normalize_local_kv(key_values, local_length=shard.local_length)
        self._layer_numbers = tuple(self._key_values)
        self._gradients: dict[int, KVPair] = {}
        self._released = False
        self._generation = 0

    @property
    def segment_id(self) -> SegmentId:
        return self._segment_id

    @property
    def shard(self) -> PrefixShard:
        return self._shard

    @property
    def sequence_length(self) -> int:
        """Compatibility alias for the global segment length."""

        return self.global_length

    @property
    def global_length(self) -> int:
        return self._shard.global_length

    @property
    def local_length(self) -> int:
        return self._shard.local_length

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return self._layer_numbers

    @property
    def released(self) -> bool:
        return self._released

    @property
    def key_values(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        return MappingProxyType(self._key_values)

    @property
    def gradients(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        return MappingProxyType(self._gradients)

    def validate(self) -> None:
        self._ensure_live()
        normalized = _normalize_local_kv(self._key_values, local_length=self.local_length)
        if tuple(normalized) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} KV layers changed from {self.layer_numbers} to {tuple(normalized)}"
            )
        for layer_number, pair in normalized.items():
            stored_pair = self._key_values[layer_number]
            if pair[0] is not stored_pair[0] or pair[1] is not stored_pair[1]:
                raise RuntimeError(
                    f"segment {self.segment_id} layer {layer_number} KV storage is no longer compact"
                )

    def make_anchors(self) -> KVPrefixAnchors:
        self._ensure_live()
        anchors = {
            layer_number: (
                key.detach().requires_grad_(True),
                value.detach().requires_grad_(True),
            )
            for layer_number, (key, value) in self._key_values.items()
        }
        return KVPrefixAnchors(
            segment_id=self.segment_id,
            shard=self.shard,
            generation=self._generation,
            key_values=MappingProxyType(anchors),
        )

    def accumulate_layer_gradients(
        self,
        layer_number: int,
        key_grad: Tensor,
        value_grad: Tensor,
    ) -> None:
        self._ensure_live()
        if layer_number not in self._key_values:
            raise KeyError(f"segment {self.segment_id} has no KV for layer {layer_number}")
        key, value = self._key_values[layer_number]
        if key_grad.shape != key.shape or value_grad.shape != value.shape:
            raise ValueError(
                f"segment {self.segment_id} layer {layer_number} gradient shape mismatch: "
                f"expected {key.shape}, got dK={key_grad.shape}, dV={value_grad.shape}"
            )
        if key_grad.device != key.device or value_grad.device != value.device:
            raise ValueError(f"segment {self.segment_id} layer {layer_number} gradient device mismatch")
        if key_grad.dtype != key.dtype or value_grad.dtype != value.dtype:
            raise ValueError(f"segment {self.segment_id} layer {layer_number} gradient dtype mismatch")
        if layer_number not in self._gradients:
            self._gradients[layer_number] = (
                key_grad.detach().clone(memory_format=torch.contiguous_format),
                value_grad.detach().clone(memory_format=torch.contiguous_format),
            )
        else:
            accumulated_key, accumulated_value = self._gradients[layer_number]
            accumulated_key.add_(key_grad.detach())
            accumulated_value.add_(value_grad.detach())

    def accumulate_anchor_gradients(self, anchors: KVPrefixAnchors) -> None:
        self._ensure_live()
        if not isinstance(anchors, KVPrefixAnchors):
            raise TypeError(f"anchors must be KVPrefixAnchors, got {type(anchors).__name__}")
        if (
            anchors.segment_id != self.segment_id
            or anchors.shard != self.shard
            or anchors.generation != self._generation
        ):
            raise RuntimeError(f"KV prefix anchors for segment {anchors.segment_id} are stale")
        if tuple(anchors.key_values) != self.layer_numbers:
            raise RuntimeError(
                f"KV prefix anchor layers must be {self.layer_numbers}, got {tuple(anchors.key_values)}"
            )
        for layer_number, (key_anchor, value_anchor) in anchors.key_values.items():
            if key_anchor.grad is None or value_anchor.grad is None:
                raise RuntimeError(f"layer {layer_number} KV prefix anchor gradient is missing")
            self.accumulate_layer_gradients(layer_number, key_anchor.grad, value_anchor.grad)

    def consume_gradients(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        if self._gradients and tuple(self._gradients) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} gradient layers must be {self.layer_numbers}, "
                f"got {tuple(self._gradients)}"
            )
        gradients = self._gradients
        self._gradients = {}
        self._generation += 1
        return MappingProxyType(gradients)

    def release(self) -> None:
        if self._released:
            return
        self._key_values.clear()
        self._gradients.clear()
        self._released = True
        self._generation += 1

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError(f"KV prefix state for segment {self.segment_id} has been released")


@dataclass(slots=True)
class PrefixStateEntry:
    segment: SegmentSpec
    state: PrefixState


class PrefixStateStack:
    """Topology-only stack for heterogeneous prefix-state implementations."""

    def __init__(self) -> None:
        self._entries: list[PrefixStateEntry] = []
        self._by_segment_id: dict[SegmentId, PrefixStateEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def segment_ids(self) -> tuple[SegmentId, ...]:
        return tuple(entry.segment.segment_id for entry in self._entries)

    @property
    def prefix_length(self) -> int:
        return sum(entry.state.global_length for entry in self._entries)

    def top(self) -> PrefixStateEntry:
        if not self._entries:
            raise RuntimeError("prefix-state stack is empty")
        return self._entries[-1]

    def get(self, segment_id: SegmentId) -> PrefixStateEntry:
        try:
            return self._by_segment_id[segment_id]
        except KeyError as exc:
            raise KeyError(f"segment {segment_id} is not on the prefix-state stack") from exc

    def push_state(
        self,
        segment: SegmentSpec,
        state: PrefixState,
        *,
        entry_type: type[PrefixStateEntry] = PrefixStateEntry,
    ) -> PrefixStateEntry:
        if not isinstance(segment, SegmentSpec):
            raise TypeError(f"segment must be SegmentSpec, got {type(segment).__name__}")
        if not isinstance(state, PrefixState):
            raise TypeError(f"state must implement PrefixState, got {type(state).__name__}")
        state.validate()
        if state.released:
            raise RuntimeError(f"cannot push released state for segment {segment.segment_id}")
        if state.segment_id != segment.segment_id:
            raise ValueError(
                f"state segment_id {state.segment_id} does not match segment {segment.segment_id}"
            )
        if state.global_length != segment.length:
            raise ValueError(
                f"segment {segment.segment_id} state global length must be {segment.length}, "
                f"got {state.global_length}"
            )
        if segment.segment_id in self._by_segment_id:
            raise RuntimeError(f"segment {segment.segment_id} is already on the prefix-state stack")
        expected_parent = self._entries[-1].segment.segment_id if self._entries else None
        if segment.parent_id != expected_parent:
            raise ValueError(
                f"cannot push segment {segment.segment_id}: expected parent {expected_parent}, got {segment.parent_id}"
            )
        if segment.prefix_length != self.prefix_length or segment.position_start != self.prefix_length:
            raise ValueError(
                f"cannot push segment {segment.segment_id}: stack prefix length is {self.prefix_length}, "
                f"got prefix_length={segment.prefix_length}, position_start={segment.position_start}"
            )

        entry = entry_type(segment, state)
        self._entries.append(entry)
        self._by_segment_id[segment.segment_id] = entry
        return entry

    def pop_state(self, segment_id: SegmentId) -> PrefixStateEntry:
        if not self._entries:
            raise RuntimeError("cannot pop an empty prefix-state stack")
        if self._entries[-1].segment.segment_id != segment_id:
            raise RuntimeError(
                f"cannot pop segment {segment_id}: stack top is {self._entries[-1].segment.segment_id}"
            )
        entry = self._entries.pop()
        del self._by_segment_id[segment_id]
        return entry

    def assert_empty(self) -> None:
        if self._entries or self._by_segment_id:
            raise RuntimeError(f"prefix-state stack is not empty: {self.segment_ids}")
