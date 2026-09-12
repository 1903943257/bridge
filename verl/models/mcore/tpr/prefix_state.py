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

"""Prefix-state lifecycle primitives shared by TPR model layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from .context import KVPair
from .segment_plan import SegmentId, SegmentSpec
from .shard import PrefixShard, SequenceShard


def _validate_segment_id(segment_id: SegmentId) -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
        raise ValueError(f"segment_id must be a non-negative integer, got {segment_id!r}")


def _validate_layer_number(layer_number: int) -> None:
    if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer_number!r}")


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
    def shard(self) -> SequenceShard: ...

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
class GDNLayerState:
    """Causal-convolution and recurrent continuation state for one GDN layer."""

    conv_state: Tensor
    recurrent_state: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.conv_state, Tensor) or not isinstance(self.recurrent_state, Tensor):
            raise TypeError("GDN conv_state and recurrent_state must be torch tensors")
        if self.conv_state.ndim != 3:
            raise ValueError(
                "GDN conv_state must have shape [batch, channels, history], "
                f"got {tuple(self.conv_state.shape)}"
            )
        if self.recurrent_state.ndim != 4:
            raise ValueError(
                "GDN recurrent_state must have shape [batch, heads, key_dim, value_dim], "
                f"got {tuple(self.recurrent_state.shape)}"
            )
        if any(dimension <= 0 for dimension in self.conv_state.shape):
            raise ValueError(f"GDN conv_state dimensions must be positive, got {tuple(self.conv_state.shape)}")
        if any(dimension <= 0 for dimension in self.recurrent_state.shape):
            raise ValueError(
                f"GDN recurrent_state dimensions must be positive, got {tuple(self.recurrent_state.shape)}"
            )
        if self.conv_state.shape[0] != self.recurrent_state.shape[0]:
            raise ValueError(
                "GDN continuation states must have the same batch size, "
                f"got conv={self.conv_state.shape[0]} and recurrent={self.recurrent_state.shape[0]}"
            )
        if self.conv_state.device != self.recurrent_state.device:
            raise ValueError(
                "GDN continuation states must be on the same device, "
                f"got conv={self.conv_state.device} and recurrent={self.recurrent_state.device}"
            )
        if not self.conv_state.is_floating_point() or not self.recurrent_state.is_floating_point():
            raise ValueError("GDN conv_state and recurrent_state must be floating-point tensors")


@dataclass(frozen=True, slots=True)
class GDNPrefixAnchors:
    """Autograd leaves for all GDN layer states saved by one segment."""

    segment_id: SegmentId
    generation: int
    layer_states: Mapping[int, GDNLayerState]


def _normalize_gdn_layer_states(
    layer_states: Mapping[int, GDNLayerState],
    *,
    graph_free: bool,
) -> dict[int, GDNLayerState]:
    if not isinstance(layer_states, Mapping) or not layer_states:
        raise ValueError("GDN layer_states must be a non-empty mapping")

    normalized: dict[int, GDNLayerState] = {}
    reference_device: torch.device | None = None
    reference_batch: int | None = None
    for layer_number, layer_state in layer_states.items():
        _validate_layer_number(layer_number)
        if not isinstance(layer_state, GDNLayerState):
            raise TypeError(
                f"layer {layer_number} state must be GDNLayerState, got {type(layer_state).__name__}"
            )
        conv_state = layer_state.conv_state
        recurrent_state = layer_state.recurrent_state
        if graph_free and (
            conv_state.requires_grad
            or recurrent_state.requires_grad
            or conv_state.grad_fn is not None
            or recurrent_state.grad_fn is not None
        ):
            raise ValueError(f"layer {layer_number} cached GDN state must be detached and graph-free")
        if reference_device is None:
            reference_device = conv_state.device
            reference_batch = conv_state.shape[0]
        elif conv_state.device != reference_device or conv_state.shape[0] != reference_batch:
            raise ValueError("all GDN layers in a prefix state must use the same device and batch size")
        normalized[layer_number] = GDNLayerState(
            _compact_graph_free_tensor(conv_state) if graph_free else conv_state,
            _compact_graph_free_tensor(recurrent_state) if graph_free else recurrent_state,
        )
    return dict(sorted(normalized.items()))


class GDNPrefixState:
    """Graph-free GDN continuation states and delayed gradients for one segment.

    Stage 3 deliberately stores no packed-sequence or context-parallel metadata.
    ``sequence_length`` belongs only to the segment lifecycle; GDN continuation
    itself consists solely of each layer's convolution and recurrent tensors.
    """

    state_kind = "gdn"

    def __init__(
        self,
        segment_id: SegmentId,
        sequence_length: int,
        layer_states: Mapping[int, GDNLayerState],
    ) -> None:
        _validate_segment_id(segment_id)
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")

        self._segment_id = segment_id
        self._sequence_length = sequence_length
        self._layer_states = _normalize_gdn_layer_states(layer_states, graph_free=True)
        self._layer_numbers = tuple(self._layer_states)
        self._gradients: dict[int, GDNLayerState] = {}
        self._released = False
        self._generation = 0

    @classmethod
    def save(
        cls,
        segment_id: SegmentId,
        sequence_length: int,
        layer_states: Mapping[int, GDNLayerState],
    ) -> GDNPrefixState:
        """Detach and compact state tensors produced by a graph-carrying Push."""

        normalized = _normalize_gdn_layer_states(layer_states, graph_free=False)
        saved_states = {
            layer_number: GDNLayerState(
                layer_state.conv_state.detach().clone(memory_format=torch.contiguous_format),
                layer_state.recurrent_state.detach().clone(memory_format=torch.contiguous_format),
            )
            for layer_number, layer_state in normalized.items()
        }
        return cls(segment_id, sequence_length, saved_states)

    @property
    def segment_id(self) -> SegmentId:
        return self._segment_id

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return self._layer_numbers

    @property
    def released(self) -> bool:
        return self._released

    @property
    def layer_states(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        return MappingProxyType(self._layer_states)

    @property
    def gradients(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        return MappingProxyType(self._gradients)

    def restore(self, layer_number: int) -> GDNLayerState:
        """Return one saved layer state for independent branch continuation."""

        self._ensure_live()
        _validate_layer_number(layer_number)
        try:
            return self._layer_states[layer_number]
        except KeyError as exc:
            raise KeyError(f"segment {self.segment_id} has no GDN state for layer {layer_number}") from exc

    def validate(self) -> None:
        self._ensure_live()
        normalized = _normalize_gdn_layer_states(self._layer_states, graph_free=True)
        if tuple(normalized) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} GDN layers changed from {self.layer_numbers} to {tuple(normalized)}"
            )
        for layer_number, layer_state in normalized.items():
            stored_state = self._layer_states[layer_number]
            if (
                layer_state.conv_state is not stored_state.conv_state
                or layer_state.recurrent_state is not stored_state.recurrent_state
            ):
                raise RuntimeError(
                    f"segment {self.segment_id} layer {layer_number} GDN storage is no longer compact"
                )

    def make_anchors(self) -> GDNPrefixAnchors:
        """Create branch-local autograd leaves without mutating the saved state."""

        self._ensure_live()
        anchors = {
            layer_number: GDNLayerState(
                layer_state.conv_state.detach().requires_grad_(True),
                layer_state.recurrent_state.detach().requires_grad_(True),
            )
            for layer_number, layer_state in self._layer_states.items()
        }
        return GDNPrefixAnchors(
            segment_id=self.segment_id,
            generation=self._generation,
            layer_states=MappingProxyType(anchors),
        )

    def accumulate_layer_gradients(
        self,
        layer_number: int,
        conv_grad: Tensor,
        recurrent_grad: Tensor,
    ) -> None:
        self._ensure_live()
        _validate_layer_number(layer_number)
        try:
            saved_state = self._layer_states[layer_number]
        except KeyError as exc:
            raise KeyError(f"segment {self.segment_id} has no GDN state for layer {layer_number}") from exc
        gradient = GDNLayerState(conv_grad, recurrent_grad)
        for name, actual, expected in (
            ("conv", gradient.conv_state, saved_state.conv_state),
            ("recurrent", gradient.recurrent_state, saved_state.recurrent_state),
        ):
            if actual.shape != expected.shape:
                raise ValueError(
                    f"segment {self.segment_id} layer {layer_number} {name} gradient shape mismatch: "
                    f"expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
                )
            if actual.device != expected.device or actual.dtype != expected.dtype:
                raise ValueError(
                    f"segment {self.segment_id} layer {layer_number} {name} gradient "
                    "device/dtype mismatch"
                )

        if layer_number not in self._gradients:
            self._gradients[layer_number] = GDNLayerState(
                conv_grad.detach().clone(memory_format=torch.contiguous_format),
                recurrent_grad.detach().clone(memory_format=torch.contiguous_format),
            )
        else:
            accumulated = self._gradients[layer_number]
            accumulated.conv_state.add_(conv_grad.detach())
            accumulated.recurrent_state.add_(recurrent_grad.detach())

    def accumulate_anchor_gradients(self, anchors: GDNPrefixAnchors) -> None:
        self._ensure_live()
        if not isinstance(anchors, GDNPrefixAnchors):
            raise TypeError(f"anchors must be GDNPrefixAnchors, got {type(anchors).__name__}")
        if anchors.segment_id != self.segment_id or anchors.generation != self._generation:
            raise RuntimeError(f"GDN prefix anchors for segment {anchors.segment_id} are stale")
        if tuple(anchors.layer_states) != self.layer_numbers:
            raise RuntimeError(
                f"GDN prefix anchor layers must be {self.layer_numbers}, got {tuple(anchors.layer_states)}"
            )
        for layer_number, layer_state in anchors.layer_states.items():
            if layer_state.conv_state.grad is None or layer_state.recurrent_state.grad is None:
                raise RuntimeError(f"layer {layer_number} GDN prefix anchor gradient is missing")
            self.accumulate_layer_gradients(
                layer_number,
                layer_state.conv_state.grad,
                layer_state.recurrent_state.grad,
            )

    def consume_gradients(self) -> Mapping[int, GDNLayerState]:
        self._ensure_live()
        if self._gradients and tuple(self._gradients) != self.layer_numbers:
            raise RuntimeError(
                f"segment {self.segment_id} GDN gradient layers must be {self.layer_numbers}, "
                f"got {tuple(self._gradients)}"
            )
        gradients = self._gradients
        self._gradients = {}
        self._generation += 1
        return MappingProxyType(gradients)

    def release(self) -> None:
        if self._released:
            return
        self._layer_states.clear()
        self._gradients.clear()
        self._released = True
        self._generation += 1

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError(f"GDN prefix state for segment {self.segment_id} has been released")


@dataclass(frozen=True, slots=True)
class KVPrefixAnchors:
    """Autograd leaves for one local KV prefix state."""

    segment_id: SegmentId
    shard: SequenceShard
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
        shard: SequenceShard | None = None,
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
        elif not isinstance(shard, SequenceShard):
            raise TypeError(f"shard must implement SequenceShard, got {type(shard).__name__}")
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
    def shard(self) -> SequenceShard:
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
