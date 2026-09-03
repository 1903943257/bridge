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

"""KV path stack and gradient relay buffers for TPR segment execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import torch
from torch import Tensor

from .context import KVPair
from .segment_plan import SegmentId, SegmentSpec


def _validate_layer_number(layer_number: int) -> None:
    if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer_number!r}")


def _normalize_segment_kv(
    key_values: Mapping[int, KVPair],
    *,
    sequence_length: int,
    require_graph_free: bool,
) -> Mapping[int, KVPair]:
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
        if key.shape[0] != sequence_length:
            raise ValueError(
                f"layer {layer_number} KV sequence length must be {sequence_length}, got {key.shape[0]}"
            )
        if key.shape[1] != 1:
            raise ValueError(f"layer {layer_number} KV batch size must be 1, got {key.shape[1]}")
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError(
                f"layer {layer_number} K/V device and dtype must match, "
                f"got K=({key.device}, {key.dtype}), V=({value.device}, {value.dtype})"
            )
        if require_graph_free and (key.requires_grad or value.requires_grad or key.grad_fn or value.grad_fn):
            raise ValueError(f"layer {layer_number} cached K/V must be detached and graph-free")
        if reference_device is None:
            reference_device, reference_dtype = key.device, key.dtype
        elif key.device != reference_device or key.dtype != reference_dtype:
            raise ValueError("all KV layers in a segment must use the same device and dtype")
        normalized[layer_number] = pair
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class SegmentKV:
    """Graph-free KV produced by one segment for every attention layer."""

    segment_id: SegmentId
    sequence_length: int
    key_values: Mapping[int, KVPair]

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, int) or isinstance(self.segment_id, bool) or self.segment_id < 0:
            raise ValueError(f"segment_id must be a non-negative integer, got {self.segment_id!r}")
        if (
            not isinstance(self.sequence_length, int)
            or isinstance(self.sequence_length, bool)
            or self.sequence_length <= 0
        ):
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        object.__setattr__(
            self,
            "key_values",
            _normalize_segment_kv(
                self.key_values,
                sequence_length=self.sequence_length,
                require_graph_free=True,
            ),
        )

    @property
    def layer_numbers(self) -> tuple[int, ...]:
        return tuple(self.key_values)


@dataclass(frozen=True, slots=True)
class PastKVSlice:
    """Location of one segment inside concatenated past KV anchors."""

    segment_id: SegmentId
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PastKVAnchors:
    """Leaf past-KV tensors plus metadata needed to relay their gradients."""

    key_values: Mapping[int, KVPair]
    slices: tuple[PastKVSlice, ...]
    stack_signature: tuple[SegmentId, ...]

    @property
    def prefix_length(self) -> int:
        return 0 if not self.slices else self.slices[-1].end


@dataclass(slots=True)
class KVStackEntry:
    segment: SegmentSpec
    kv: SegmentKV
    _gradients: dict[int, KVPair] = field(default_factory=dict, repr=False)

    @property
    def gradients(self) -> Mapping[int, KVPair]:
        return MappingProxyType(self._gradients)

    def accumulate(self, layer_number: int, key_grad: Tensor, value_grad: Tensor) -> None:
        key, value = self.kv.key_values[layer_number]
        if key_grad.shape != key.shape or value_grad.shape != value.shape:
            raise ValueError(
                f"segment {self.segment.segment_id} layer {layer_number} gradient shape mismatch: "
                f"expected {key.shape}, got dK={key_grad.shape}, dV={value_grad.shape}"
            )
        if key_grad.device != key.device or value_grad.device != value.device:
            raise ValueError(f"segment {self.segment.segment_id} layer {layer_number} gradient device mismatch")
        if key_grad.dtype != key.dtype or value_grad.dtype != value.dtype:
            raise ValueError(f"segment {self.segment.segment_id} layer {layer_number} gradient dtype mismatch")
        if layer_number not in self._gradients:
            self._gradients[layer_number] = (key_grad.detach().clone(), value_grad.detach().clone())
        else:
            accumulated_key, accumulated_value = self._gradients[layer_number]
            accumulated_key.add_(key_grad.detach())
            accumulated_value.add_(value_grad.detach())


class KVStack:
    """Own the graph-free KV and relayed KV gradients on the active DFS path."""

    def __init__(self) -> None:
        self._entries: list[KVStackEntry] = []
        self._by_segment_id: dict[SegmentId, KVStackEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def segment_ids(self) -> tuple[SegmentId, ...]:
        return tuple(entry.segment.segment_id for entry in self._entries)

    @property
    def prefix_length(self) -> int:
        return sum(entry.segment.length for entry in self._entries)

    def top(self) -> KVStackEntry:
        if not self._entries:
            raise RuntimeError("KV stack is empty")
        return self._entries[-1]

    def get(self, segment_id: SegmentId) -> KVStackEntry:
        try:
            return self._by_segment_id[segment_id]
        except KeyError as exc:
            raise KeyError(f"segment {segment_id} is not on the KV stack") from exc

    def push(self, segment: SegmentSpec, key_values: Mapping[int, KVPair]) -> KVStackEntry:
        if not isinstance(segment, SegmentSpec):
            raise TypeError(f"segment must be SegmentSpec, got {type(segment).__name__}")
        if segment.segment_id in self._by_segment_id:
            raise RuntimeError(f"segment {segment.segment_id} is already on the KV stack")
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

        kv = SegmentKV(segment.segment_id, segment.length, key_values)
        if self._entries:
            expected_layers = self._entries[0].kv.layer_numbers
            if kv.layer_numbers != expected_layers:
                raise ValueError(
                    f"segment {segment.segment_id} KV layers must be {expected_layers}, got {kv.layer_numbers}"
                )
            for layer_number in expected_layers:
                previous_key, _ = self._entries[-1].kv.key_values[layer_number]
                current_key, _ = kv.key_values[layer_number]
                if previous_key.shape[1:] != current_key.shape[1:]:
                    raise ValueError(f"layer {layer_number} KV non-sequence shape differs across segments")
                if previous_key.device != current_key.device or previous_key.dtype != current_key.dtype:
                    raise ValueError(f"layer {layer_number} KV device or dtype differs across segments")

        entry = KVStackEntry(segment, kv)
        self._entries.append(entry)
        self._by_segment_id[segment.segment_id] = entry
        return entry

    def pop(self, segment_id: SegmentId) -> KVStackEntry:
        if not self._entries:
            raise RuntimeError("cannot pop an empty KV stack")
        if self._entries[-1].segment.segment_id != segment_id:
            raise RuntimeError(f"cannot pop segment {segment_id}: stack top is {self._entries[-1].segment.segment_id}")
        entry = self._entries.pop()
        del self._by_segment_id[segment_id]
        return entry

    def build_past_key_values(self) -> Mapping[int, KVPair]:
        """Concatenate graph-free KV on the active root-to-parent path."""

        if not self._entries:
            return MappingProxyType({})
        result: dict[int, KVPair] = {}
        for layer_number in self._entries[0].kv.layer_numbers:
            keys = [entry.kv.key_values[layer_number][0] for entry in self._entries]
            values = [entry.kv.key_values[layer_number][1] for entry in self._entries]
            result[layer_number] = (torch.cat(keys, dim=0), torch.cat(values, dim=0))
        return MappingProxyType(result)

    def build_past_anchors(self) -> PastKVAnchors:
        """Build detached leaf tensors whose grads can be split back to ancestors."""

        concatenated = self.build_past_key_values()
        anchors = {
            layer_number: (
                key.detach().requires_grad_(True),
                value.detach().requires_grad_(True),
            )
            for layer_number, (key, value) in concatenated.items()
        }
        slices: list[PastKVSlice] = []
        offset = 0
        for entry in self._entries:
            end = offset + entry.segment.length
            slices.append(PastKVSlice(entry.segment.segment_id, offset, end))
            offset = end
        return PastKVAnchors(MappingProxyType(anchors), tuple(slices), self.segment_ids)

    def accumulate_anchor_gradients(self, anchors: PastKVAnchors) -> None:
        """Split full-prefix anchor grads and add them to ancestor buffers."""

        if not isinstance(anchors, PastKVAnchors):
            raise TypeError(f"anchors must be PastKVAnchors, got {type(anchors).__name__}")
        if anchors.stack_signature != self.segment_ids:
            raise RuntimeError(
                f"past KV anchors are stale: built for {anchors.stack_signature}, current stack is {self.segment_ids}"
            )
        if not self._entries:
            if anchors.key_values or anchors.slices:
                raise RuntimeError("empty stack received non-empty past KV anchors")
            return

        expected_layers = self._entries[0].kv.layer_numbers
        if tuple(anchors.key_values) != expected_layers:
            raise RuntimeError(
                f"past KV anchor layers must be {expected_layers}, got {tuple(anchors.key_values)}"
            )
        for layer_number, (key_anchor, value_anchor) in anchors.key_values.items():
            if key_anchor.grad is None or value_anchor.grad is None:
                raise RuntimeError(f"layer {layer_number} past KV anchor gradient is missing")
            for kv_slice in anchors.slices:
                entry = self.get(kv_slice.segment_id)
                entry.accumulate(
                    layer_number,
                    key_anchor.grad[kv_slice.start : kv_slice.end],
                    value_anchor.grad[kv_slice.start : kv_slice.end],
                )

    def get_new_kv_gradients(self, segment_id: SegmentId) -> Mapping[int, KVPair]:
        return self.get(segment_id).gradients

    def assert_empty(self) -> None:
        if self._entries or self._by_segment_id:
            raise RuntimeError(f"KV stack is not empty: {self.segment_ids}")
