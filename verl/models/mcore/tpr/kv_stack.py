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
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from .context import KVPair
from .prefix_state import KVPrefixState, PrefixStateEntry, PrefixStateStack
from .segment_plan import SegmentId, SegmentSpec
from .shard import SequenceShard


# Compatibility name retained for existing executor and profiling imports.
SegmentKV = KVPrefixState


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


class KVStackEntry(PrefixStateEntry):
    """KV-specialized compatibility entry over a generic PrefixStateEntry."""

    __slots__ = ()

    @property
    def kv(self) -> KVPrefixState:
        if not isinstance(self.state, KVPrefixState):
            raise TypeError(f"KVStackEntry requires KVPrefixState, got {type(self.state).__name__}")
        return self.state

    @property
    def gradients(self) -> Mapping[int, KVPair]:
        return self.kv.gradients

    def accumulate(self, layer_number: int, key_grad: Tensor, value_grad: Tensor) -> None:
        self.kv.accumulate_layer_gradients(layer_number, key_grad, value_grad)


class KVStack(PrefixStateStack):
    """Own the graph-free KV and relayed KV gradients on the active DFS path."""

    def __init__(self) -> None:
        super().__init__()

    def top(self) -> KVStackEntry:
        entry = super().top()
        if not isinstance(entry, KVStackEntry):
            raise TypeError(f"KVStack contains {type(entry).__name__}")
        return entry

    def get(self, segment_id: SegmentId) -> KVStackEntry:
        try:
            entry = super().get(segment_id)
        except KeyError as exc:
            raise KeyError(f"segment {segment_id} is not on the KV stack") from exc
        if not isinstance(entry, KVStackEntry):
            raise TypeError(f"KVStack contains {type(entry).__name__}")
        return entry

    def push(
        self,
        segment: SegmentSpec,
        key_values: Mapping[int, KVPair],
        *,
        shard: SequenceShard | None = None,
    ) -> KVStackEntry:
        kv = KVPrefixState(segment.segment_id, segment.length, key_values, shard=shard)
        if self._entries:
            first = self._entries[0]
            previous = self._entries[-1]
            if not isinstance(first, KVStackEntry) or not isinstance(previous, KVStackEntry):
                raise TypeError("KVStack contains a non-KV entry")
            expected_layers = first.kv.layer_numbers
            if kv.layer_numbers != expected_layers:
                raise ValueError(
                    f"segment {segment.segment_id} KV layers must be {expected_layers}, got {kv.layer_numbers}"
                )
            for layer_number in expected_layers:
                previous_key, _ = previous.kv.key_values[layer_number]
                current_key, _ = kv.key_values[layer_number]
                if previous_key.shape[1:] != current_key.shape[1:]:
                    raise ValueError(f"layer {layer_number} KV non-sequence shape differs across segments")
                if previous_key.device != current_key.device or previous_key.dtype != current_key.dtype:
                    raise ValueError(f"layer {layer_number} KV device or dtype differs across segments")

        entry = self.push_state(segment, kv, entry_type=KVStackEntry)
        if not isinstance(entry, KVStackEntry):
            raise TypeError(f"KVStack created {type(entry).__name__}")
        return entry

    def pop(self, segment_id: SegmentId) -> KVStackEntry:
        try:
            entry = self.pop_state(segment_id)
        except RuntimeError as exc:
            message = str(exc).replace("prefix-state stack", "KV stack")
            raise RuntimeError(message) from exc
        if not isinstance(entry, KVStackEntry):
            raise TypeError(f"KVStack contains {type(entry).__name__}")
        return entry

    def build_past_key_values(self) -> Mapping[int, KVPair]:
        """Concatenate graph-free KV on the active root-to-parent path."""

        if not self._entries:
            return MappingProxyType({})
        if any(entry.kv.shard.cp_size != 1 for entry in self._entries if isinstance(entry, KVStackEntry)):
            raise RuntimeError("sharded KV must be consumed through a context-parallel attention backend")
        result: dict[int, KVPair] = {}
        for layer_number in self._entries[0].kv.layer_numbers:
            entries = [entry for entry in self._entries if isinstance(entry, KVStackEntry)]
            if len(entries) != len(self._entries):
                raise TypeError("KVStack contains a non-KV entry")
            keys = [entry.kv.key_values[layer_number][0] for entry in entries]
            values = [entry.kv.key_values[layer_number][1] for entry in entries]
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
        try:
            super().assert_empty()
        except RuntimeError as exc:
            raise RuntimeError(str(exc).replace("prefix-state stack", "KV stack")) from exc
