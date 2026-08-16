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

"""CPU-side segment topology and execution protocol for DTA training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import torch
from torch import Tensor

SegmentId = int


def _validate_segment_id(segment_id: SegmentId, *, field_name: str = "segment_id") -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {segment_id!r}")


@dataclass(frozen=True, slots=True)
class SegmentLossTerm:
    """One weighted next-token loss owned by a segment query position.

    Multiple terms may use the same ``query_offset``. This represents a
    branching position whose logit predicts the first token of several child
    segments without retaining a separate logit-gradient relay buffer.
    """

    query_offset: int
    target_token_id: int
    weight: float = 1.0
    sample_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query_offset, int) or isinstance(self.query_offset, bool) or self.query_offset < 0:
            raise ValueError(f"query_offset must be a non-negative integer, got {self.query_offset!r}")
        if (
            not isinstance(self.target_token_id, int)
            or isinstance(self.target_token_id, bool)
            or self.target_token_id < 0
        ):
            raise ValueError(f"target_token_id must be a non-negative integer, got {self.target_token_id!r}")
        if not isinstance(self.weight, (int, float)) or isinstance(self.weight, bool) or not math.isfinite(self.weight):
            raise ValueError(f"weight must be finite, got {self.weight!r}")
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight!r}")
        if self.sample_id is not None:
            _validate_segment_id(self.sample_id, field_name="sample_id")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    """An immutable description of one non-empty contiguous token segment."""

    segment_id: SegmentId
    parent_id: SegmentId | None
    token_ids: Tensor
    position_start: int
    prefix_length: int
    loss_terms: tuple[SegmentLossTerm, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_segment_id(self.segment_id)
        if self.parent_id is not None:
            _validate_segment_id(self.parent_id, field_name="parent_id")
            if self.parent_id == self.segment_id:
                raise ValueError(f"segment {self.segment_id} cannot be its own parent")
        if not isinstance(self.token_ids, Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if self.token_ids.device.type != "cpu" or self.token_ids.dtype != torch.long or self.token_ids.ndim != 1:
            raise ValueError(
                "token_ids must be a one-dimensional CPU torch.long tensor, "
                f"got shape={tuple(self.token_ids.shape)}, dtype={self.token_ids.dtype}, device={self.token_ids.device}"
            )
        if self.token_ids.numel() == 0:
            raise ValueError("token_ids must contain at least one token")
        if torch.any(self.token_ids < 0).item():
            raise ValueError("token_ids must be non-negative")
        if not isinstance(self.position_start, int) or isinstance(self.position_start, bool) or self.position_start < 0:
            raise ValueError(f"position_start must be a non-negative integer, got {self.position_start!r}")
        if not isinstance(self.prefix_length, int) or isinstance(self.prefix_length, bool) or self.prefix_length < 0:
            raise ValueError(f"prefix_length must be a non-negative integer, got {self.prefix_length!r}")

        normalized_terms = tuple(self.loss_terms)
        for term in normalized_terms:
            if not isinstance(term, SegmentLossTerm):
                raise TypeError(f"loss_terms must contain SegmentLossTerm objects, got {type(term).__name__}")
            if term.query_offset >= self.length:
                raise ValueError(
                    f"segment {self.segment_id} loss query_offset {term.query_offset} is outside length {self.length}"
                )
        object.__setattr__(self, "loss_terms", normalized_terms)

        # Isolate the plan from later mutation of the caller's tensor. Tensor
        # identity is not part of the scheduler protocol.
        object.__setattr__(self, "token_ids", self.token_ids.detach().clone())

    @property
    def length(self) -> int:
        return self.token_ids.numel()

    @property
    def position_end(self) -> int:
        """Exclusive absolute position at the end of this segment."""

        return self.position_start + self.length


class SegmentEventKind(str, Enum):
    PUSH = "push"
    POP = "pop"


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    kind: SegmentEventKind
    segment_id: SegmentId

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SegmentEventKind):
            raise TypeError(f"kind must be SegmentEventKind, got {type(self.kind).__name__}")
        _validate_segment_id(self.segment_id)


def PushSegment(segment_id: SegmentId) -> SegmentEvent:  # noqa: N802
    """Create an event that enters and forwards a segment."""

    return SegmentEvent(SegmentEventKind.PUSH, segment_id)


def PopSegment(segment_id: SegmentId) -> SegmentEvent:  # noqa: N802
    """Create an event that recomputes/backwards and releases a segment."""

    return SegmentEvent(SegmentEventKind.POP, segment_id)


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Validated rooted segment tree stored entirely as CPU metadata."""

    segments: Mapping[SegmentId, SegmentSpec] | Sequence[SegmentSpec]
    root_id: SegmentId
    total_loss_weight: float | None = None
    _children: Mapping[SegmentId, tuple[SegmentId, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_segment_id(self.root_id, field_name="root_id")
        source: Iterable[SegmentSpec]
        source = self.segments.values() if isinstance(self.segments, Mapping) else self.segments

        normalized: dict[SegmentId, SegmentSpec] = {}
        for segment in source:
            if not isinstance(segment, SegmentSpec):
                raise TypeError(f"segments must contain SegmentSpec objects, got {type(segment).__name__}")
            if segment.segment_id in normalized:
                raise ValueError(f"duplicate segment_id {segment.segment_id}")
            normalized[segment.segment_id] = segment
        if not normalized:
            raise ValueError("segments must not be empty")
        if self.root_id not in normalized:
            raise ValueError(f"root_id {self.root_id} does not exist")

        roots = [segment.segment_id for segment in normalized.values() if segment.parent_id is None]
        if roots != [self.root_id]:
            raise ValueError(f"plan must have exactly root {self.root_id}, got roots={sorted(roots)}")

        children: dict[SegmentId, list[SegmentId]] = {segment_id: [] for segment_id in normalized}
        for segment in normalized.values():
            if segment.parent_id is None:
                if segment.prefix_length != 0 or segment.position_start != 0:
                    raise ValueError("root segment must have prefix_length=0 and position_start=0")
                continue
            if segment.parent_id not in normalized:
                raise ValueError(f"segment {segment.segment_id} has missing parent {segment.parent_id}")
            parent = normalized[segment.parent_id]
            if segment.prefix_length != parent.position_end or segment.position_start != parent.position_end:
                raise ValueError(
                    f"segment {segment.segment_id} must start at parent {parent.segment_id} end {parent.position_end}, "
                    f"got prefix_length={segment.prefix_length}, position_start={segment.position_start}"
                )
            children[segment.parent_id].append(segment.segment_id)

        visited: set[SegmentId] = set()
        active: set[SegmentId] = set()

        def visit(segment_id: SegmentId) -> None:
            if segment_id in active:
                raise ValueError(f"cycle detected at segment {segment_id}")
            if segment_id in visited:
                return
            active.add(segment_id)
            for child_id in children[segment_id]:
                visit(child_id)
            active.remove(segment_id)
            visited.add(segment_id)

        visit(self.root_id)
        if visited != set(normalized):
            raise ValueError(f"segments are not reachable from root: {sorted(set(normalized) - visited)}")

        inferred_weight = sum(term.weight for segment in normalized.values() for term in segment.loss_terms)
        total_loss_weight = inferred_weight if self.total_loss_weight is None else self.total_loss_weight
        if (
            not isinstance(total_loss_weight, (int, float))
            or isinstance(total_loss_weight, bool)
            or not math.isfinite(total_loss_weight)
            or total_loss_weight <= 0
        ):
            raise ValueError(f"total_loss_weight must be finite and positive, got {total_loss_weight!r}")

        object.__setattr__(self, "segments", MappingProxyType(normalized))
        object.__setattr__(self, "_children", MappingProxyType({key: tuple(value) for key, value in children.items()}))
        object.__setattr__(self, "total_loss_weight", float(total_loss_weight))

    def get(self, segment_id: SegmentId) -> SegmentSpec:
        try:
            return self.segments[segment_id]
        except KeyError as exc:
            raise KeyError(f"unknown segment_id {segment_id}") from exc

    def parent_of(self, segment_id: SegmentId) -> SegmentSpec | None:
        segment = self.get(segment_id)
        return None if segment.parent_id is None else self.segments[segment.parent_id]

    def children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        self.get(segment_id)
        return tuple(self.segments[child_id] for child_id in self._children[segment_id])

    def path_to(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        path: list[SegmentSpec] = []
        current: SegmentSpec | None = self.get(segment_id)
        while current is not None:
            path.append(current)
            current = None if current.parent_id is None else self.segments[current.parent_id]
        return tuple(reversed(path))

    def depth_of(self, segment_id: SegmentId) -> int:
        return len(self.path_to(segment_id)) - 1

    def dfs_events(self) -> Iterator[SegmentEvent]:
        """Yield a deterministic child-insertion-order DFS Push/Pop stream."""

        def traverse(segment_id: SegmentId) -> Iterator[SegmentEvent]:
            yield PushSegment(segment_id)
            for child_id in self._children[segment_id]:
                yield from traverse(child_id)
            yield PopSegment(segment_id)

        yield from traverse(self.root_id)

    def validate_events(self, events: Iterable[SegmentEvent]) -> tuple[SegmentEvent, ...]:
        """Validate stack discipline and exactly-once execution for a plan."""

        normalized = tuple(events)
        stack: list[SegmentId] = []
        pushed: set[SegmentId] = set()
        popped: set[SegmentId] = set()
        for event in normalized:
            if not isinstance(event, SegmentEvent):
                raise TypeError(f"events must contain SegmentEvent objects, got {type(event).__name__}")
            segment = self.get(event.segment_id)
            if event.kind is SegmentEventKind.PUSH:
                if event.segment_id in pushed:
                    raise ValueError(f"segment {event.segment_id} is pushed more than once")
                expected_parent = stack[-1] if stack else None
                if segment.parent_id != expected_parent:
                    raise ValueError(
                        f"cannot push segment {event.segment_id}: expected parent on stack {segment.parent_id}, "
                        f"got {expected_parent}"
                    )
                stack.append(event.segment_id)
                pushed.add(event.segment_id)
            else:
                if not stack or stack[-1] != event.segment_id:
                    raise ValueError(f"cannot pop segment {event.segment_id}: current stack is {stack}")
                if any(child_id not in popped for child_id in self._children[event.segment_id]):
                    raise ValueError(f"cannot pop segment {event.segment_id} before all children")
                stack.pop()
                popped.add(event.segment_id)

        expected = set(self.segments)
        if stack or pushed != expected or popped != expected:
            raise ValueError(
                f"incomplete event stream: stack={stack}, unpushed={sorted(expected - pushed)}, "
                f"unpopped={sorted(expected - popped)}"
            )
        return normalized

