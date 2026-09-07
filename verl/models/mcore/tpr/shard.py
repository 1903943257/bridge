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

"""Backend-neutral sequence ownership for context-parallel TPR segments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

SequenceRange = tuple[int, int]


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_parallel_coordinates(cp_rank: int, cp_size: int) -> None:
    _validate_integer("cp_size", cp_size, minimum=1)
    _validate_integer("cp_rank", cp_rank, minimum=0)
    if cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")


def _normalize_dimension(tensor: Tensor, dim: int) -> int:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be an integer, got {dim!r}")
    normalized = dim if dim >= 0 else tensor.ndim + dim
    if normalized < 0 or normalized >= tensor.ndim:
        raise IndexError(f"dimension {dim} is out of range for tensor with {tensor.ndim} dimensions")
    return normalized


def _validate_global_offset(global_offset: int, global_length: int) -> None:
    _validate_integer("global_offset", global_offset, minimum=0)
    if global_offset >= global_length:
        raise ValueError(f"global_offset must be in [0, {global_length}), got {global_offset}")


@runtime_checkable
class SequenceShard(Protocol):
    """Logical placement of one global Segment on a single CP rank.

    The contract describes data ownership only. Ring steps, communication
    ranks, block identifiers, and head-sharded layouts belong to CP backends.
    """

    @property
    def global_length(self) -> int: ...

    @property
    def local_length(self) -> int: ...

    @property
    def cp_rank(self) -> int: ...

    @property
    def cp_size(self) -> int: ...

    @property
    def global_ranges(self) -> tuple[SequenceRange, ...]: ...

    @property
    def is_full(self) -> bool: ...

    def select(self, tensor: Tensor, *, dim: int = 0) -> Tensor: ...

    def global_indices(self, *, device: torch.device | str | None = None) -> Tensor: ...

    def owns(self, global_offset: int) -> bool: ...

    def global_to_local(self, global_offset: int) -> int | None: ...

    def local_to_global(self, local_offset: int) -> int: ...


@dataclass(frozen=True, slots=True)
class PrefixShard:
    """Uniform contiguous sequence shard retained for AllGather compatibility."""

    global_length: int
    local_start: int
    local_end: int
    cp_rank: int = 0
    cp_size: int = 1

    def __post_init__(self) -> None:
        _validate_integer("global_length", self.global_length, minimum=1)
        _validate_integer("local_start", self.local_start, minimum=0)
        _validate_integer("local_end", self.local_end, minimum=0)
        _validate_parallel_coordinates(self.cp_rank, self.cp_size)
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
        _validate_integer("global_length", global_length, minimum=1)
        _validate_parallel_coordinates(cp_rank, cp_size)
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
    def global_ranges(self) -> tuple[SequenceRange, ...]:
        return ((self.local_start, self.local_end),)

    @property
    def is_full(self) -> bool:
        return self.cp_size == 1

    def select(self, tensor: Tensor, *, dim: int = 0) -> Tensor:
        dim = _normalize_dimension(tensor, dim)
        if tensor.shape[dim] != self.global_length:
            raise ValueError(
                f"tensor sequence dimension must be {self.global_length}, got {tensor.shape[dim]}"
            )
        return tensor.narrow(dim, self.local_start, self.local_length)

    def global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        return torch.arange(self.local_start, self.local_end, dtype=torch.long, device=device)

    def owns(self, global_offset: int) -> bool:
        _validate_global_offset(global_offset, self.global_length)
        return self.local_start <= global_offset < self.local_end

    def global_to_local(self, global_offset: int) -> int | None:
        if not self.owns(global_offset):
            return None
        return global_offset - self.local_start

    def local_to_global(self, local_offset: int) -> int:
        _validate_integer("local_offset", local_offset, minimum=0)
        if local_offset >= self.local_length:
            raise ValueError(f"local_offset must be in [0, {self.local_length}), got {local_offset}")
        return self.local_start + local_offset


@dataclass(frozen=True, slots=True)
class RangeSequenceShard:
    """A backend-neutral shard composed of ordered, disjoint global ranges."""

    global_length: int
    ranges: tuple[SequenceRange, ...]
    cp_rank: int
    cp_size: int

    def __post_init__(self) -> None:
        _validate_integer("global_length", self.global_length, minimum=1)
        _validate_parallel_coordinates(self.cp_rank, self.cp_size)
        normalized = tuple(tuple(item) for item in self.ranges)
        if not normalized:
            raise ValueError("ranges must contain at least one non-empty range")
        previous_end = 0
        for index, item in enumerate(normalized):
            if len(item) != 2:
                raise ValueError(f"ranges[{index}] must be a (start, end) pair, got {item!r}")
            start, end = item
            _validate_integer(f"ranges[{index}].start", start, minimum=0)
            _validate_integer(f"ranges[{index}].end", end, minimum=1)
            if start >= end:
                raise ValueError(f"ranges[{index}] must be non-empty, got [{start}, {end})")
            if end > self.global_length:
                raise ValueError(
                    f"ranges[{index}] ends at {end}, beyond global_length {self.global_length}"
                )
            if index and start < previous_end:
                raise ValueError("ranges must be ordered and non-overlapping")
            previous_end = end
        object.__setattr__(self, "ranges", normalized)

    @property
    def global_ranges(self) -> tuple[SequenceRange, ...]:
        return self.ranges

    @property
    def local_length(self) -> int:
        return sum(end - start for start, end in self.ranges)

    @property
    def is_full(self) -> bool:
        return self.ranges == ((0, self.global_length),)

    def select(self, tensor: Tensor, *, dim: int = 0) -> Tensor:
        dim = _normalize_dimension(tensor, dim)
        if tensor.shape[dim] != self.global_length:
            raise ValueError(
                f"tensor sequence dimension must be {self.global_length}, got {tensor.shape[dim]}"
            )
        chunks = tuple(tensor.narrow(dim, start, end - start) for start, end in self.ranges)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=dim)

    def global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        chunks = tuple(
            torch.arange(start, end, dtype=torch.long, device=device) for start, end in self.ranges
        )
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def owns(self, global_offset: int) -> bool:
        _validate_global_offset(global_offset, self.global_length)
        return any(start <= global_offset < end for start, end in self.ranges)

    def global_to_local(self, global_offset: int) -> int | None:
        _validate_global_offset(global_offset, self.global_length)
        local_start = 0
        for start, end in self.ranges:
            if start <= global_offset < end:
                return local_start + global_offset - start
            local_start += end - start
        return None

    def local_to_global(self, local_offset: int) -> int:
        _validate_integer("local_offset", local_offset, minimum=0)
        if local_offset >= self.local_length:
            raise ValueError(f"local_offset must be in [0, {self.local_length}), got {local_offset}")
        remaining = local_offset
        for start, end in self.ranges:
            length = end - start
            if remaining < length:
                return start + remaining
            remaining -= length
        raise AssertionError("validated local offset was not mapped")
