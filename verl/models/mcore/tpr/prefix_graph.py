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

"""Strict-LIFO ownership for retained Prefix backward graphs.

This module intentionally has no dependency on torch, the segment executor,
or the activation-offload implementation.  A record owns opaque autograd
objects and one native offload session from Push until its matching Pop.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PrefixGraphSession(Protocol):
    """The only native-session operation owned by a graph record."""

    def close(self) -> None: ...


def _validate_segment_id(segment_id: int) -> None:
    if not isinstance(segment_id, int) or isinstance(segment_id, bool) or segment_id < 0:
        raise ValueError(f"segment_id must be a non-negative integer, got {segment_id!r}")


def _normalize_signature(signature: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(signature, tuple):
        raise TypeError(f"parent_stack_signature must be a tuple, got {type(signature).__name__}")
    for segment_id in signature:
        _validate_segment_id(segment_id)
    if len(set(signature)) != len(signature):
        raise ValueError(f"parent_stack_signature contains duplicate segments: {signature}")
    return signature


def _validate_session(session: PrefixGraphSession) -> None:
    if not callable(getattr(session, "close", None)):
        raise TypeError("native_session must provide close()")


class PrefixGraphRecord:
    """Own one retained Prefix graph and its native offload payload.

    Records have a one-way lifecycle: ``active -> consumed -> closed`` for a
    normal Pop, or ``active -> closed`` for cancellation.  ``close`` is
    deliberately idempotent so exception cleanup cannot close the native
    session twice.  Closing also drops every graph-carrying Python reference.
    """

    __slots__ = (
        "_segment_id",
        "_generation",
        "_parent_stack_signature",
        "_backward_loss_root",
        "_graph_key_values",
        "_parent_anchors",
        "_detached_loss",
        "_native_session",
        "_state",
    )

    def __init__(
        self,
        *,
        segment_id: int,
        generation: int,
        parent_stack_signature: tuple[int, ...],
        backward_loss_root: Any,
        graph_key_values: Mapping[int, Any],
        parent_anchors: Any,
        detached_loss: Any,
        native_session: PrefixGraphSession,
    ) -> None:
        _validate_segment_id(segment_id)
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError(f"generation must be a non-negative integer, got {generation!r}")
        parent_stack_signature = _normalize_signature(parent_stack_signature)
        if not isinstance(graph_key_values, Mapping):
            raise TypeError(f"graph_key_values must be a mapping, got {type(graph_key_values).__name__}")
        _validate_session(native_session)

        self._segment_id = segment_id
        self._generation = generation
        self._parent_stack_signature = parent_stack_signature
        self._backward_loss_root = backward_loss_root
        self._graph_key_values = dict(graph_key_values)
        self._parent_anchors = parent_anchors
        self._detached_loss = detached_loss
        self._native_session: PrefixGraphSession | None = native_session
        self._state = "active"

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def parent_stack_signature(self) -> tuple[int, ...]:
        return self._parent_stack_signature

    @property
    def active(self) -> bool:
        return self._state == "active"

    @property
    def consumed(self) -> bool:
        return self._state == "consumed"

    @property
    def closed(self) -> bool:
        return self._state == "closed"

    def _ensure_open(self) -> None:
        if self.closed:
            raise RuntimeError(f"Prefix graph for segment {self.segment_id} is closed")

    @property
    def backward_loss_root(self) -> Any:
        self._ensure_open()
        return self._backward_loss_root

    @property
    def graph_key_values(self) -> Mapping[int, Any]:
        self._ensure_open()
        return MappingProxyType(self._graph_key_values)

    @property
    def parent_anchors(self) -> Any:
        self._ensure_open()
        return self._parent_anchors

    @property
    def detached_loss(self) -> Any:
        self._ensure_open()
        return self._detached_loss

    @property
    def native_session(self) -> PrefixGraphSession:
        self._ensure_open()
        if self._native_session is None:  # Defensive; close clears the field.
            raise RuntimeError(f"Prefix graph for segment {self.segment_id} has no native session")
        return self._native_session

    def _consume(self) -> None:
        if self._state != "active":
            raise RuntimeError(
                f"Prefix graph for segment {self.segment_id} cannot be consumed from state {self._state}"
            )
        self._state = "consumed"

    def close(self) -> None:
        """Release native payload and graph references exactly once."""

        if self.closed:
            return
        session = self._native_session
        self._state = "closed"
        self._native_session = None
        try:
            if session is not None:
                session.close()
        finally:
            self._backward_loss_root = None
            self._graph_key_values.clear()
            self._parent_anchors = None
            self._detached_loss = None

    def __enter__(self) -> PrefixGraphRecord:
        if not self.consumed:
            raise RuntimeError(
                f"Prefix graph for segment {self.segment_id} must be consumed before entering backward scope"
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False


class PrefixGraphStore:
    """Own retained Prefix graphs in the same strict-LIFO order as KVStack."""

    __slots__ = ("_records", "_next_generation")

    def __init__(self) -> None:
        self._records: list[PrefixGraphRecord] = []
        self._next_generation = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def segment_ids(self) -> tuple[int, ...]:
        return tuple(record.segment_id for record in self._records)

    def top(self) -> PrefixGraphRecord:
        if not self._records:
            raise RuntimeError("Prefix graph store is empty")
        return self._records[-1]

    def push(
        self,
        *,
        segment_id: int,
        parent_stack_signature: tuple[int, ...],
        backward_loss_root: Any,
        graph_key_values: Mapping[int, Any],
        parent_anchors: Any,
        detached_loss: Any,
        native_session: PrefixGraphSession,
    ) -> PrefixGraphRecord:
        """Take ownership of one Push graph and its native session.

        Ownership of ``native_session`` transfers at method entry.  A rejected
        Push therefore closes it before propagating the validation error.
        """

        record: PrefixGraphRecord | None = None
        try:
            _validate_segment_id(segment_id)
            parent_stack_signature = _normalize_signature(parent_stack_signature)
            if parent_stack_signature != self.segment_ids:
                raise RuntimeError(
                    "Prefix graph parent stack mismatch: "
                    f"expected {self.segment_ids}, got {parent_stack_signature}"
                )
            if segment_id in self.segment_ids:
                raise RuntimeError(f"segment {segment_id} already has an active Prefix graph")
            record = PrefixGraphRecord(
                segment_id=segment_id,
                generation=self._next_generation,
                parent_stack_signature=parent_stack_signature,
                backward_loss_root=backward_loss_root,
                graph_key_values=graph_key_values,
                parent_anchors=parent_anchors,
                detached_loss=detached_loss,
                native_session=native_session,
            )
        except BaseException:
            try:
                if record is not None:
                    record.close()
                elif callable(getattr(native_session, "close", None)):
                    native_session.close()
            except BaseException as cleanup_error:
                raise RuntimeError("failed to close rejected Prefix graph session") from cleanup_error
            raise

        self._next_generation += 1
        self._records.append(record)
        return record

    def _pop_validated(
        self,
        *,
        segment_id: int,
        parent_stack_signature: tuple[int, ...],
    ) -> PrefixGraphRecord:
        _validate_segment_id(segment_id)
        parent_stack_signature = _normalize_signature(parent_stack_signature)
        record = self.top()
        if record.segment_id != segment_id:
            raise RuntimeError(
                f"cannot Pop Prefix graph segment {segment_id}: store top is {record.segment_id}"
            )
        expected_parent = tuple(item.segment_id for item in self._records[:-1])
        if record.parent_stack_signature != expected_parent:
            raise RuntimeError(
                f"Prefix graph store metadata is corrupt for segment {segment_id}: "
                f"record parent is {record.parent_stack_signature}, store parent is {expected_parent}"
            )
        if parent_stack_signature != record.parent_stack_signature:
            raise RuntimeError(
                f"Prefix graph parent stack is stale for segment {segment_id}: "
                f"expected {record.parent_stack_signature}, got {parent_stack_signature}"
            )
        self._records.pop()
        return record

    def consume(
        self,
        *,
        segment_id: int,
        parent_stack_signature: tuple[int, ...],
    ) -> PrefixGraphRecord:
        """Remove and consume the matching top record for exactly one Pop.

        Use the result as a context manager so native/session cleanup happens
        on both successful backward and exceptional exit.
        """

        record = self._pop_validated(
            segment_id=segment_id,
            parent_stack_signature=parent_stack_signature,
        )
        record._consume()
        return record

    def abort_top(
        self,
        *,
        segment_id: int,
        parent_stack_signature: tuple[int, ...],
    ) -> None:
        """Discard the matching active top record without running backward."""

        record = self._pop_validated(
            segment_id=segment_id,
            parent_stack_signature=parent_stack_signature,
        )
        record.close()

    def close_all(self) -> None:
        """Emergency cleanup in reverse Push order, attempting every close."""

        failures: list[BaseException] = []
        while self._records:
            record = self._records.pop()
            try:
                record.close()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(f"failed to close {len(failures)} Prefix graph session(s)") from failures[0]

    def assert_empty(self) -> None:
        if self._records:
            raise RuntimeError(f"Prefix graph store is not empty: active segments {self.segment_ids}")
