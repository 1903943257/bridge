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

"""Build token-exact compressed trajectory tries for TPR training."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """A zero-copy reference into one row of the original training batch."""

    row: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or isinstance(self.row, bool) or self.row < 0:
            raise ValueError(f"row must be a non-negative integer, got {self.row!r}")
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError(f"start must be a non-negative integer, got {self.start!r}")
        if not isinstance(self.end, int) or isinstance(self.end, bool) or self.end <= self.start:
            raise ValueError(f"end must be an integer > start, got {self.end!r}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TrajectoryNode:
    """One compressed token segment in a trajectory radix tree."""

    node_id: int
    parent_id: int | None
    segment: SegmentRef
    member_rows: tuple[int, ...]
    terminal_rows: tuple[int, ...]
    children: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, int) or isinstance(self.node_id, bool) or self.node_id < 0:
            raise ValueError(f"node_id must be a non-negative integer, got {self.node_id!r}")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, int)
            or isinstance(self.parent_id, bool)
            or self.parent_id < 0
            or self.parent_id == self.node_id
        ):
            raise ValueError(f"invalid parent_id {self.parent_id!r} for node {self.node_id}")
        if not self.member_rows:
            raise ValueError(f"node {self.node_id} must contain at least one member row")
        if len(set(self.member_rows)) != len(self.member_rows):
            raise ValueError(f"node {self.node_id} member_rows must be unique")
        if not set(self.terminal_rows).issubset(self.member_rows):
            raise ValueError(f"node {self.node_id} terminal_rows must be a subset of member_rows")


@dataclass(frozen=True, slots=True)
class TrajectoryTree:
    """One executable compressed trie whose real root starts at token position zero."""

    uid: str
    tree_index: int
    root_id: int
    member_rows: tuple[int, ...]
    nodes: Mapping[int, TrajectoryNode]

    def __post_init__(self) -> None:
        normalized = dict(self.nodes)
        if self.root_id not in normalized:
            raise ValueError(f"root_id {self.root_id} does not exist")
        root = normalized[self.root_id]
        if root.parent_id is not None:
            raise ValueError("trajectory tree root must not have a parent")
        if root.segment.start != 0:
            raise ValueError("trajectory tree executable root must start at token position 0")
        if root.member_rows != self.member_rows:
            raise ValueError("trajectory tree member_rows must match root member_rows")

        for node_id, node in normalized.items():
            if node_id != node.node_id:
                raise ValueError(f"node mapping key {node_id} does not match node_id {node.node_id}")
            if node.parent_id is not None and node.parent_id not in normalized:
                raise ValueError(f"node {node_id} has missing parent {node.parent_id}")
            for child_id in node.children:
                if child_id not in normalized:
                    raise ValueError(f"node {node_id} has missing child {child_id}")
                child = normalized[child_id]
                if child.parent_id != node_id:
                    raise ValueError(f"node {node_id} child {child_id} has inconsistent parent")
                if child.segment.start != node.segment.end:
                    raise ValueError(
                        f"node {child_id} must start at parent {node_id} end {node.segment.end}, "
                        f"got {child.segment.start}"
                    )

        object.__setattr__(self, "nodes", MappingProxyType(normalized))

    @property
    def key(self) -> str:
        return f"{self.uid}:{self.tree_index}"

    def get(self, node_id: int) -> TrajectoryNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown trajectory node {node_id}") from exc


@dataclass(slots=True)
class _TrieNode:
    """Temporary one-token trie node used only while building CPU topology."""

    position: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[int, "_TrieNode"] = field(default_factory=dict)


def _rows(batch: Any, name: str) -> list[Any]:
    try:
        value = batch[name]
    except Exception as exc:
        raise ValueError(f"training batch is missing required field {name!r}") from exc

    if isinstance(value, (list, tuple)):
        return list(value)

    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        return list(unbind())

    raise TypeError(f"field {name!r} does not expose row-wise values: {type(value)!r}")


def _as_1d_input_ids(value: Any, *, row: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"input_ids[{row}] must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 1:
        raise ValueError(f"input_ids[{row}] must be 1-D, got shape={tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"input_ids[{row}] must not be empty")
    if value.dtype == torch.bool or torch.is_floating_point(value) or torch.is_complex(value):
        raise ValueError(f"input_ids[{row}] must use an integer dtype, got {value.dtype}")
    if torch.any(value < 0).item():
        raise ValueError(f"input_ids[{row}] must contain non-negative token ids")
    return value


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"trajectory key must be str, got {type(key)!r}")
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"unexpected trajectory key format: {key!r}")
    try:
        trajectory_index = int(fields[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has non-integer index: {key!r}") from exc
    return fields[0], fields[1], trajectory_index


def _insert_sequence(dummy_root: _TrieNode, *, row: int, token_ids: torch.Tensor) -> None:
    node = dummy_root
    for position, token_id in enumerate(token_ids.detach().cpu().tolist()):
        token_id = int(token_id)
        child = node.children.get(token_id)
        if child is None:
            child = _TrieNode(position=position)
            node.children[token_id] = child
        elif child.position != position:
            raise RuntimeError(
                f"internal trie position mismatch for token {token_id}: "
                f"expected {position}, got {child.position}"
            )
        child.member_rows.append(row)
        node = child
    node.terminal_rows.append(row)


def _compress_real_root(real_root: _TrieNode, *, uid: str, tree_index: int) -> TrajectoryTree:
    nodes: dict[int, TrajectoryNode] = {}
    next_node_id = 0

    def compress(start_node: _TrieNode, parent_id: int | None) -> int:
        nonlocal next_node_id

        start = start_node.position
        current = start_node
        while not current.terminal_rows and len(current.children) == 1:
            child = next(iter(current.children.values()))
            if current.member_rows != child.member_rows:
                break
            current = child

        end = current.position + 1
        node_id = next_node_id
        next_node_id += 1
        member_rows = tuple(current.member_rows)
        terminal_rows = tuple(current.terminal_rows)
        source_row = member_rows[0]

        # Reserve the parent before recursing so descendants can point back to it.
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            parent_id=parent_id,
            segment=SegmentRef(row=source_row, start=start, end=end),
            member_rows=member_rows,
            terminal_rows=terminal_rows,
            children=(),
        )

        child_ids = tuple(compress(child, node_id) for child in current.children.values())
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            parent_id=parent_id,
            segment=SegmentRef(row=source_row, start=start, end=end),
            member_rows=member_rows,
            terminal_rows=terminal_rows,
            children=child_ids,
        )
        return node_id

    root_id = compress(real_root, None)
    return TrajectoryTree(
        uid=uid,
        tree_index=tree_index,
        root_id=root_id,
        member_rows=nodes[root_id].member_rows,
        nodes=nodes,
    )


def build_trajectory_trees_legacy(
    keys: list[str] | tuple[str, ...],
    batch: Any,
) -> tuple[TrajectoryTree, ...]:
    """Build token-exact compressed tries from a ReplayBuffer/TQ training batch.

    uid is used only as a candidate-sharing boundary. Within each uid group,
    topology is determined solely by exact prefixes of the complete input_ids
    sequence. Prompt/response boundaries do not affect compression.

    The internal ordinary trie uses a dummy root. Because the current executable
    SegmentPlan requires a non-empty root beginning at absolute position zero,
    each first-token branch under the dummy root is returned as its own
    TrajectoryTree. A uid group may therefore produce multiple executable trees
    when its rows do not share the first token.

    Nodes keep row/range references into the original batch instead of copying
    token tensors. terminal_rows records logical trajectories that end at a node
    even if that node also has descendants.
    """

    keys = list(keys)
    if not keys:
        return ()

    input_rows = _rows(batch, "input_ids")
    if len(keys) != len(input_rows):
        raise ValueError(f"keys/batch row mismatch: keys={len(keys)} rows={len(input_rows)}")

    seen_keys: set[str] = set()
    grouped_rows: dict[str, list[int]] = {}
    validated_input_ids: list[torch.Tensor] = []
    for row, (key, input_ids) in enumerate(zip(keys, input_rows, strict=True)):
        if key in seen_keys:
            raise ValueError(f"duplicate trajectory key: {key!r}")
        seen_keys.add(key)
        uid, _, _ = _parse_key(key)
        grouped_rows.setdefault(uid, []).append(row)
        validated_input_ids.append(_as_1d_input_ids(input_ids, row=row))

    trees: list[TrajectoryTree] = []
    for uid, member_rows in grouped_rows.items():
        dummy_root = _TrieNode(position=-1)
        for row in member_rows:
            _insert_sequence(dummy_root, row=row, token_ids=validated_input_ids[row])

        for tree_index, real_root in enumerate(dummy_root.children.values()):
            trees.append(_compress_real_root(real_root, uid=uid, tree_index=tree_index))

    return tuple(trees)


def build_trajectory_trees(
    keys: list[str] | tuple[str, ...],
    batch: Any,
) -> tuple[TrajectoryTree, ...]:
    """Build exact TPR trajectory trees via the direct-compressed radix path.

    The experimental fast builder passed exact legacy-topology and PPO
    reference differential tests. Keep build_trajectory_trees_legacy available
    for explicit regression comparison only; regular TPR callers use radix.
    This affects *only* CPU topology construction: the resulting public
    TrajectoryTree/SegmentRef, UID boundaries, loss ownership and runtime
    executor stay unchanged.
    """
    from .trajectory_tree_radix import build_trajectory_trees_radix

    return build_trajectory_trees_radix(keys, batch)


def build_prompt_sibling_trees(
    keys: list[str] | tuple[str, ...],
    batch: Any,
) -> tuple[TrajectoryTree, ...]:
    """Compatibility alias for the old V0 builder.

    The returned topology is now a generic compressed token trie and is no
    longer constrained to prompt -> response-leaf structure.
    """

    return build_trajectory_trees(keys, batch)
