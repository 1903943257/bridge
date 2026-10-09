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

"""Direct-compressed LCP/radix builder for the EXISTING TPR tree.

Motivation: trajectory_tree._insert_sequence materializes one temporary
_TrieNode per distinct token before compressing. This builder inserts entire
sequence spans into a radix trie, using the LCP of a new row and an existing
span to split nodes only at a real fork/termination.

DTA inspires the compact LCP *representation*, not our execution semantics.
No DTA TokenTrie, DTAEngine, KV, or distributed runtime is imported.

Crucially this returns the *same* TrajectoryTree / TrajectoryNode / SegmentRef
as build_trajectory_trees (including node ids, UID-scoped trees, children in
first-appearance order, and logical duplicate/terminal row ownership).
The original public build_trajectory_trees now selects this builder by default;
the old per-token builder is retained as build_trajectory_trees_legacy for
explicit regression comparisons, with the same downstream TPR execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trajectory_tree import (
    SegmentRef,
    TrajectoryNode,
    TrajectoryTree,
    _as_1d_input_ids,
    _parse_key,
    _rows,
)


@dataclass(slots=True)
class _RadixNode:
    start: int
    end: int
    source_row: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[int, "_RadixNode"] = field(default_factory=dict)


def _lcp_end(
    current: list[int], existing: list[int], start: int, stop: int
) -> int:
    """First differing absolute token position in [start, stop)."""
    position = start
    while position < stop and current[position] == existing[position]:
        position += 1
    return position


def _insert_compressed(
    root: _RadixNode, row: int, token_rows: list[list[int]]
) -> None:
    """Insert one logical row, splitting one compressed span at each fork."""
    tokens = token_rows[row]
    parent = root
    pos = 0
    n = len(tokens)
    while pos < n:
        token = tokens[pos]
        child = parent.children.get(token)
        if child is None:
            # No old branch: allocate exactly one leaf for the entire tail.
            parent.children[token] = _RadixNode(
                start=pos, end=n, source_row=row,
                member_rows=[row], terminal_rows=[row],
            )
            return

        fork = _lcp_end(
            tokens, token_rows[child.source_row], pos, min(n, child.end)
        )
        if fork == child.end:
            # The complete existing compressed segment is shared.
            child.member_rows.append(row)
            pos = child.end
            if pos == n:
                child.terminal_rows.append(row)
                return
            parent = child
            continue

        # The new row diverges *inside* an existing compressed segment, or
        # terminates inside it. Split into common prefix + old suffix (+ new
        # suffix when present). Unlike a normal trie, no per-token nodes.
        if fork <= pos:
            raise AssertionError("radix child index disagrees with token")
        old_token = token_rows[child.source_row][fork]
        common = _RadixNode(
            start=pos,
            end=fork,
            source_row=child.source_row,
            member_rows=[*child.member_rows, row],
            children={old_token: child},
        )
        child.start = fork
        # Assigning the existing dictionary key preserves original insertion
        # order of siblings; new siblings append after the old branch.
        parent.children[token] = common
        if fork == n:
            common.terminal_rows.append(row)
        else:
            new_token = tokens[fork]
            if new_token == old_token:
                raise AssertionError("split tokens must differ")
            common.children[new_token] = _RadixNode(
                start=fork, end=n, source_row=row,
                member_rows=[row], terminal_rows=[row],
            )
        return

    raise AssertionError("nonempty trajectory terminated without a leaf")


def _to_tpr_tree(
    root: _RadixNode, *, uid: str, tree_index: int
) -> TrajectoryTree:
    """Lower directly compressed metadata to the unchanged TPR public types."""
    nodes: dict[int, TrajectoryNode] = {}

    def visit(node: _RadixNode, parent_id: int | None) -> int:
        node_id = len(nodes)
        # Reserve the parent before its children to keep exactly the old
        # builder's deterministic preorder node ids.
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            parent_id=parent_id,
            segment=SegmentRef(row=node.source_row, start=node.start, end=node.end),
            member_rows=tuple(node.member_rows),
            terminal_rows=tuple(node.terminal_rows),
            children=(),
        )
        children = tuple(visit(child, node_id) for child in node.children.values())
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            parent_id=parent_id,
            segment=SegmentRef(row=node.source_row, start=node.start, end=node.end),
            member_rows=tuple(node.member_rows),
            terminal_rows=tuple(node.terminal_rows),
            children=children,
        )
        return node_id

    root_id = visit(root, None)
    return TrajectoryTree(
        uid=uid, tree_index=tree_index, root_id=root_id,
        member_rows=nodes[root_id].member_rows, nodes=nodes,
    )


def build_trajectory_trees_radix(
    keys: list[str] | tuple[str, ...], batch: Any
) -> tuple[TrajectoryTree, ...]:
    """Direct-compressed default TPR builder; only CPU tree construction changes.

    Accepts the same keys/batch as the original builder. In particular, UIDs
    are *candidate sharing boundaries* and the dummy root is never executed.
    """
    keys = list(keys)
    if not keys:
        return ()

    input_rows = _rows(batch, "input_ids")
    if len(keys) != len(input_rows):
        raise ValueError(
            f"keys/batch row mismatch: keys={len(keys)} rows={len(input_rows)}"
        )
    seen_keys: set[str] = set()
    grouped_rows: dict[str, list[int]] = {}
    tokens: list[list[int]] = []
    for row, (key, value) in enumerate(zip(keys, input_rows, strict=True)):
        if key in seen_keys:
            raise ValueError(f"duplicate trajectory key: {key!r}")
        seen_keys.add(key)
        uid, _, _ = _parse_key(key)
        grouped_rows.setdefault(uid, []).append(row)
        validated = _as_1d_input_ids(value, row=row)
        tokens.append(validated.detach().cpu().tolist())

    trees = []
    for uid, rows in grouped_rows.items():
        # Dummy root has no executable token and is never materialized.
        root = _RadixNode(start=0, end=0, source_row=rows[0])
        for row in rows:
            _insert_compressed(root, row, tokens)
        for index, child in enumerate(root.children.values()):
            trees.append(_to_tpr_tree(child, uid=uid, tree_index=index))
    return tuple(trees)
