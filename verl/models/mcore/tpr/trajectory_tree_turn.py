# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Turn-boundary-only, token-exact trajectory prefix sharing.

A "turn unit" is a complete Qwen3 ChatML message ending in <|im_end|>
(token 151645), or an explicitly captured prompt boundary. The final
possibly unfinished response tail is kept as ONE indivisible leaf unit.
Sharing requires entire units to match exactly; a partial shared message
never creates a short internal Push/Pop node.

Adjacent completed turns shorter than min_segment_tokens are grouped until
a complete-turn boundary at or beyond that threshold. This is a coarse,
optional physical execution topology, NOT a change to response supervision.
UIDs still bound candidate sharing; every input token and terminal row is
preserved. Other tokenizers must not select this Qwen-specific policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .trajectory_tree import (
    SegmentRef,
    TrajectoryNode,
    TrajectoryTree,
    _as_1d_input_ids,
    _parse_key,
    _rows,
)

QWEN3_IM_END_TOKEN_ID = 151645


@dataclass(slots=True)
class _TurnNode:
    end: int
    source_row: int
    member_rows: list[int] = field(default_factory=list)
    terminal_rows: list[int] = field(default_factory=list)
    children: dict[tuple[int, ...], "_TurnNode"] = field(default_factory=dict)


def _turn_units(
    tokens: list[int],
    *,
    prompt_length: int | None,
    min_segment_tokens: int,
    end_token_id: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Token-exact units ending at complete ChatML messages / prompt cut.

    End-of-input is always a boundary, but does not automatically introduce a
    shared internal node if a longer sibling has a different unfinished tail.
    """
    length = len(tokens)
    candidates = {i + 1 for i, token in enumerate(tokens)
                  if token == end_token_id}
    if prompt_length is not None and 0 < prompt_length < length:
        candidates.add(prompt_length)
    candidates.add(length)
    previous = 0
    units = []
    for end in sorted(candidates):
        # Only complete message/prompt boundaries can be shared internally.
        # A last short tail is always kept, as terminal rows need every token.
        if end < length and end - previous < min_segment_tokens:
            continue
        if end <= previous:
            raise AssertionError("turn unit boundaries must increase")
        units.append((tuple(tokens[previous:end]), end))
        previous = end
    if previous != length or not units:
        raise AssertionError("turn units must reconstruct entire trajectory")
    return tuple(units)


def _insert(root: _TurnNode, row: int,
            units: tuple[tuple[tuple[int, ...], int], ...]) -> None:
    current = root
    for signature, end in units:
        child = current.children.get(signature)
        if child is None:
            child = _TurnNode(end=end, source_row=row)
            current.children[signature] = child
        elif child.end != end:
            raise AssertionError("identical turn tokens have different absolute ends")
        child.member_rows.append(row)
        current = child
    current.terminal_rows.append(row)


def _compress(node: _TurnNode, parent_id: int | None,
              nodes: dict[int, TrajectoryNode], start: int) -> int:
    current = node
    while not current.terminal_rows and len(current.children) == 1:
        candidate = next(iter(current.children.values()))
        if current.member_rows != candidate.member_rows:
            break
        current = candidate
    node_id = len(nodes)
    segment = SegmentRef(row=node.source_row, start=start, end=current.end)
    nodes[node_id] = TrajectoryNode(
        node_id=node_id, parent_id=parent_id, segment=segment,
        member_rows=tuple(current.member_rows),
        terminal_rows=tuple(current.terminal_rows), children=(),
    )
    child_ids = tuple(
        _compress(child, node_id, nodes, current.end)
        for child in current.children.values()
    )
    nodes[node_id] = TrajectoryNode(
        node_id=node_id, parent_id=parent_id, segment=segment,
        member_rows=tuple(current.member_rows),
        terminal_rows=tuple(current.terminal_rows), children=child_ids,
    )
    return node_id


def build_trajectory_trees_turn(
    keys: list[str] | tuple[str, ...],
    batch: Any,
    *,
    min_segment_tokens: int = 128,
    end_token_id: int = QWEN3_IM_END_TOKEN_ID,
) -> tuple[TrajectoryTree, ...]:
    """Build exact TPR topology sharing only completed Turn units.

    Qwen3-only experimental strategy. A captured prompt endpoint may be used
    even when the ChatML prompt ends in an open assistant header. When no
    common valid unit exists, returns separate full-sequence trees rather
    than fabricating any shared prefix.
    """
    if not isinstance(min_segment_tokens, int) or isinstance(min_segment_tokens, bool) or min_segment_tokens < 1:
        raise ValueError("min_segment_tokens must be a positive integer")
    if not isinstance(end_token_id, int) or isinstance(end_token_id, bool) or end_token_id < 0:
        raise ValueError("end_token_id must be a non-negative integer")
    keys = list(keys)
    if not keys:
        return ()
    inputs = _rows(batch, "input_ids")
    if len(inputs) != len(keys):
        raise ValueError("turn builder input rows and trajectory keys disagree")
    try:
        prompts = _rows(batch, "prompts")
    except (ValueError, KeyError):
        prompts = None
    if prompts is not None and len(prompts) != len(keys):
        raise ValueError("turn builder prompt rows and trajectory keys disagree")

    groups: dict[str, list[int]] = {}
    units_by_row = []
    seen = set()
    for row, key in enumerate(keys):
        if key in seen:
            raise ValueError(f"duplicate trajectory key: {key!r}")
        seen.add(key)
        uid, _, _ = _parse_key(key)
        groups.setdefault(uid, []).append(row)
        input_ids = _as_1d_input_ids(inputs[row], row=row)
        token_ids = input_ids.detach().cpu().tolist()
        prompt_length = None
        if prompts is not None:
            prompt = _as_1d_input_ids(prompts[row], row=row)
            prompt_length = int(prompt.numel())
            if prompt_length > len(token_ids) or not torch.equal(input_ids[:prompt_length].cpu(), prompt.cpu()):
                raise ValueError(f"turn builder prompt is not an exact input prefix for row={row}")
        units_by_row.append(_turn_units(
            token_ids, prompt_length=prompt_length,
            min_segment_tokens=min_segment_tokens, end_token_id=end_token_id,
        ))

    forest: list[TrajectoryTree] = []
    for uid, rows in groups.items():
        dummy = _TurnNode(end=0, source_row=rows[0])
        for row in rows:
            _insert(dummy, row, units_by_row[row])
        for tree_index, child in enumerate(dummy.children.values()):
            nodes: dict[int, TrajectoryNode] = {}
            root_id = _compress(child, None, nodes, 0)
            forest.append(TrajectoryTree(
                uid=uid, tree_index=tree_index, root_id=root_id,
                member_rows=nodes[root_id].member_rows, nodes=nodes,
            ))
    return tuple(forest)
