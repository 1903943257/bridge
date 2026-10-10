# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Lower exact token tries to executable TPR plans and logical PPO token references.

The topology never owns PPO values or normalization factors. A response token
at absolute position t is predicted by the query at t-1, which may belong to
the *parent* segment when t is the first token of a child.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import torch

from .segment_plan import SegmentPlan, SegmentSpec
from .trajectory_tree import TrajectoryTree, build_trajectory_trees, build_trajectory_trees_legacy


@dataclass(frozen=True, slots=True)
class SegmentObjectiveRef:
    segment_id: int
    query_offset: int
    target_token_id: int
    sample_row: int
    response_offset: int


@dataclass(frozen=True, slots=True)
class TreeExecutionPlan:
    tree: TrajectoryTree
    segment_plan: SegmentPlan
    objective_refs: tuple[SegmentObjectiveRef, ...]

    def refs_for_segment(self, segment_id: int) -> tuple[SegmentObjectiveRef, ...]:
        self.segment_plan.get(segment_id)
        return tuple(ref for ref in self.objective_refs if ref.segment_id == segment_id)

    @property
    def logical_loss_tokens(self) -> int:
        return len(self.objective_refs)


@dataclass(frozen=True, slots=True)
class ForestExecutionPlan:
    trees: tuple[TreeExecutionPlan, ...]
    logical_loss_tokens: int

    def __post_init__(self):
        if sum(tree.logical_loss_tokens for tree in self.trees) != self.logical_loss_tokens:
            raise ValueError("forest logical_loss_tokens disagrees with objective refs")

    @property
    def segment_count(self) -> int:
        return sum(len(tree.segment_plan.segments) for tree in self.trees)


def _rows(batch: Any, key: str) -> list[torch.Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"batch is missing {key!r}") from exc
    values = list(value) if isinstance(value, (list, tuple)) else list(value.unbind())
    if not all(isinstance(item, torch.Tensor) and item.ndim == 1 for item in values):
        raise ValueError(f"{key!r} must contain one-dimensional tensor rows")
    return values


def build_tree_execution_plans(
    keys: list[str] | tuple[str, ...],
    batch: Any,
    *,
    trees: tuple[TrajectoryTree, ...] | None = None,
    require_loss_mask_alignment: bool = False,
    tree_builder: str = "radix",
) -> ForestExecutionPlan:
    """Build a forest of topology-only SegmentPlans and PPO ownership references.

    response_mask is right-aligned with input_ids, matching VERL's RL
    prompt+response layout. Every True response-mask element maps to exactly
    one ref; duplicate branches/samples intentionally retain separate refs.
    Optional loss-mask alignment catches mismatch against the native
    batch_num_tokens denominator before entering the Megatron schedule.
    Default radix builds direct compressed TPR topology. The old per-token
    builder remains available as tree_builder="legacy" for CPU regression.
    """
    if tree_builder not in ("legacy", "radix"):
        raise ValueError(f"unsupported TPR tree_builder={tree_builder!r}; expected legacy or radix")
    keys = list(keys)
    if trees is None:
        if tree_builder == "legacy":
            trees = build_trajectory_trees_legacy(keys, batch)
        else:
            trees = build_trajectory_trees(keys, batch)
    input_rows = _rows(batch, "input_ids")
    response_masks = _rows(batch, "response_mask")
    if len(keys) != len(input_rows) or len(keys) != len(response_masks):
        raise ValueError("keys, input_ids and response_mask must have identical row counts")

    try:
        response_rows = _rows(batch, "responses")
    except ValueError:
        response_rows = None

    selected_positions: list[list[int]] = []
    expected_token_count = 0
    for row, (tokens, mask) in enumerate(zip(input_rows, response_masks, strict=True)):
        if mask.numel() > tokens.numel():
            raise ValueError(f"response_mask[{row}] is longer than input_ids[{row}]")
        if response_rows is not None:
            response = response_rows[row]
            if response.numel() != mask.numel():
                raise ValueError(f"responses/response_mask length mismatch on row {row}")
            if not torch.equal(tokens[-response.numel():], response):
                raise ValueError(f"response is not an input_ids suffix on row {row}")
        valid = torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
        prefix_len = tokens.numel() - mask.numel()
        if valid and prefix_len + valid[0] == 0:
            raise ValueError(f"row {row} has no preceding query token for first supervised token")
        selected_positions.append(valid)
        expected_token_count += len(valid)

    if require_loss_mask_alignment:
        loss_masks = _rows(batch, "loss_mask")
        if len(loss_masks) != len(keys):
            raise ValueError("loss_mask rows do not match keys")
        for row, (loss_mask, response_mask) in enumerate(zip(loss_masks, response_masks, strict=True)):
            if loss_mask.numel() == response_mask.numel():
                aligned = loss_mask
            elif loss_mask.numel() == input_rows[row].numel():
                aligned = loss_mask[-response_mask.numel():]
                if bool(loss_mask[:-response_mask.numel()].to(bool).any()):
                    raise ValueError(f"row {row} loss_mask includes prompt tokens")
            else:
                raise ValueError(f"row {row} loss_mask shape is incompatible with response_mask")
            if not torch.equal(aligned.to(bool), response_mask.to(bool)):
                raise ValueError(f"row {row} response_mask and loss_mask disagree")

    consumed_rows: set[int] = set()
    result: list[TreeExecutionPlan] = []
    for tree in trees:
        segments = []
        for node in tree.nodes.values():
            ref = node.segment
            token_ids = input_rows[ref.row][ref.start:ref.end].detach().to(device="cpu", dtype=torch.long)
            segments.append(
                SegmentSpec(
                    segment_id=node.node_id,
                    parent_id=node.parent_id,
                    token_ids=token_ids,
                    position_start=ref.start,
                    prefix_length=ref.start,
                )
            )
        # Physical topology is independent of CE or PPO objective metadata.
        plan = SegmentPlan(segments, root_id=tree.root_id, topology_only=True)
        refs: list[SegmentObjectiveRef] = []
        terminal_to_path = {}

        def visit(node_id: int, path: tuple[int, ...]) -> None:
            node = tree.get(node_id)
            current_path = (*path, node_id)
            for row in node.terminal_rows:
                if row in terminal_to_path:
                    raise ValueError(f"trajectory row {row} has multiple terminal nodes")
                terminal_to_path[row] = current_path
            for child_id in node.children:
                visit(child_id, current_path)

        visit(tree.root_id, ())
        if set(terminal_to_path) != set(tree.member_rows):
            raise ValueError(f"tree {tree.key} terminal rows do not cover members")

        for row in tree.member_rows:
            if row in consumed_rows:
                raise ValueError(f"row {row} appears in more than one tree")
            consumed_rows.add(row)
            input_ids = input_rows[row]
            response_len = response_masks[row].numel()
            response_start = input_ids.numel() - response_len
            path = terminal_to_path[row]
            ends = [tree.get(node_id).segment.end for node_id in path]
            for response_offset in selected_positions[row]:
                target_position = response_start + response_offset
                query_position = target_position - 1
                owner_idx = bisect_right(ends, query_position)
                if owner_idx >= len(path):
                    raise ValueError(f"row {row} query position {query_position} is outside tree")
                node_id = path[owner_idx]
                node = tree.get(node_id)
                if not node.segment.start <= query_position < node.segment.end:
                    raise ValueError(f"row {row} has uncovered query position {query_position}")
                target = int(input_ids[target_position].item())
                refs.append(
                    SegmentObjectiveRef(
                        segment_id=node_id,
                        query_offset=query_position - node.segment.start,
                        target_token_id=target,
                        sample_row=row,
                        response_offset=response_offset,
                    )
                )
        result.append(TreeExecutionPlan(tree=tree, segment_plan=plan, objective_refs=tuple(refs)))

    if consumed_rows != set(range(len(keys))):
        raise ValueError(f"forest does not cover all input rows: {set(range(len(keys))) - consumed_rows}")
    forest = ForestExecutionPlan(tuple(result), logical_loss_tokens=expected_token_count)
    return forest
