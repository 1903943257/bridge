# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DP placement -> the UNMODIFIED TPR execution plan, CPU contract.

No DTA TokenTrie/CompressedTrie is imported. DP only assigns original rows.
Every DP shard builds the exact same TrajectoryTree and SegmentPlan classes
as ordinary single-rank TPR PPO training.
"""

import torch

from verl.models.mcore.tpr.dp_placement import (
    _uid_scoped_tree_cost,
    plan_tpr_dta_dp,
)
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _source_rows():
    # Two independent TPR UIDs deliberately share [1, 2, 3]; execution is
    # still UID-scoped. The third alpha row is a terminal prefix of the first.
    seqs = [
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 6),
        (1, 2, 3),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 7, 8),
        (1, 2, 3, 7, 9),
    ]
    keys = [f"alpha_trace_{i}" for i in range(3)] + [
        f"beta_trace_{i}" for i in range(3)
    ]
    return seqs, keys


def _make_batch(seqs):
    rows = [torch.tensor(seq, dtype=torch.long) for seq in seqs]
    masks = [torch.ones(min(2, len(seq) - 1), dtype=torch.bool) for seq in seqs]
    return {
        "input_ids": rows,
        "response_mask": masks,
        "responses": [row[-len(mask):].clone() for row, mask in zip(rows, masks, strict=True)],
        "loss_mask": [mask.clone() for mask in masks],
    }


def _check_existing_tpr_forest(subset, keys):
    batch = _make_batch(subset)
    forest = build_tree_execution_plans(
        keys, batch, require_loss_mask_alignment=True
    )
    # All these objects come from the original TPR implementation.
    assert sum(tree.logical_loss_tokens for tree in forest.trees) == sum(
        mask.sum().item() for mask in batch["response_mask"]
    )
    actual_tree_tokens = 0
    observed = set()
    for execution in forest.trees:
        plan = execution.segment_plan
        events = plan.validate_events(plan.dfs_events())
        assert events
        actual_tree_tokens += sum(s.length for s in plan.segments.values())
        for ref in execution.objective_refs:
            assert 0 <= ref.sample_row < len(subset)
            logical = (ref.sample_row, ref.response_offset)
            assert logical not in observed
            observed.add(logical)
            token = batch["input_ids"][ref.sample_row]
            mask = batch["response_mask"][ref.sample_row]
            target_idx = token.numel() - mask.numel() + ref.response_offset
            segment = plan.get(ref.segment_id)
            assert segment.position_start + ref.query_offset == target_idx - 1
            assert ref.target_token_id == token[target_idx].item()
    expected = {
        (row, offset)
        for row, mask in enumerate(batch["response_mask"])
        for offset in range(mask.numel())
    }
    assert observed == expected
    return forest, actual_tree_tokens


def test_tpr_dp_split_rebuilds_identical_original_executable_forests():
    seqs, keys = _source_rows()
    placement = plan_tpr_dta_dp(seqs, keys, dp_size=2)
    assert placement.equal_rows_per_rank
    assert placement.policy == "tpr_dta"
    assert sorted(i for shard in placement.partitions for i in shard) == list(range(len(seqs)))

    _, whole_cost = _check_existing_tpr_forest(seqs, keys)
    assert placement.global_tree_tokens == whole_cost

    rank_costs = []
    global_objective_owners = []
    for original_indices in placement.partitions:
        local_seqs = [seqs[i] for i in original_indices]
        local_keys = [keys[i] for i in original_indices]
        forest, rank_cost = _check_existing_tpr_forest(local_seqs, local_keys)
        rank_costs.append(rank_cost)
        for execution in forest.trees:
            for ref in execution.objective_refs:
                global_objective_owners.append(
                    (original_indices[ref.sample_row], ref.response_offset)
                )
    assert tuple(rank_costs) == placement.tree_tokens_by_rank
    assert sum(rank_costs) - whole_cost == placement.duplicated_tree_tokens
    assert sorted(global_objective_owners) == sorted(
        (row, offset) for row, seq in enumerate(seqs)
        for offset in range(min(2, len(seq) - 1))
    )


def test_duplicate_rows_survive_dp_split_and_tpr_objective_lowering():
    seqs = [(5, 6, 7, 8)] * 4
    keys = [f"shared_trace_{i}" for i in range(4)]
    placement = plan_tpr_dta_dp(seqs, keys, dp_size=2)
    assert [len(part) for part in placement.partitions] == [2, 2]
    logical_owners = []
    for indices in placement.partitions:
        forest, cost = _check_existing_tpr_forest(
            [seqs[i] for i in indices],
            [keys[i] for i in indices],
        )
        assert cost == 4
        logical_owners.extend(
            (indices[ref.sample_row], ref.response_offset)
            for execution in forest.trees
            for ref in execution.objective_refs
        )
    assert sorted(logical_owners) == sorted(
        (i, offset) for i in range(4) for offset in range(2)
    )
