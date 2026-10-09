# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Differential tests: radix/LCP builder must equal original executable TPR tree."""

import random

import pytest
import torch

from verl.models.mcore.tpr.trajectory_tree import build_trajectory_trees
from verl.models.mcore.tpr.trajectory_tree_radix import build_trajectory_trees_radix
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _batch(rows):
    tokens = [torch.tensor(row, dtype=torch.long) for row in rows]
    # Last 1 or 2 positions are supervised; query must exist in each row.
    masks = [torch.ones(min(2, len(row) - 1), dtype=torch.bool) for row in rows]
    return {
        "input_ids": tokens,
        "response_mask": masks,
        "responses": [tokens[i][-len(mask):].clone() for i, mask in enumerate(masks)],
        "loss_mask": [mask.clone() for mask in masks],
    }


def _assert_equivalent(rows, uids):
    batch = _batch(rows)
    keys = [f"{uid}_trace_{i}" for i, uid in enumerate(uids)]
    old = build_trajectory_trees(keys, batch)
    new = build_trajectory_trees_radix(keys, batch)
    # Strong identity: exactly the same UID/tree-order, segment spans,
    # original source rows, terminal/duplicate members and node IDs.
    assert new == old

    baseline = build_tree_execution_plans(
        keys, batch, trees=old, require_loss_mask_alignment=True
    )
    experimental = build_tree_execution_plans(
        keys, batch, trees=new, require_loss_mask_alignment=True
    )
    assert experimental.logical_loss_tokens == baseline.logical_loss_tokens
    assert experimental.segment_count == baseline.segment_count
    for original, fast in zip(baseline.trees, experimental.trees, strict=True):
        assert fast.tree == original.tree
        assert fast.objective_refs == original.objective_refs
        assert fast.segment_plan.root_id == original.segment_plan.root_id
        assert fast.segment_plan.validate_events(
            fast.segment_plan.dfs_events()
        ) == original.segment_plan.validate_events(
            original.segment_plan.dfs_events()
        )
        for node_id in original.segment_plan.segments:
            left = original.segment_plan.get(node_id)
            right = fast.segment_plan.get(node_id)
            assert left.parent_id == right.parent_id
            assert left.position_start == right.position_start
            assert left.prefix_length == right.prefix_length
            assert torch.equal(left.token_ids, right.token_ids)


@pytest.mark.parametrize(
    "rows,uids",
    [
        # Nested forks and first-token roots; UID order has interleaving.
        ([[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 7], [8, 9]],
         ["a", "a", "a", "a"]),
        # Duplicate exact trajectory, prefix-of-longer, then an extension.
        ([[1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
         ["a"] * 4),
        # Strict prefix arrives AFTER the longer path, forcing a split
        # with an internal terminal node; other branch is already present.
        ([[1, 2, 3, 4, 5], [1, 2, 3, 7], [1, 2], [1, 2, 3, 4, 6]],
         ["a"] * 4),
        # Two UIDs share token values but are separate execution trees.
        ([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3, 6]],
         ["a", "b", "a", "b"]),
        # Multiple first tokens, exact duplicate and no intermediate forks.
        ([[5, 1], [7, 2], [5, 1], [9, 3]], ["a"] * 4),
        # One UID with one trajectory, and another UID with two siblings.
        ([[1, 2], [5, 6, 7], [5, 6, 8]], ["a", "b", "b"]),
    ],
)
def test_radix_builder_matches_original_tree_and_ppo_lowering(rows, uids):
    _assert_equivalent(rows, uids)


@pytest.mark.parametrize("seed", [0, 1, 7, 13, 23])
def test_radix_differential_random_multilevel_prefixes(seed):
    rng = random.Random(seed)
    rows = []
    uids = []
    for i in range(32):
        uid = f"uid{rng.randrange(4)}"
        prefix = [10 + (i % 3), 20 + rng.randrange(3), 30 + rng.randrange(3)]
        tail = [rng.randrange(1, 9) for _ in range(rng.randrange(2, 12))]
        sequence = prefix + tail
        if rows and i % 7 == 0:
            # Deliberately choose a duplicate or a strict prefix from an
            # earlier row of the same UID.
            prior = [r for r, u in zip(rows, uids) if u == uid]
            if prior:
                sequence = list(rng.choice(prior))
                if len(sequence) > 2 and i % 2:
                    sequence = sequence[:rng.randrange(2, len(sequence))]
        rows.append(sequence)
        uids.append(uid)
    _assert_equivalent(rows, uids)


def test_radix_rejects_bad_input_just_like_original():
    same_keys = ["x_trace_0", "x_trace_0"]
    batch = {"input_ids": [torch.tensor([1, 2]), torch.tensor([1, 3])]}
    with pytest.raises(ValueError, match="duplicate trajectory key"):
        build_trajectory_trees_radix(same_keys, batch)
    with pytest.raises(ValueError, match="must not be empty"):
        build_trajectory_trees_radix(
            ["x_trace_0"], {"input_ids": [torch.empty(0, dtype=torch.long)]}
        )
    with pytest.raises(ValueError, match="row mismatch"):
        build_trajectory_trees_radix(["x_trace_0"], batch)


def test_radix_selector_in_original_ppo_forest_entry_is_opt_in():
    rows = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3], [9, 8, 7]]
    keys = [f"uid_trace_{i}" for i in range(len(rows))]
    batch = _batch(rows)

    baseline = build_tree_execution_plans(
        keys, batch, tree_builder="legacy", require_loss_mask_alignment=True
    )
    radix = build_tree_execution_plans(
        keys, batch, tree_builder="radix", require_loss_mask_alignment=True
    )
    assert len(baseline.trees) == len(radix.trees)
    assert baseline.logical_loss_tokens == radix.logical_loss_tokens
    for old, new in zip(baseline.trees, radix.trees, strict=True):
        assert old.tree == new.tree
        assert old.objective_refs == new.objective_refs
        assert old.segment_plan.validate_events(
            old.segment_plan.dfs_events()
        ) == new.segment_plan.validate_events(
            new.segment_plan.dfs_events()
        )
        for seg_id in old.segment_plan.segments:
            assert torch.equal(
                old.segment_plan.get(seg_id).token_ids,
                new.segment_plan.get(seg_id).token_ids,
            )


def test_radix_selector_rejects_unknown_strategy_before_execution():
    with pytest.raises(ValueError, match="unsupported TPR tree_builder"):
        build_tree_execution_plans([], {"input_ids": []}, tree_builder="typo")
