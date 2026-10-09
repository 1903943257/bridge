# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""DTA planner trie vs VERL TPR executable trie semantic contracts."""

import torch

from verl.models.mcore.tpr._vendor.areal_dta.token_trie import TokenTrie
from verl.models.mcore.tpr.dp_placement import _tree_token_cost
from verl.models.mcore.tpr.trajectory_tree import build_trajectory_trees


def test_duplicate_and_terminal_prefix_rows_remain_logically_distinct():
    # AReaL leafization stores duplicate/contained trajectories as attachments
    # on a longer leaf.  TPR's execution tree instead keeps each row as a
    # distinct terminal_rows entry on the corresponding executable node.
    sequences = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 5]]
    tokens = [torch.tensor(row, dtype=torch.long) for row in sequences]

    areal = TokenTrie(tokens)
    attachments = sorted(
        attachment["_sequence_batch_id"]
        for leaf_attachments in areal.attach_lists
        for attachment, _ in leaf_attachments
    )
    assert len(areal.inputs) == 2  # only two maximal distinct leaves
    assert attachments == list(range(len(sequences)))

    keys = [f"uid0_rollout_{i}" for i in range(len(sequences))]
    trees = build_trajectory_trees(keys, {"input_ids": tokens})
    assert len(trees) == 1
    tree = trees[0]
    assert set(tree.member_rows) == set(range(len(sequences)))
    terminals = sorted(
        row for node in tree.nodes.values() for row in node.terminal_rows
    )
    assert terminals == list(range(len(sequences)))

    # An internal node terminates row 2, while its descendants still continue.
    assert any(node.terminal_rows == (2,) and node.children for node in tree.nodes.values())
    # Both duplicate rows terminate on one full-length physical node.
    assert any(set(node.terminal_rows) == {0, 1} for node in tree.nodes.values())
    assert sum(node.segment.length for node in tree.nodes.values()) == _tree_token_cost(sequences)


def test_uid_boundary_is_tpr_policy_not_areal_global_prefix_rule():
    sequences = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4], [1, 2, 3]]
    tokens = [torch.tensor(row, dtype=torch.long) for row in sequences]
    keys = ["job0_rollout_0", "job0_rollout_1",
            "job1_rollout_2", "job1_rollout_3"]
    trees = build_trajectory_trees(keys, {"input_ids": tokens})

    # TPR deliberately does not share a prefix across UID boundaries today.
    assert len(trees) == 2
    local_executed_tree_tokens = sum(
        node.segment.length
        for tree in trees
        for node in tree.nodes.values()
    )
    assert local_executed_tree_tokens == 9

    # AReaL placement can reason globally across both UIDs because it only
    # assigns complete trajectories; that does NOT mean local TPR execution
    # automatically shares the cross-UID prefix after dispatch.
    assert _tree_token_cost(sequences) == 5
    assert local_executed_tree_tokens > _tree_token_cost(sequences)
