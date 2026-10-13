"""CPU-only Qwen3 ChatML Turn-boundary forest/PP0 ownership gates."""
import torch
import pytest

from verl.models.mcore.tpr.trajectory_tree import build_trajectory_trees
from verl.models.mcore.tpr.trajectory_tree_turn import (
    QWEN3_IM_END_TOKEN_ID,
    build_trajectory_trees_turn,
)
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


EOT = QWEN3_IM_END_TOKEN_ID
KEYS = ("same_tpr_0", "same_tpr_1", "same_tpr_2")


def batch(inputs, prompt_lengths=None):
    value = {"input_ids": [torch.tensor(tokens, dtype=torch.long) for tokens in inputs]}
    if prompt_lengths is not None:
        value["prompts"] = [
            torch.tensor(tokens[:size], dtype=torch.long)
            for tokens, size in zip(inputs, prompt_lengths)
        ]
        value["responses"] = [
            torch.tensor(tokens[size:], dtype=torch.long)
            for tokens, size in zip(inputs, prompt_lengths)
        ]
        value["response_mask"] = [
            torch.ones(len(tokens)-size, dtype=torch.bool)
            for tokens, size in zip(inputs, prompt_lengths)
        ]
    return value


def paths_reconstruct(trees, data):
    covered = {}
    for tree in trees:
        def visit(node_id, path):
            node = tree.get(node_id)
            ref = node.segment
            segment = data["input_ids"][ref.row][ref.start:ref.end].tolist()
            merged = path + segment
            for row in node.terminal_rows:
                assert row not in covered
                assert merged == data["input_ids"][row].tolist()
                covered[row] = True
            for child_id in node.children:
                visit(child_id, merged)
        visit(tree.root_id, [])
    assert set(covered) == set(range(len(data["input_ids"])))


def counted(trees):
    return sum(len(t.nodes) for t in trees)


def global_pp0_refs(forest):
    refs = []
    for executable in forest.trees:
        plan = executable.segment_plan
        for ref in executable.objective_refs:
            segment = plan.get(ref.segment_id)
            refs.append((
                ref.sample_row, ref.response_offset, ref.target_token_id,
                segment.position_start + ref.query_offset,
            ))
    return sorted(refs)


def test_short_partial_shared_message_is_not_pushed():
    # Share prompt, then 9 complete tokens of one *unfinished* assistant turn;
    # branch inside that turn. Exact trie caches 9 tokens, Turn trie must not.
    prompt = [11, 12, 13, 14]
    common_partial = [21, 22, 23, 24, 25, 26, 27, 28, 29]
    data = batch([
        prompt + common_partial + [31, EOT],
        prompt + common_partial + [32, EOT],
        prompt + [33, EOT],
    ], prompt_lengths=[len(prompt)]*3)
    exact = build_trajectory_trees(KEYS, data)
    turned = build_trajectory_trees_turn(KEYS, data, min_segment_tokens=2)
    paths_reconstruct(turned, data)
    assert counted(exact) > counted(turned)
    assert len(turned) == 1
    tree = turned[0]
    root = tree.get(tree.root_id)
    assert root.segment.end == len(prompt)
    assert len(root.children) == 3
    assert all(not tree.get(k).children for k in root.children)


def test_complete_turns_may_share_and_short_complete_turn_is_coalesced():
    prompt = [11, 12, 13, 14]
    t1 = [7]*140 + [EOT]
    short_t2 = [8, 9, EOT]
    short_t3 = [10, 11, EOT]
    out = [
        prompt + t1 + short_t2 + short_t3 + [12, EOT],
        prompt + t1 + short_t2 + short_t3 + [13, EOT],
        prompt + t1 + [14, EOT],
    ]
    data = batch(out, prompt_lengths=[4]*3)
    trees = build_trajectory_trees_turn(KEYS, data, min_segment_tokens=128)
    paths_reconstruct(trees, data)
    assert len(trees) == 1
    tree = trees[0]
    root = tree.get(tree.root_id)
    # Common first assistant turn 141 tokens may be shared.
    assert root.segment.end == len(prompt) + len(t1)
    # The two tiny *complete* turns don't each get their own Push/Pop;
    # next terminal tails are merged, so no intermediate 3- or 6-token node.
    assert not any(node.children and node.segment.length in (3, 6)
                   for node in tree.nodes.values())


def test_pp0_objective_ownership_and_masks_identical():
    prompt = [11, 12, 13, 14]
    inputs = [
        prompt + [7]*140 + [EOT, 21,22,23,31,EOT],
        prompt + [7]*140 + [EOT, 21,22,23,32,EOT],
        prompt + [7]*140 + [EOT, 41,42,43,EOT],
    ]
    data = batch(inputs, prompt_lengths=[len(prompt)]*3)
    exact = build_tree_execution_plans(KEYS, data)
    turned = build_tree_execution_plans(
        KEYS, data, tree_policy="qwen_turn",
        turn_min_segment_tokens=128,
    )
    paths_reconstruct(tuple(t.tree for t in turned.trees), data)
    assert turned.logical_loss_tokens == exact.logical_loss_tokens
    assert global_pp0_refs(turned) == global_pp0_refs(exact)
    assert turned.segment_count <= exact.segment_count


def test_no_full_shared_turn_leaves_independent_trees():
    # Same initial 5 tokens, but divergent within the very first message.
    data = batch([[1,2,3,4,5,6,EOT], [1,2,3,4,5,7,EOT]])
    keys = ("one_tpr_0", "one_tpr_1")
    trees = build_trajectory_trees_turn(keys, data)
    assert len(trees) == 2
    assert all(len(t.nodes) == 1 for t in trees)
    paths_reconstruct(trees, data)


def test_terminal_row_and_duplicates_remain_exact():
    shared = [1, 2, 3, EOT]
    data = batch([shared, shared + [4,5,EOT], shared])
    trees = build_trajectory_trees_turn(KEYS, data, min_segment_tokens=2)
    paths_reconstruct(trees, data)
    assert len(trees) == 1
    root = trees[0].get(trees[0].root_id)
    assert root.terminal_rows == (0,2)


def test_separate_uids_and_invalid_policy_rejected():
    data = batch([[1,2,EOT,3,EOT],[1,2,EOT,4,EOT]])
    trees = build_trajectory_trees_turn(("a_tpr_0","b_tpr_1"), data)
    assert len(trees) == 2
    with pytest.raises(ValueError, match="unknown TPR tree_policy"):
        build_tree_execution_plans(("a_tpr_0","b_tpr_1"), data,
                                   tree_policy="inferred_from_random_token")


def test_false_boundary_and_wrong_prompt_are_rejected_or_not_reused():
    data = batch([[1,2,3,4],[1,2,3,5]])
    trees = build_trajectory_trees_turn(("a_tpr_0","a_tpr_1"), data)
    assert len(trees) == 2
    paths_reconstruct(trees, data)
    bad = batch([[1,2,3,4]], prompt_lengths=[2])
    bad["prompts"][0][0] = 999
    with pytest.raises(ValueError, match="not an exact input prefix"):
        build_trajectory_trees_turn(["x_tpr_0"], bad)
