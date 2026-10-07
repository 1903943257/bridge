import pytest
import torch

from verl.models.mcore.tpr.trajectory_tree import (
    build_prompt_sibling_trees,
    build_trajectory_trees,
)


def _make_batch(input_ids):
    return {
        "input_ids": [torch.tensor(tokens, dtype=torch.long) for tokens in input_ids],
    }


def _tokens(tree, node_id, batch):
    node = tree.get(node_id)
    ref = node.segment
    return batch["input_ids"][ref.row][ref.start : ref.end].tolist()


def test_builds_multilevel_compressed_trie_from_complete_input_ids():
    batch = _make_batch(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 7],
        ]
    )
    keys = ["uid-a_0_0", "uid-a_1_0", "uid-a_2_0"]

    (tree,) = build_trajectory_trees(keys, batch)

    root = tree.get(tree.root_id)
    assert tree.uid == "uid-a"
    assert tree.member_rows == (0, 1, 2)
    assert (root.segment.start, root.segment.end) == (0, 3)
    assert _tokens(tree, root.node_id, batch) == [1, 2, 3]
    assert root.member_rows == (0, 1, 2)
    assert root.terminal_rows == ()
    assert len(root.children) == 2

    shared_4 = tree.get(root.children[0])
    direct_7 = tree.get(root.children[1])
    assert _tokens(tree, shared_4.node_id, batch) == [4]
    assert shared_4.member_rows == (0, 1)
    assert shared_4.terminal_rows == ()
    assert len(shared_4.children) == 2
    assert [_tokens(tree, child_id, batch) for child_id in shared_4.children] == [[5], [6]]

    assert _tokens(tree, direct_7.node_id, batch) == [7]
    assert direct_7.member_rows == (2,)
    assert direct_7.terminal_rows == (2,)
    assert direct_7.children == ()


def test_compression_is_not_limited_by_prompt_boundary():
    prompts = [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])]
    responses = [torch.tensor([4, 5, 6]), torch.tensor([4, 5, 7])]
    batch = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": [
            torch.cat((prompts[0], responses[0])),
            torch.cat((prompts[1], responses[1])),
        ],
    }

    (tree,) = build_trajectory_trees(["u_0_0", "u_1_0"], batch)
    root = tree.get(tree.root_id)

    # Prompt length is 3, but the exact shared prefix extends two generated
    # tokens into the response. Topology must therefore compress through it.
    assert (root.segment.start, root.segment.end) == (0, 5)
    assert _tokens(tree, root.node_id, batch) == [1, 2, 3, 4, 5]


def test_terminal_rows_preserve_sequence_that_ends_inside_another_path():
    batch = _make_batch(
        [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
        ]
    )

    (tree,) = build_trajectory_trees(["u_0_0", "u_1_0"], batch)
    root = tree.get(tree.root_id)

    assert _tokens(tree, root.node_id, batch) == [1, 2, 3]
    assert root.member_rows == (0, 1)
    assert root.terminal_rows == (0,)
    assert len(root.children) == 1

    child = tree.get(root.children[0])
    assert _tokens(tree, child.node_id, batch) == [4, 5]
    assert child.member_rows == (1,)
    assert child.terminal_rows == (1,)


def test_uid_is_only_candidate_partition_not_prefix_definition():
    batch = _make_batch(
        [
            [1, 2, 3],
            [9, 8],
            [1, 2, 4],
        ]
    )

    trees = build_trajectory_trees(["u1_0_0", "u1_1_0", "u1_2_0"], batch)

    # Same uid does not force unrelated rows into one fake tree. The dummy root
    # is builder-only, so distinct first-token branches become executable trees.
    assert len(trees) == 2
    assert [tree.uid for tree in trees] == ["u1", "u1"]
    assert trees[0].member_rows == (0, 2)
    assert trees[1].member_rows == (1,)
    assert trees[0].get(trees[0].root_id).segment.start == 0
    assert trees[1].get(trees[1].root_id).segment.start == 0


def test_multiple_uids_remain_separate_candidate_groups_in_first_seen_order():
    batch = _make_batch(
        [
            [1, 2, 3],
            [1, 2, 9],
            [1, 2, 4],
        ]
    )

    trees = build_trajectory_trees(["u1_0_0", "u2_0_0", "u1_1_0"], batch)

    assert [tree.uid for tree in trees] == ["u1", "u2"]
    assert trees[0].member_rows == (0, 2)
    assert trees[1].member_rows == (1,)


def test_exact_duplicate_trajectories_keep_distinct_logical_terminal_rows():
    batch = _make_batch(
        [
            [1, 2, 3],
            [1, 2, 3],
        ]
    )

    (tree,) = build_trajectory_trees(["u_0_0", "u_1_0"], batch)
    root = tree.get(tree.root_id)

    assert _tokens(tree, root.node_id, batch) == [1, 2, 3]
    assert root.member_rows == (0, 1)
    assert root.terminal_rows == (0, 1)
    assert root.children == ()


def test_old_builder_name_is_a_compatibility_alias_with_new_semantics():
    batch = _make_batch([[1, 2, 3, 4], [1, 2, 3, 5]])
    keys = ["u_0_0", "u_1_0"]

    new_trees = build_trajectory_trees(keys, batch)
    compatibility_trees = build_prompt_sibling_trees(keys, batch)

    assert compatibility_trees == new_trees


def test_rejects_duplicate_keys_and_empty_sequences():
    batch = _make_batch([[1], [2]])
    with pytest.raises(ValueError, match="duplicate trajectory key"):
        build_trajectory_trees(["u_0_0", "u_0_0"], batch)

    empty_batch = {"input_ids": [torch.tensor([], dtype=torch.long)]}
    with pytest.raises(ValueError, match="must not be empty"):
        build_trajectory_trees(["u_0_0"], empty_batch)
