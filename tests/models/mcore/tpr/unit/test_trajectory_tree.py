import pytest
import torch

from verl.models.mcore.tpr.trajectory_tree import build_prompt_sibling_trees


def _make_batch(prompts, responses):
    prompts = [torch.tensor(x, dtype=torch.long) for x in prompts]
    responses = [torch.tensor(x, dtype=torch.long) for x in responses]
    response_mask = [torch.ones(len(x), dtype=torch.long) for x in responses]
    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": [torch.cat((p, r)) for p, r in zip(prompts, responses, strict=True)],
        "response_mask": response_mask,
        "loss_mask": [x.clone() for x in response_mask],
        "rollout_log_probs": [torch.zeros(len(x)) for x in responses],
        "rm_scores": [torch.zeros(len(x)) for x in responses],
    }


def test_builds_prompt_root_and_response_leaves_without_reordering():
    batch = _make_batch([[1, 2, 3]] * 3, [[4, 5], [6], [7, 8, 9]])
    keys = ["uid-a_0_0", "uid-a_1_0", "uid-a_2_0"]

    (tree,) = build_prompt_sibling_trees(keys, batch)

    assert tree.uid == "uid-a"
    assert tree.root.row == 0
    assert (tree.root.start, tree.root.end) == (0, 3)
    assert tree.member_rows == (0, 1, 2)
    assert [leaf.key for leaf in tree.leaves] == keys
    assert [(leaf.segment.start, leaf.segment.end) for leaf in tree.leaves] == [(3, 5), (3, 4), (3, 6)]


def test_groups_multiple_prompt_uids_in_first_seen_order():
    batch = _make_batch([[1], [9], [1]], [[2], [8, 7], [3]])
    trees = build_prompt_sibling_trees(["u1_0_0", "u2_0_0", "u1_1_0"], batch)

    assert [tree.uid for tree in trees] == ["u1", "u2"]
    assert trees[0].member_rows == (0, 2)
    assert trees[1].member_rows == (1,)


def test_rejects_non_identical_prompt_within_uid():
    batch = _make_batch([[1, 2], [1, 3]], [[4], [5]])
    with pytest.raises(ValueError, match="exact prompt root"):
        build_prompt_sibling_trees(["u_0_0", "u_1_0"], batch)


def test_rejects_input_ids_that_do_not_equal_prompt_plus_response():
    batch = _make_batch([[1, 2]], [[3, 4]])
    batch["input_ids"][0] = torch.tensor([1, 2, 3, 9])
    with pytest.raises(ValueError, match="response suffix mismatch"):
        build_prompt_sibling_trees(["u_0_0"], batch)


def test_rejects_response_field_alignment_mismatch():
    batch = _make_batch([[1, 2]], [[3, 4]])
    batch["rollout_log_probs"][0] = torch.zeros(1)
    with pytest.raises(ValueError, match="response-field length mismatch"):
        build_prompt_sibling_trees(["u_0_0"], batch)


def test_rejects_response_and_loss_mask_mismatch():
    batch = _make_batch([[1, 2]], [[3, 4]])
    batch["loss_mask"][0][0] = 0
    with pytest.raises(ValueError, match="response_mask == loss_mask"):
        build_prompt_sibling_trees(["u_0_0"], batch)
