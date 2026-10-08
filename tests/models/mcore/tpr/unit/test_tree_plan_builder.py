import pytest
import torch

from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _batch():
    tokens = [
        torch.tensor([10, 11, 12, 20, 30]),
        torch.tensor([10, 11, 12, 21, 31]),
        torch.tensor([10, 11, 12, 20, 32]),
        torch.tensor([99, 100, 101]),
    ]
    masks = [
        torch.tensor([1, 1, 1], dtype=torch.bool),
        torch.tensor([1, 1, 1], dtype=torch.bool),
        torch.tensor([1, 1, 1], dtype=torch.bool),
        torch.tensor([0, 1], dtype=torch.bool),
    ]
    return {
        "input_ids": tokens,
        "response_mask": masks,
        "responses": [t[-len(m):] for t, m in zip(tokens, masks, strict=True)],
        "loss_mask": [m.clone() for m in masks],
    }


def _keys():
    return [f"uid_session_{i}" for i in range(4)]


def test_forest_lowering_covers_every_logical_loss_token_exactly_once():
    batch = _batch()
    forest = build_tree_execution_plans(_keys(), batch, require_loss_mask_alignment=True)
    assert len(forest.trees) == 2  # divergent very first tokens
    assert forest.logical_loss_tokens == 10
    assert forest.segment_count >= 4

    seen = set()
    for executable in forest.trees:
        tree = executable.tree
        plan = executable.segment_plan
        assert all(not node.loss_terms for node in plan.segments.values())
        for ref in executable.objective_refs:
            key = (ref.sample_row, ref.response_offset)
            assert key not in seen
            seen.add(key)
            query = plan.get(ref.segment_id)
            absolute_query = query.position_start + ref.query_offset
            target_position = len(batch["input_ids"][ref.sample_row]) - len(
                batch["response_mask"][ref.sample_row]
            ) + ref.response_offset
            assert absolute_query == target_position - 1
            assert ref.target_token_id == int(batch["input_ids"][ref.sample_row][target_position])

    expected = {
        (row, i)
        for row, mask in enumerate(batch["response_mask"])
        for i in torch.nonzero(mask).flatten().tolist()
    }
    assert seen == expected


def test_branch_first_token_uses_parent_last_query_and_retains_duplicate_rows():
    batch = _batch()
    forest = build_tree_execution_plans(_keys(), batch)
    tree = next(plan for plan in forest.trees if 0 in plan.tree.member_rows)
    root = tree.segment_plan.get(tree.segment_plan.root_id)
    assert root.length == 3  # shared [10, 11, 12]

    # The first child token is read from the parent's query at absolute pos=2.
    branch_refs = [ref for ref in tree.objective_refs if ref.response_offset == 1]
    assert {(ref.sample_row, ref.target_token_id) for ref in branch_refs} == {
        (0, 20), (1, 21), (2, 20)
    }
    assert {ref.segment_id for ref in branch_refs} == {root.segment_id}
    assert {ref.query_offset for ref in branch_refs} == {2}

    # Physical branch q->20 is shared; two logical objective refs are retained.
    repeated = [ref for ref in branch_refs if ref.target_token_id == 20]
    assert {ref.sample_row for ref in repeated} == {0, 2}


def test_terminal_at_internal_node_has_own_objectives():
    batch = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4])],
        "response_mask": [torch.tensor([1]), torch.tensor([1, 1])],
        "responses": [torch.tensor([3]), torch.tensor([3, 4])],
    }
    forest = build_tree_execution_plans(["u_s_0", "u_s_1"], batch)
    assert forest.logical_loss_tokens == 3
    assert {(r.sample_row, r.response_offset) for r in forest.trees[0].objective_refs} == {
        (0, 0), (1, 0), (1, 1)
    }


def test_reject_missing_prev_query_and_misaligned_loss_masks():
    batch = _batch()
    batch["response_mask"][0] = torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool)
    batch["responses"][0] = batch["input_ids"][0]
    with pytest.raises(ValueError, match="no preceding query"):
        build_tree_execution_plans(_keys(), batch)

    batch = _batch()
    batch["loss_mask"][1][0] = False
    with pytest.raises(ValueError, match="disagree"):
        build_tree_execution_plans(_keys(), batch, require_loss_mask_alignment=True)


def test_tree_plans_are_topology_only_and_preserve_old_ce_contract():
    batch = _batch()
    forest = build_tree_execution_plans(_keys(), batch)
    assert all(executable.segment_plan.topology_only for executable in forest.trees)
    assert all(executable.segment_plan.total_loss_weight is None for executable in forest.trees)
    # Legacy CE plans remain strict: topology-only is explicit and opt-in.
