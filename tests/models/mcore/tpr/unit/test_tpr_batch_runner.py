from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.tpr.tpr_batch_runner import TPRBatchRunner
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _forest():
    return build_tree_execution_plans(
        ["u_s_0", "u_s_1", "v_s_0"],
        {
            "input_ids": [
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 2, 4]),
                torch.tensor([7, 8, 9]),
            ],
            "response_mask": [
                torch.tensor([1], dtype=torch.bool),
                torch.tensor([1], dtype=torch.bool),
                torch.tensor([1], dtype=torch.bool),
            ],
        },
    )


def test_forest_runs_every_tree_with_one_finalization():
    forest = _forest()
    calls = []

    @contextmanager
    def no_sync():
        calls.append("enter_no_sync")
        yield
        calls.append("exit_no_sync")

    def run_one_tree(plan):
        calls.append(("tree", plan.tree.key))
        return SimpleNamespace(normalized_loss=torch.tensor(0.25 * plan.logical_loss_tokens))

    def finalize():
        calls.append("finalize")

    runner = TPRBatchRunner(
        run_tree=run_one_tree, finalize_gradients=finalize, no_sync_context=no_sync
    )
    result = runner.run(forest)
    assert result.tree_count == len(forest.trees) == 2
    assert result.segment_count == forest.segment_count
    assert result.logical_loss_tokens == 3
    assert result.normalized_loss.item() == pytest.approx(0.75)
    assert calls == [
        "enter_no_sync",
        ("tree", forest.trees[0].tree.key),
        ("tree", forest.trees[1].tree.key),
        "exit_no_sync",
        "finalize",
    ]
    with pytest.raises(RuntimeError, match="single-use"):
        runner.run(forest)


def test_forest_failure_never_finalizes_gradients():
    forest = _forest()
    calls = []
    def fail_on_tree(plan):
        calls.append(plan.tree.key)
        if len(calls) == 2:
            raise RuntimeError("tree failed")
        return SimpleNamespace(normalized_loss=torch.tensor(1.0))

    runner = TPRBatchRunner(
        run_tree=fail_on_tree,
        finalize_gradients=lambda: calls.append("finalize"),
    )
    with pytest.raises(RuntimeError, match="tree failed"):
        runner.run(forest)
    assert "finalize" not in calls


def test_invalid_tree_loss_does_not_finalize():
    forest = _forest()
    count = []
    runner = TPRBatchRunner(
        run_tree=lambda plan: SimpleNamespace(normalized_loss=1.0),
        finalize_gradients=lambda: count.append("finalize"),
    )
    with pytest.raises(TypeError, match="scalar Tensor"):
        runner.run(forest)
    assert count == []
