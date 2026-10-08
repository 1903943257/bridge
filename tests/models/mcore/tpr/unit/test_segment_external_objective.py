"""CPU-level checks for the CE-preserving SegmentExecutor objective hook."""

import pytest
import torch

from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec


def _segment():
    return SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        position_start=0,
        prefix_length=0,
    )


def _executor_for_hook(segment, callback, count):
    # These tests deliberately exercise only the loss seam without constructing
    # Megatron/NPU models; the public constructor is validated separately.
    executor = object.__new__(SegmentExecutor)
    executor.segment_loss_fn = callback
    executor.segment_loss_term_counts = {segment.segment_id: count}
    executor.segment_loss_metrics = []
    return executor


def test_injected_objective_is_used_without_ce_denominator():
    segment = _segment()
    logits = torch.randn(1, 3, 5, requires_grad=True)

    def objective(spec, tensor):
        assert spec is segment
        return tensor[0, 1, 2].square(), {"source": "native-ppo"}

    executor = _executor_for_hook(segment, objective, count=1)
    loss_sum, normalized = executor._compute_loss(segment, logits)
    assert loss_sum is normalized
    assert executor._owned_loss_count(segment) == 1
    assert executor.segment_loss_metrics == [(0, {"source": "native-ppo"})]

    loss_sum.backward()
    assert logits.grad is not None
    assert logits.grad[0, 1, 2] != 0
    assert torch.count_nonzero(logits.grad).item() == 1


def test_no_objective_ref_uses_connected_zero_not_ce():
    segment = _segment()
    logits = torch.randn(1, 3, 5, requires_grad=True)

    def invalid_callback(*args):
        raise AssertionError("no loss callback should run for an empty segment")

    executor = _executor_for_hook(segment, invalid_callback, count=0)
    _, loss = executor._compute_loss(segment, logits)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
    assert executor.segment_loss_metrics == []


def test_external_objective_must_return_differentiable_scalar():
    segment = _segment()
    logits = torch.randn(1, 3, 5, requires_grad=True)
    executor = _executor_for_hook(segment, lambda *_: (torch.tensor(0.0), {}), count=1)
    with pytest.raises(TypeError, match="differentiable scalar"):
        executor._compute_loss(segment, logits)


def test_ce_free_plan_explicit_opt_in():
    segment = _segment()
    plan = SegmentPlan([segment], root_id=0, topology_only=True)
    assert plan.total_loss_weight is None
    assert plan.topology_only
    with pytest.raises(ValueError, match="total_loss_weight"):
        SegmentPlan([segment], root_id=0)
