# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from test_segment_executor import _FakeDTAModel
from verl.models.mcore.dta import (
    FixedTopologyScheduler,
    PopSegment,
    PushSegment,
    SchedulerState,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)

_PREFIX_LENGTH = 1024
_SUFFIX_1_LENGTH = 512
_SUFFIX_2_LENGTH = 256
_TOTAL_LOSS_WEIGHT = 2814


def _tokens(start, length):
    return torch.arange(start, start + length, dtype=torch.long) % 8


def _internal_terms(tokens, *, weight=1.0):
    return tuple(
        SegmentLossTerm(index, int(tokens[index + 1]), weight=weight)
        for index in range(tokens.numel() - 1)
    )


def _branching_plan(*, reverse_children=False):
    prefix_tokens = _tokens(1, _PREFIX_LENGTH)
    suffix_1_tokens = _tokens(3, _SUFFIX_1_LENGTH)
    suffix_2_tokens = _tokens(5, _SUFFIX_2_LENGTH)
    prefix_terms = _internal_terms(prefix_tokens, weight=2.0) + (
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(suffix_1_tokens[0])),
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(suffix_2_tokens[0])),
    )
    prefix = SegmentSpec(0, None, prefix_tokens, 0, 0, prefix_terms)
    suffix_1 = SegmentSpec(
        1,
        0,
        suffix_1_tokens,
        _PREFIX_LENGTH,
        _PREFIX_LENGTH,
        _internal_terms(suffix_1_tokens),
    )
    suffix_2 = SegmentSpec(
        2,
        0,
        suffix_2_tokens,
        _PREFIX_LENGTH,
        _PREFIX_LENGTH,
        _internal_terms(suffix_2_tokens),
    )
    children = [suffix_2, suffix_1] if reverse_children else [suffix_1, suffix_2]
    return SegmentPlan([prefix, *children], root_id=0)


def _run(plan, *, events=None, model=None):
    model = _FakeDTAModel() if model is None else model
    executor = SegmentExecutor(model, plan, expected_layer_numbers=(1, 2))
    scheduler = FixedTopologyScheduler(plan, executor, events=events)
    return model, executor, scheduler, scheduler.run()


def test_fixed_scheduler_executes_branching_dfs_and_aggregates_results():
    plan = _branching_plan()
    model, executor, scheduler, result = _run(plan)

    assert result.event_trace == (
        PushSegment(0),
        PushSegment(1),
        PopSegment(1),
        PushSegment(2),
        PopSegment(2),
        PopSegment(0),
    )
    assert [item.segment_id for item in result.forward_results] == [0, 1, 2]
    assert [item.segment_id for item in result.backward_results] == [1, 2, 0]
    assert result.pushed_segment_count == result.popped_segment_count == 3
    assert result.peak_path_tokens == _PREFIX_LENGTH + _SUFFIX_1_LENGTH
    assert plan.total_loss_weight == _TOTAL_LOSS_WEIGHT
    torch.testing.assert_close(result.normalized_loss, result.loss_sum / _TOTAL_LOSS_WEIGHT)
    assert torch.isfinite(result.normalized_loss)
    assert model.scale.grad is not None and torch.isfinite(model.scale.grad)
    executor.kv_stack.assert_empty()
    assert scheduler.state is SchedulerState.COMPLETED


def test_reversing_sibling_order_preserves_loss_and_parameter_gradient():
    torch.manual_seed(2026)
    plan_a = _branching_plan()
    model_a, _, _, result_a = _run(plan_a)
    grad_a = model_a.scale.grad.detach().clone()

    torch.manual_seed(2026)
    plan_b = _branching_plan(reverse_children=True)
    model_b, _, _, result_b = _run(plan_b)
    grad_b = model_b.scale.grad.detach().clone()

    torch.testing.assert_close(result_a.normalized_loss, result_b.normalized_loss)
    torch.testing.assert_close(grad_a, grad_b, atol=1e-5, rtol=1e-5)


def test_explicit_valid_event_order_is_honored():
    plan = _branching_plan()
    events = (
        PushSegment(0),
        PushSegment(2),
        PopSegment(2),
        PushSegment(1),
        PopSegment(1),
        PopSegment(0),
    )
    _, _, _, result = _run(plan, events=events)
    assert result.event_trace == events


def test_scheduler_is_one_shot():
    plan = _branching_plan()
    _, _, scheduler, _ = _run(plan)
    with pytest.raises(RuntimeError, match="READY"):
        scheduler.run()


def test_execution_failure_marks_scheduler_failed():
    plan = _branching_plan()
    model = _FakeDTAModel(fail=True)
    executor = SegmentExecutor(model, plan, expected_layer_numbers=(1, 2))
    scheduler = FixedTopologyScheduler(plan, executor)

    with pytest.raises(RuntimeError, match="injected"):
        scheduler.run()
    assert scheduler.state is SchedulerState.FAILED
    assert executor.failed


