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

from verl.models.mcore.dta import (
    PopSegment,
    PushSegment,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)


def _branching_plan() -> SegmentPlan:
    prefix = SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.arange(1024, dtype=torch.long),
        position_start=0,
        prefix_length=0,
        loss_terms=(
            SegmentLossTerm(query_offset=0, target_token_id=1),
            SegmentLossTerm(query_offset=1023, target_token_id=2000, sample_id=0),
            SegmentLossTerm(query_offset=1023, target_token_id=3000, sample_id=1),
        ),
    )
    child_a = SegmentSpec(
        segment_id=1,
        parent_id=0,
        token_ids=torch.arange(2000, 2512, dtype=torch.long),
        position_start=1024,
        prefix_length=1024,
        loss_terms=(SegmentLossTerm(query_offset=0, target_token_id=2001),),
    )
    child_b = SegmentSpec(
        segment_id=2,
        parent_id=0,
        token_ids=torch.arange(3000, 3256, dtype=torch.long),
        position_start=1024,
        prefix_length=1024,
        loss_terms=(SegmentLossTerm(query_offset=0, target_token_id=3001),),
    )
    return SegmentPlan([prefix, child_a, child_b], root_id=0)


def test_branching_plan_preserves_topology_positions_and_loss_ownership():
    plan = _branching_plan()

    assert plan.get(0).length == 1024
    assert plan.get(1).length == 512
    assert plan.get(2).length == 256
    assert plan.get(1).position_start == plan.get(0).position_end == 1024
    assert [segment.segment_id for segment in plan.children_of(0)] == [1, 2]
    assert [segment.segment_id for segment in plan.path_to(2)] == [0, 2]
    assert plan.depth_of(0) == 0
    assert plan.depth_of(2) == 1
    assert [term.target_token_id for term in plan.get(0).loss_terms if term.query_offset == 1023] == [2000, 3000]
    assert plan.total_loss_weight == 5.0


def test_token_ids_are_cpu_long_and_isolated_from_caller_mutation():
    token_ids = torch.tensor([1, 2], dtype=torch.long)
    segment = SegmentSpec(0, None, token_ids, position_start=0, prefix_length=0, loss_terms=(SegmentLossTerm(0, 2),))
    token_ids[0] = 99

    assert segment.token_ids.tolist() == [1, 2]
    with pytest.raises(ValueError, match="one-dimensional CPU torch.long"):
        SegmentSpec(0, None, torch.tensor([1.0]), position_start=0, prefix_length=0)


def test_dfs_event_protocol_matches_push_visit_pop_order():
    plan = _branching_plan()
    events = tuple(plan.dfs_events())

    assert events == (
        PushSegment(0),
        PushSegment(1),
        PopSegment(1),
        PushSegment(2),
        PopSegment(2),
        PopSegment(0),
    )
    assert plan.validate_events(events) == events


@pytest.mark.parametrize(
    "events",
    [
        (PushSegment(0), PushSegment(1), PopSegment(0)),
        (PushSegment(0), PushSegment(2), PopSegment(2), PopSegment(0)),
        (PushSegment(0), PushSegment(1), PopSegment(1), PopSegment(0)),
    ],
)
def test_invalid_or_incomplete_event_stream_is_rejected(events):
    with pytest.raises(ValueError):
        _branching_plan().validate_events(events)


def test_plan_rejects_bad_parent_position_and_missing_parent():
    root = SegmentSpec(0, None, torch.tensor([1, 2]), position_start=0, prefix_length=0, loss_terms=(SegmentLossTerm(0, 2),))
    wrong_position = SegmentSpec(1, 0, torch.tensor([3]), position_start=1, prefix_length=1)
    with pytest.raises(ValueError, match="must start at parent"):
        SegmentPlan([root, wrong_position], root_id=0)

    missing_parent = SegmentSpec(1, 9, torch.tensor([3]), position_start=2, prefix_length=2)
    with pytest.raises(ValueError, match="missing parent"):
        SegmentPlan([root, missing_parent], root_id=0)


def test_plan_rejects_invalid_loss_and_zero_total_weight():
    with pytest.raises(ValueError, match="outside length"):
        SegmentSpec(
            0,
            None,
            torch.tensor([1]),
            position_start=0,
            prefix_length=0,
            loss_terms=(SegmentLossTerm(1, 2),),
        )

    root = SegmentSpec(0, None, torch.tensor([1]), position_start=0, prefix_length=0)
    with pytest.raises(ValueError, match="total_loss_weight"):
        SegmentPlan([root], root_id=0)
