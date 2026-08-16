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
    DTA_REQUEST_KEY,
    DTAForwardBackwardRequest,
    PopSegment,
    PushSegment,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)


def _plan():
    root = SegmentSpec(
        0,
        None,
        torch.tensor([1, 2]),
        0,
        0,
        (SegmentLossTerm(0, 2),),
    )
    child = SegmentSpec(1, 0, torch.tensor([3]), 2, 2)
    return SegmentPlan([root, child], root_id=0)


def test_request_key_is_explicit_and_request_uses_default_dfs():
    request = DTAForwardBackwardRequest(_plan())
    assert DTA_REQUEST_KEY == "dta_forward_backward_request"
    assert request.events is None


def test_request_validates_and_freezes_explicit_events():
    plan = _plan()
    events = [PushSegment(0), PushSegment(1), PopSegment(1), PopSegment(0)]
    request = DTAForwardBackwardRequest(plan, events)
    events.clear()

    assert request.events == (PushSegment(0), PushSegment(1), PopSegment(1), PopSegment(0))


def test_request_rejects_incomplete_events_and_wrong_plan_type():
    plan = _plan()
    with pytest.raises(ValueError, match="incomplete"):
        DTAForwardBackwardRequest(plan, [PushSegment(0)])
    with pytest.raises(TypeError, match="SegmentPlan"):
        DTAForwardBackwardRequest(object())


