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

"""Request object used by the verl MegatronEngine TPR thin entry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .segment_plan import SegmentEvent, SegmentPlan

TPR_REQUEST_KEY = "tpr_forward_backward_request"


@dataclass(frozen=True, slots=True)
class TPRForwardBackwardRequest:
    """A prebuilt tree plan and optional fixed execution order."""

    plan: SegmentPlan
    events: tuple[SegmentEvent, ...] | None = None

    def __init__(self, plan: SegmentPlan, events: Sequence[SegmentEvent] | None = None) -> None:
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        normalized_events = None if events is None else plan.validate_events(events)
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "events", normalized_events)
