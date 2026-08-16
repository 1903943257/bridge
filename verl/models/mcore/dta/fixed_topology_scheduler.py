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

"""One-shot scheduler for a precomputed DTA segment topology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from torch import Tensor

from .segment_executor import SegmentBackwardResult, SegmentExecutor, SegmentForwardResult
from .segment_plan import SegmentEvent, SegmentEventKind, SegmentPlan


class SchedulerState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TreeScheduleResult:
    event_trace: tuple[SegmentEvent, ...]
    forward_results: tuple[SegmentForwardResult, ...]
    backward_results: tuple[SegmentBackwardResult, ...]
    loss_sum: Tensor
    normalized_loss: Tensor
    peak_path_tokens: int

    @property
    def pushed_segment_count(self) -> int:
        return len(self.forward_results)

    @property
    def popped_segment_count(self) -> int:
        return len(self.backward_results)


class FixedTopologyScheduler:
    """Execute one validated, immutable DFS Push/Pop event stream."""

    def __init__(
        self,
        plan: SegmentPlan,
        executor: SegmentExecutor,
        *,
        events: Sequence[SegmentEvent] | None = None,
    ) -> None:
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        if not isinstance(executor, SegmentExecutor):
            raise TypeError(f"executor must be SegmentExecutor, got {type(executor).__name__}")
        if executor.plan is not plan:
            raise ValueError("executor and scheduler must share the same SegmentPlan instance")
        if executor.failed:
            raise ValueError("executor is already failed")
        if len(executor.kv_stack):
            raise ValueError("executor KV stack must be empty before scheduling")

        candidate_events = tuple(plan.dfs_events()) if events is None else tuple(events)
        self.events = plan.validate_events(candidate_events)
        self.plan = plan
        self.executor = executor
        self.state = SchedulerState.READY

    def run(self) -> TreeScheduleResult:
        if self.state is not SchedulerState.READY:
            raise RuntimeError(f"scheduler can only run from READY, current state is {self.state.value}")
        self.state = SchedulerState.RUNNING
        forward_results: list[SegmentForwardResult] = []
        backward_results: list[SegmentBackwardResult] = []
        peak_path_tokens = 0
        try:
            for event in self.events:
                if event.kind is SegmentEventKind.PUSH:
                    forward_results.append(self.executor.push(event.segment_id))
                    peak_path_tokens = max(peak_path_tokens, self.executor.kv_stack.prefix_length)
                else:
                    backward_results.append(self.executor.pop(event.segment_id))
            self.executor.kv_stack.assert_empty()
            if not backward_results:
                raise RuntimeError("schedule produced no backward results")

            loss_sum = backward_results[0].loss_sum
            normalized_loss = backward_results[0].normalized_loss
            for result in backward_results[1:]:
                loss_sum = loss_sum + result.loss_sum
                normalized_loss = normalized_loss + result.normalized_loss
            schedule_result = TreeScheduleResult(
                event_trace=self.events,
                forward_results=tuple(forward_results),
                backward_results=tuple(backward_results),
                loss_sum=loss_sum,
                normalized_loss=normalized_loss,
                peak_path_tokens=peak_path_tokens,
            )
        except Exception:
            self.state = SchedulerState.FAILED
            raise
        self.state = SchedulerState.COMPLETED
        return schedule_result


