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

from .segment_executor import LeafVisitResult, SegmentBackwardResult, SegmentExecutor, SegmentForwardResult
from .segment_plan import SegmentEvent, SegmentEventKind, SegmentPlan


class SchedulerState(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PhysicalExecutionKind(str, Enum):
    PUSH = "push"
    POP = "pop"
    VISIT_LEAF = "visit_leaf"


@dataclass(frozen=True, slots=True)
class PhysicalExecution:
    kind: PhysicalExecutionKind
    segment_id: int


@dataclass(frozen=True, slots=True)
class TreeScheduleResult:
    event_trace: tuple[SegmentEvent, ...]
    execution_trace: tuple[PhysicalExecution, ...]
    forward_results: tuple[SegmentForwardResult, ...]
    backward_results: tuple[SegmentBackwardResult, ...]
    direct_leaf_results: tuple[LeafVisitResult, ...]
    loss_sum: Tensor
    normalized_loss: Tensor
    peak_path_tokens: int

    @property
    def pushed_segment_count(self) -> int:
        return len(self.forward_results)

    @property
    def popped_segment_count(self) -> int:
        return len(self.backward_results)

    @property
    def direct_leaf_count(self) -> int:
        return len(self.direct_leaf_results)

    @property
    def executed_segment_count(self) -> int:
        return self.pushed_segment_count + self.direct_leaf_count


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
        direct_leaf_results: list[LeafVisitResult] = []
        all_loss_results: list[SegmentBackwardResult] = []
        execution_trace: list[PhysicalExecution] = []
        peak_path_tokens = 0
        try:
            event_index = 0
            while event_index < len(self.events):
                event = self.events[event_index]
                if self._is_direct_leaf_pair(event_index):
                    result = self.executor.visit_leaf(event.segment_id)
                    direct_leaf_results.append(result)
                    all_loss_results.append(result.backward)
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.VISIT_LEAF, event.segment_id))
                    peak_path_tokens = max(
                        peak_path_tokens,
                        result.forward.prefix_length + result.forward.suffix_length,
                    )
                    event_index += 2
                    continue
                if event.kind is SegmentEventKind.PUSH:
                    forward_results.append(self.executor.push(event.segment_id))
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.PUSH, event.segment_id))
                    peak_path_tokens = max(peak_path_tokens, self.executor.kv_stack.prefix_length)
                else:
                    result = self.executor.pop(event.segment_id)
                    backward_results.append(result)
                    all_loss_results.append(result)
                    execution_trace.append(PhysicalExecution(PhysicalExecutionKind.POP, event.segment_id))
                event_index += 1
            self.executor.kv_stack.assert_empty()
            if not all_loss_results:
                raise RuntimeError("schedule produced no backward results")

            loss_sum = all_loss_results[0].loss_sum
            normalized_loss = all_loss_results[0].normalized_loss
            for result in all_loss_results[1:]:
                loss_sum = loss_sum + result.loss_sum
                normalized_loss = normalized_loss + result.normalized_loss
            schedule_result = TreeScheduleResult(
                event_trace=self.events,
                execution_trace=tuple(execution_trace),
                forward_results=tuple(forward_results),
                backward_results=tuple(backward_results),
                direct_leaf_results=tuple(direct_leaf_results),
                loss_sum=loss_sum,
                normalized_loss=normalized_loss,
                peak_path_tokens=peak_path_tokens,
            )
        except Exception:
            self.state = SchedulerState.FAILED
            raise
        self.state = SchedulerState.COMPLETED
        return schedule_result

    def _is_direct_leaf_pair(self, event_index: int) -> bool:
        if event_index + 1 >= len(self.events):
            return False
        push_event = self.events[event_index]
        pop_event = self.events[event_index + 1]
        return (
            push_event.kind is SegmentEventKind.PUSH
            and pop_event.kind is SegmentEventKind.POP
            and push_event.segment_id == pop_event.segment_id
            and not self.plan.children_of(push_event.segment_id)
        )
