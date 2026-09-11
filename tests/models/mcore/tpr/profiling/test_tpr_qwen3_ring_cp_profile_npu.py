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

"""Opt-in Qwen3-1.7B Reference CP versus TPR Ring CP=2 profile.

The Reference executes every complete ``P + S`` trajectory independently.
TPR executes one shared Prefix followed by all Suffixes.  Model construction,
checkpoint loading, plan construction, warmup, and gradient clearing are not
timed.  Each measured sample includes forward, loss, backward, and one final
CP parameter-gradient synchronization.

Run all cases from the verl repository root with two visible NPUs::

    TPR_RUN_QWEN_RING_CP_PROFILE=1 \
    TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B \
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29551 \
        -m pytest -s -v -x \
        tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py

Set ``TPR_QWEN_RING_CP_PROFILE_CASES`` to a comma-separated list of case IDs
to split a long run, for example ``p4096_s512_n2``.  The main comparison keeps
three warmups and ten uninstrumented samples.  A separate one-warmup,
three-sample NPU-event pass collects the latency breakdown; set
``TPR_QWEN_RING_CP_PROFILE_BREAKDOWN=0`` to disable that diagnostic pass.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import verl.models.mcore.tpr.parallel.ring_attention as ring_attention
from verl.models.mcore.tpr import (
    FixedTopologyScheduler,
    PhysicalExecutionKind,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)
from verl.utils.device import is_torch_npu_available

from ..parallel._tpr_cp_test_utils import cp_runtime
from ..parallel.test_tpr_qwen3_cp_equivalence_npu import (
    _QWEN_MODEL_CASES,
    _assert_qwen_cp_architecture,
    _load_hf_config,
    _make_qwen_cp_model,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_RING_CP_PROFILE") != "1",
    reason="Set TPR_RUN_QWEN_RING_CP_PROFILE=1 for the Qwen Ring CP profile",
)

_EXPECTED_WORLD_SIZE = 2
_WARMUP_RUNS = 3
_MEASURE_RUNS = 10
_BREAKDOWN_WARMUP_RUNS = 1
_BREAKDOWN_MEASURE_RUNS = 3
_RUN_BREAKDOWN = os.getenv("TPR_QWEN_RING_CP_PROFILE_BREAKDOWN", "1") == "1"
_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class _ProfileCase:
    prefix_length: int
    suffix_length: int
    trajectory_count: int

    @property
    def case_id(self) -> str:
        return (
            f"p{self.prefix_length}_s{self.suffix_length}_n{self.trajectory_count}"
        )

    @property
    def reference_logical_tokens(self) -> int:
        return self.trajectory_count * (self.prefix_length + self.suffix_length)

    @property
    def tpr_logical_executed_tokens(self) -> int:
        return self.prefix_length + self.trajectory_count * self.suffix_length

    @property
    def loss_term_count(self) -> int:
        return self.trajectory_count * (
            self.prefix_length + self.suffix_length - 1
        )


@dataclass(frozen=True, slots=True)
class _RunObservation:
    local_normalized_loss: torch.Tensor
    execution_trace: tuple[tuple[PhysicalExecutionKind, int], ...]


@dataclass(frozen=True, slots=True)
class _ProfileStats:
    median_ms: float
    mean_ms: float
    peak_allocated_bytes: int
    baseline_allocated_bytes: int

    @property
    def incremental_peak_bytes(self) -> int:
        return max(0, self.peak_allocated_bytes - self.baseline_allocated_bytes)


@dataclass(frozen=True, slots=True)
class _BreakdownStats:
    wall_median_ms: float
    categories: dict[str, float]
    calls: dict[str, int]


@dataclass(frozen=True, slots=True)
class _PathBreakdown:
    prefix_attention_forward_ms: float
    prefix_attention_backward_ms: float
    prefix_non_attention_forward_ms: float
    prefix_non_attention_backward_ms: float
    branch_attention_forward_ms: float
    branch_attention_backward_ms: float
    branch_non_attention_forward_ms: float
    branch_non_attention_backward_ms: float
    ring_communication_forward_ms: float
    ring_communication_backward_ms: float
    ring_fa_forward_ms: float
    ring_fa_backward_ms: float
    ring_merge_forward_ms: float
    ring_merge_backward_ms: float
    tpr_overhead_forward_ms: float
    tpr_overhead_backward_ms: float
    other_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class _ProfileResult:
    case: _ProfileCase
    reference: _ProfileStats
    tpr: _ProfileStats
    reference_breakdown: _BreakdownStats | None = None
    tpr_breakdown: _BreakdownStats | None = None

    @property
    def median_speedup(self) -> float:
        return self.reference.median_ms / self.tpr.median_ms

    @property
    def mean_speedup(self) -> float:
        return self.reference.mean_ms / self.tpr.mean_ms


_PROFILE_CASES = (
    _ProfileCase(4096, 512, 2),
    _ProfileCase(4096, 512, 8),
    _ProfileCase(8192, 1024, 8),
)


class _NPUEventRecorder:
    """Aggregate asynchronous NPU intervals without synchronizing inner scopes."""

    def __init__(self) -> None:
        self._stacks: dict[str, list] = {}
        self._pairs: dict[str, list[tuple]] = {}

    def begin(self, category: str) -> None:
        event = torch.npu.Event(enable_timing=True)
        event.record()
        self._stacks.setdefault(category, []).append(event)

    def end(self, category: str) -> None:
        stack = self._stacks.get(category)
        if not stack:
            raise RuntimeError(f"unmatched NPU timing event for {category}")
        start = stack.pop()
        end = torch.npu.Event(enable_timing=True)
        end.record()
        self._pairs.setdefault(category, []).append((start, end))

    def call(self, category: str, function, *args, **kwargs):
        self.begin(category)
        try:
            return function(*args, **kwargs)
        finally:
            self.end(category)

    def summarize(self) -> tuple[dict[str, float], dict[str, int]]:
        if any(self._stacks.values()):
            raise RuntimeError("unclosed NPU timing event")
        totals = {
            category: sum(start.elapsed_time(end) for start, end in pairs)
            for category, pairs in self._pairs.items()
        }
        calls = {category: len(pairs) for category, pairs in self._pairs.items()}
        return totals, calls


def _selected_profile_cases() -> tuple[_ProfileCase, ...]:
    value = os.getenv("TPR_QWEN_RING_CP_PROFILE_CASES", "all").strip().lower()
    if value in ("", "all"):
        return _PROFILE_CASES
    requested = {item.strip() for item in value.split(",") if item.strip()}
    known = {case.case_id: case for case in _PROFILE_CASES}
    unknown = requested.difference(known)
    if unknown:
        raise ValueError(
            "unknown TPR_QWEN_RING_CP_PROFILE_CASES entries "
            f"{sorted(unknown)}; expected a subset of {tuple(known)}"
        )
    return tuple(case for case in _PROFILE_CASES if case.case_id in requested)


def _tokens(start: int, length: int, vocab_size: int) -> torch.Tensor:
    return (torch.arange(start, start + length, dtype=torch.long) % vocab_size).contiguous()


def _internal_loss_terms(
    tokens: torch.Tensor,
    *,
    sample_id: int,
    weight: float = 1.0,
) -> tuple[SegmentLossTerm, ...]:
    return tuple(
        SegmentLossTerm(
            query_offset,
            int(tokens[query_offset + 1]),
            weight=weight,
            sample_id=sample_id,
        )
        for query_offset in range(tokens.numel() - 1)
    )


def _make_reference_plan(
    trajectory: torch.Tensor,
    *,
    sample_id: int,
    total_loss_weight: int,
) -> SegmentPlan:
    return SegmentPlan(
        (
            SegmentSpec(
                sample_id,
                None,
                trajectory,
                0,
                0,
                _internal_loss_terms(trajectory, sample_id=sample_id),
            ),
        ),
        root_id=sample_id,
        total_loss_weight=total_loss_weight,
    )


def _make_tpr_plan(
    prefix: torch.Tensor,
    suffixes: tuple[torch.Tensor, ...],
) -> SegmentPlan:
    prefix_length = prefix.numel()
    trajectory_count = len(suffixes)
    prefix_terms = _internal_loss_terms(
        prefix,
        sample_id=0,
        weight=float(trajectory_count),
    ) + tuple(
        SegmentLossTerm(
            prefix_length - 1,
            int(suffix[0]),
            sample_id=sample_id,
        )
        for sample_id, suffix in enumerate(suffixes, start=1)
    )
    segments = [SegmentSpec(0, None, prefix, 0, 0, prefix_terms)]
    segments.extend(
        SegmentSpec(
            sample_id,
            0,
            suffix,
            prefix_length,
            prefix_length,
            _internal_loss_terms(suffix, sample_id=sample_id),
        )
        for sample_id, suffix in enumerate(suffixes, start=1)
    )
    return SegmentPlan(segments, root_id=0)


def _make_case_plans(
    case: _ProfileCase,
    *,
    vocab_size: int,
) -> tuple[tuple[SegmentPlan, ...], SegmentPlan]:
    prefix = _tokens(17, case.prefix_length, vocab_size)
    suffixes = tuple(
        _tokens(
            100_003 + sample_id * (case.suffix_length + 17),
            case.suffix_length,
            vocab_size,
        )
        for sample_id in range(1, case.trajectory_count + 1)
    )
    total_loss_weight = case.loss_term_count
    reference_plans = tuple(
        _make_reference_plan(
            torch.cat((prefix, suffix)),
            sample_id=sample_id,
            total_loss_weight=total_loss_weight,
        )
        for sample_id, suffix in enumerate(suffixes, start=1)
    )
    tpr_plan = _make_tpr_plan(prefix, suffixes)
    if any(plan.total_loss_weight != total_loss_weight for plan in reference_plans):
        raise AssertionError("Reference plans must use the shared logical loss denominator")
    if tpr_plan.total_loss_weight != total_loss_weight:
        raise AssertionError(
            f"TPR loss weight must be {total_loss_weight}, got {tpr_plan.total_loss_weight}"
        )
    return reference_plans, tpr_plan


def _run_plan(model, plan, runtime, expected_layer_numbers) -> _RunObservation:
    executor = SegmentExecutor(
        model,
        plan,
        expected_layer_numbers=expected_layer_numbers,
        cp_group=runtime.cp_group,
        cp_backend="ring",
    )
    if executor.cp_backend is None or executor.cp_backend.backend_name != "ring":
        raise AssertionError("profile did not resolve the Ring CP backend")
    result = FixedTopologyScheduler(plan, executor).run()
    return _RunObservation(
        local_normalized_loss=result.normalized_loss,
        execution_trace=tuple(
            (execution.kind, execution.segment_id)
            for execution in result.execution_trace
        ),
    )


def _finalize_cp_parameter_gradients(model, runtime) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing parameter gradient before CP finalize: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
        parameter.grad.div_(runtime.cp_size)


def _make_reference_runner(model, plans, runtime, expected_layer_numbers):
    def run() -> _RunObservation:
        normalized_loss = None
        trace = []
        for plan in plans:
            observation = _run_plan(model, plan, runtime, expected_layer_numbers)
            normalized_loss = (
                observation.local_normalized_loss
                if normalized_loss is None
                else normalized_loss + observation.local_normalized_loss
            )
            trace.extend(observation.execution_trace)
        _finalize_cp_parameter_gradients(model, runtime)
        if normalized_loss is None:
            raise AssertionError("Reference profile must execute at least one trajectory")
        return _RunObservation(normalized_loss, tuple(trace))

    return run


def _make_tpr_runner(model, plan, runtime, expected_layer_numbers):
    def run() -> _RunObservation:
        observation = _run_plan(model, plan, runtime, expected_layer_numbers)
        _finalize_cp_parameter_gradients(model, runtime)
        return observation

    return run


def _validate_observation(
    observation: _RunObservation,
    *,
    expected_trace: tuple[tuple[PhysicalExecutionKind, int], ...],
) -> None:
    if observation.execution_trace != expected_trace:
        raise AssertionError(
            f"unexpected profile execution trace: {observation.execution_trace}"
        )
    if not math.isfinite(float(observation.local_normalized_loss)):
        raise AssertionError("profile produced a non-finite local normalized loss")


def _global_max_float(value: float, runtime) -> float:
    tensor = torch.tensor(value, dtype=torch.float32, device=runtime.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    return float(tensor.item())


def _global_max_int(value: int, runtime) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device=runtime.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    return int(tensor.item())


def _release_iteration_state(model, runtime) -> None:
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    dist.barrier(group=runtime.cp_group)
    torch.npu.synchronize()


def _profile_runner(
    run,
    model,
    runtime,
    *,
    expected_trace: tuple[tuple[PhysicalExecutionKind, int], ...],
) -> _ProfileStats:
    for warmup_index in range(_WARMUP_RUNS):
        model.zero_grad(set_to_none=True)
        dist.barrier(group=runtime.cp_group)
        torch.npu.synchronize()
        observation = run()
        torch.npu.synchronize()
        if warmup_index == 0:
            _validate_observation(observation, expected_trace=expected_trace)
        del observation

    _release_iteration_state(model, runtime)
    local_baseline = int(torch.npu.memory_allocated())
    torch.npu.reset_peak_memory_stats()

    latencies_ms = []
    for _ in range(_MEASURE_RUNS):
        model.zero_grad(set_to_none=True)
        dist.barrier(group=runtime.cp_group)
        torch.npu.synchronize()
        started = time.perf_counter()
        observation = run()
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        del observation
        latencies_ms.append(_global_max_float(elapsed_ms, runtime))

    local_peak = int(torch.npu.max_memory_allocated())
    return _ProfileStats(
        median_ms=statistics.median(latencies_ms),
        mean_ms=statistics.mean(latencies_ms),
        peak_allocated_bytes=_global_max_int(local_peak, runtime),
        baseline_allocated_bytes=_global_max_int(local_baseline, runtime),
    )


_BREAKDOWN_SUFFIXES = (
    "operation",
    "model_forward",
    "backward",
    "loss_forward",
    "kv_backward",
    "ring_forward",
    "ring_backward",
    "comm_forward",
    "comm_backward",
    "fa_forward",
    "fa_backward",
    "merge_forward",
)


def _breakdown_phases(path: str) -> tuple[str, ...]:
    if path == "reference":
        return ("full",)
    if path == "tpr":
        return ("prefix_push", "branch_visit", "prefix_pop")
    raise ValueError(f"unknown breakdown path {path!r}")


def _collect_breakdown(
    run,
    model,
    runtime,
    *,
    path: str,
    expected_trace: tuple[tuple[PhysicalExecutionKind, int], ...],
) -> _BreakdownStats:
    """Collect a low-sample NPU-event breakdown without inner synchronization."""

    for warmup_index in range(_BREAKDOWN_WARMUP_RUNS):
        model.zero_grad(set_to_none=True)
        dist.barrier(group=runtime.cp_group)
        torch.npu.synchronize()
        observation = run()
        torch.npu.synchronize()
        if warmup_index == 0:
            _validate_observation(observation, expected_trace=expected_trace)
        del observation

    phases = _breakdown_phases(path)
    expected_categories = tuple(
        f"{phase}.{suffix}"
        for phase in phases
        for suffix in _BREAKDOWN_SUFFIXES
    ) + ("gradient_finalize",)
    wall_samples = []
    category_samples = []
    call_samples = []

    for _ in range(_BREAKDOWN_MEASURE_RUNS):
        recorder = _NPUEventRecorder()
        active_phase = [None]
        original_push = SegmentExecutor.push
        original_visit_leaf = SegmentExecutor.visit_leaf
        original_pop = SegmentExecutor.pop
        original_compute_loss = SegmentExecutor._compute_loss
        original_accumulate = SegmentExecutor._accumulate_past_anchor_gradients
        original_autograd_backward = torch.autograd.backward
        original_finalize = _finalize_cp_parameter_gradients
        original_ring_forward = ring_attention._RingTPRAttention.forward
        original_ring_backward = ring_attention._RingTPRAttention.backward
        original_circulate = ring_attention._circulate_kv
        original_reduce = ring_attention._reduce_ring_gradients_to_owner
        original_fa_forward = ring_attention._block_attention_forward
        original_fa_backward = ring_attention._block_attention_backward
        original_merge = ring_attention._merge_attention

        def phase_for(method_name: str) -> str:
            if path == "reference":
                return "full"
            return {
                "push": "prefix_push",
                "visit_leaf": "branch_visit",
                "pop": "prefix_pop",
            }[method_name]

        def in_phase(category_suffix: str, function, *args, **kwargs):
            phase = active_phase[0]
            if phase is None:
                return function(*args, **kwargs)
            return recorder.call(f"{phase}.{category_suffix}", function, *args, **kwargs)

        def call_operation(method_name: str, function, executor, segment_id):
            if executor.model is not model:
                return function(executor, segment_id)
            previous_phase = active_phase[0]
            active_phase[0] = phase_for(method_name)
            try:
                return in_phase("operation", function, executor, segment_id)
            finally:
                active_phase[0] = previous_phase

        def timed_push(executor, segment_id):
            return call_operation("push", original_push, executor, segment_id)

        def timed_visit_leaf(executor, segment_id):
            return call_operation(
                "visit_leaf",
                original_visit_leaf,
                executor,
                segment_id,
            )

        def timed_pop(executor, segment_id):
            return call_operation("pop", original_pop, executor, segment_id)

        def timed_compute_loss(executor, *args, **kwargs):
            if executor.model is not model:
                return original_compute_loss(executor, *args, **kwargs)
            return in_phase(
                "loss_forward",
                original_compute_loss,
                executor,
                *args,
                **kwargs,
            )

        def timed_accumulate(executor, *args, **kwargs):
            if executor.model is not model:
                return original_accumulate(executor, *args, **kwargs)
            return in_phase(
                "kv_backward",
                original_accumulate,
                executor,
                *args,
                **kwargs,
            )

        def timed_autograd_backward(*args, **kwargs):
            return in_phase("backward", original_autograd_backward, *args, **kwargs)

        def timed_finalize(model_arg, runtime_arg):
            if model_arg is not model:
                return original_finalize(model_arg, runtime_arg)
            return recorder.call(
                "gradient_finalize",
                original_finalize,
                model_arg,
                runtime_arg,
            )

        def timed_ring_forward(ctx, *args):
            return in_phase("ring_forward", original_ring_forward, ctx, *args)

        def timed_ring_backward(ctx, *args):
            return in_phase("ring_backward", original_ring_backward, ctx, *args)

        def timed_circulate(*args, **kwargs):
            return in_phase("comm_forward", original_circulate, *args, **kwargs)

        def timed_reduce(*args, **kwargs):
            return in_phase("comm_backward", original_reduce, *args, **kwargs)

        def timed_fa_forward(*args, **kwargs):
            return in_phase("fa_forward", original_fa_forward, *args, **kwargs)

        def timed_fa_backward(*args, **kwargs):
            return in_phase("fa_backward", original_fa_backward, *args, **kwargs)

        def timed_merge(*args, **kwargs):
            return in_phase("merge_forward", original_merge, *args, **kwargs)

        def model_forward_start(*_args):
            phase = active_phase[0]
            if phase is None:
                raise RuntimeError("profile model forward occurred outside an execution phase")
            recorder.begin(f"{phase}.model_forward")

        def model_forward_end(*_args):
            phase = active_phase[0]
            if phase is None:
                raise RuntimeError("profile model forward ended outside an execution phase")
            recorder.end(f"{phase}.model_forward")

        handles = (
            model.register_forward_pre_hook(model_forward_start),
            model.register_forward_hook(model_forward_end),
        )
        module = sys.modules[__name__]
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(SegmentExecutor, "push", timed_push))
                stack.enter_context(
                    patch.object(SegmentExecutor, "visit_leaf", timed_visit_leaf)
                )
                stack.enter_context(patch.object(SegmentExecutor, "pop", timed_pop))
                stack.enter_context(
                    patch.object(SegmentExecutor, "_compute_loss", timed_compute_loss)
                )
                stack.enter_context(
                    patch.object(
                        SegmentExecutor,
                        "_accumulate_past_anchor_gradients",
                        timed_accumulate,
                    )
                )
                stack.enter_context(
                    patch.object(torch.autograd, "backward", timed_autograd_backward)
                )
                stack.enter_context(
                    patch.object(module, "_finalize_cp_parameter_gradients", timed_finalize)
                )
                stack.enter_context(
                    patch.object(
                        ring_attention._RingTPRAttention,
                        "forward",
                        staticmethod(timed_ring_forward),
                    )
                )
                stack.enter_context(
                    patch.object(
                        ring_attention._RingTPRAttention,
                        "backward",
                        staticmethod(timed_ring_backward),
                    )
                )
                stack.enter_context(
                    patch.object(ring_attention, "_circulate_kv", timed_circulate)
                )
                stack.enter_context(
                    patch.object(
                        ring_attention,
                        "_reduce_ring_gradients_to_owner",
                        timed_reduce,
                    )
                )
                stack.enter_context(
                    patch.object(
                        ring_attention,
                        "_block_attention_forward",
                        timed_fa_forward,
                    )
                )
                stack.enter_context(
                    patch.object(
                        ring_attention,
                        "_block_attention_backward",
                        timed_fa_backward,
                    )
                )
                stack.enter_context(
                    patch.object(ring_attention, "_merge_attention", timed_merge)
                )

                model.zero_grad(set_to_none=True)
                dist.barrier(group=runtime.cp_group)
                torch.npu.synchronize()
                started = time.perf_counter()
                observation = run()
                torch.npu.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000.0
                del observation
        finally:
            for handle in handles:
                handle.remove()

        local_totals, local_calls = recorder.summarize()
        wall_samples.append(_global_max_float(wall_ms, runtime))
        category_samples.append(
            {
                category: _global_max_float(local_totals.get(category, 0.0), runtime)
                for category in expected_categories
            }
        )
        call_samples.append(
            {category: local_calls.get(category, 0) for category in expected_categories}
        )

    for category in expected_categories:
        observed_calls = {sample[category] for sample in call_samples}
        if len(observed_calls) != 1:
            raise AssertionError(
                f"unstable breakdown call count for {category}: {sorted(observed_calls)}"
            )
    return _BreakdownStats(
        wall_median_ms=statistics.median(wall_samples),
        categories={
            category: statistics.median(sample[category] for sample in category_samples)
            for category in expected_categories
        },
        calls={category: call_samples[0][category] for category in expected_categories},
    )


def _format_gib(value: int) -> str:
    return f"{value / _GIB:.3f} GiB"


def _category(stats: _BreakdownStats, phase: str, suffix: str) -> float:
    return stats.categories.get(f"{phase}.{suffix}", 0.0)


def _phase_breakdown(stats: _BreakdownStats, phase: str) -> dict[str, float]:
    attention_forward = _category(stats, phase, "ring_forward")
    attention_backward = _category(stats, phase, "ring_backward")
    communication_forward = _category(stats, phase, "comm_forward")
    communication_backward = _category(stats, phase, "comm_backward")
    fa_forward = _category(stats, phase, "fa_forward")
    fa_backward = _category(stats, phase, "fa_backward")
    return {
        "attention_forward": attention_forward,
        "attention_backward": attention_backward,
        "non_attention_forward": max(
            0.0,
            _category(stats, phase, "model_forward") - attention_forward,
        ),
        "non_attention_backward": max(
            0.0,
            _category(stats, phase, "backward") - attention_backward,
        ),
        "communication_forward": communication_forward,
        "communication_backward": communication_backward,
        "fa_forward": fa_forward,
        "fa_backward": fa_backward,
        # Backward merge/accumulation is implemented inside the custom Ring
        # backward rather than through _merge_attention.  The Ring-total
        # remainder therefore gives the least intrusive common definition.
        "merge_forward": max(
            0.0,
            attention_forward - communication_forward - fa_forward,
        ),
        "merge_backward": max(
            0.0,
            attention_backward - communication_backward - fa_backward,
        ),
    }


def _sum_phase_values(
    phase_values: dict[str, dict[str, float]],
    phases: tuple[str, ...],
    key: str,
) -> float:
    return sum(phase_values[phase][key] for phase in phases)


def _derive_path_breakdown(
    case: _ProfileCase,
    profile: _ProfileStats,
    breakdown: _BreakdownStats,
    *,
    path: str,
) -> _PathBreakdown:
    phases = _breakdown_phases(path)
    values = {phase: _phase_breakdown(breakdown, phase) for phase in phases}
    if path == "reference":
        full = values["full"]
        prefix_fraction = case.prefix_length / (
            case.prefix_length + case.suffix_length
        )
        branch_fraction = 1.0 - prefix_fraction
        prefix_attention_forward = full["attention_forward"] * prefix_fraction
        prefix_attention_backward = full["attention_backward"] * prefix_fraction
        prefix_non_attention_forward = (
            full["non_attention_forward"] * prefix_fraction
        )
        prefix_non_attention_backward = (
            full["non_attention_backward"] * prefix_fraction
        )
        branch_attention_forward = full["attention_forward"] * branch_fraction
        branch_attention_backward = full["attention_backward"] * branch_fraction
        branch_non_attention_forward = (
            full["non_attention_forward"] * branch_fraction
        )
        branch_non_attention_backward = (
            full["non_attention_backward"] * branch_fraction
        )
        overhead_forward = 0.0
        overhead_backward = 0.0
    else:
        prefix_phases = ("prefix_push", "prefix_pop")
        branch_phases = ("branch_visit",)
        prefix_attention_forward = _sum_phase_values(
            values, prefix_phases, "attention_forward"
        )
        prefix_attention_backward = _sum_phase_values(
            values, prefix_phases, "attention_backward"
        )
        prefix_non_attention_forward = _sum_phase_values(
            values, prefix_phases, "non_attention_forward"
        )
        prefix_non_attention_backward = _sum_phase_values(
            values, prefix_phases, "non_attention_backward"
        )
        branch_attention_forward = _sum_phase_values(
            values, branch_phases, "attention_forward"
        )
        branch_attention_backward = _sum_phase_values(
            values, branch_phases, "attention_backward"
        )
        branch_non_attention_forward = _sum_phase_values(
            values, branch_phases, "non_attention_forward"
        )
        branch_non_attention_backward = _sum_phase_values(
            values, branch_phases, "non_attention_backward"
        )
        overhead_forward = 0.0
        overhead_backward = 0.0
        for phase in phases:
            residual = max(
                0.0,
                _category(breakdown, phase, "operation")
                - _category(breakdown, phase, "model_forward")
                - _category(breakdown, phase, "backward")
                - _category(breakdown, phase, "loss_forward"),
            )
            kv_backward = min(
                residual,
                _category(breakdown, phase, "kv_backward"),
            )
            overhead_forward += residual - kv_backward
            overhead_backward += kv_backward

    communication_forward = _sum_phase_values(
        values, phases, "communication_forward"
    )
    communication_backward = _sum_phase_values(
        values, phases, "communication_backward"
    )
    fa_forward = _sum_phase_values(values, phases, "fa_forward")
    fa_backward = _sum_phase_values(values, phases, "fa_backward")
    merge_forward = _sum_phase_values(values, phases, "merge_forward")
    merge_backward = _sum_phase_values(values, phases, "merge_backward")
    covered_top_level = sum(
        (
            prefix_attention_forward,
            prefix_attention_backward,
            prefix_non_attention_forward,
            prefix_non_attention_backward,
            branch_attention_forward,
            branch_attention_backward,
            branch_non_attention_forward,
            branch_non_attention_backward,
            overhead_forward,
            overhead_backward,
        )
    )
    return _PathBreakdown(
        prefix_attention_forward,
        prefix_attention_backward,
        prefix_non_attention_forward,
        prefix_non_attention_backward,
        branch_attention_forward,
        branch_attention_backward,
        branch_non_attention_forward,
        branch_non_attention_backward,
        communication_forward,
        communication_backward,
        fa_forward,
        fa_backward,
        merge_forward,
        merge_backward,
        overhead_forward,
        overhead_backward,
        profile.median_ms - covered_top_level,
        profile.median_ms,
    )


def _format_forward_backward(forward_ms: float, backward_ms: float) -> str:
    return f"{forward_ms:.1f}/{backward_ms:.1f}"


def _print_breakdown_summary(results: tuple[_ProfileResult, ...], *, rank: int) -> None:
    if rank != 0 or not results or results[0].reference_breakdown is None:
        return
    print("\nRing CP latency breakdown (all paired cells are forward/backward ms)")
    print(
        "Reference Prefix/Branch is a token-weighted estimate because each P+S "
        "trajectory is one fused execution. TPR phase attribution is measured."
    )
    print(
        "Ring Comm/FA are measured drill-downs of Attention and are not additive "
        "with Prefix or Branch Attn. Total is the uninstrumented 10-run median."
    )
    print(
        "Other is the residual containing loss work, final parameter-gradient "
        "synchronization, scheduler/host gaps, and any uncovered device work."
    )
    print(
        "Case | Path | Prefix | Branch Attn | Branch Non-Attn | Ring Comm | "
        "Ring FA | TPR Overhead | Other | Total"
    )
    print("-" * 150)
    for result in results:
        assert result.reference_breakdown is not None
        assert result.tpr_breakdown is not None
        paths = (
            (
                "Reference",
                _derive_path_breakdown(
                    result.case,
                    result.reference,
                    result.reference_breakdown,
                    path="reference",
                ),
            ),
            (
                "TPR",
                _derive_path_breakdown(
                    result.case,
                    result.tpr,
                    result.tpr_breakdown,
                    path="tpr",
                ),
            ),
        )
        for path_name, item in paths:
            raw_breakdown = (
                result.reference_breakdown
                if path_name == "Reference"
                else result.tpr_breakdown
            )
            instrumented_wall = raw_breakdown.wall_median_ms
            prefix_forward = (
                item.prefix_attention_forward_ms
                + item.prefix_non_attention_forward_ms
            )
            prefix_backward = (
                item.prefix_attention_backward_ms
                + item.prefix_non_attention_backward_ms
            )
            branch_attention = _format_forward_backward(
                item.branch_attention_forward_ms,
                item.branch_attention_backward_ms,
            )
            branch_non_attention = _format_forward_backward(
                item.branch_non_attention_forward_ms,
                item.branch_non_attention_backward_ms,
            )
            ring_communication = _format_forward_backward(
                item.ring_communication_forward_ms,
                item.ring_communication_backward_ms,
            )
            prefix_attention = _format_forward_backward(
                item.prefix_attention_forward_ms,
                item.prefix_attention_backward_ms,
            )
            prefix_non_attention = _format_forward_backward(
                item.prefix_non_attention_forward_ms,
                item.prefix_non_attention_backward_ms,
            )
            print(
                f"{result.case.case_id} | {path_name} | "
                f"{_format_forward_backward(prefix_forward, prefix_backward)} | "
                f"{branch_attention} | {branch_non_attention} | "
                f"{ring_communication} | "
                f"{_format_forward_backward(item.ring_fa_forward_ms, item.ring_fa_backward_ms)} | "
                f"{_format_forward_backward(item.tpr_overhead_forward_ms, item.tpr_overhead_backward_ms)} | "
                f"{item.other_ms:.1f} | {item.total_ms:.1f}"
            )
            print(
                f"  detail: Prefix Attn={prefix_attention}, "
                f"Prefix Non-Attn={prefix_non_attention}, "
                "Ring Merge/framework="
                f"{_format_forward_backward(item.ring_merge_forward_ms, item.ring_merge_backward_ms)}, "
                f"gradient finalize={raw_breakdown.categories['gradient_finalize']:.1f} ms, "
                f"instrumented wall={instrumented_wall:.1f} ms"
            )
        _print_attribution(result, paths[0][1], paths[1][1])


def _print_attribution(
    result: _ProfileResult,
    reference: _PathBreakdown,
    tpr: _PathBreakdown,
) -> None:
    assert result.reference_breakdown is not None
    assert result.tpr_breakdown is not None
    case = result.case
    tpr_raw = result.tpr_breakdown
    push_forward = _category(tpr_raw, "prefix_push", "model_forward")
    pop_forward = _category(tpr_raw, "prefix_pop", "model_forward")
    pop_backward = _category(tpr_raw, "prefix_pop", "backward")
    shape_aligned_reference_prefix = case.trajectory_count * (
        pop_forward + pop_backward
    )
    actual_tpr_prefix = push_forward + pop_forward + pop_backward
    estimated_prefix_saved = shape_aligned_reference_prefix - actual_tpr_prefix

    reference_branch_fraction = case.suffix_length / (
        case.prefix_length + case.suffix_length
    )
    reference_branch_compute = (
        reference.branch_non_attention_forward_ms
        + reference.branch_non_attention_backward_ms
        + (
            reference.ring_fa_forward_ms
            + reference.ring_fa_backward_ms
            + reference.ring_merge_forward_ms
            + reference.ring_merge_backward_ms
        )
        * reference_branch_fraction
    )
    tpr_branch_compute = (
        tpr.branch_non_attention_forward_ms
        + tpr.branch_non_attention_backward_ms
        + _category(tpr_raw, "branch_visit", "fa_forward")
        + _category(tpr_raw, "branch_visit", "fa_backward")
        + _phase_breakdown(tpr_raw, "branch_visit")["merge_forward"]
        + _phase_breakdown(tpr_raw, "branch_visit")["merge_backward"]
    )
    penalties = {
        "repeated Ring communication": max(
            0.0,
            tpr.ring_communication_forward_ms
            + tpr.ring_communication_backward_ms
            - reference.ring_communication_forward_ms
            - reference.ring_communication_backward_ms,
        ),
        "small-shape FA/GEMM": max(
            0.0,
            tpr_branch_compute - reference_branch_compute,
        ),
        "Prefix recompute/backward": pop_forward + pop_backward,
        "scheduler/KV/uncovered": (
            tpr.tpr_overhead_forward_ms
            + tpr.tpr_overhead_backward_ms
            + max(0.0, tpr.other_ms - reference.other_ms)
        ),
    }
    dominant_name, dominant_ms = max(penalties.items(), key=lambda item: item[1])
    reference_communication_calls = sum(
        result.reference_breakdown.calls.get(f"full.{suffix}", 0)
        for suffix in ("comm_forward", "comm_backward")
    )
    tpr_communication_calls = sum(
        result.tpr_breakdown.calls.get(f"{phase}.{suffix}", 0)
        for phase in _breakdown_phases("tpr")
        for suffix in ("comm_forward", "comm_backward")
    )
    print(
        "  Prefix traversal count: Reference="
        f"{case.trajectory_count} forward + {case.trajectory_count} backward; "
        "TPR=2 forward (Push + Pop recompute) + 1 backward."
    )
    print(
        "  Prefix compute saved (shape-aligned estimate): "
        f"{estimated_prefix_saved:.1f} ms; counterfactual Reference="
        f"{shape_aligned_reference_prefix:.1f} ms, TPR-measured={actual_tpr_prefix:.1f} ms."
    )
    print(
        "  Ring transport calls on rank 0 (forward + backward wrappers): "
        f"Reference={reference_communication_calls}, TPR={tpr_communication_calls}."
    )
    print(
        "  TPR physical phases: "
        f"Push={_category(tpr_raw, 'prefix_push', 'operation'):.1f} ms, "
        f"Visit={_category(tpr_raw, 'branch_visit', 'operation'):.1f} ms, "
        f"Pop={_category(tpr_raw, 'prefix_pop', 'operation'):.1f} ms "
        f"(Pop backward={pop_backward:.1f} ms)."
    )
    print(
        "  Benefit consumers (diagnostic, non-additive): "
        + ", ".join(f"{name}={value:.1f} ms" for name, value in penalties.items())
    )
    print(f"  Dominant measured consumer: {dominant_name} ({dominant_ms:.1f} ms)")


def _print_case_result(result: _ProfileResult, *, rank: int) -> None:
    if rank != 0:
        return
    case = result.case
    print(
        f"Ring CP profile case {case.case_id}:\n"
        f"  Reference logical tokens N*(P+S): {case.reference_logical_tokens}\n"
        f"  TPR logical executed tokens P+N*S: {case.tpr_logical_executed_tokens}\n"
        f"  logical loss terms: {case.loss_term_count}\n"
        f"  Reference median/mean: {result.reference.median_ms:.3f} / "
        f"{result.reference.mean_ms:.3f} ms\n"
        f"  TPR median/mean: {result.tpr.median_ms:.3f} / "
        f"{result.tpr.mean_ms:.3f} ms\n"
        f"  TPR speedup median/mean: {result.median_speedup:.3f}x / "
        f"{result.mean_speedup:.3f}x\n"
        f"  Reference peak allocated: {_format_gib(result.reference.peak_allocated_bytes)} "
        f"(incremental {_format_gib(result.reference.incremental_peak_bytes)})\n"
        f"  TPR peak allocated: {_format_gib(result.tpr.peak_allocated_bytes)} "
        f"(incremental {_format_gib(result.tpr.incremental_peak_bytes)})"
    )


def _print_summary(results: tuple[_ProfileResult, ...], *, rank: int) -> None:
    if rank != 0:
        return
    print("\nQwen3-1.7B BF16 Ring CP=2 Reference vs TPR profile")
    print(f"warmup={_WARMUP_RUNS}, measured={_MEASURE_RUNS}; time is max across CP ranks")
    print(
        "P | S | N | Ref ms (median/mean) | TPR ms (median/mean) | "
        "Speedup (median) | Ref Peak Mem | TPR Peak Mem"
    )
    print("-" * 118)
    for result in results:
        case = result.case
        print(
            f"{case.prefix_length} | {case.suffix_length} | {case.trajectory_count} | "
            f"{result.reference.median_ms:.3f}/{result.reference.mean_ms:.3f} | "
            f"{result.tpr.median_ms:.3f}/{result.tpr.mean_ms:.3f} | "
            f"{result.median_speedup:.3f}x | "
            f"{_format_gib(result.reference.peak_allocated_bytes)} | "
            f"{_format_gib(result.tpr.peak_allocated_bytes)}"
        )


def test_qwen3_1_7b_reference_cp_vs_tpr_ring_cp_profile(cp_runtime):
    runtime = cp_runtime
    if runtime.cp_size != _EXPECTED_WORLD_SIZE:
        raise AssertionError(
            f"Ring profile requires CP={_EXPECTED_WORLD_SIZE}, got {runtime.cp_size}"
        )

    model_case = next(
        case for case in _QWEN_MODEL_CASES if case.name == "qwen3_1_7b"
    )
    hf_config = _load_hf_config(model_case)
    model = _make_qwen_cp_model(runtime, model_case, hf_config)
    _assert_qwen_cp_architecture(model, runtime, model_case, hf_config)
    if next(model.parameters()).dtype != torch.bfloat16:
        raise AssertionError("Ring CP profile model must use BF16 parameters")
    expected_layer_numbers = tuple(
        layer.self_attention.layer_number for layer in model.decoder.layers
    )
    dist.barrier(group=runtime.cp_group)
    torch.npu.synchronize()

    results = []
    for case in _selected_profile_cases():
        reference_plans, tpr_plan = _make_case_plans(
            case,
            vocab_size=hf_config.vocab_size,
        )
        reference_runner = _make_reference_runner(
            model,
            reference_plans,
            runtime,
            expected_layer_numbers,
        )
        tpr_runner = _make_tpr_runner(
            model,
            tpr_plan,
            runtime,
            expected_layer_numbers,
        )
        reference_trace = tuple(
            (PhysicalExecutionKind.VISIT_LEAF, sample_id)
            for sample_id in range(1, case.trajectory_count + 1)
        )
        tpr_trace = (
            (PhysicalExecutionKind.PUSH, 0),
            *(
                (PhysicalExecutionKind.VISIT_LEAF, sample_id)
                for sample_id in range(1, case.trajectory_count + 1)
            ),
            (PhysicalExecutionKind.POP, 0),
        )

        reference_stats = _profile_runner(
            reference_runner,
            model,
            runtime,
            expected_trace=reference_trace,
        )
        _release_iteration_state(model, runtime)
        tpr_stats = _profile_runner(
            tpr_runner,
            model,
            runtime,
            expected_trace=tpr_trace,
        )
        _release_iteration_state(model, runtime)

        reference_breakdown = None
        tpr_breakdown = None
        if _RUN_BREAKDOWN:
            reference_breakdown = _collect_breakdown(
                reference_runner,
                model,
                runtime,
                path="reference",
                expected_trace=reference_trace,
            )
            _release_iteration_state(model, runtime)
            tpr_breakdown = _collect_breakdown(
                tpr_runner,
                model,
                runtime,
                path="tpr",
                expected_trace=tpr_trace,
            )
            _release_iteration_state(model, runtime)

        result = _ProfileResult(
            case,
            reference_stats,
            tpr_stats,
            reference_breakdown,
            tpr_breakdown,
        )
        results.append(result)
        _print_case_result(result, rank=runtime.rank)

    final_results = tuple(results)
    _print_summary(final_results, rank=runtime.rank)
    _print_breakdown_summary(final_results, rank=runtime.rank)
