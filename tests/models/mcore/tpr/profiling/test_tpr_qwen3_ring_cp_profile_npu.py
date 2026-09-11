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
to split a long run, for example ``p4096_s512_n2``.
"""

from __future__ import annotations

import gc
import math
import os
import statistics
import time
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

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
class _ProfileResult:
    case: _ProfileCase
    reference: _ProfileStats
    tpr: _ProfileStats

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


def _format_gib(value: int) -> str:
    return f"{value / _GIB:.3f} GiB"


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

        result = _ProfileResult(case, reference_stats, tpr_stats)
        results.append(result)
        _print_case_result(result, rank=runtime.rank)

    _print_summary(tuple(results), rank=runtime.rank)
