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

"""Opt-in Qwen3-1.7B/4B Reference versus TPR profile with CP=1/2/4.

The Reference executes every complete ``P + S`` trajectory independently.
TPR executes one shared Prefix followed by all Suffixes.  Model construction,
checkpoint loading, plan construction, warmup, and gradient clearing are not
timed.  Each measured sample includes forward, loss, backward, and one final
CP parameter-gradient synchronization (omitted for CP=1).

Select TPR_QWEN_PROFILE_SIZE=1.7B or 4B (default: 1.7B).
Checkpoints default to /workspace/hf_models/Qwen3-<size>; override with
TPR_QWEN_MODEL_PATH or the size-specific TPR_QWEN_1_7B_PATH /
TPR_QWEN_4B_PATH. Synthetic and 0.6B profiles are intentionally unsupported. CP is WORLD_SIZE, set by
torchrun --nproc_per_node=1, 2 or 4. CP=1 uses local rectangular attention;
CP>1 uses Ring. All sizes use the same MindSpeed bootstrap and loss path.

Run all cases from the verl repository root with two visible NPUs::

    TPR_RUN_QWEN_RING_CP_PROFILE=1 \
    TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B \
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29551 \
        -m pytest -s -v -x \
        tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py

Edit ``_PROFILE_CASES`` below to choose cases; every listed case is run.
The main comparison keeps
three warmups and ten uninstrumented samples.  A separate one-warmup,
three-sample NPU-event pass collects the latency breakdown; set
``TPR_QWEN_RING_CP_PROFILE_BREAKDOWN=0`` to disable that diagnostic pass.
"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

import verl.models.mcore.tpr.parallel.ring_attention as ring_attention
import verl.models.mcore.tpr.attention as local_attention
from verl.models.mcore.tpr import (
    FixedTopologyScheduler,
    PhysicalExecutionKind,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)
from verl.utils.device import is_torch_npu_available

from ._qwen3_profile_target import resolve_qwen3_profile_target
from ..parallel._ring_block_probe import ring_block_probe
from ..parallel.test_tpr_qwen3_cp_equivalence_npu import (
    _make_qwen_cp_model,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_RING_CP_PROFILE") != "1",
    reason="Set TPR_RUN_QWEN_RING_CP_PROFILE=1 for the Qwen Ring CP profile",
)

_EXPECTED_WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
_WARMUP_RUNS = int(os.getenv("TPR_QWEN_RING_CP_PROFILE_WARMUP", "3"))
_MEASURE_RUNS = int(os.getenv("TPR_QWEN_RING_CP_PROFILE_REPEATS", "10"))
_BREAKDOWN_WARMUP_RUNS = 1
_BREAKDOWN_MEASURE_RUNS = 3
_RUN_BREAKDOWN = os.getenv("TPR_QWEN_RING_CP_PROFILE_BREAKDOWN", "1") == "1"
_ENABLE_OFFLOAD = os.getenv("TPR_QWEN_RING_CP_PROFILE_OFFLOAD", "0") == "1"
_PROFILE_PATH = os.getenv("TPR_QWEN_RING_CP_PROFILE_PATH", "both").strip().lower()
if _PROFILE_PATH not in ("both", "reference", "tpr"):
    raise ValueError("TPR_QWEN_RING_CP_PROFILE_PATH must be both, reference or tpr")
_SWAP_MODULES = os.getenv("TPR_SWAP_MODULES", "self_attention,mlp")
_LOSS_CHUNK_SIZE_RAW = int(os.getenv("TPR_LOSS_CHUNK_SIZE", "0"))
_LOSS_CHUNK_SIZE = None if _LOSS_CHUNK_SIZE_RAW <= 0 else _LOSS_CHUNK_SIZE_RAW
_GIB = 1024**3


@pytest.fixture(scope="module")
def profile_runtime():
    if _EXPECTED_WORLD_SIZE not in (1, 2, 4):
        raise ValueError("This profile supports WORLD_SIZE=1, 2 or 4")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    # Match the production NPU bootstrap before Megatron initializes its
    # model-parallel RNG tracker. MindSpeed also redirects the legacy
    # ``torch.cuda`` RNG calls in this Megatron revision to torch_npu.
    pytest_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = pytest_argv

    from mindspeed.args_utils import get_full_args

    vars(get_full_args()).pop("", None)
    repatch(
        {
            "context_parallel_size": _EXPECTED_WORLD_SIZE,
            "context_parallel_algo": "megatron_cp_algo",
        }
    )
    args = get_full_args()
    if _ENABLE_OFFLOAD:
        vars(args).update(
            swap_attention=False,
            context_parallel_size=_EXPECTED_WORLD_SIZE,
            pipeline_model_parallel_size=1,
            eval_interval=0,
            curr_iteration=1,
            noop_layers=None,
            swap_modules=_SWAP_MODULES,
        )

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_EXPECTED_WORLD_SIZE,
            expert_model_parallel_size=1,
        )

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(260907)

    cp_group = parallel_state.get_context_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    if cp_group.size() != _EXPECTED_WORLD_SIZE or tp_group.size() != 1 or pp_group.size() != 1:
        raise AssertionError(
            f"unexpected topology: TP={tp_group.size()}, PP={pp_group.size()}, CP={cp_group.size()}"
        )
    yield SimpleNamespace(
        rank=dist.get_rank(cp_group),
        cp_size=cp_group.size(),
        device=torch.device("npu", local_rank),
        cp_group=cp_group,
        tp_group=tp_group,
        pp_group=pp_group,
    )

    dist.barrier(group=cp_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


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
    std_ms: float
    peak_reserved_bytes: int
    parameter_count: int
    adapter_probe_calls: int
    layer_count: int
    physical_phase_count: int
    max_incremental_peak_bytes: int

    @property
    def incremental_peak_bytes(self) -> int:
        return self.max_incremental_peak_bytes


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


def _parse_profile_cases() -> tuple[_ProfileCase, ...]:
    raw = os.getenv("TPR_QWEN_RING_CP_PROFILE_CASES", "").strip()
    if not raw:
        return (_ProfileCase(16384, 16384, 8),)
    cases = []
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                "TPR_QWEN_RING_CP_PROFILE_CASES entries must be P:S or P:S:N"
            )
        prefix, suffix = (int(parts[0]), int(parts[1]))
        siblings = int(parts[2]) if len(parts) == 3 else 2
        if prefix <= 0 or suffix <= 0 or siblings <= 0:
            raise ValueError("profile P/S/N values must be positive")
        cases.append(_ProfileCase(prefix, suffix, siblings))
    return tuple(cases)


_PROFILE_CASES = _parse_profile_cases()


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
        cp_group=runtime.cp_group if runtime.cp_size > 1 else None,
        cp_backend="ring" if runtime.cp_size > 1 else None,
        loss_chunk_size=_LOSS_CHUNK_SIZE,
    )
    if runtime.cp_size > 1 and (executor.cp_backend is None or executor.cp_backend.backend_name != "ring"):
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
    if runtime.cp_size == 1:
        return
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
    # Probe separately so instrumentation cannot affect the latency samples.
    layer_trace = []
    adapter_calls = 0
    adapter_owner = ring_attention._RingTPRAttention if runtime.cp_size > 1 else local_attention
    adapter_name = "forward" if runtime.cp_size > 1 else "rectangular_causal_attention"
    original_adapter = getattr(adapter_owner, adapter_name)

    def counted_adapter(*args, **kwargs):
        nonlocal adapter_calls
        adapter_calls += 1
        return original_adapter(*args, **kwargs)

    handles = [
        layer.self_attention.register_forward_pre_hook(
            lambda module, _args: layer_trace.append(module.layer_number)
        )
        for layer in model.decoder.layers
    ]
    try:
        model.zero_grad(set_to_none=True)
        dist.barrier(group=runtime.cp_group)
        with patch.object(
            adapter_owner, adapter_name,
            staticmethod(counted_adapter) if runtime.cp_size > 1 else counted_adapter
        ):
            with ring_block_probe(model, runtime):
                observation = run()
                torch.npu.synchronize()
        _validate_observation(observation, expected_trace=expected_trace)
        del observation
    finally:
        for handle in handles:
            handle.remove()
    layer_numbers = tuple(layer.self_attention.layer_number for layer in model.decoder.layers)
    assert tuple(layer_trace) == layer_numbers * len(expected_trace)
    assert adapter_calls == len(layer_trace)

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
    local_reserved_peak = int(torch.npu.max_memory_reserved())
    return _ProfileStats(
        median_ms=statistics.median(latencies_ms),
        mean_ms=statistics.mean(latencies_ms),
        peak_allocated_bytes=_global_max_int(local_peak, runtime),
        baseline_allocated_bytes=_global_max_int(local_baseline, runtime),
        std_ms=statistics.pstdev(latencies_ms),
        peak_reserved_bytes=_global_max_int(local_reserved_peak, runtime),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        adapter_probe_calls=adapter_calls,
        layer_count=len(layer_numbers),
        physical_phase_count=len(expected_trace),
        max_incremental_peak_bytes=_global_max_int(max(0, local_peak - local_baseline), runtime),
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
    "attention_forward",
    "attention_backward",
    "mlp_forward",
    "mlp_backward",
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

        handles = [
            model.register_forward_pre_hook(model_forward_start),
            model.register_forward_hook(model_forward_end),
        ]

        def add_module_hooks(module, category):
            def begin(*_args):
                recorder.begin(f"{active_phase[0]}.{category}_forward")

            def end(*_args):
                recorder.end(f"{active_phase[0]}.{category}_forward")

            def backward_begin(*_args):
                recorder.begin(f"{active_phase[0]}.{category}_backward")

            def backward_end(*_args):
                recorder.end(f"{active_phase[0]}.{category}_backward")

            handles.extend((
                module.register_forward_pre_hook(begin),
                module.register_forward_hook(end),
                module.register_full_backward_pre_hook(backward_begin),
                module.register_full_backward_hook(backward_end),
            ))

        for layer in model.decoder.layers:
            add_module_hooks(layer.self_attention, "attention")
            add_module_hooks(layer.mlp, "mlp")
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
    # This table derives attention from Ring-specific events. The local path
    # is covered by the module-level controlled breakdown instead.
    if _EXPECTED_WORLD_SIZE == 1:
        return
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


def _print_controlled_breakdown(name, breakdown, *, path, sibling_count):
    phases = _breakdown_phases(path)
    print(f"{name} diagnostic breakdown (median of {_BREAKDOWN_MEASURE_RUNS} runs):")
    print(f"  synchronized wall: {breakdown.wall_median_ms:.3f} ms")

    def row(label, keys, *, per_leaf=False):
        total = sum(breakdown.categories.get(key, 0.0) for key in keys)
        calls = sum(breakdown.calls.get(key, 0) for key in keys)
        suffix = f", {total / sibling_count:.3f} ms/leaf" if per_leaf else ""
        print(f"  {label:<24} {total:>10.3f} ms ({calls} calls{suffix})")
        return total

    if path == "tpr":
        row("root_push", ("prefix_push.operation",))
        leaf = row("leaf_visit", ("branch_visit.operation",), per_leaf=True)
        leaf_backward = row("leaf_visit_backward", ("branch_visit.backward",))
        pop = row("root_pop", ("prefix_pop.operation",))
        pop_backward = row("root_pop_backward", ("prefix_pop.backward",))
    model_forward = row("model_forward", tuple(f"{phase}.model_forward" for phase in phases))
    if path == "reference":
        row("reference_backward", ("full.backward",))
    for category in ("attention_forward", "attention_backward", "mlp_forward", "mlp_backward",
                     "loss_forward", "fa_forward", "fa_backward", "comm_forward", "comm_backward"):
        row(category, tuple(f"{phase}.{category}" for phase in phases))
    row("gradient_finalize", ("gradient_finalize",))
    if path == "tpr":
        print(f"  leaf non-backward        {leaf - leaf_backward:>10.3f} ms (derived)")
        print(f"  root_pop non-backward    {pop - pop_backward:>10.3f} ms (derived)")
    backward = sum(breakdown.categories.get(f"{phase}.backward", 0.0) for phase in phases)
    print(f"  wall - model graph      {breakdown.wall_median_ms - model_forward - backward:>10.3f} ms (approximate)")


def _stats_payload(stats: _ProfileStats) -> dict:
    return {
        "median_ms": stats.median_ms,
        "mean_ms": stats.mean_ms,
        "std_ms": stats.std_ms,
        "baseline_allocated": stats.baseline_allocated_bytes,
        "peak_allocated": stats.peak_allocated_bytes,
        "incremental_peak": stats.incremental_peak_bytes,
        "peak_reserved": stats.peak_reserved_bytes,
        "parameter_count": stats.parameter_count,
        "adapter_probe_calls": stats.adapter_probe_calls,
        "layer_count": stats.layer_count,
        "physical_phase_count": stats.physical_phase_count,
    }


def _print_path_result(case: _ProfileCase, path: str, stats: _ProfileStats, *, rank: int) -> None:
    if rank != 0:
        return
    print("TPR_REF_TPR_PATH_RESULT " + json.dumps({
        "path": path,
        "cp_size": _EXPECTED_WORLD_SIZE,
        "prefix": case.prefix_length,
        "suffix": case.suffix_length,
        "siblings": case.trajectory_count,
        "offload": _ENABLE_OFFLOAD,
        "swap_modules": _SWAP_MODULES,
        "loss_chunk_size": _LOSS_CHUNK_SIZE,
        "stats": _stats_payload(stats),
    }), flush=True)


def _print_case_result(result: _ProfileResult, *, rank: int) -> None:
    if rank != 0:
        return
    case = result.case
    reference, tpr = result.reference, result.tpr
    assert reference.parameter_count == tpr.parameter_count
    print(f"\nP={case.prefix_length}, S={case.suffix_length}, N={case.trajectory_count}, CP={_EXPECTED_WORLD_SIZE}")
    print(f"Model parameters: {reference.parameter_count / 1e9:.3f}B")
    print("Latency: per-sample maximum across CP ranks; memory: maximum per-rank statistic.")
    print("Probe calls: attention adapter entries on rank 0; FA block calls describe Ring only.")
    for name, stats in (("Reference", reference), ("TPR", tpr)):
        print(f"{name}:")
        print(f"  median latency:     {stats.median_ms:.3f} ms")
        print(f"  mean latency:       {stats.mean_ms:.3f} ms")
        print(f"  std:                {stats.std_ms:.3f} ms")
        print(f"  baseline allocated: {_format_gib(stats.baseline_allocated_bytes)}")
        print(f"  peak allocated:     {_format_gib(stats.peak_allocated_bytes)}")
        print(f"  incremental peak:   {_format_gib(stats.incremental_peak_bytes)}")
        print(f"  peak reserved:      {_format_gib(stats.peak_reserved_bytes)} (auxiliary)")
        print(f"  adapter probe calls: {stats.adapter_probe_calls}")
    print(f"TPR adapter trace verified: {tpr.layer_count} layers, "
          f"{tpr.physical_phase_count} physical phases, {tpr.adapter_probe_calls} adapter calls")
    for name, path, breakdown in (("Reference", "reference", result.reference_breakdown),
                                  ("TPR", "tpr", result.tpr_breakdown)):
        if breakdown is not None:
            _print_controlled_breakdown(name, breakdown, path=path, sibling_count=case.trajectory_count)
    ratio = reference.incremental_peak_bytes / max(tpr.incremental_peak_bytes, 1)
    reduction = 1.0 - tpr.incremental_peak_bytes / max(reference.incremental_peak_bytes, 1)
    print(f"Speedup (median): {result.median_speedup:.3f}x")
    print(f"Incremental-peak ratio: {ratio:.3f}x")
    print(f"Incremental-peak reduction: {reduction * 100:.2f}%")
    print("TPR_REF_TPR_RESULT " + json.dumps({
        "cp_size": _EXPECTED_WORLD_SIZE,
        "prefix": case.prefix_length,
        "suffix": case.suffix_length,
        "siblings": case.trajectory_count,
        "offload": _ENABLE_OFFLOAD,
        "swap_modules": _SWAP_MODULES,
        "loss_chunk_size": _LOSS_CHUNK_SIZE,
        "reference": {
            "median_ms": reference.median_ms,
            "mean_ms": reference.mean_ms,
            "std_ms": reference.std_ms,
            "baseline_allocated": reference.baseline_allocated_bytes,
            "peak_allocated": reference.peak_allocated_bytes,
            "incremental_peak": reference.incremental_peak_bytes,
            "peak_reserved": reference.peak_reserved_bytes,
        },
        "tpr": {
            "median_ms": tpr.median_ms,
            "mean_ms": tpr.mean_ms,
            "std_ms": tpr.std_ms,
            "baseline_allocated": tpr.baseline_allocated_bytes,
            "peak_allocated": tpr.peak_allocated_bytes,
            "incremental_peak": tpr.incremental_peak_bytes,
            "peak_reserved": tpr.peak_reserved_bytes,
        },
        "speedup": result.median_speedup,
        "incremental_peak_reduction_pct": reduction * 100.0,
    }), flush=True)


def _print_summary(results: tuple[_ProfileResult, ...], *, rank: int) -> None:
    if rank != 0:
        return
    print(f"\nControlled fused-reference summary (CP={_EXPECTED_WORLD_SIZE})\n")
    print("| P | S | N | Ref median ms | TPR median ms | Speedup | Ref incr. GiB | TPR incr. GiB | Incr. reduction |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        case, reference, tpr = result.case, result.reference, result.tpr
        reduction = 1.0 - tpr.incremental_peak_bytes / max(reference.incremental_peak_bytes, 1)
        print(f"| {case.prefix_length} | {case.suffix_length} | {case.trajectory_count} | "
              f"{reference.median_ms:.3f} | {tpr.median_ms:.3f} | {result.median_speedup:.3f}x | "
              f"{reference.incremental_peak_bytes / _GIB:.3f} | {tpr.incremental_peak_bytes / _GIB:.3f} | "
              f"{reduction * 100:.2f}% |")


def test_qwen3_reference_cp_vs_tpr_ring_cp_profile(profile_runtime):
    runtime = profile_runtime
    if runtime.cp_size != _EXPECTED_WORLD_SIZE:
        raise AssertionError(
            f"Ring profile requires CP={_EXPECTED_WORLD_SIZE}, got {runtime.cp_size}"
        )

    target = resolve_qwen3_profile_target()
    size = target.size
    model_path = target.path
    hf_config = target.hf_config
    model_case = SimpleNamespace(path=model_path)
    model = _make_qwen_cp_model(runtime, model_case, hf_config)
    config = model.config
    config.swap_attention = _ENABLE_OFFLOAD
    config.swap_modules = _SWAP_MODULES
    assert config.context_parallel_size == runtime.cp_size
    for actual, expected in (
        (config.num_layers, hf_config.num_hidden_layers),
        (config.hidden_size, hf_config.hidden_size),
        (config.ffn_hidden_size, hf_config.intermediate_size),
        (config.num_attention_heads, hf_config.num_attention_heads),
        (config.num_query_groups, hf_config.num_key_value_heads),
        (config.kv_channels, hf_config.head_dim),
    ):
        assert actual == expected
    assert len(model.decoder.layers) == hf_config.num_hidden_layers
    assert model.share_embeddings_and_output_weights == hf_config.tie_word_embeddings
    parameters = target.assert_model_scale(model)
    if runtime.rank == 0:
        print(f"Qwen3-{size}: checkpoint={model_path}, CP={runtime.cp_size}")
        print(f"Prefix FULL coalescing: {os.getenv('TPR_RING_COALESCE_PREFIX_FULL', '0') == '1'}")
        print(f"Prefix Q coalescing requested: {os.getenv('TPR_RING_COALESCE_PREFIX_QUERY', '0') == '1'}")
        print(f"Activation offload: {_ENABLE_OFFLOAD}; swap_modules={_SWAP_MODULES}")
        print(f"Loss chunk size: {_LOSS_CHUNK_SIZE}")
        print(f"Profile cases: {[case.case_id for case in _PROFILE_CASES]}")
        print(f"Profile path: {_PROFILE_PATH}")
        if runtime.cp_size == 1:
            print("Local rectangular attention; Ring-only diagnostic counters are zero.")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise AssertionError("Ring CP profile model must use BF16 parameters")
    expected_layer_numbers = tuple(
        layer.self_attention.layer_number for layer in model.decoder.layers
    )
    dist.barrier(group=runtime.cp_group)
    torch.npu.synchronize()

    results = []
    for case in _PROFILE_CASES:
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

        if _PROFILE_PATH in ("both", "reference"):
            if runtime.rank == 0:
                print("TPR_REF_TPR_STAGE " + json.dumps({
                    "path": "reference", "cp_size": runtime.cp_size,
                    "prefix": case.prefix_length, "suffix": case.suffix_length,
                    "siblings": case.trajectory_count,
                }), flush=True)
            reference_stats = _profile_runner(
                reference_runner,
                model,
                runtime,
                expected_trace=reference_trace,
            )
            _release_iteration_state(model, runtime)
            _print_path_result(case, "reference", reference_stats, rank=runtime.rank)
        else:
            reference_stats = None

        if _PROFILE_PATH in ("both", "tpr"):
            if runtime.rank == 0:
                print("TPR_REF_TPR_STAGE " + json.dumps({
                    "path": "tpr", "cp_size": runtime.cp_size,
                    "prefix": case.prefix_length, "suffix": case.suffix_length,
                    "siblings": case.trajectory_count,
                }), flush=True)
            tpr_stats = _profile_runner(
                tpr_runner,
                model,
                runtime,
                expected_trace=tpr_trace,
            )
            _release_iteration_state(model, runtime)
            _print_path_result(case, "tpr", tpr_stats, rank=runtime.rank)
        else:
            tpr_stats = None

        # Standalone path mode is the formal capacity/performance mode. Each
        # side runs in a fresh torchrun, so an OOM on one path cannot suppress
        # the other. Combined mode retains the historical in-process comparison.
        if _PROFILE_PATH != "both":
            continue

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

    if _PROFILE_PATH == "both":
        final_results = tuple(results)
        _print_breakdown_summary(final_results, rank=runtime.rank)
        _print_summary(final_results, rank=runtime.rank)
