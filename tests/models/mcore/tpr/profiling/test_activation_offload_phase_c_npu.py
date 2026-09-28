"""Phase C1 Prefix-graph offload acceptance on one NPU.

This module is intentionally independent from the Phase A/B entry points.  It
compares the existing recompute policy with the retained Push-graph policy and
observes the real MindSpeed SwapPrefetch transfers.  Model/spec imports stay
below the MindSpeed bootstrap fixture because importing them during collection
caches incomplete backend providers on the target server.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gc
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_OFFLOAD_C") != "1",
    reason="Set TPR_RUN_OFFLOAD_C=1",
)


@pytest.fixture(scope="module")
def runtime():
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    assert int(os.getenv("WORLD_SIZE", "1")) == 1, "Phase C1 is CP=1 only"
    torch.npu.set_device(int(os.getenv("LOCAL_RANK", "0")))
    dist.init_process_group(backend="hccl")
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    from mindspeed.args_utils import get_full_args

    vars(get_full_args()).pop("", None)
    repatch(
        dict(
            context_parallel_size=1,
            experimental_attention_variant=None,
            use_flash_attn=True,
        )
    )
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(123)
    try:
        yield SimpleNamespace(
            device=torch.device("npu", torch.npu.current_device()),
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
        )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.fixture
def native_args(runtime, monkeypatch):
    from mindspeed import args_utils
    from verl.models.mcore.tpr import activation_offload

    args = SimpleNamespace(**vars(args_utils.get_full_args()))
    vars(args).update(
        swap_attention=True,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        eval_interval=0,
        curr_iteration=1,
        noop_layers=None,
        swap_modules=os.getenv("TPR_SWAP_MODULES", "self_attention,mlp"),
    )
    monkeypatch.setattr(activation_offload, "_args", lambda: args)
    prefetch = activation_offload._native_prefetch()
    monkeypatch.setattr(prefetch, "get_args", lambda: args)
    return args


def _make_model(runtime, monkeypatch, *, max_sequence_length):
    from ._qwen3_profile_target import resolve_qwen3_profile_target
    from ..correctness import test_tpr_qwen3_compatibility_npu as qwen_fixture
    from ..correctness.test_tpr_qwen3_compatibility_npu import _make_qwen_model
    from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention
    from verl.models.mcore.tpr.attention import TPRSelfAttention

    target = resolve_qwen3_profile_target()
    monkeypatch.setattr(qwen_fixture, "QWEN_MODEL_PATH", target.path)
    model = _make_qwen_model(
        runtime.device,
        tpr=True,
        max_sequence_length=max_sequence_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    parameter_count = target.assert_model_scale(model)
    model.config.cross_entropy_loss_fusion = True
    model.config.cross_entropy_fusion_impl = "native"
    assert model.config.context_parallel_size == 1
    assert getattr(model.config, "attention_dropout", 0.0) == 0.0
    assert getattr(model.config, "hidden_dropout", 0.0) == 0.0
    assert all(
        isinstance(layer.self_attention, TPRSelfAttention)
        and getattr(layer.self_attention, "tpr_state_kind", None) != "gdn"
        for layer in model.decoder.layers
    )
    return model, target, parameter_count


def _internal_terms(tokens, *, weight=1.0, sample_id=None):
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm

    return tuple(
        SegmentLossTerm(index, int(tokens[index + 1]), weight=weight, sample_id=sample_id)
        for index in range(tokens.numel() - 1)
    )


def _flat_plan(prefix_length, suffix_length, *, siblings=2):
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec

    prefix = torch.arange(prefix_length, dtype=torch.long) % 2048
    suffixes = tuple(
        (torch.arange(suffix_length, dtype=torch.long) + 37 + 56 * index) % 2048
        for index in range(siblings)
    )
    root_terms = _internal_terms(prefix, weight=float(siblings)) + tuple(
        SegmentLossTerm(prefix_length - 1, int(suffix[0]), sample_id=index + 1)
        for index, suffix in enumerate(suffixes)
    )
    segments = [SegmentSpec(0, None, prefix, 0, 0, root_terms)]
    segments.extend(
        SegmentSpec(
            index + 1,
            0,
            suffix,
            prefix_length,
            prefix_length,
            _internal_terms(suffix, sample_id=index + 1),
        )
        for index, suffix in enumerate(suffixes)
    )
    return SegmentPlan(segments, root_id=0)


def _nested_plan(prefix_length, suffix_length):
    """root -> middle -> two leaves, plus one root leaf."""
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec

    root = torch.arange(prefix_length, dtype=torch.long) % 2048
    middle = (torch.arange(suffix_length, dtype=torch.long) + 37) % 2048
    leaves = tuple(
        (torch.arange(suffix_length, dtype=torch.long) + offset) % 2048
        for offset in (93, 149, 205)
    )
    root_terms = _internal_terms(root, weight=3.0) + (
        SegmentLossTerm(prefix_length - 1, int(middle[0]), sample_id=2),
        SegmentLossTerm(prefix_length - 1, int(middle[0]), sample_id=3),
        SegmentLossTerm(prefix_length - 1, int(leaves[2][0]), sample_id=4),
    )
    middle_terms = _internal_terms(middle, weight=2.0) + (
        SegmentLossTerm(suffix_length - 1, int(leaves[0][0]), sample_id=2),
        SegmentLossTerm(suffix_length - 1, int(leaves[1][0]), sample_id=3),
    )
    root_end = prefix_length
    middle_end = prefix_length + suffix_length
    return SegmentPlan(
        [
            SegmentSpec(0, None, root, 0, 0, root_terms),
            SegmentSpec(1, 0, middle, root_end, root_end, middle_terms),
            SegmentSpec(2, 1, leaves[0], middle_end, middle_end, _internal_terms(leaves[0], sample_id=2)),
            SegmentSpec(3, 1, leaves[1], middle_end, middle_end, _internal_terms(leaves[1], sample_id=3)),
            SegmentSpec(4, 0, leaves[2], root_end, root_end, _internal_terms(leaves[2], sample_id=4)),
        ],
        root_id=0,
    )


class _NativeTransferProbe:
    def __init__(self, monkeypatch):
        from mindspeed.core.memory.swap_attention.prefetch import SwapTensor

        self.counts = {"d2h_bytes": 0, "h2d_bytes": 0}
        self._payload_refs = []
        self._peak_live_pinned_payload_bytes = 0
        release = SwapTensor.wait_d2h_finished
        reload = SwapTensor.launch_h2d

        def released(item, *args, **kwargs):
            before = item.stat
            result = release(item, *args, **kwargs)
            if before == "d2h" and item.stat == "host":
                assert item.tensor.storage().size() == 0
                assert item.tensor_cpu.is_pinned()
                size = item.storage_size * item.tensor.element_size()
                self.counts["d2h_bytes"] += size
                self._payload_refs.append((size, weakref.ref(item.tensor_cpu)))
                self._peak_live_pinned_payload_bytes = max(
                    self._peak_live_pinned_payload_bytes,
                    self.live_pinned_payload_bytes(),
                )
            return result

        def reloaded(item, *args, **kwargs):
            before = item.stat
            result = reload(item, *args, **kwargs)
            if before == "host" and item.stat == "h2d":
                self.counts["h2d_bytes"] += item.storage_size * item.tensor.element_size()
            return result

        monkeypatch.setattr(SwapTensor, "wait_d2h_finished", released)
        monkeypatch.setattr(SwapTensor, "launch_h2d", reloaded)

    def snapshot(self):
        return dict(self.counts)

    def live_pinned_payload_bytes(self):
        return sum(size for size, reference in self._payload_refs if reference() is not None)

    def reset_peak_live_pinned_payload_bytes(self):
        self._peak_live_pinned_payload_bytes = self.live_pinned_payload_bytes()

    def peak_live_pinned_payload_bytes(self):
        return self._peak_live_pinned_payload_bytes


def _delta(after, before):
    return {name: after[name] - before[name] for name in before}


def _proc_memory():
    result = {"rss_bytes": None, "pinned_bytes": None}
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                name, _, value = line.partition(":")
                if name in ("VmRSS", "VmPin"):
                    kib = int(value.strip().split()[0])
                    result["rss_bytes" if name == "VmRSS" else "pinned_bytes"] = kib * 1024
    except OSError:
        pass
    return result


@dataclass
class _RunResult:
    tensors: tuple[dict, dict, dict, dict] | None
    total_model_forwards: int
    stage_rows: tuple[dict, ...]
    pop_parameter_contributions: dict[str, torch.Tensor] | None = None


def _run(
    model,
    plan,
    *,
    policy,
    probe,
    collect=True,
    trace_pop_parameter_names: tuple[str, ...] = (),
    emit_stage_rows=True,
    profile_stage_memory=False,
):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor

    model.zero_grad(set_to_none=True)
    logs = {}
    boundaries = {}
    pop_parameter_contributions = {}
    stage_rows = []
    model_forwards = defaultdict(int)
    current_stage = ["outside"]

    named_parameters = dict(model.named_parameters())
    missing_trace_parameters = [
        name for name in trace_pop_parameter_names if name not in named_parameters
    ]
    if missing_trace_parameters:
        raise KeyError(
            "unknown traced Phase C parameter(s): "
            + ", ".join(missing_trace_parameters)
        )

    def snapshot_traced_parameter_grads():
        snapshots = {}
        for name in trace_pop_parameter_names:
            parameter = named_parameters[name]
            if parameter.grad is None:
                snapshots[name] = torch.zeros_like(parameter.detach(), device="cpu")
            else:
                snapshots[name] = parameter.grad.detach().cpu().clone()
        return snapshots

    def count_model_forward(module, inputs):
        model_forwards[current_stage[0]] += 1

    model_hook = model.register_forward_pre_hook(count_model_forward)

    class ObservedExecutor(SegmentExecutor):
        def _stage_call(self, action, segment_id, function):
            stage = f"{action}.{segment_id}"
            previous = current_stage[0]
            current_stage[0] = stage
            before = probe.snapshot()
            before_forwards = model_forwards[stage]
            memory = {}
            if profile_stage_memory:
                torch.npu.synchronize()
                memory.update(
                    allocated_before_bytes=int(torch.npu.memory_allocated()),
                    reserved_before_bytes=int(torch.npu.memory_reserved()),
                    live_pinned_before_bytes=probe.live_pinned_payload_bytes(),
                )
                torch.npu.reset_peak_memory_stats()
                probe.reset_peak_live_pinned_payload_bytes()
            stage_start = time.perf_counter()
            if emit_stage_rows:
                print(
                    "TPR_PHASE_C_STAGE "
                    + json.dumps(dict(policy=policy, stage=stage, event="begin")),
                    flush=True,
                )
            try:
                result = function()
                torch.npu.synchronize()
            except Exception as error:
                print(
                    "TPR_PHASE_C_STAGE "
                    + json.dumps(
                        dict(
                            policy=policy,
                            stage=stage,
                            event="failure",
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    ),
                    flush=True,
                )
                raise
            finally:
                current_stage[0] = previous
            stage_latency_ms = (time.perf_counter() - stage_start) * 1000.0
            transfers = _delta(probe.snapshot(), before)
            if profile_stage_memory:
                memory.update(
                    allocated_after_bytes=int(torch.npu.memory_allocated()),
                    reserved_after_bytes=int(torch.npu.memory_reserved()),
                    peak_allocated_bytes=int(torch.npu.max_memory_allocated()),
                    peak_reserved_bytes=int(torch.npu.max_memory_reserved()),
                    peak_live_pinned_payload_bytes=probe.peak_live_pinned_payload_bytes(),
                    live_pinned_after_bytes=probe.live_pinned_payload_bytes(),
                )
                memory["allocated_delta_bytes"] = (
                    memory["allocated_after_bytes"] - memory["allocated_before_bytes"]
                )
                memory["reserved_delta_bytes"] = (
                    memory["reserved_after_bytes"] - memory["reserved_before_bytes"]
                )
                memory["stage_incremental_peak_allocated_bytes"] = (
                    memory["peak_allocated_bytes"] - memory["allocated_before_bytes"]
                )
                memory["stage_incremental_peak_reserved_bytes"] = (
                    memory["peak_reserved_bytes"] - memory["reserved_before_bytes"]
                )
            row = dict(
                policy=policy,
                stage=stage,
                event="end",
                latency_ms=stage_latency_ms,
                model_forwards=model_forwards[stage] - before_forwards,
                **memory,
                **transfers,
            )
            stage_rows.append(row)
            if emit_stage_rows:
                print("TPR_PHASE_C_STAGE " + json.dumps(row), flush=True)
            return result

        def push(self, segment_id):
            return self._stage_call(
                "push", segment_id, lambda: super(ObservedExecutor, self).push(segment_id)
            )

        def visit_leaf(self, segment_id):
            return self._stage_call(
                "visit", segment_id, lambda: super(ObservedExecutor, self).visit_leaf(segment_id)
            )

        def pop(self, segment_id):
            return self._stage_call(
                "pop", segment_id, lambda: super(ObservedExecutor, self).pop(segment_id)
            )

        def _compute_loss(self, segment, logits):
            if collect and segment.loss_terms:
                assert isinstance(logits, torch.Tensor), "C1 correctness expects unchunked logits"
                indices = torch.tensor(
                    [term.query_offset for term in segment.loss_terms],
                    dtype=torch.long,
                    device=logits.device,
                )
                targets = torch.tensor(
                    [term.target_token_id for term in segment.loss_terms],
                    dtype=torch.long,
                    device=logits.device,
                )
                # Diagnostic logprobs must not add their own saved tensors to
                # the retained Prefix session under observation.
                with torch.no_grad():
                    logs[segment.segment_id] = -F.cross_entropy(
                        logits[0].index_select(0, indices).float(),
                        targets,
                        reduction="none",
                    ).cpu()
            return super()._compute_loss(segment, logits)

    loss_chunk_size_env = os.getenv("TPR_PHASE_C_LOSS_CHUNK_SIZE")
    loss_chunk_size = None if loss_chunk_size_env in (None, "") else int(loss_chunk_size_env)
    executor = ObservedExecutor(
        model,
        plan,
        prefix_backward_policy=policy,
        loss_chunk_size=loss_chunk_size,
    )
    losses = []

    def snapshot_prefix_gradients(segment_id):
        if not collect:
            return
        entry = executor.kv_stack.get(segment_id)
        for layer, pair in entry.gradients.items():
            for name, value in zip(("key", "value"), pair):
                boundaries[f"segment={segment_id}.layer={layer}.{name}"] = value.detach().cpu().clone()

    def execute(segment_id):
        children = plan.children_of(segment_id)
        if not children:
            losses.append(executor.visit_leaf(segment_id).backward.normalized_loss.detach())
            return
        executor.push(segment_id)
        for child in children:
            execute(child.segment_id)
        snapshot_prefix_gradients(segment_id)
        before_pop = snapshot_traced_parameter_grads()
        losses.append(executor.pop(segment_id).normalized_loss.detach())
        after_pop = snapshot_traced_parameter_grads()
        for name in trace_pop_parameter_names:
            pop_parameter_contributions[f"segment={segment_id}.{name}"] = (
                after_pop[name] - before_pop[name]
            )

    try:
        execute(plan.root_id)
        executor.kv_stack.assert_empty()
        assert not executor.gdn_states
        executor.assert_prefix_graphs_empty()
    except Exception:
        executor.abort()
        executor.assert_prefix_graphs_empty()
        raise
    finally:
        model_hook.remove()

    _assert_stage_contract(policy, stage_rows)
    expected_forwards = len(plan.segments) + (
        sum(bool(plan.children_of(segment_id)) for segment_id in plan.segments)
        if policy == "recompute"
        else 0
    )
    total_forwards = sum(model_forwards.values())
    assert total_forwards == expected_forwards, (
        f"{policy}: expected {expected_forwards} model forwards, got {dict(model_forwards)}"
    )

    if not collect:
        return _RunResult(None, total_forwards, tuple(stage_rows), None)
    grads = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_grads = sorted(trainable - grads.keys())
    assert not missing_grads, f"{policy}: trainable parameters without gradients: {missing_grads[:20]}"
    tensors = (
        {"total": torch.stack(losses).sum().cpu()},
        logs,
        grads,
        boundaries,
    )
    return _RunResult(
        tensors,
        total_forwards,
        tuple(stage_rows),
        pop_parameter_contributions,
    )


def _assert_stage_contract(policy, stage_rows):
    assert policy in ("recompute", "offload")
    for row in stage_rows:
        action = row["stage"].split(".", 1)[0]
        if action == "push":
            assert row["model_forwards"] == 1
            if policy == "recompute":
                assert row["d2h_bytes"] == 0 and row["h2d_bytes"] == 0
            else:
                assert row["d2h_bytes"] > 0 and row["h2d_bytes"] == 0
        elif action == "visit":
            assert row["model_forwards"] == 1
            assert row["d2h_bytes"] > 0 and row["h2d_bytes"] > 0
        elif action == "pop":
            if policy == "recompute":
                assert row["model_forwards"] == 1
                assert row["d2h_bytes"] > 0 and row["h2d_bytes"] > 0
            else:
                assert row["model_forwards"] == 0
                assert row["d2h_bytes"] == 0 and row["h2d_bytes"] > 0
        else:
            raise AssertionError(f"unknown Phase C stage {row['stage']}")


def _difference_metrics(expected, actual):
    expected = expected.reshape(-1)
    actual = actual.reshape(-1)
    max_abs = squared_error = squared_reference = 0.0
    mismatched = 0
    finite = True
    for offset in range(0, expected.numel(), 1024 * 1024):
        left = expected[offset : offset + 1024 * 1024].float()
        right = actual[offset : offset + 1024 * 1024].float()
        diff = (right - left).abs()
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        max_abs = max(max_abs, diff.max().item())
        mismatched += int((diff > 2e-4 + 2e-3 * left.abs()).sum().item())
        squared_error += diff.square().sum(dtype=torch.float64).item()
        squared_reference += left.square().sum(dtype=torch.float64).item()
    return dict(
        max_abs=max_abs,
        relative_l2=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
        mismatch_fraction=mismatched / max(expected.numel(), 1),
        mismatched=mismatched,
        elements=expected.numel(),
        finite=finite,
    )


def _build_parameter_noise_baseline(reference, repeats, *, case):
    from ._offload_b_gate import merge_baseline

    baseline = {}
    for repeat_index, repeat in enumerate(repeats, start=1):
        assert reference.keys() == repeat.keys(), (
            f"{case}/parameter_grad calibration keys changed on repeat {repeat_index}"
        )
        for name, expected in reference.items():
            value = repeat[name]
            if expected.shape != value.shape:
                raise AssertionError(
                    f"{case}/parameter_grad/{name}: calibration shape changed "
                    f"{tuple(expected.shape)} -> {tuple(value.shape)}"
                )
            metrics = _difference_metrics(expected, value)
            if not metrics["finite"]:
                raise AssertionError(
                    f"{case}/parameter_grad/{name}: non-finite calibration repeat"
                )
            baseline[name] = merge_baseline(baseline.get(name), metrics)

    worst = sorted(
        (
            (str(name), metrics)
            for name, metrics in baseline.items()
            if metrics["mismatched"] > 0
        ),
        key=lambda item: item[1]["relative_l2"],
        reverse=True,
    )[:5]
    print(
        "TPR_PHASE_C_CALIBRATION "
        + json.dumps(
            dict(
                case=case,
                repeats=len(repeats),
                checked=len(reference),
                noisy=sum(metrics["mismatched"] > 0 for metrics in baseline.values()),
                worst=worst,
            )
        ),
        flush=True,
    )
    return baseline


def _extend_small_parameter_noise_baseline(
    reference,
    repeat,
    baseline,
    *,
    case,
    repeat_index,
):
    from ._offload_b_gate import is_small_tensor, merge_baseline

    assert reference.keys() == repeat.keys(), (
        f"{case}/parameter_grad small calibration keys changed on repeat {repeat_index}"
    )
    updated = 0
    for name, expected in reference.items():
        value = repeat[name]
        if expected.shape != value.shape:
            raise AssertionError(
                f"{case}/parameter_grad/{name}: small calibration shape changed "
                f"{tuple(expected.shape)} -> {tuple(value.shape)}"
            )
        metrics = _difference_metrics(expected, value)
        if not metrics["finite"]:
            raise AssertionError(
                f"{case}/parameter_grad/{name}: non-finite small calibration repeat"
            )
        if not is_small_tensor(metrics):
            continue
        previous = baseline.get(name)
        merged = merge_baseline(previous, metrics)
        baseline[name] = merged
        updated += int(previous is None or merged["relative_l2"] > previous["relative_l2"])

    print(
        "TPR_PHASE_C_SMALL_CALIBRATION "
        + json.dumps(
            dict(
                case=case,
                repeat=repeat_index,
                checked=sum(value.numel() < 4096 for value in reference.values()),
                updated=updated,
            )
        ),
        flush=True,
    )


def _assert_acceptance_match(reference, actual, parameter_baseline, *, case, run):
    from ._offload_b_gate import gradient_gate

    failures = []
    for category, expected_map, actual_map in zip(
        ("loss", "logprob", "parameter_grad", "prefix_grad"),
        reference,
        actual,
    ):
        assert expected_map.keys() == actual_map.keys(), (
            f"{case}/{run}/{category}: tensor keys changed"
        )
        assert expected_map, f"{case}/{run}/{category}: empty comparison"
        failed = []
        for name, expected in expected_map.items():
            value = actual_map[name]
            if expected.shape != value.shape:
                metrics = {
                    "shape_expected": tuple(expected.shape),
                    "shape_actual": tuple(value.shape),
                }
                failed.append((str(name), metrics))
                failures.append(f"{case}/{run}/{category}/{name}")
                continue

            metrics = _difference_metrics(expected, value)
            if category == "parameter_grad":
                passed, limits, severity = gradient_gate(
                    metrics,
                    parameter_baseline.get(name),
                )
                if not passed:
                    detail = dict(
                        **metrics,
                        limits=limits,
                        severity=severity,
                        baseline=parameter_baseline.get(name),
                    )
                    failed.append((str(name), detail))
                    failures.append(f"{case}/{run}/{category}/{name}")
            else:
                try:
                    torch.testing.assert_close(value, expected, rtol=2e-3, atol=2e-4)
                except AssertionError:
                    failed.append((str(name), metrics))
                    failures.append(f"{case}/{run}/{category}/{name}")

        worst = sorted(
            failed,
            key=lambda item: item[1].get("severity", item[1].get("relative_l2", float("inf"))),
            reverse=True,
        )[:5]
        print(
            "TPR_PHASE_C_COMPARE "
            + json.dumps(
                dict(
                    case=case,
                    run=run,
                    category=category,
                    gate=("baseline_aware" if category == "parameter_grad" else "strict"),
                    checked=len(expected_map),
                    failed=len(failed),
                    worst=worst,
                )
            ),
            flush=True,
        )
    assert not failures, "Phase C1 acceptance failures: " + ", ".join(failures[:10])


def _repeat_gradient_summary(expected, actual, *, comparison):
    assert expected.keys() == actual.keys(), f"{comparison}: parameter gradient keys changed"
    failed = []
    finite = True
    for name, reference in expected.items():
        value = actual[name]
        if reference.shape != value.shape:
            failed.append((str(name), {"shape_mismatch": True}))
            finite = False
            continue
        metrics = _difference_metrics(reference, value)
        finite = finite and metrics["finite"]
        if metrics["mismatch_fraction"] > 0:
            failed.append((str(name), metrics))
    worst = sorted(
        failed,
        key=lambda item: item[1].get("relative_l2", float("inf")),
        reverse=True,
    )[:5]
    summary = dict(
        comparison=comparison,
        checked=len(expected),
        failed_strict=len(failed),
        finite=finite,
        worst=worst,
    )
    print("TPR_PHASE_C_GRAD_REPEAT " + json.dumps(summary), flush=True)
    return summary


def _pop_contribution_summary(expected, actual, *, comparison):
    assert expected is not None and actual is not None
    assert expected.keys() == actual.keys(), f"{comparison}: Pop contribution keys changed"
    failed = []
    finite = True
    for name, reference in expected.items():
        value = actual[name]
        metrics = _difference_metrics(reference, value)
        finite = finite and metrics["finite"]
        if metrics["mismatch_fraction"] > 0:
            failed.append((name, metrics))
    worst = sorted(
        failed,
        key=lambda item: item[1]["relative_l2"],
        reverse=True,
    )[:5]
    summary = dict(
        comparison=comparison,
        checked=len(expected),
        failed_strict=len(failed),
        finite=finite,
        worst=worst,
    )
    print("TPR_PHASE_C_PREFIX_PARAM " + json.dumps(summary), flush=True)
    return summary


def test_phase_c1_parameter_gradient_repeat_diagnostics(
    runtime,
    native_args,
    monkeypatch,
):
    if os.getenv("TPR_PHASE_C_GRAD_DIAGNOSTIC") != "1":
        pytest.skip("Set TPR_PHASE_C_GRAD_DIAGNOSTIC=1")

    prefix = int(os.getenv("TPR_PHASE_C_PREFIX", "1024"))
    suffix = int(os.getenv("TPR_PHASE_C_SUFFIX", "1024"))
    plan = _flat_plan(prefix, suffix)
    trace_default = (
        "decoder.layers.0.input_layernorm.weight,"
        "decoder.layers.0.self_attention.q_layernorm.weight,"
        "decoder.layers.0.self_attention.linear_qkv.weight,"
        "decoder.layers.0.mlp.linear_fc1.weight"
    )
    trace_names = tuple(
        name.strip()
        for name in os.getenv(
            "TPR_PHASE_C_TRACE_PARAMETERS",
            trace_default,
        ).split(",")
        if name.strip()
    )

    torch.manual_seed(123)
    model, target, parameter_count = _make_model(
        runtime,
        monkeypatch,
        max_sequence_length=prefix + suffix,
    )
    model.config.swap_attention = True
    model.config.swap_modules = native_args.swap_modules
    probe = _NativeTransferProbe(monkeypatch)

    runs = {}
    for label, policy in (
        ("recompute1", "recompute"),
        ("recompute2", "recompute"),
        ("offload1", "offload"),
        ("offload2", "offload"),
    ):
        runs[label] = _run(
            model,
            plan,
            policy=policy,
            probe=probe,
            trace_pop_parameter_names=trace_names,
        )
        assert runs[label].tensors is not None

    parameter_summaries = [
        _repeat_gradient_summary(
            runs["recompute1"].tensors[2],
            runs["recompute2"].tensors[2],
            comparison="recompute1_vs_recompute2",
        ),
        _repeat_gradient_summary(
            runs["offload1"].tensors[2],
            runs["offload2"].tensors[2],
            comparison="offload1_vs_offload2",
        ),
        _repeat_gradient_summary(
            runs["recompute1"].tensors[2],
            runs["offload1"].tensors[2],
            comparison="recompute1_vs_offload1",
        ),
        _repeat_gradient_summary(
            runs["recompute2"].tensors[2],
            runs["offload2"].tensors[2],
            comparison="recompute2_vs_offload2",
        ),
    ]
    contribution_summaries = [
        _pop_contribution_summary(
            runs["recompute1"].pop_parameter_contributions,
            runs["offload1"].pop_parameter_contributions,
            comparison="prefix_recompute1_vs_offload1",
        ),
        _pop_contribution_summary(
            runs["recompute2"].pop_parameter_contributions,
            runs["offload2"].pop_parameter_contributions,
            comparison="prefix_recompute2_vs_offload2",
        ),
    ]

    assert all(row["finite"] for row in parameter_summaries + contribution_summaries)
    print(
        "TPR_PHASE_C_GRAD_DIAGNOSTIC "
        + json.dumps(
            dict(
                model=target.label,
                checkpoint=str(target.path),
                parameter_count=parameter_count,
                prefix=prefix,
                suffix=suffix,
                traced_parameters=trace_names,
                parameter_summaries=parameter_summaries,
                contribution_summaries=contribution_summaries,
            )
        ),
        flush=True,
    )


@pytest.mark.parametrize("case", ["flat", "nested"])
def test_phase_c1_correctness(runtime, native_args, monkeypatch, case):
    prefix = int(os.getenv("TPR_PHASE_C_PREFIX", "1024"))
    suffix = int(os.getenv("TPR_PHASE_C_SUFFIX", "1024"))
    plan = _flat_plan(prefix, suffix) if case == "flat" else _nested_plan(prefix, suffix)
    max_sequence_length = prefix + suffix if case == "flat" else prefix + 2 * suffix

    torch.manual_seed(123)
    model, target, parameter_count = _make_model(
        runtime,
        monkeypatch,
        max_sequence_length=max_sequence_length,
    )
    model.config.swap_attention = True
    model.config.swap_modules = native_args.swap_modules
    probe = _NativeTransferProbe(monkeypatch)

    recompute1 = _run(model, plan, policy="recompute", probe=probe)
    recompute2 = _run(model, plan, policy="recompute", probe=probe)
    recompute3 = _run(model, plan, policy="recompute", probe=probe)

    assert (
        recompute1.tensors is not None
        and recompute2.tensors is not None
        and recompute3.tensors is not None
    )
    reference = recompute1.tensors
    parameter_baseline = _build_parameter_noise_baseline(
        reference[2],
        (recompute2.tensors[2], recompute3.tensors[2]),
        case=case,
    )

    # Phase B observed a longer-tail repeat distribution for tiny gradients
    # such as q/k LayerNorm weights. Calibrate that existing envelope with
    # recompute-only samples; offload never contributes to its own budget.
    small_calibration_repeats = int(
        os.getenv("TPR_PHASE_C_SMALL_CALIBRATION_REPEATS", "12")
    )
    if small_calibration_repeats < 0:
        raise ValueError(
            "TPR_PHASE_C_SMALL_CALIBRATION_REPEATS must be non-negative"
        )
    for repeat_index in range(1, small_calibration_repeats + 1):
        small_repeat = _run(model, plan, policy="recompute", probe=probe)
        assert small_repeat.tensors is not None
        _extend_small_parameter_noise_baseline(
            reference[2],
            small_repeat.tensors[2],
            parameter_baseline,
            case=case,
            repeat_index=repeat_index,
        )
        del small_repeat

    offload1 = _run(model, plan, policy="offload", probe=probe)
    offload2 = _run(model, plan, policy="offload", probe=probe)
    recompute_after = _run(model, plan, policy="recompute", probe=probe)
    runs = {
        "recompute2": recompute2,
        "recompute3": recompute3,
        "offload1": offload1,
        "offload2": offload2,
        "recompute_after": recompute_after,
    }
    assert all(result.tensors is not None for result in runs.values())

    # The two general calibration repeats must themselves satisfy the final
    # envelope, while loss/logprob/Prefix dKV stay under the strict gate.
    for label in ("recompute2", "recompute3"):
        _assert_acceptance_match(
            reference,
            runs[label].tensors,
            parameter_baseline,
            case=case,
            run=label,
        )

    # Held-out acceptance: neither offload nor recompute_after ever calibrates.
    for label in ("offload1", "offload2", "recompute_after"):
        _assert_acceptance_match(
            reference,
            runs[label].tensors,
            parameter_baseline,
            case=case,
            run=label,
        )

    if case == "flat":
        sibling_count = len(plan.children_of(plan.root_id))
        assert recompute1.total_model_forwards == sibling_count + 2
        assert offload1.total_model_forwards == sibling_count + 1
    print(
        "TPR_PHASE_C_CORRECTNESS "
        + json.dumps(
            dict(
                case=case,
                model=target.label,
                checkpoint=str(target.path),
                parameter_count=parameter_count,
                recompute_forwards=recompute1.total_model_forwards,
                offload_forwards=offload1.total_model_forwards,
                calibration_repeats=2,
                small_calibration_repeats=small_calibration_repeats,
                offload_repeats=2,
                recompute_after=True,
                **probe.snapshot(),
            )
        ),
        flush=True,
    )


def test_phase_c1_performance(runtime, native_args, monkeypatch):
    if os.getenv("TPR_PHASE_C_PERF") != "1":
        pytest.skip("Set TPR_PHASE_C_PERF=1")

    policy = os.getenv("TPR_PHASE_C_PERF_POLICY", "")
    if policy not in ("recompute", "offload"):
        raise ValueError(
            "TPR_PHASE_C_PERF_POLICY must be 'recompute' or 'offload'"
        )
    prefix = int(os.getenv("TPR_PHASE_C_PERF_PREFIX", "8192"))
    suffix = int(os.getenv("TPR_PHASE_C_PERF_SUFFIX", "1024"))
    siblings = int(os.getenv("TPR_PHASE_C_PERF_SIBLINGS", "2"))
    warmup = int(os.getenv("TPR_PHASE_C_PERF_WARMUP", "1"))
    repeats = int(os.getenv("TPR_PHASE_C_PERF_REPEATS", "3"))
    if prefix <= 0 or suffix <= 0 or siblings <= 0:
        raise ValueError("Phase C perf lengths/siblings must be positive")
    if warmup < 0 or repeats <= 0:
        raise ValueError("Phase C perf requires warmup >= 0 and repeats > 0")

    plan = _flat_plan(prefix, suffix, siblings=siblings)
    torch.manual_seed(123)
    model, target, parameter_count = _make_model(
        runtime,
        monkeypatch,
        max_sequence_length=prefix + suffix,
    )
    model.config.swap_attention = True
    model.config.swap_modules = native_args.swap_modules
    probe = _NativeTransferProbe(monkeypatch)

    for _ in range(warmup):
        _run(
            model,
            plan,
            policy=policy,
            probe=probe,
            collect=False,
            emit_stage_rows=False,
        )
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

    samples = []
    for iteration in range(repeats):
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        baseline_allocated = int(torch.npu.memory_allocated())
        baseline_reserved = int(torch.npu.memory_reserved())
        torch.npu.reset_peak_memory_stats()
        probe.reset_peak_live_pinned_payload_bytes()
        before_transfers = probe.snapshot()
        proc_before = _proc_memory()

        start = time.perf_counter()
        result = _run(
            model,
            plan,
            policy=policy,
            probe=probe,
            collect=False,
            emit_stage_rows=False,
            profile_stage_memory=True,
        )
        torch.npu.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        for row in result.stage_rows:
            print(
                "TPR_PHASE_C_PERF_STAGE "
                + json.dumps(
                    dict(
                        iteration=iteration,
                        prefix=prefix,
                        suffix=suffix,
                        siblings=siblings,
                        **row,
                    )
                ),
                flush=True,
            )

        transfers = _delta(probe.snapshot(), before_transfers)
        peak_allocated = max(row["peak_allocated_bytes"] for row in result.stage_rows)
        peak_reserved = max(row["peak_reserved_bytes"] for row in result.stage_rows)
        peak_pinned_payload = max(
            row["peak_live_pinned_payload_bytes"] for row in result.stage_rows
        )
        proc_after = _proc_memory()
        sample = dict(
            iteration=iteration,
            latency_ms=latency_ms,
            model_forwards=result.total_model_forwards,
            baseline_allocated_bytes=baseline_allocated,
            baseline_reserved_bytes=baseline_reserved,
            peak_allocated_bytes=peak_allocated,
            incremental_peak_allocated_bytes=peak_allocated - baseline_allocated,
            peak_reserved_bytes=peak_reserved,
            peak_live_pinned_payload_bytes=peak_pinned_payload,
            settled_live_pinned_payload_bytes=probe.live_pinned_payload_bytes(),
            rss_before_bytes=proc_before["rss_bytes"],
            rss_after_bytes=proc_after["rss_bytes"],
            **transfers,
        )
        samples.append(sample)
        print("TPR_PHASE_C_PERF_SAMPLE " + json.dumps(
            dict(policy=policy, prefix=prefix, suffix=suffix, siblings=siblings, **sample)
        ), flush=True)
        assert sample["settled_live_pinned_payload_bytes"] == 0

    latencies = [sample["latency_ms"] for sample in samples]
    peak_allocated = [sample["peak_allocated_bytes"] for sample in samples]
    incremental_peak = [sample["incremental_peak_allocated_bytes"] for sample in samples]
    peak_reserved = [sample["peak_reserved_bytes"] for sample in samples]
    peak_pinned = [sample["peak_live_pinned_payload_bytes"] for sample in samples]
    final = dict(
        policy=policy,
        loss_chunk_size=(
            None
            if os.getenv("TPR_PHASE_C_LOSS_CHUNK_SIZE") in (None, "")
            else int(os.getenv("TPR_PHASE_C_LOSS_CHUNK_SIZE"))
        ),
        model=target.label,
        checkpoint=str(target.path),
        parameter_count=parameter_count,
        prefix=prefix,
        suffix=suffix,
        siblings=siblings,
        warmup=warmup,
        repeats=repeats,
        model_forwards=samples[0]["model_forwards"],
        latency_ms_median=statistics.median(latencies),
        latency_ms_mean=statistics.mean(latencies),
        latency_ms_min=min(latencies),
        latency_ms_max=max(latencies),
        baseline_allocated_bytes_median=int(statistics.median(
            sample["baseline_allocated_bytes"] for sample in samples
        )),
        peak_allocated_bytes_median=int(statistics.median(peak_allocated)),
        peak_allocated_bytes_max=max(peak_allocated),
        incremental_peak_allocated_bytes_median=int(statistics.median(incremental_peak)),
        incremental_peak_allocated_bytes_max=max(incremental_peak),
        peak_reserved_bytes_max=max(peak_reserved),
        peak_live_pinned_payload_bytes_max=max(peak_pinned),
        d2h_bytes_median=int(statistics.median(sample["d2h_bytes"] for sample in samples)),
        h2d_bytes_median=int(statistics.median(sample["h2d_bytes"] for sample in samples)),
    )
    print("TPR_PHASE_C_PERF " + json.dumps(final), flush=True)


def test_phase_c1_repeated_lifecycle(runtime, native_args, monkeypatch):
    prefix = int(os.getenv("TPR_PHASE_C_LEAK_PREFIX", "512"))
    suffix = int(os.getenv("TPR_PHASE_C_LEAK_SUFFIX", "512"))
    repeats = int(os.getenv("TPR_PHASE_C_LEAK_REPEATS", "3"))
    plan = _flat_plan(prefix, suffix)

    torch.manual_seed(123)
    model, target, parameter_count = _make_model(
        runtime,
        monkeypatch,
        max_sequence_length=prefix + suffix,
    )
    model.config.swap_attention = True
    model.config.swap_modules = native_args.swap_modules
    probe = _NativeTransferProbe(monkeypatch)

    # Warm native/pinned allocators before defining the settled baseline.
    _run(model, plan, policy="offload", probe=probe, collect=False)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    assert probe.live_pinned_payload_bytes() == 0
    baseline = dict(
        allocated_bytes=int(torch.npu.memory_allocated()),
        reserved_bytes=int(torch.npu.memory_reserved()),
        live_pinned_payload_bytes=probe.live_pinned_payload_bytes(),
        **_proc_memory(),
    )

    samples = []
    for iteration in range(repeats):
        _run(model, plan, policy="offload", probe=probe, collect=False)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        sample = dict(
            iteration=iteration,
            allocated_bytes=int(torch.npu.memory_allocated()),
            reserved_bytes=int(torch.npu.memory_reserved()),
            live_pinned_payload_bytes=probe.live_pinned_payload_bytes(),
            **_proc_memory(),
        )
        samples.append(sample)
        print("TPR_PHASE_C_MEMORY " + json.dumps(sample), flush=True)
        assert sample["live_pinned_payload_bytes"] == 0

    npu_tolerance = int(os.getenv("TPR_PHASE_C_NPU_LEAK_TOLERANCE_MIB", "64")) * 1024**2
    cpu_tolerance = int(os.getenv("TPR_PHASE_C_CPU_LEAK_TOLERANCE_MIB", "128")) * 1024**2
    assert max(sample["allocated_bytes"] for sample in samples) <= baseline["allocated_bytes"] + npu_tolerance
    assert max(sample["reserved_bytes"] for sample in samples) <= baseline["reserved_bytes"] + npu_tolerance
    if baseline["pinned_bytes"] is not None:
        assert max(sample["pinned_bytes"] for sample in samples) <= baseline["pinned_bytes"] + npu_tolerance
    if baseline["rss_bytes"] is not None:
        assert max(sample["rss_bytes"] for sample in samples) <= baseline["rss_bytes"] + cpu_tolerance

    print(
        "TPR_PHASE_C_LIFECYCLE "
        + json.dumps(
            dict(
                model=target.label,
                checkpoint=str(target.path),
                parameter_count=parameter_count,
                repeats=repeats,
                baseline=baseline,
                final=samples[-1],
                **probe.snapshot(),
            )
        ),
        flush=True,
    )
