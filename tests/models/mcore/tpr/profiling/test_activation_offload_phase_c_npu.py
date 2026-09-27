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
import sys
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


def _run(model, plan, *, policy, probe, collect=True):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor

    model.zero_grad(set_to_none=True)
    logs = {}
    boundaries = {}
    stage_rows = []
    model_forwards = defaultdict(int)
    current_stage = ["outside"]

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
            transfers = _delta(probe.snapshot(), before)
            row = dict(
                policy=policy,
                stage=stage,
                event="end",
                model_forwards=model_forwards[stage] - before_forwards,
                **transfers,
            )
            stage_rows.append(row)
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

    executor = ObservedExecutor(model, plan, prefix_backward_policy=policy)
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
        losses.append(executor.pop(segment_id).normalized_loss.detach())

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
        return _RunResult(None, total_forwards, tuple(stage_rows))
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
    return _RunResult(tensors, total_forwards, tuple(stage_rows))


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
        finite=finite,
    )


def _assert_strict_match(reference, actual, *, case):
    failures = []
    for category, expected_map, actual_map in zip(
        ("loss", "logprob", "parameter_grad", "prefix_grad"),
        reference,
        actual,
    ):
        assert expected_map.keys() == actual_map.keys(), f"{case}/{category}: tensor keys changed"
        assert expected_map, f"{case}/{category}: empty comparison"
        failed = []
        for name, expected in expected_map.items():
            value = actual_map[name]
            try:
                torch.testing.assert_close(value, expected, rtol=2e-3, atol=2e-4)
            except AssertionError:
                metrics = (
                    {"shape_expected": tuple(expected.shape), "shape_actual": tuple(value.shape)}
                    if expected.shape != value.shape
                    else _difference_metrics(expected, value)
                )
                failed.append((str(name), metrics))
                failures.append(f"{case}/{category}/{name}")
        worst = sorted(
            failed,
            key=lambda item: item[1].get("relative_l2", float("inf")),
            reverse=True,
        )[:5]
        print(
            "TPR_PHASE_C_COMPARE "
            + json.dumps(
                dict(
                    case=case,
                    category=category,
                    checked=len(expected_map),
                    failed=len(failed),
                    worst=worst,
                )
            ),
            flush=True,
        )
    assert not failures, "Phase C1 strict correctness failures: " + ", ".join(failures[:10])


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

    reference = _run(model, plan, policy="recompute", probe=probe)
    actual = _run(model, plan, policy="offload", probe=probe)
    assert reference.tensors is not None and actual.tensors is not None
    _assert_strict_match(reference.tensors, actual.tensors, case=case)

    if case == "flat":
        sibling_count = len(plan.children_of(plan.root_id))
        assert reference.total_model_forwards == sibling_count + 2
        assert actual.total_model_forwards == sibling_count + 1
    print(
        "TPR_PHASE_C_CORRECTNESS "
        + json.dumps(
            dict(
                case=case,
                model=target.label,
                checkpoint=str(target.path),
                parameter_count=parameter_count,
                recompute_forwards=reference.total_model_forwards,
                offload_forwards=actual.total_model_forwards,
                **probe.snapshot(),
            )
        ),
        flush=True,
    )


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
