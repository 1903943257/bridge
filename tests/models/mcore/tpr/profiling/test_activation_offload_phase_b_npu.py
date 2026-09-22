"""Opt-in native swap acceptance for Ring CP2/4; see ACTIVATION_OFFLOAD_PHASE_B.md."""

from dataclasses import replace
from datetime import timedelta
import gc
import json
import os
import resource
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(os.getenv("TPR_RUN_OFFLOAD_B") != "1", reason="Set TPR_RUN_OFFLOAD_B=1")


@pytest.fixture(scope="module")
def runtime():
    import torch_npu  # noqa: F401
    size = int(os.environ["WORLD_SIZE"])
    assert size in (2, 4)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)
    repatch(dict(context_parallel_size=size, experimental_attention_variant=None, use_flash_attn=True))
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                            pipeline_model_parallel_size=1, context_parallel_size=size)
    model_parallel_cuda_manual_seed(123)
    try:
        yield SimpleNamespace(cp_size=size, rank=dist.get_rank(),
                              device=torch.device("npu", torch.npu.current_device()),
                              cp_group=parallel_state.get_context_parallel_group(),
                              tp_group=parallel_state.get_tensor_model_parallel_group(),
                              pp_group=parallel_state.get_pipeline_model_parallel_group())
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.fixture
def native_args(runtime, monkeypatch):
    from mindspeed.args_utils import get_full_args
    from verl.models.mcore.tpr import activation_offload
    args = SimpleNamespace(**vars(get_full_args()))
    vars(args).update(swap_attention=False, context_parallel_size=runtime.cp_size,
                      pipeline_model_parallel_size=1, eval_interval=0, curr_iteration=1,
                      noop_layers=None, swap_modules=os.getenv("TPR_SWAP_MODULES", "self_attention,mlp"))
    monkeypatch.setattr(activation_offload, "_args", lambda: args)
    monkeypatch.setattr(activation_offload._native_prefetch(), "get_args", lambda: args)
    # Identical math path for BOTH off/on. No merge-specific swap policy.
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_FULL", "1")
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", "1")
    return args


def _model(runtime, length, native_args):
    # These imports MUST follow MindSpeed bootstrap, not pytest collection.
    from ._qwen3_profile_target import resolve_qwen3_profile_target
    from ..parallel.test_tpr_qwen3_cp_equivalence_npu import _make_qwen_cp_model
    target = resolve_qwen3_profile_target()
    assert target.size == "1.7B"
    target.hf_config.max_position_embeddings = max(target.hf_config.max_position_embeddings, length)
    model = _make_qwen_cp_model(runtime, SimpleNamespace(path=target.path), target.hf_config)
    target.assert_model_scale(model)
    model.config.cross_entropy_loss_fusion = True
    model.config.cross_entropy_fusion_impl = "native"
    model.config.swap_attention = False
    model.config.swap_modules = native_args.swap_modules
    return model


def _plan(prefix, suffix, *, sparse=False):
    from .test_tpr_qwen3_ring_cp_profile_npu import _ProfileCase, _make_case_plans
    from verl.models.mcore.tpr.segment_plan import SegmentPlan
    _, plan = _make_case_plans(_ProfileCase(prefix, suffix, 2), vocab_size=2048)
    if sparse:
        # Only the owner of query zero has local loss; all ranks still backward.
        plan = SegmentPlan([replace(s, loss_terms=tuple(t for t in s.loss_terms if t.query_offset == 0))
                            for s in plan.segments.values()], root_id=0)
    return plan


class _Probe:
    """Test-only native counters and optional events; never owns tensor storage."""

    def __init__(self, monkeypatch, *, timing=False, stage="capacity", synchronize=False):
        from mindspeed.core.memory.swap_attention.prefetch import SwapTensor
        from verl.models.mcore.tpr.parallel import ring_attention
        self.phase = "setup"
        self.stage = stage
        self.synchronize = synchronize
        self.counts = dict(d2h_bytes=0, released_bytes=0, h2d_bytes=0)
        self.restore_peak = 0
        self.by_layer = {}
        self.events = []
        self.errors = []
        self.phases = []
        for method, before, after, key in (
            ("launch_d2h", "device", "d2h", "d2h_bytes"),
            ("wait_d2h_finished", "d2h", "host", "released_bytes"),
            ("launch_h2d", "host", "h2d", "h2d_bytes"),
        ):
            original = getattr(SwapTensor, method)

            def wrapper(item, *args, _original=original, _before=before, _after=after, _key=key, **kwargs):
                previous = item.stat
                result = _original(item, *args, **kwargs)
                if previous == _before and item.stat == _after:
                    size = item.storage_size * item.tensor.element_size()
                    self.counts[_key] += size
                    label = f"{self.phase}/{item.layer_name}/{_key}"
                    self.by_layer[label] = self.by_layer.get(label, 0) + size
                    if _key == "released_bytes":
                        if item.tensor.storage().size() != 0 or not item.tensor_cpu.is_pinned():
                            self.errors.append("native release/pinned-storage contract failed")
                    if _key == "h2d_bytes":
                        self.restore_peak = max(self.restore_peak, torch.npu.memory_allocated())
                return result

            monkeypatch.setattr(SwapTensor, method, wrapper)
        if timing:
            for name, category in (("_circulate_kv", "ring_comm"),
                                   ("_reduce_ring_gradients_to_owner", "ring_comm"),
                                   ("_block_attention_forward", "fa"),
                                   ("_block_attention_backward", "fa"),
                                   ("_merge_attention", "merge")):
                original = getattr(ring_attention, name)

                def timed(*args, _original=original, _category=category, **kwargs):
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        end.record()
                        self.events.append((_category, start, end))

                monkeypatch.setattr(ring_attention, name, timed)

    def marker(self, event, **extra):
        # Correctness logs are intentionally minimal. Emit phase detail only
        # for exceptions; normal progress is summarized once per stage.
        if event == "failure":
            print("TPR_OFFLOAD_B_FAILURE " + json.dumps(dict(
                rank=dist.get_rank(), stage=self.stage, phase=self.phase, **extra)), flush=True)

    def call(self, phase, enabled, function, *, transfer_gate=True):
        self.phase = phase
        before = self.counts.copy()
        try:
            result = function()
            if self.synchronize:
                # Correctness only: surface asynchronous errors at their phase.
                torch.npu.synchronize()
        except Exception as error:
            self.marker("failure", error=str(error)[:500])
            raise
        delta = {key: value - before[key] for key, value in self.counts.items()}
        self.phases.append(dict(phase=phase, **delta))
        if not transfer_gate:
            return result
        if phase.startswith("push") or not enabled:
            if any(delta.values()):
                self.errors.append(f"{phase}: unexpected transfers {delta}")
        elif delta["released_bytes"] <= 0 or delta["h2d_bytes"] <= 0:
            self.errors.append(f"{phase}: native swap did not release/reload {delta}")
        return result

    def report(self):
        times = dict(ring_comm=0.0, fa=0.0, merge=0.0)
        for category, start, end in self.events:
            times[category] += start.elapsed_time(end)
        detail = {"native_hits_by_layer": self.by_layer} if os.getenv("TPR_OFFLOAD_B_VERBOSE") == "1" else {}
        return dict(**self.counts, restore_sample_peak_allocated=self.restore_peak,
                    phase_transfers=self.phases, native_hit_entries=len(self.by_layer), **detail,
                    stream_interval_ms=times if self.events else None,
                    transfer_time_ms=None, exposed_wait_ms=None)


def _run(model, plan, runtime, probe, *, observe):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from .test_tpr_qwen3_ring_cp_profile_npu import _finalize_cp_parameter_gradients
    logs = {}

    class ObservedExecutor(SegmentExecutor):
        def _compute_loss(self, segment, logits):
            if observe:
                terms = self._owned_loss_terms(segment)
                shard = self._segment_shard(segment)
                # Only diagnostics: production backward still calls native fused CE.
                with torch.no_grad():
                    offsets = torch.tensor([shard.global_to_local(t.query_offset) for t in terms],
                                           device=logits.device, dtype=torch.long)
                    targets = torch.tensor([t.target_token_id for t in terms], device=logits.device)
                    if hasattr(logits, "per_term_loss"):
                        logs[segment.segment_id] = -logits.per_term_loss.detach().float().cpu()
                    else:
                        logs[segment.segment_id] = (-torch.nn.functional.cross_entropy(
                            logits[0].index_select(0, offsets).float(),
                            targets.long(),
                            reduction="none",
                        )).cpu()
                else:
                    logs[segment.segment_id] = torch.empty(0)
            return super()._compute_loss(segment, logits)

    loss_chunk_size = int(os.getenv("TPR_LOSS_CHUNK_SIZE", "1024"))
    executor = ObservedExecutor(
        model,
        plan,
        cp_group=runtime.cp_group,
        cp_backend="ring",
        loss_chunk_size=loss_chunk_size,
    )
    if observe and runtime.rank == 0 and probe.stage == "off":
        print("TPR_OFFLOAD_B_SHARDS " + json.dumps(dict(cp_size=runtime.cp_size, segments={
            str(s.segment_id): dict(logical_length=s.length,
                                    physical_local_length=executor._segment_shard(s).local_length,
                                    local_loss_terms=len(executor._owned_loss_terms(s)))
            for s in plan.segments.values()})), flush=True)
    enabled = model.config.swap_attention
    probe.call("push.0", enabled, lambda: executor.push(0))
    losses = []
    for child in (1, 2):
        result = probe.call(f"visit.{child}", enabled, lambda: executor.visit_leaf(child))
        losses.append(result.backward.normalized_loss)
    boundary = {}
    if observe:
        for layer, pair in executor.kv_stack.top().gradients.items():
            for name, value in zip(("key", "value"), pair):
                boundary[f"{layer}.{name}"] = value.detach().cpu().clone()
    losses.append(probe.call("pop.0", enabled, lambda: executor.pop(0)).normalized_loss)
    executor.kv_stack.assert_empty()
    def finalize():
        _finalize_cp_parameter_gradients(model, runtime)
        loss = torch.stack(losses).sum().detach()
        dist.all_reduce(loss, group=runtime.cp_group)
        return loss

    loss = probe.call("finalize", enabled, finalize, transfer_gate=False)
    if not observe:
        return loss
    grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters()}
    return {"loss": loss.cpu()}, logs, grads, boundary


def _report(runtime, record):
    record = dict(rank=runtime.rank, cp_size=runtime.cp_size, qkv_merge=True, **record)
    print("TPR_OFFLOAD_B_RANK " + json.dumps(record), flush=True)
    records = [None] * runtime.cp_size
    dist.all_gather_object(records, record, group=runtime.cp_group)
    if runtime.rank == 0:
        maximum = {key: max(row[key] for row in records)
                   for key, value in record.items() if type(value) in (int, float) and key != "rank"}
        if record.get("stream_interval_ms") is not None:
            maximum["stream_interval_ms"] = {
                key: max(row["stream_interval_ms"][key] for row in records)
                for key in record["stream_interval_ms"]}
        print("TPR_OFFLOAD_B_MAX " + json.dumps(dict(max_rank=maximum)), flush=True)


def _compare(reference, actual, baseline, *, stage):
    from .test_activation_offload_phase_a_npu import _difference_metrics
    from ._offload_b_gate import gradient_gate, merge_baseline

    failures = 0
    failed_details = []
    category_summary = {}
    calibration = stage in ("off_repeat", "off_repeat2")
    gradient_categories = ("parameter_grad", "prefix_grad")

    for category, expected, observed in zip(
        ("loss", "logprob", "parameter_grad", "prefix_grad"), reference, actual
    ):
        bad = 0
        keys_ok = bool(expected) and expected.keys() == observed.keys()
        bad += int(not keys_ok)
        category_baseline = baseline.setdefault(category, {}) if category in gradient_categories else None

        for name in expected.keys() & observed.keys():
            if expected[name].shape != observed[name].shape:
                bad += 1
                failed_details.append(dict(
                    category=category, tensor=str(name), severity=float("inf"),
                    reason="shape_mismatch"))
                continue

            metrics = _difference_metrics(expected[name], observed[name])
            metrics["mismatch_fraction"] = metrics["mismatched"] / max(metrics["elements"], 1)
            limits = {}

            if category in gradient_categories:
                if stage == "off":
                    passed, severity = metrics["finite"], 0.0
                elif calibration:
                    passed, severity = metrics["finite"], 0.0
                    category_baseline[name] = merge_baseline(category_baseline.get(name), metrics)
                else:
                    passed, limits, severity = gradient_gate(metrics, category_baseline.get(name))
            else:
                passed = metrics["finite"] and metrics["mismatched"] == 0
                severity = metrics["max_abs"] if passed else float("inf")

            if not passed:
                bad += 1
                failed_details.append(dict(
                    category=category, tensor=str(name), severity=severity,
                    relative_l2=metrics["relative_l2"], max_abs=metrics["max_abs"],
                    mismatch_fraction=metrics["mismatch_fraction"], limits=limits))

        failures += bad
        category_summary[category] = bad

    worst = sorted(failed_details, key=lambda row: row["severity"], reverse=True)[:3]
    return failures, dict(by_category=category_summary, worst=worst)


@pytest.mark.parametrize("padded,sparse", [(False, False), (True, False), (True, True)])
def test_ring_offload_correctness(runtime, native_args, monkeypatch, padded, sparse):
    length = int(os.getenv("TPR_OFFLOAD_B_CHECK_LENGTH", "2048"))
    plan = _plan(length - int(padded), length - 3 * int(padded), sparse=sparse)
    model = _model(runtime, 2 * length, native_args)
    reference = None
    baseline = {}
    failures = 0
    # Two OFF repeats calibrate native Ring/NPU backward noise before swap.
    # Final OFF checks that repeated native swap lifecycles leave no regression.
    for stage, enabled in (("off", False), ("off_repeat", False), ("off_repeat2", False),
                           ("on", True), ("on_repeat", True), ("off_after", False)):
        model.config.swap_attention = enabled
        model.zero_grad(set_to_none=True)
        with monkeypatch.context() as patches:
            probe = _Probe(patches, stage=stage, synchronize=True)
            actual = _run(model, plan, runtime, probe, observe=True)
        if reference is None:
            reference = actual
        stage_failures, comparison = _compare(reference, actual, baseline, stage=stage)
        stage_failures += len(probe.errors)
        failures += stage_failures
        torch.npu.synchronize()

        local_result = dict(
            rank=runtime.rank,
            failures=stage_failures,
            by_category=comparison["by_category"],
            worst=comparison["worst"],
            d2h_bytes=probe.counts["d2h_bytes"],
            released_bytes=probe.counts["released_bytes"],
            h2d_bytes=probe.counts["h2d_bytes"],
            probe_errors=probe.errors[:3],
        )
        stage_results = [None] * runtime.cp_size
        dist.all_gather_object(stage_results, local_result, group=runtime.cp_group)
        if runtime.rank == 0:
            categories = ("loss", "logprob", "parameter_grad", "prefix_grad")
            unique_worst = {}
            for row in stage_results:
                for item in row["worst"]:
                    key = (item["category"], item["tensor"])
                    if key not in unique_worst or item["severity"] > unique_worst[key]["severity"]:
                        unique_worst[key] = item
            worst = sorted(
                unique_worst.values(), key=lambda row: row["severity"], reverse=True
            )[:3]
            print("TPR_OFFLOAD_B_RESULT " + json.dumps(dict(
                stage=stage,
                failures_max=max(row["failures"] for row in stage_results),
                failures_by_rank=[row["failures"] for row in stage_results],
                by_category_max={
                    category: max(row["by_category"][category] for row in stage_results)
                    for category in categories
                },
                transfer_max={
                    key: max(row[key] for row in stage_results)
                    for key in ("d2h_bytes", "released_bytes", "h2d_bytes")
                },
                probe_errors=[error for row in stage_results for error in row["probe_errors"]][:3],
                worst=worst,
            )), flush=True)
        if actual is not reference:
            del actual
    failure_count = torch.tensor(failures, device=runtime.device, dtype=torch.int32)
    dist.all_reduce(failure_count, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    if runtime.rank == 0:
        assert failure_count.item() == 0, (
            f"Offload correctness failed; max-rank failures={failure_count.item()}; "
            "see TPR_OFFLOAD_B_RESULT"
        )


def test_ring_offload_capacity(runtime, native_args, monkeypatch):
    if os.getenv("TPR_OFFLOAD_B_CAPACITY") != "1":
        pytest.skip("Set TPR_OFFLOAD_B_CAPACITY=1; run each cell in a fresh torchrun")
    prefix, suffix = int(os.getenv("TPR_PREFIX", "16384")), int(os.getenv("TPR_SUFFIX", "16384"))
    enabled = os.getenv("TPR_OFFLOAD", "0") == "1"
    plan = _plan(prefix, suffix)
    model = _model(runtime, prefix + suffix, native_args)
    model.config.swap_attention = enabled
    warmup = int(os.getenv("TPR_OFFLOAD_B_WARMUP", "1"))
    repeats = int(os.getenv("TPR_OFFLOAD_B_REPEATS", "3"))
    assert warmup >= 0 and repeats > 0
    for iteration in range(warmup + repeats):
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        dist.barrier(group=runtime.cp_group)
        baseline = torch.npu.memory_allocated()
        baseline_reserved = torch.npu.memory_reserved()
        torch.npu.reset_peak_memory_stats()
        with monkeypatch.context() as patches:
            probe = _Probe(patches, timing=os.getenv("TPR_OFFLOAD_B_TIMING") == "1",
                           stage=f"capacity.{iteration}")
            start = time.perf_counter()
            try:
                _run(model, plan, runtime, probe, observe=False)
                torch.npu.synchronize()
            except Exception as error:
                # No collective after a rank-local OOM: peers may be inside HCCL.
                print("TPR_OFFLOAD_B_FAILURE " + json.dumps(dict(
                    rank=runtime.rank, cp_size=runtime.cp_size, prefix=prefix, suffix=suffix,
                    offload=enabled, iteration=iteration, stage=probe.phase,
                    oom="out of memory" in str(error).lower(), error=str(error))), flush=True)
                raise
            latency = time.perf_counter() - start
            peak = torch.npu.max_memory_allocated()
            reserved = torch.npu.max_memory_reserved()
        _report(runtime, dict(prefix=prefix, suffix=suffix, offload=enabled, iteration=iteration,
                              warmup=iteration < warmup, latency_s=latency,
                              peak_allocated=peak, peak_reserved=reserved,
                              incremental_peak=peak - baseline,
                              incremental_reserved=reserved - baseline_reserved,
                              cpu_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                              oom_stage=None, **probe.report()))
        failed = torch.tensor(bool(probe.errors), device=runtime.device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=runtime.cp_group)
        assert not failed.item(), f"Native transfer gate failed: {probe.errors}"
