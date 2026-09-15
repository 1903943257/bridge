"""Stage 4.5: AllGather-only CP2 Prefix Reuse performance, not a numerical PASS.

Ref: independent whole trajectories; TPR: production Push/Visit/Pop scheduler.
Both use the SAME whole-QKV AllGather diagnostic backend and stateful GDN.
No optimizer, Ring attention/P2P, projection shape controls, or clamp.
"""

from collections import defaultdict
from contextlib import contextmanager, nullcontext
import gc
import json
import os
from statistics import median
import time

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available
from baseline import _qwen35_baseline_utils as baseline
from ._prefix_reuse_metrics import benchmark_cases, summarize_case
from ._whole_fa_allgather import install_whole_fa_allgather

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires two Ascend NPUs", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = baseline.initialize_npu_runtime(world_size=2)
    yield value
    baseline.destroy_npu_runtime(value)


def _plans(n, p, s):
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec, SegmentLossTerm

    gen = torch.Generator().manual_seed(450001 + p * 101 + s)
    prefix = torch.randint(1, baseline.VOCAB_SIZE, (p,), generator=gen)
    segments = [SegmentSpec(0, None, prefix, 0, 0, ())]
    refs = []
    for sid in range(1, n + 1):
        suffix = torch.randint(1, baseline.VOCAB_SIZE, (s,), generator=gen)
        terms = tuple(SegmentLossTerm(i, int(suffix[i + 1])) for i in range(s - 1))
        segments.append(SegmentSpec(sid, 0, suffix, p, p, terms))
        full_terms = tuple(SegmentLossTerm(t.query_offset + p, t.target_token_id) for t in terms)
        refs.append(SegmentPlan([SegmentSpec(0, None, torch.cat((prefix, suffix)), 0, 0, full_terms)], root_id=0))
    tree = SegmentPlan(segments, root_id=0)
    assert tree.total_loss_weight == sum(plan.total_loss_weight for plan in refs) == n * (s - 1)
    return refs, tree


class _Profile:
    """Separate synchronized wall-clock profile; never used for speedup.

    Exclusive nesting prevents counting communication again as model compute.
    Synchronization perturbs overlap; these are attribution estimates, NOT
    pure kernel time. Inclusive prefix/forward/backward totals overlap buckets.
    """

    def __init__(self):
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.stack = []

    @contextmanager
    def span(self, name):
        torch.npu.synchronize()
        frame = [time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - frame[0]) * 1000
            self.stack.pop()
            self.inclusive[name] += elapsed
            self.exclusive[name] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, patch, owner, attr, name):
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)

        patch.setattr(owner, attr, wrapped)

    def install(self, patch, model):
        from verl.models.mcore.tpr.parallel import allgather_attention as ag
        from verl.models.mcore.tpr import segment_executor as se
        from verl.models.mcore.tpr.prefix_state import GDNPrefixState, KVPrefixState
        from verl.models.mcore.tpr.kv_stack import KVStack

        self.wrap(patch, model, "forward", "model_forward")
        self.wrap(patch, torch.autograd, "backward", "backward")
        self.wrap(patch, ag, "_all_gather_into_tensor", "allgather")
        self.wrap(patch, ag, "_reduce_scatter_tensor", "reduce_scatter")
        # Captures both GDN forward and inverse-autograd A2A collectives.
        self.wrap(patch, dist, "all_to_all_single", "a2a")
        self.wrap(patch, se, "build_sharded_past_anchors", "state_ops")
        self.wrap(patch, se, "accumulate_sharded_past_anchor_gradients", "state_ops")
        self.wrap(patch, se, "_compact_kv_cache", "state_ops")
        self.wrap(patch, se.SegmentExecutor, "_gdn_parent_anchors", "state_ops")
        self.wrap(patch, se.SegmentExecutor, "_compute_loss", "loss_compute")
        for owner, names in ((GDNPrefixState, ("make_anchors", "accumulate_anchor_gradients",
                                                "consume_gradients", "release")),
                             (KVPrefixState, ("release",)),
                             (KVStack, ("push", "pop"))):
            for name in names:
                self.wrap(patch, owner, name, "state_ops")
        # Preserve the classmethod descriptor when timing graph-free GDN save.
        original_save = GDNPrefixState.save.__func__

        def save(cls, *args, **kwargs):
            with self.span("state_ops"):
                return original_save(cls, *args, **kwargs)

        patch.setattr(GDNPrefixState, "save", classmethod(save))

    def fields(self):
        buckets = ("model_forward", "backward", "allgather", "reduce_scatter", "a2a",
                   "parameter_sum", "state_ops", "loss_compute", "prefix_recompute", "schedule")
        values = {f"profile_exclusive_{key}_ms": self.exclusive[key] for key in buckets}
        values.update({f"profile_inclusive_{key}_ms": self.inclusive[key]
                       for key in ("model_forward", "backward", "prefix_recompute", "schedule")})
        return values


class _PhaseEvents:
    """Coarse async stream timings, following the existing TPR NPU profiler."""

    def __init__(self):
        self.pairs = defaultdict(list)

    @contextmanager
    def span(self, name):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pairs[name].append((start, end))

    def wrap(self, patch, owner, attr, name):
        original = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)

        patch.setattr(owner, attr, wrapped)

    def fields(self):
        # Called only after the total measurement's end synchronize.
        return {f"{key}_device_ms": sum(start.elapsed_time(end) for start, end in self.pairs[key])
                for key in ("forward", "backward", "prefix_recompute", "parameter_sum")}


def _execute(model, runtime, refs, tree, mode, profile, events):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.fixed_topology_scheduler import FixedTopologyScheduler

    class Executor(SegmentExecutor):
        def _forward(self, segment, **kwargs):
            recompute = mode == "tpr" and segment.segment_id == 0 and not kwargs["no_grad"]
            timer = profile or events
            span = timer.span("prefix_recompute") if recompute else nullcontext()
            with span:
                return super()._forward(segment, **kwargs)

    def executor(plan, scale):
        return Executor(model, plan, expected_layer_numbers=(4, 8, 12, 16, 20, 24),
                        cp_group=runtime.cp_group, cp_backend="ring", loss_scale_func=lambda loss: loss * scale)

    # Backend name carries zigzag placement only. FA callable is replaced by AG.
    # Cancel Executor CP multiplier: explicit parameter SUM, not DP averaging.
    if mode == "ref":
        losses = []
        for plan in refs:
            ex = executor(plan, 1 / (2 * len(refs)))
            result = ex.visit_leaf(0)
            losses.append(result.backward.normalized_loss / len(refs))
            assert not len(ex.kv_stack) and not ex.gdn_states
        loss = torch.stack(losses).sum()
    else:
        ex = executor(tree, 0.5)
        result = FixedTopologyScheduler(tree, ex).run()
        assert result.pushed_segment_count == result.popped_segment_count == 1
        assert result.direct_leaf_count == len(refs)
        assert not len(ex.kv_stack) and not ex.gdn_states
        loss = result.normalized_loss
    with (profile or events).span("parameter_sum"):
        baseline.allreduce_parameter_gradients(model, runtime.cp_group)
    return loss.detach()


def _rank_max(values, runtime):
    names = sorted(values)
    tensor = torch.tensor([values[k] for k in names], device=runtime.device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    return dict(zip(names, tensor.cpu().tolist()))


def _sample(model, runtime, refs, tree, mode, patch, profiled):
    import mindspeed.core.ssm.gated_delta_net as gdn
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    model.zero_grad(set_to_none=True)
    profile = _Profile() if profiled else None
    events = None if profiled else _PhaseEvents()
    with patch.context() as local_patch:
        counts = install_whole_fa_allgather(local_patch)
        transport, _ = ring._load_mindspeed_ring_primitives()

        def forbidden_ring(*args, **kwargs):
            raise AssertionError("Stage 4.5 must not execute Ring P2P")

        local_patch.setattr(transport, "async_send_recv", forbidden_ring)
        if profile:
            profile.install(local_patch, model)
        else:
            events.wrap(local_patch, model, "forward", "forward")
            events.wrap(local_patch, torch.autograd, "backward", "backward")
        with baseline.AllToAllProbe(gdn) as a2a:
            dist.barrier(group=runtime.cp_group)
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            allocated_start = torch.npu.memory_allocated()
            reserved_start = torch.npu.memory_reserved()
            started = time.perf_counter()
            with profile.span("schedule") if profile else nullcontext():
                loss = _execute(model, runtime, refs, tree, mode, profile, events)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            # Capture before finite checks, reporting collectives, or CPU copies.
            values = dict(total_ms=elapsed,
                          allocated_start_MiB=allocated_start / 2**20,
                          reserved_start_MiB=reserved_start / 2**20,
                          peak_allocated_MiB=torch.npu.max_memory_allocated() / 2**20,
                          peak_reserved_MiB=torch.npu.max_memory_reserved() / 2**20,
                          incremental_allocated_MiB=(torch.npu.max_memory_allocated() - allocated_start) / 2**20)
        n = len(refs)
        forwards = n if mode == "ref" else n + 2
        assert a2a.count("cp2hp") == 108 * forwards and a2a.count("hp2cp") == 18 * forwards
        assert counts["fa_allgather"] == 6 * forwards
        assert counts["all_gather"] == (18 * n if mode == "ref" else 30 * n + 36)
        # Graph-free push has no VJP. CP Pop includes a zero-logit objective
        # to keep collective participation identical, so all six root FAs
        # still run backward, even though the prefix owns no training loss.
        assert counts["reduce_scatter"] == (18 * n if mode == "ref" else 30 * n + 18)
    # Outside measured interval. Verify both ranks fail together on non-finite.
    finite = torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            # Bound diagnostic temporaries: a full-vocabulary embedding-sized
            # boolean mask would pollute the next sample's cached reservation.
            for chunk in parameter.grad.detach().reshape(-1).split(1024 * 1024):
                finite = finite & torch.isfinite(chunk).all()
    finite = finite.to(torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=runtime.cp_group)
    assert finite.item(), "non-finite loss/gradient; performance result invalid"
    dist.all_reduce(loss, group=runtime.cp_group)
    values["loss"] = loss.item()
    if profile:
        values.update(profile.fields())
    else:
        values.update(events.fields())
    values = _rank_max(values, runtime)
    return values, dict(counts, cp2hp=a2a.count("cp2hp"), hp2cp=a2a.count("hp2cp"), ring_p2p=0)


@pytest.mark.skipif(os.getenv("STAGE45_PERF", "0") != "1", reason="Opt-in benchmark: STAGE45_PERF=1")
def test_qwen35_allgather_prefix_reuse_performance(runtime, monkeypatch):
    cases = benchmark_cases(os.getenv("STAGE45_BRANCHES", "2,4,8,16"),
                            os.getenv("STAGE45_LENGTHS", "8192:1024,8192:8192"))
    repeats = int(os.getenv("STAGE45_REPEATS", "3"))
    warmup = int(os.getenv("STAGE45_WARMUP", "1"))
    assert repeats >= 3 and warmup >= 1
    monkeypatch.setattr(baseline, "SEQUENCE_LENGTH", max(p + s for _, p, s in cases))
    torch.manual_seed(450001)
    model = baseline.make_qwen35_model(runtime, cp_size=2, tpr=True)
    baseline.broadcast_module_state(model, src=0)
    baseline.assert_hybrid_architecture(model)
    assert model.config.hidden_dropout == model.config.attention_dropout == 0
    if runtime.rank == 0:
        print("STAGE-4.5 AllGather=whole-QKV+replicated-FA; CP2 BF16 18GDN+6FA; "
              "Ref=independent-whole; TPR=FixedTopologyScheduler; no optimizer/Engine wrapper; "
              "S-only next-token objective; no Ring/clamp/Linear-control; "
              "clean total includes one parameter SUM; synchronized profile is separate", flush=True)
    summary = []
    try:
        for case_index, (n, p, s) in enumerate(cases):
            refs, tree = _plans(n, p, s)
            samples = {}
            profiles = {}
            communications = {}
            # Alternate Ref/TPR ordering across cases; each starts an independent
            # warm-cache regime (empty_cache BEFORE warmup, not timed reps).
            modes = ("ref", "tpr") if case_index % 2 == 0 else ("tpr", "ref")
            for profiled in (False, True):
                for mode in modes:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.npu.empty_cache()
                    if runtime.rank == 0:
                        print(f"STAGE-4.5 RUN N={n} P={p} S={s} {mode} profile={profiled}", flush=True)
                    rows = []
                    for rep in range(warmup + repeats):
                        row, comm = _sample(model, runtime, refs, tree, mode, monkeypatch, profiled)
                        if rep >= warmup:
                            rows.append(row)
                    (profiles if profiled else samples)[mode] = rows
                    communications[mode] = comm
            result = summarize_case(n, p, s, samples["ref"], samples["tpr"])
            result["profile_median"] = {
                mode: {key: median(row[key] for row in rows) for key in rows[0]}
                for mode, rows in profiles.items()
            }
            result["communication_per_rank"] = communications
            result["samples"] = samples
            summary.append(result)
            if runtime.rank == 0:
                print("STAGE45_CASE " + json.dumps(result, sort_keys=True), flush=True)
        if runtime.rank == 0:
            print("STAGE45_SUMMARY N P S ref_ms tpr_ms ref_F/B_ms tpr_F/B_ms ideal measured saving_realization loss_rel")
            for row in summary:
                print(f"{row['N']} {row['P']} {row['S']} {row['ref']['total_ms']:.3f} "
                      f"{row['tpr']['total_ms']:.3f} "
                      f"{row['ref']['forward_device_ms']:.3f}/{row['ref']['backward_device_ms']:.3f} "
                      f"{row['tpr']['forward_device_ms']:.3f}/{row['tpr']['backward_device_ms']:.3f} "
                      f"{row['ideal_token_speedup']:.3f} "
                      f"{row['measured_speedup']:.3f} {row['saving_realization_fraction']:.3f} "
                      f"{row['loss_relative_diff']:.6e}")
            print("STAGE45 performance measurement complete; NOT a numerical correctness/training PASS")
    finally:
        model.zero_grad(set_to_none=True)
