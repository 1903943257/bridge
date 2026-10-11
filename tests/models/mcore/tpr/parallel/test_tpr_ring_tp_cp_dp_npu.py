# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-Qwen3-1.7B TP2+RingCP2 then DP2+RingCP2 module gates.

Original two 4-NPU runs (no SP):
  TPR_RUN_RING_COMBO=tp_cp torchrun --nproc_per_node=4 -m pytest -s -vv -x \
    tests/models/mcore/tpr/parallel/test_tpr_ring_tp_cp_dp_npu.py
  TPR_RUN_RING_COMBO=dp_cp torchrun --nproc_per_node=4 -m pytest -s -vv -x \
    tests/models/mcore/tpr/parallel/test_tpr_ring_tp_cp_dp_npu.py

Diagnostic controls preserving the exact numerical gates and Ring math:
  # Swap the two physical token workloads between DP0 and DP1 (4 NPUs).
  TPR_RUN_RING_COMBO=dp_cp TPR_RING_SWAP_DP_DATA=1 \
    torchrun --nproc_per_node=4 -m pytest -s -vv -x \
    tests/models/mcore/tpr/parallel/test_tpr_ring_tp_cp_dp_npu.py
  # Reproduce original DP1 tokens on isolated TP1 x RingCP2, DP1 (2 NPUs).
  TPR_RUN_RING_COMBO=cp_control TPR_RING_CONTROL_DATASET=1 \
    torchrun --nproc_per_node=2 -m pytest -s -vv -x \
    tests/models/mcore/tpr/parallel/test_tpr_ring_tp_cp_dp_npu.py
  # Switch only the isolated diagnostic CP2 test to the existing
  # AllGather CP backend; never switch tp_cp/dp_cp acceptance cases.
  TPR_RUN_RING_COMBO=cp_control TPR_RING_CONTROL_DATASET=1 \
    TPR_CP_CONTROL_BACKEND=allgather torchrun --nproc_per_node=2 \
    -m pytest -s -vv -x \
    tests/models/mcore/tpr/parallel/test_tpr_ring_tp_cp_dp_npu.py

The baseline computes two *independent* full trajectories, each using the
repository's own Ring CP backend. The candidate shares Prefix KV across
siblings using that exact backend. Native Megatron TP vocab CE/projections,
actual CP Ring P2P, and native TP/CP/DP process groups are exercised.

SP, Megatron's DP DDP/optimizer finalizer, production Engine/Trainer routing,
real TQ long-context accuracy and mixed TP+SP+CP are OUT OF SCOPE. The
parameter-gradient CP accumulation here is the existing test-oracle convention
and is NOT claimed to be native Megatron optimizer gradient finalization.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.models.mcore.tpr.fixed_topology_scheduler import FixedTopologyScheduler
from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.parallel import ring_attention
from verl.models.mcore.tpr.tp_validation import (
    assert_tp_plan_agreement, digest_segment_plan,
)
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from ._tpr_cp_test_utils import (
    _equivalence_tpr_plan,
    _independent_plan,
    _logical_logprob_indices,
)
from .test_tpr_tp2_sp_dp2_npu import _distinct_dp_trajectories


_MODE = os.getenv("TPR_RUN_RING_COMBO", "")
pytestmark = pytest.mark.skipif(
    _MODE not in ("tp_cp", "dp_cp", "cp_control"),
    reason="set TPR_RUN_RING_COMBO=tp_cp, dp_cp or cp_control for Ring CP tests",
)
_QWEN_PATH = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))


@pytest.fixture(scope="module")
def ring_combo_runtime():
    expected_world_size = 2 if _MODE == "cp_control" else 4
    if int(os.getenv("WORLD_SIZE", "1")) != expected_world_size:
        pytest.skip(
            f"{_MODE} Ring CP requires torchrun --nproc_per_node="
            f"{expected_world_size}"
        )
    import torch_npu  # noqa: F401

    tp_size = 2 if _MODE == "tp_cp" else 1
    cp_size = 2
    dp_size = 2 if _MODE == "dp_cp" else 1
    assert tp_size * cp_size * dp_size == expected_world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    repatch({
        "tensor_model_parallel_size": tp_size,
        "context_parallel_size": cp_size,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(261011)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    assert dist.get_world_size(group=tp_group) == tp_size
    assert dist.get_world_size(group=cp_group) == cp_size
    assert dist.get_world_size(group=dp_group) == dp_size
    runtime = SimpleNamespace(
        rank=parallel_state.get_tensor_model_parallel_rank(),
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(),
        dp_rank=parallel_state.get_data_parallel_rank(),
        tp_size=tp_size, cp_size=cp_size, dp_size=dp_size,
        device=torch.device("npu", local_rank),
        tp_group=tp_group, cp_group=cp_group, dp_group=dp_group, pp_group=pp_group,
    )
    yield runtime
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@contextmanager
def _count_real_ring_kernel_calls():
    """Assert Ring FWD and BWD were really used, not a backend fallback."""
    original_fwd = ring_attention._block_attention_forward
    original_bwd = ring_attention._block_attention_backward
    calls = {"forward": 0, "backward": 0}

    def fwd(*args, **kwargs):
        calls["forward"] += 1
        return original_fwd(*args, **kwargs)

    def bwd(*args, **kwargs):
        calls["backward"] += 1
        return original_bwd(*args, **kwargs)

    ring_attention._block_attention_forward = fwd
    ring_attention._block_attention_backward = bwd
    try:
        yield calls
    finally:
        ring_attention._block_attention_forward = original_fwd
        ring_attention._block_attention_backward = original_bwd


def _run_ring_forest(model, plans, runtime, logical_indices, logical_count, *, cp_backend="ring"):
    """Run independent trajectories or a shared TPR tree with Ring CP.

    Only diagnostic logprob ownership and post-backward CP grad aggregation
    are test-side operations. No custom Ring attention, P2P, TP or DP gradient
    communication is performed.
    """
    model.zero_grad(set_to_none=True)
    local_loss = torch.zeros((), dtype=torch.float32, device=runtime.device)
    logprobs = torch.zeros(logical_count, dtype=torch.float32, device=runtime.device)
    counts = torch.zeros(logical_count, dtype=torch.int64, device=runtime.device)
    traces = []
    with _count_real_ring_kernel_calls() as ring_calls:
        for plan in plans:
            executor = SegmentExecutor(
                model, plan, cp_group=runtime.cp_group, cp_backend=cp_backend,
                expected_layer_numbers=tuple(range(1, 29)),
            )
            assert executor.cp_backend.backend_name == cp_backend
            original_loss = executor._compute_loss

            def capture_loss(segment, logits):
                terms = executor._owned_loss_terms(segment)
                if terms:
                    shard = executor._segment_shard(segment)
                    local_offsets = [shard.global_to_local(term.query_offset) for term in terms]
                    if any(index is None for index in local_offsets):
                        raise AssertionError("Ring CP loss owner is outside local shard")
                    offsets = torch.tensor(local_offsets, device=runtime.device, dtype=torch.long)
                    labels = torch.tensor(
                        [term.target_token_id for term in terms],
                        device=runtime.device, dtype=torch.long,
                    )
                    # Use GPTModel's native Megatron vocab-parallel cross
                    # entropy (logprob == -CE). F.log_softmax on the local
                    # TP vocab shard would produce incorrect token logprobs.
                    with torch.no_grad():
                        token_logits = logits[0].index_select(0, offsets).detach()
                        token_logprobs = -model.compute_language_model_loss(
                            labels.unsqueeze(0),
                            token_logits.unsqueeze(1),
                        ).reshape(-1).float()
                        logical = torch.tensor(
                            [logical_indices[(
                                term.sample_id,
                                segment.position_start + term.query_offset,
                            )] for term in terms],
                            dtype=torch.long, device=runtime.device,
                        )
                        logprobs.index_copy_(0, logical, token_logprobs)
                        counts.index_add_(0, logical, torch.ones_like(logical))
                return original_loss(segment, logits)

            executor._compute_loss = capture_loss
            try:
                result = FixedTopologyScheduler(plan, executor).run()
            finally:
                executor._compute_loss = original_loss
            local_loss += result.normalized_loss.detach()
            traces.extend((step.kind.value, step.segment_id) for step in result.execution_trace)

    # Logprob & loss shards sum only over CP; TP peers repeat the same logical
    # targets but own distinct vocab/head shards, never sum CE across TP.
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(logprobs, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    ownership_ok = bool(torch.all(counts == 1).item())

    # Existing CP-equivalence test convention: sum and average per-CP local
    # parameter gradients for comparing two executions at matching topology.
    # This is a test oracle, not native Megatron DP/optimizer synchronization.
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing Ring CP parameter gradient: {name}")
        grad = parameter.grad.detach().float().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
        grad.div_(runtime.cp_size)
        gradients[name] = grad.cpu()
    return float(local_loss.item()), logprobs.cpu(), gradients, ownership_ok, ring_calls, traces


def test_qwen3_native_tp_or_dp_with_ring_cp(ring_combo_runtime):
    runtime = ring_combo_runtime
    # Only the two-rank CP control can opt into existing AllGather CP,
    # allowing a same-data differential test without changing Ring acceptance.
    cp_backend = (
        os.getenv("TPR_CP_CONTROL_BACKEND", "ring")
        if _MODE == "cp_control" else "ring"
    )
    if cp_backend not in ("ring", "allgather"):
        raise ValueError(f"unsupported diagnostic CP backend {cp_backend!r}")
    # The original acceptance run is unchanged. Optional controls vary only
    # which of the two already-validated physical token workloads is supplied:
    # no attention algorithm, gradient reduction or numerical gate is altered.
    if _MODE == "cp_control":
        dataset_id = int(os.getenv("TPR_RING_CONTROL_DATASET", "1"))
    elif _MODE == "dp_cp" and os.getenv("TPR_RING_SWAP_DP_DATA", "0") == "1":
        dataset_id = 1 - runtime.dp_rank
    else:
        dataset_id = runtime.dp_rank
    if dataset_id not in (0, 1):
        raise ValueError(f"TPR_RING dataset_id must be 0 or 1, got {dataset_id}")
    prefix, suffix1, suffix2, signature = _distinct_dp_trajectories(dataset_id)
    # Ring CP has two zigzag ranges per CP rank; divisible lengths avoid
    # padding and keep both rank-local logical shard lengths equal.
    assert all(x.numel() % (2 * runtime.cp_size) == 0 for x in (prefix, suffix1, suffix2))
    first = torch.cat((prefix, suffix1))
    second = torch.cat((prefix, suffix2))
    plan = _equivalence_tpr_plan(prefix, suffix1, suffix2)
    digest = digest_segment_plan(plan)
    if runtime.tp_size > 1:
        assert_tp_plan_agreement(digest, tp_size=runtime.tp_size)

    # Same physical tree on every CP rank and TP peer of a DP replica.
    # Different physical tree on different DP replicas. Use WORLD only for
    # control-plane signatures, not model gradients or loss normalization.
    metadata = torch.tensor(
        [runtime.dp_rank, runtime.tp_rank, runtime.cp_rank, signature],
        dtype=torch.int64, device=runtime.device,
    )
    metadata_world = [
        torch.empty_like(metadata) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(metadata_world, metadata)
    rows = [x.cpu().tolist() for x in metadata_world]
    signatures_by_dp = {}
    for dp, tp, cp, sig in rows:
        signatures_by_dp.setdefault(dp, set()).add(sig)
    groups_aligned = all(len(values) == 1 for values in signatures_by_dp.values())
    workloads_distinct = (
        len(signatures_by_dp) == runtime.dp_size
        and (runtime.dp_size == 1
             or len({next(iter(vals)) for vals in signatures_by_dp.values()}) == runtime.dp_size)
    )
    if not groups_aligned or not workloads_distinct:
        raise AssertionError(f"native TP/DP/CP plan ownership mismatch: {rows}")

    torch.manual_seed(20261011)
    reference, hf = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN_PATH, tp_size=runtime.tp_size,
        sequence_parallel=False, cp_size=runtime.cp_size,
    )
    torch.manual_seed(20261011)
    candidate, _ = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN_PATH, tp_size=runtime.tp_size,
        sequence_parallel=False, cp_size=runtime.cp_size, load_weights=False,
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)
    assert hf.vocab_size == 151936
    assert reference.config.context_parallel_size == runtime.cp_size == 2

    logical_indices, count = _logical_logprob_indices(first, second)
    independent_plans = (
        _independent_plan(first, sample_id=1, total_loss_weight=plan.total_loss_weight),
        _independent_plan(second, sample_id=2, total_loss_weight=plan.total_loss_weight),
    )
    dist.barrier()
    reference_loss, reference_logp, reference_grad, reference_owners, ref_ring, ref_trace = (
        _run_ring_forest(
            reference, independent_plans, runtime, logical_indices, count,
            cp_backend=cp_backend,
        )
    )
    actual_loss, actual_logp, actual_grad, actual_owners, actual_ring, actual_trace = (
        _run_ring_forest(
            candidate, (plan,), runtime, logical_indices, count,
            cp_backend=cp_backend,
        )
    )
    ce_gap = abs(reference_loss - actual_loss)
    logit_rel_l2 = (
        torch.linalg.vector_norm(actual_logp - reference_logp).item()
        / max(torch.linalg.vector_norm(reference_logp).item(), 1e-24)
    )
    same_names = reference_grad.keys() == actual_grad.keys()
    names = reference_grad.keys() & actual_grad.keys()
    grad_delta_sq = sum((reference_grad[name] - actual_grad[name]).square().sum().item()
                        for name in names)
    grad_ref_sq = sum(reference_grad[name].square().sum().item() for name in names)
    grad_rel_l2 = (grad_delta_sq / max(grad_ref_sq, 1e-24)) ** 0.5

    # CP source semantics: independent full-sequence visits vs shared tree's
    # Push(P), Visit(S1), Visit(S2), Pop(P) (Prefix is recomputed at Pop).
    trace_ok = (
        len(ref_trace) == 2 and len(actual_trace) == 4
        and actual_trace[0][0] == "push" and actual_trace[-1][0] == "pop"
    )
    ring_ok = all(
        (counts["forward"] > 0 and counts["backward"] > 0)
        if cp_backend == "ring"
        else (counts["forward"] == 0 and counts["backward"] == 0)
        for counts in (ref_ring, actual_ring)
    )
    # In the AllGather control, lack of Ring kernel calls is EXPECTED,
    # and executor.cp_backend was asserted to be AllGather above.
    finite = all(math.isfinite(v) for v in (
        reference_loss, actual_loss, logit_rel_l2, grad_rel_l2,
    ))
    gates = {
        "reference_loss_owners": bool(reference_owners),
        "tpr_loss_owners": bool(actual_owners),
        "same_parameter_names": bool(same_names),
        "nonempty_gradients": bool(names),
        "tree_trace": bool(trace_ok),
        "real_ring_fwd_bwd": bool(ring_ok),
        "finite_metrics": bool(finite),
        "loss_gap_lt_0.02": ce_gap < 0.02,
        "logprob_rel_l2_lt_0.05": logit_rel_l2 < 0.05,
        "grad_rel_l2_lt_0.05": grad_rel_l2 < 0.05,
    }
    local_ok = all(gates.values())
    if not local_ok:
        # Diagnostics are strictly observational and run after the native
        # Ring forward/backward and CP gradient-oracle collectives complete.
        # Never relax a threshold or mutate model gradients on a failing DP.
        contributors = []
        for name in names:
            error_sq = (
                (reference_grad[name] - actual_grad[name]).square().sum().item()
            )
            parameter_ref_sq = reference_grad[name].square().sum().item()
            contributors.append((
                name,
                (error_sq / max(grad_ref_sq, 1e-24)) ** 0.5,
                (error_sq / max(parameter_ref_sq, 1e-24)) ** 0.5,
                parameter_ref_sq ** 0.5,
            ))
        contributors.sort(key=lambda item: item[1], reverse=True)
        logprob_abs = (actual_logp - reference_logp).abs().reshape(-1)
        worst_count = min(8, logprob_abs.numel())
        worst_positions = torch.topk(logprob_abs, worst_count).indices.tolist()
        logical_lookup = {index: key for key, index in logical_indices.items()}
        top_logprobs = [
            (
                logical_lookup[position],
                float(reference_logp[position]),
                float(actual_logp[position]),
                float(logprob_abs[position]),
            )
            for position in worst_positions
        ]
        print(
            f"TPR_RING_COMBO_DIAG mode={_MODE} cp_backend={cp_backend} "
            f"world_rank={dist.get_rank()} dp_rank={runtime.dp_rank} "
            f"tp_rank={runtime.tp_rank} cp_rank={runtime.cp_rank} "
            f"dataset_id={dataset_id} "
            f"failed_gates={[name for name, ok in gates.items() if not ok]} "
            f"loss_gap={ce_gap:.8f} logprob_rel_l2={logit_rel_l2:.8f} "
            f"grad_rel_l2={grad_rel_l2:.8f} "
            f"reference_grad_norm={grad_ref_sq ** 0.5:.8f} "
            f"gradient_top_contributors={contributors[:8]} "
            f"logprob_top_positions={top_logprobs} "
            f"reference_ring_calls={ref_ring} tpr_ring_calls={actual_ring}",
            flush=True,
        )
    metrics = torch.tensor(
        [runtime.dp_rank, runtime.tp_rank, runtime.cp_rank,
         reference_loss, actual_loss, ce_gap, logit_rel_l2, grad_rel_l2,
         int(groups_aligned), int(workloads_distinct),
         int(reference_owners and actual_owners), int(trace_ok),
         int(ring_ok), int(dataset_id), int(local_ok)],
        dtype=torch.float32, device=runtime.device,
    )
    summaries = [
        torch.empty_like(metrics) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(summaries, metrics)
    observed = [m.cpu().tolist() for m in summaries]
    if dist.get_rank() == 0:
        for global_rank, row in enumerate(observed):
            print(
                f"TPR_RING_COMBO mode={_MODE} cp_backend={cp_backend} world_rank={global_rank} "
                f"dp_rank={int(row[0])} tp_rank={int(row[1])} cp_rank={int(row[2])} "
                f"dataset_id={int(row[13])} "
                f"reference_ce={row[3]:.7f} tpr_ce={row[4]:.7f} "
                f"loss_gap={row[5]:.7f} logprob_rel_l2={row[6]:.7f} "
                f"grad_rel_l2={row[7]:.7f} "
                f"plan_groups_aligned={bool(row[8])} dp_workloads_distinct={bool(row[9])} "
                f"loss_ownership_ok={bool(row[10])} tree_trace_ok={bool(row[11])} "
                f"backend_dispatch_ok={bool(row[12])} gate_pass={bool(row[14])}",
                flush=True,
            )
    dist.barrier()
    failures = [index for index, row in enumerate(observed) if row[-1] != 1]
    assert not failures, f"{_MODE} Ring CP + TP/DP module gate failed on {failures}"
