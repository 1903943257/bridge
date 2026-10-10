# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real Qwen3-1.7B, real recorded TQ: isolated DP2 / TP2xDP2 module acceptance.

This is deliberately NOT a production VERL DP dispatcher or clipped PPO
update. It tests the existing TPR SegmentExecutor, shared-prefix semantics,
native Megatron model/TP collectives, and the *real* TP/DP process groups.

Standalone pure DP2 with pretrained Qwen checkpoint and real TQ rows:
    TPR_RUN_MODULE_DP=1 TPR_MODULE_TP_SIZE=1 \
      torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29532 \
      -m pytest -vv -s tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py

Then combined native TP2 x DP2 = 4 NPUs:
    TPR_RUN_MODULE_DP=1 TPR_MODULE_TP_SIZE=2 \
      torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29534 \
      -m pytest -vv -s tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py

The module runs TPR and independent full-trajectory references for each
DP replica. DP replicas intentionally get *different* real TQ rows; within a
replica TP peers must execute the *same* plan. The DP-group all-reduce below
is a tiny control-plane handshake only, NOT a replacement for Megatron
distributed optimizer/gradient finalization. Runtime DP training stays gated.
"""

from __future__ import annotations

import hashlib
import os
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state

from verl.models.mcore.tpr.fixed_topology_scheduler import FixedTopologyScheduler
from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.tp_validation import (
    assert_tp_plan_agreement,
    digest_segment_plan,
)
from verl.models.mcore.tpr.tree_plan_builder import _rows
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from ._tpr_cp_test_utils import _equivalence_tpr_plan

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_MODULE_DP") != "1",
    reason="set TPR_RUN_MODULE_DP=1 for real-Qwen isolated DP2/TP2xDP2 NPU gate",
)

_REAL_TQ = Path(os.getenv(
    "TPR_REAL_TQ_BATCH",
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt",
))
_QWEN = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))


@pytest.fixture(scope="module")
def parallel_runtime():
    tp_size = int(os.getenv("TPR_MODULE_TP_SIZE", "1"))
    assert tp_size in (1, 2)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert world_size == tp_size * 2, (
        f"DP2 module test expects world_size=TP*2={tp_size * 2}, got {world_size}"
    )
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    # Use the installed MindSpeed Megatron adaptor exactly as in the
    # previously user-PASSED real-Qwen pure TP2 test.
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    repatch({
        "tensor_model_parallel_size": tp_size,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(20261010)

    tp_group = parallel_state.get_tensor_model_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    assert dist.get_world_size(group=tp_group) == tp_size
    assert dist.get_world_size(group=dp_group) == 2
    assert dp_rank in (0, 1)
    runtime = SimpleNamespace(
        rank=tp_rank, dp_rank=dp_rank, device=torch.device("npu", local_rank),
        tp_group=tp_group, dp_group=dp_group, cp_group=cp_group,
        pp_group=pp_group, tp_size=tp_size,
    )
    yield runtime
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _real_tq_two_trajectories(runtime):
    """Select truly distinct real-token DP workloads at recorded branch points.

    The first 160 tokens of different recorded TQ rows may be identical even
    if their *logical row IDs* differ. Using those shared prompt tokens made
    both DP ranks report identical CE/gradient values while never exercising
    different physical forests. We now select actual sibling divergence
    windows, using unchanged recorded token IDs and no random/synthetic data.

    The 160-token window is an explicitly cropped, local-position test:
    it does not model the original full long-context causal history.
    """
    if not _REAL_TQ.is_file():
        pytest.skip(f"real UniAgent TQ dump is unavailable: {_REAL_TQ}")
    dump = torch.load(str(_REAL_TQ), weights_only=False, map_location="cpu")
    keys = tuple(dump["keys"])
    data = dump["tensordict"]
    seqs = _rows(data, "input_ids")
    assert len(seqs) >= 8 and len(keys) == len(seqs), (
        "real TQ gate requires the recorded >=8 distinct trajectory rows"
    )
    prefix_len, suffix_len = 128, 32
    tokens = [value.detach().cpu().to(torch.long) for value in seqs]

    def candidate_windows(rows):
        windows = []
        for first_row, second_row in combinations(rows, 2):
            left, right = tokens[first_row], tokens[second_row]
            common_length = min(left.numel(), right.numel())
            if common_length < prefix_len + suffix_len:
                continue
            mismatches = torch.nonzero(
                left[:common_length] != right[:common_length],
                as_tuple=False,
            ).flatten()
            if not mismatches.numel():
                continue  # Identical token path or strict-prefix row.
            fork = int(mismatches[0])
            if fork < prefix_len or fork + suffix_len > common_length:
                continue  # Cannot form the required true 128+32 fork window.
            start = fork - prefix_len
            prefix = left[start:fork].contiguous()
            assert torch.equal(prefix, right[start:fork])
            suffixes = (
                left[fork:fork + suffix_len].contiguous(),
                right[fork:fork + suffix_len].contiguous(),
            )
            assert not torch.equal(*suffixes), "fork suffixes must differ"
            physical = torch.cat((prefix, *suffixes))
            signature = hashlib.sha256(physical.numpy().tobytes()).hexdigest()
            windows.append((
                prefix, suffixes[0], suffixes[1],
                (first_row, second_row), fork, start, signature,
            ))
        return windows

    # Restrict each logical DP replica to its own disjoint real TQ rows;
    # select deterministic candidate pairs from each group. Require DP1's
    # *physical* 160-token tree to differ from DP0's, not just its row IDs.
    dp0 = candidate_windows(range(0, 4))
    dp1 = candidate_windows(range(4, 8))
    if not dp0 or not dp1:
        raise AssertionError(
            "recorded TQ lacks divergent sibling pairs with 128 shared + "
            "32 post-fork real tokens in each of the disjoint DP row groups"
        )
    chosen0 = dp0[0]
    chosen1 = next((w for w in dp1 if w[-1] != chosen0[-1]), None)
    if chosen1 is None:
        raise AssertionError(
            "recorded TQ DP groups have no distinct fork-window physical "
            "workloads; refuse a false-positive heterogeneous-DP PASS"
        )
    chosen = chosen0 if runtime.dp_rank == 0 else chosen1
    prefix, first, second, selected, fork, start, signature = chosen
    assert prefix.numel() == prefix_len
    assert first.numel() == second.numel() == suffix_len
    assert min(int(torch.min(v).item()) for v in (prefix, first, second)) >= 0
    assert max(int(torch.max(v).item()) for v in (prefix, first, second)) < 151936
    return prefix, first, second, selected, fork, start, signature

def _reference(model, prefix, first, second, tp_size):
    model.zero_grad(set_to_none=True)
    total_terms = 2 * (len(prefix) + len(first) - 1)
    loss = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    for suffix in (first, second):
        seq = torch.cat((prefix, suffix)).to(loss.device)
        positions = torch.arange(seq.numel(), device=loss.device).unsqueeze(0)
        logits = model(
            input_ids=seq.unsqueeze(0), position_ids=positions,
            attention_mask=None,
        )
        assert logits.shape[-1] == 151936 // tp_size
        ce = model.compute_language_model_loss(
            seq[1:].unsqueeze(0),
            logits[0, :-1, :].unsqueeze(1),
        ).reshape(-1)
        term = ce.float().sum() / total_terms
        term.backward()
        loss += term.detach()
    return float(loss.item()), {
        key: parameter.grad.detach().float().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _tpr(model, runtime, prefix, first, second):
    plan = _equivalence_tpr_plan(prefix, first, second)
    if runtime.tp_size > 1:
        assert_tp_plan_agreement(digest_segment_plan(plan), tp_size=runtime.tp_size)

    model.zero_grad(set_to_none=True)
    layers = tuple(range(1, model.config.num_layers + 1))
    # A Megatron CP=1 process group exists, but TPR's cp_group argument
    # means "enable distributed CP" and must be None for CP=1. Passing
    # the singleton group incorrectly asks SegmentExecutor to construct a
    # CP backend and fails before the first real TQ forward. DP topology
    # is independent of this choice: TP1xDP2 and TP2xDP2 both use CP=1.
    assert dist.get_world_size(group=runtime.cp_group) == 1
    executor = SegmentExecutor(
        model, plan,
        expected_layer_numbers=layers,
        cp_group=None,
    )
    result = FixedTopologyScheduler(plan, executor).run()
    return float(result.normalized_loss.item()), {
        key: parameter.grad.detach().float().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None
    }, digest_segment_plan(plan)


def test_real_qwen_module_dp2_and_tp2dp2_local_forest_correctness(parallel_runtime):
    runtime = parallel_runtime
    prefix, first, second, selected, fork, window_start, signature = (
        _real_tq_two_trajectories(runtime)
    )
    torch.manual_seed(20261010)
    reference, hf = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN, tp_size=runtime.tp_size
    )
    assert hf.vocab_size == 151936
    torch.manual_seed(20261010)
    model, _ = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN, tp_size=runtime.tp_size, load_weights=False
    )
    model.load_state_dict(reference.state_dict(), strict=True)

    # Within each replica both TP peers share the original *logical* samples.
    # Across DP replicas, model is replicated but tree/sample sets differ.
    native_loss, native_grad = _reference(
        reference, prefix, first, second, runtime.tp_size
    )
    tpr_loss, tpr_grad, digest = _tpr(
        model, runtime, prefix, first, second
    )
    # Do not assert before rank collectives. One TP rank may exceed the
    # numerical threshold while its peers pass. An early pytest assertion
    # leaves surviving ranks inside the later DP all_gather, which then
    # reports misleading HCCL communicator-creation / timeout errors.
    # Aggregate *diagnostic scalars* before applying the UNCHANGED gates.
    ce_gap = abs(native_loss - tpr_loss)
    same_names = native_grad.keys() == tpr_grad.keys()
    names = set(native_grad) & set(tpr_grad)
    norm_ref = sum(native_grad[key].square().sum().item() for key in names)
    norm_delta = sum(
        (native_grad[key] - tpr_grad[key]).square().sum().item()
        for key in names
    )
    relative_l2 = (norm_delta / max(norm_ref, 1e-24)) ** 0.5

    # This is NOT a new gradient collective: only two scalar diagnostics
    # are added over the existing native TP group, to distinguish a
    # single-shard >5% result from a model-wide sharded gradient error.
    tp_norms = torch.tensor(
        [norm_delta, norm_ref], device=runtime.device, dtype=torch.float32
    )
    if runtime.tp_size > 1:
        dist.all_reduce(tp_norms, op=dist.ReduceOp.SUM, group=runtime.tp_group)
    tp_global_l2 = (
        float(tp_norms[0].item()) / max(float(tp_norms[1].item()), 1e-24)
    ) ** 0.5
    finite = (
        torch.isfinite(torch.tensor(
            [native_loss, tpr_loss, relative_l2, tp_global_l2]
        )).all().item()
    )
    local_ok = (
        same_names and finite and ce_gap < 0.02 and relative_l2 < 0.05
    )

    # Check DIFFERENT original row *ownership* across DP groups. Two
    # physically identical token forests may legitimately be assigned to
    # different DP replicas; comparing only the segment-token SHA256 would
    # incorrectly mark that valid assignment as a failure. This test sends
    # only two int64 row IDs, NOT KV or parameter gradients.
    ownership = torch.tensor(
        selected, device=runtime.device, dtype=torch.int64
    )
    observed = [torch.empty_like(ownership) for _ in range(2)]
    dist.all_gather(observed, ownership, group=runtime.dp_group)
    row_sets = [set(value.cpu().tolist()) for value in observed]
    ownership_ok = row_sets[0].isdisjoint(row_sets[1])

    # A different original row ID alone is insufficient: verify the
    # physical 160-token trees differ after real-TQ fork cropping.
    physical_hash = torch.tensor(
        [int(signature[:15], 16)], dtype=torch.int64, device=runtime.device
    )
    physical_hashes = [torch.empty_like(physical_hash) for _ in range(2)]
    dist.all_gather(physical_hashes, physical_hash, group=runtime.dp_group)
    physical_ok = physical_hashes[0].item() != physical_hashes[1].item()
    local_ok = local_ok and ownership_ok and physical_ok

    # All ranks have completed forward/backward and native TP/DP collectives.
    # Explicit WORLD gather is diagnostics-only; *never* use world for
    # optimizer gradients or loss-token normalization (TP duplicates data).
    local_metrics = torch.tensor(
        [
            float(runtime.dp_rank), float(runtime.rank),
            native_loss, tpr_loss, ce_gap, relative_l2,
            tp_global_l2, float(local_ok),
        ],
        device=runtime.device, dtype=torch.float32,
    )
    world_metrics = [torch.empty_like(local_metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(world_metrics, local_metrics)
    rows_by_global_rank = [item.detach().cpu().tolist() for item in world_metrics]
    failures = [
        (global_rank, row) for global_rank, row in enumerate(rows_by_global_rank)
        if row[-1] != 1.0
    ]

    # When the gate fails, rank-local attribution is more useful than
    # weakening the BF16 tolerance. The score is each parameter's
    # contribution to the *whole local shard* gradient relative L2.
    if not local_ok:
        offenders = []
        for name in names:
            error_sq = (
                native_grad[name] - tpr_grad[name]
            ).square().sum().item()
            ref_sq = native_grad[name].square().sum().item()
            offenders.append((
                name,
                (error_sq / max(norm_ref, 1e-24)) ** 0.5,
                (error_sq / max(ref_sq, 1e-24)) ** 0.5,
                ref_sq ** 0.5,
            ))
        offenders.sort(key=lambda item: item[1], reverse=True)
        print(
            f"TPR_MODULE_PARALLEL_DIAG world_rank={dist.get_rank()} "
            f"tp={runtime.tp_size} dp_rank={runtime.dp_rank} "
            f"tp_rank={runtime.rank} loss_gap={ce_gap:.8f} "
            f"shard_grad_rel_l2={relative_l2:.7f} "
            f"tp_global_grad_rel_l2={tp_global_l2:.7f} "
            f"param_names_equal={same_names} "
            f"ownership_ok={ownership_ok} physical_ok={physical_ok} "
            f"top_contributors={offenders[:8]}",
            flush=True,
        )
    if dist.get_rank() == 0:
        for global_rank, row in enumerate(rows_by_global_rank):
            print(
                f"TPR_MODULE_PARALLEL_RANK world_rank={global_rank} "
                f"dp_rank={int(row[0])} tp_rank={int(row[1])} "
                f"native_ce={row[2]:.7f} tpr_ce={row[3]:.7f} "
                f"ce_gap={row[4]:.7f} shard_grad_rel_l2={row[5]:.7f} "
                f"tp_global_grad_rel_l2={row[6]:.7f} "
                f"gate_pass={bool(row[7])}",
                flush=True,
            )

    if runtime.rank == 0:
        print(
            f"TPR_MODULE_PARALLEL status=PASS tp={runtime.tp_size} dp=2 "
            f"dp_rank={runtime.dp_rank} real_tq_rows={selected} "
            f"window_start={window_start} true_fork={fork} "
            f"physical_hash={signature[:12]} "
            f"native_ce={native_loss:.7f} tpr_ce={tpr_loss:.7f} "
            f"grad_rel_l2={relative_l2:.6g} "
            f"full_dp_gradient_sync=NOT_TESTED "
            f"production_trainer_dispatch=NOT_TESTED",
            flush=True,
        )
    # Always finish every communicator before declaring the global test
    # failed. This avoids stranding successful peers in HCCL setup.
    dist.barrier()
    assert not failures, (
        "TPR real-TQ TP/DP module gradient/loss gate failed on "
        f"{[(r, int(v[0]), int(v[1]), round(v[5], 7)) for r, v in failures]}; "
        "see TPR_MODULE_PARALLEL_DIAG above; threshold unchanged"
    )
