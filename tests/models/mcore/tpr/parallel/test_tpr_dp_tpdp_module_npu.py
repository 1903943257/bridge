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

import os
import sys
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
    # REAL TQ ONLY. No fabricated token sequences or tiny/random model.
    if not _REAL_TQ.is_file():
        pytest.skip(f"real UniAgent TQ dump is unavailable: {_REAL_TQ}")
    dump = torch.load(str(_REAL_TQ), weights_only=False, map_location="cpu")
    keys = tuple(dump["keys"])
    data = dump["tensordict"]
    seqs = _rows(data, "input_ids")
    assert len(seqs) >= 8 and len(keys) == len(seqs), (
        "real TQ gate requires the recorded >=8 distinct trajectory rows"
    )
    selected = (0, 1) if runtime.dp_rank == 0 else (4, 5)
    tokens = [seqs[index].detach().cpu().to(torch.long) for index in selected]
    prefix_len = 128
    suffix_len = 32
    assert all(t.numel() >= prefix_len + suffix_len for t in tokens)
    prefix = tokens[0][:prefix_len].contiguous()
    assert torch.equal(prefix, tokens[1][:prefix_len]), (
        "selected real TQ pair lacks the required exact 128-token LCP"
    )
    # Use recorded physical tokens, not synthetic input IDs.
    first, second = (
        tokens[0][prefix_len:prefix_len + suffix_len].contiguous(),
        tokens[1][prefix_len:prefix_len + suffix_len].contiguous(),
    )
    assert max(int(torch.max(v).item()) for v in (prefix, first, second)) < 151936
    return prefix, first, second, selected


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
    executor = SegmentExecutor(
        model, plan,
        expected_layer_numbers=layers,
        cp_group=runtime.cp_group,
    )
    result = FixedTopologyScheduler(plan, executor).run()
    return float(result.normalized_loss.item()), {
        key: parameter.grad.detach().float().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None
    }, digest_segment_plan(plan)


def test_real_qwen_module_dp2_and_tp2dp2_local_forest_correctness(parallel_runtime):
    runtime = parallel_runtime
    prefix, first, second, selected = _real_tq_two_trajectories(runtime)
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
    assert abs(native_loss - tpr_loss) < 0.02, (
        f"dp_rank={runtime.dp_rank}: real TQ CE mismatch "
        f"reference={native_loss:.7f}, tpr={tpr_loss:.7f}"
    )
    assert native_grad.keys() == tpr_grad.keys()
    norm_ref = sum(g.square().sum().item() for g in native_grad.values())
    norm_delta = sum(
        (native_grad[name] - tpr_grad[name]).square().sum().item()
        for name in native_grad
    )
    relative_l2 = (norm_delta / max(norm_ref, 1e-24)) ** 0.5
    assert relative_l2 < 0.05, (
        f"dp_rank={runtime.dp_rank}: rank-local TP{runtime.tp_size} "
        f"real TQ gradient rel_l2={relative_l2:.6g} exceeds 5% gate"
    )

    # Check DIFFERENT execution plans across DP groups with the native DP
    # process group. This sends only 8 bytes per rank; it does not attempt
    # to synchronize the model's gradients outside native Megatron DDP.
    word = int(digest[:8], 16)
    local = torch.tensor([word], device=runtime.device, dtype=torch.int64)
    observed = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(observed, local, group=runtime.dp_group)
    assert observed[0].item() != observed[1].item(), (
        "DP replicas unexpectedly received the same TPR tree / real TQ rows"
    )

    if runtime.rank == 0:
        print(
            f"TPR_MODULE_PARALLEL status=PASS tp={runtime.tp_size} dp=2 "
            f"dp_rank={runtime.dp_rank} real_tq_rows={selected} "
            f"native_ce={native_loss:.7f} tpr_ce={tpr_loss:.7f} "
            f"grad_rel_l2={relative_l2:.6g} "
            f"full_dp_gradient_sync=NOT_TESTED "
            f"production_trainer_dispatch=NOT_TESTED",
            flush=True,
        )
    dist.barrier()
