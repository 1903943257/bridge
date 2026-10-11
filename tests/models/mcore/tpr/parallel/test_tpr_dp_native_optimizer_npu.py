# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Four-NPU module gate: native Megatron DDP gradient sync + Megatron optimizer.

This isolates TP2 x DP2 with SP disabled/enabled and real pretrained Qwen3-1.7B.
It deliberately uses Megatron DDP + native finalize_model_grads and the
Megatron optimizer, not a homemade DP all-reduce or torch.optim.SGD workaround.

First gate (TP2 x DP2, native Megatron DDP):
  TPR_RUN_DP_NATIVE_OPT=1 TPR_MODULE_TP_SIZE=2 TPR_MODULE_DP_SIZE=2 \
  TPR_MODULE_SP=0 torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29538 -m pytest -vv -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_dp_native_optimizer_npu.py

Second gate (TP2 + native SP x DP2):
  TPR_RUN_DP_NATIVE_OPT=1 TPR_MODULE_TP_SIZE=2 TPR_MODULE_DP_SIZE=2 \
  TPR_MODULE_SP=1 torchrun --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port=29539 -m pytest -vv -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_dp_native_optimizer_npu.py

These are isolated module gates, NOT full VERL Trainer, clipped PPO/GRPO,
gradient accumulation across minibatches, distributed optimizer (ZeRO/
reduce-scatter), checkpointing or training-stability certification.
"""

from __future__ import annotations

import math
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import (
    DistributedDataParallel as MegatronDDP,
    DistributedDataParallelConfig,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

from verl.models.mcore.tpr.fixed_topology_scheduler import FixedTopologyScheduler
from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.tp_validation import (
    assert_tp_plan_agreement,
    digest_segment_plan,
)

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from ._tpr_cp_test_utils import _equivalence_tpr_plan
from .test_tpr_dp_tpdp_module_npu import _QWEN, parallel_runtime
from .test_tpr_tp2_sp_dp2_npu import _distinct_dp_trajectories


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_DP_NATIVE_OPT") != "1",
    reason="set TPR_RUN_DP_NATIVE_OPT=1 for the native Megatron DP optimizer gate",
)


def _gather(tensor, group):
    """Diagnostic-only snapshots; these collectives do NOT synchronize grads."""
    out = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(out, tensor.contiguous(), group=group)
    return out


def test_tp2_dp2_native_ddp_finalize_and_optimizer_one_step(parallel_runtime):
    runtime = parallel_runtime
    assert (runtime.tp_size, runtime.dp_size) == (2, 2)
    assert dist.get_world_size(group=runtime.tp_group) == 2
    assert dist.get_world_size(group=runtime.dp_group) == 2
    assert dist.get_world_size(group=runtime.cp_group) == 1
    assert runtime.sequence_parallel == (os.getenv("TPR_MODULE_SP", "0") == "1")

    # Each DP replica sees distinct *physical* tokens; its TP peers execute
    # exactly the same tree on distinct TP weight/head shards.
    prefix, suffix1, suffix2, signature = _distinct_dp_trajectories(runtime.dp_rank)
    plan = _equivalence_tpr_plan(prefix, suffix1, suffix2)
    assert_tp_plan_agreement(digest_segment_plan(plan), tp_size=2)

    torch.manual_seed(20261011)
    model, hf = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN, tp_size=2,
        sequence_parallel=runtime.sequence_parallel,
    )
    assert hf.vocab_size == 151936

    # VERL's native MCore training path: contiguous main_grad buffers under
    # Megatron DDP; no homemade DP collectives and no PyTorch DDP.
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        use_distributed_optimizer=False,  # first validate native DP all-reduce
    )
    wrapped = MegatronDDP(
        config=model.config, ddp_config=ddp_config, module=model,
    )
    wrapped.broadcast_params()
    native_opt_config = OptimizerConfig(
        optimizer="sgd",
        lr=1.0,  # visible one-step BF16 delta; not a training LR recommendation
        weight_decay=0.0,
        clip_grad=0.0,
        sgd_momentum=0.0,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
    )
    optimizer = get_megatron_optimizer(
        config=native_opt_config, model_chunks=[wrapped],
    )
    # Follow VERL ordering: native optimizer zero_grad, then DDP grad buffers.
    optimizer.zero_grad()
    wrapped.zero_grad_buffer()

    # The scheduler operates on the actual underlying GPTModel so it can
    # access rotary_pos_emb. DDP's registered autograd hooks still own the
    # parameters' main_grad buffers and finish_grad_sync path.
    executor = SegmentExecutor(
        model, plan,
        expected_layer_numbers=tuple(range(1, model.config.num_layers + 1)),
        cp_group=None,
    )
    result = FixedTopologyScheduler(plan, executor).run()
    loss = float(result.normalized_loss.item())
    qkv = model.decoder.layers[0].self_attention.linear_qkv.weight
    fc1 = model.decoder.layers[0].mlp.linear_fc1.weight
    assert qkv.main_grad is not None and fc1.main_grad is not None, (
        "Megatron DDP backward must populate FP32 main_grad, not just param.grad"
    )
    assert qkv.main_grad.dtype == torch.float32
    assert fc1.main_grad.dtype == torch.float32

    # Read a small diagnostic slice *before* native DP sync, gather both DP
    # replicas (same TP rank), then compute an independent expected mean.
    # The test never writes this mean into the model or its grad buffers.
    sample_size = 4096
    local_qkv_grad = qkv.main_grad.detach().reshape(-1)[:sample_size].float().clone()
    local_fc1_grad = fc1.main_grad.detach().reshape(-1)[:sample_size].float().clone()
    before_qkv = _gather(local_qkv_grad, runtime.dp_group)
    before_fc1 = _gather(local_fc1_grad, runtime.dp_group)
    expected_qkv = torch.stack(before_qkv).mean(dim=0)
    expected_fc1 = torch.stack(before_fc1).mean(dim=0)
    distinct_grad = (
        not torch.equal(before_qkv[0], before_qkv[1])
        or not torch.equal(before_fc1[0], before_fc1[1])
    )

    dp_hash = torch.tensor([signature], dtype=torch.int64, device=runtime.device)
    dp_hashes = _gather(dp_hash, runtime.dp_group)
    distinct_physical = dp_hashes[0].item() != dp_hashes[1].item()

    # THIS is the only training-gradient synchronization. It calls the
    # model's native finish_grad_sync on DP then native TP SP/layernorm
    # gradient reduction (when SP enabled), as the VERL Engine does.
    finalize_model_grads([wrapped], num_tokens=None, force_all_reduce=True)
    actual_qkv = qkv.main_grad.detach().reshape(-1)[:sample_size].float()
    actual_fc1 = fc1.main_grad.detach().reshape(-1)[:sample_size].float()
    max_qkv_mean_error = (actual_qkv - expected_qkv).abs().max().item()
    max_fc1_mean_error = (actual_fc1 - expected_fc1).abs().max().item()
    dp_sync_ok = (
        torch.allclose(actual_qkv, expected_qkv, atol=1e-5, rtol=1e-5)
        and torch.allclose(actual_fc1, expected_fc1, atol=1e-5, rtol=1e-5)
    )
    # Native DP all-reduce must leave TP-local parameter gradients identical
    # on the two replicas (it must NOT mix two distinct TP weight shards).
    matched_dp_qkv = _gather(actual_qkv.contiguous(), runtime.dp_group)
    replica_grad_identical = torch.equal(matched_dp_qkv[0], matched_dp_qkv[1])

    # On SP, a sequence-parallel-replicated norm needs the native TP
    # gradient reduction. Check the main_grad BEFORE optimizer.step(), which
    # may clear optimizer-side gradients on some Megatron revisions.
    norm = model.decoder.layers[0].input_layernorm.weight
    sp_norm_flag = bool(getattr(norm, "sequence_parallel", False))
    norm_tp_ok = True
    if runtime.sequence_parallel and sp_norm_flag:
        norm_views = _gather(
            norm.main_grad.detach().float().contiguous(),
            runtime.tp_group,
        )
        norm_tp_ok = torch.equal(norm_views[0], norm_views[1])

    # The optimizer must consume Megatron main_grad and update BF16 weights.
    # Snapshot a few parameter slices, not the full 1.7B model.
    probes = (qkv, fc1, model.decoder.layers[0].input_layernorm.weight)
    before_step = [
        param.detach().reshape(-1)[:sample_size].clone() for param in probes
    ]
    updated, grad_norm, zero_count = optimizer.step()
    after_step = [
        param.detach().reshape(-1)[:sample_size].clone() for param in probes
    ]
    update_counts = [
        int((prior != after).count_nonzero().item())
        for prior, after in zip(before_step, after_step, strict=True)
    ]
    optimizer_updated = bool(updated) and sum(update_counts) > 0
    after_dp = _gather(after_step[0].contiguous(), runtime.dp_group)
    param_replicas_identical = torch.equal(after_dp[0], after_dp[1])

    # clip_grad=0 requests no clipping and some Megatron releases return
    # grad_norm=None by design; this is not an optimizer failure.
    grad_norm_scalar = float(grad_norm) if grad_norm is not None else 0.0
    local_ok = all((
        math.isfinite(loss),
        distinct_physical, distinct_grad,
        dp_sync_ok, replica_grad_identical, norm_tp_ok,
        optimizer_updated, param_replicas_identical,
        math.isfinite(grad_norm_scalar),
    ))
    metrics = torch.tensor(
        [
            float(runtime.dp_rank), float(runtime.rank),
            float(runtime.sequence_parallel), loss,
            max_qkv_mean_error, max_fc1_mean_error,
            float(distinct_physical), float(distinct_grad),
            float(dp_sync_ok), float(replica_grad_identical),
            float(norm_tp_ok), float(optimizer_updated),
            float(param_replicas_identical), float(sum(update_counts)),
            grad_norm_scalar, float(local_ok),
        ],
        dtype=torch.float32, device=runtime.device,
    )
    world = _gather(metrics, dist.group.WORLD)
    observed = [t.cpu().tolist() for t in world]
    if dist.get_rank() == 0:
        for global_rank, row in enumerate(observed):
            print(
                f"TPR_NATIVE_DP_OPT world_rank={global_rank} "
                f"dp_rank={int(row[0])} tp_rank={int(row[1])} "
                f"sp={bool(row[2])} loss={row[3]:.7f} "
                f"qkv_mean_max_abs={row[4]:.8g} "
                f"fc1_mean_max_abs={row[5]:.8g} "
                f"workloads_distinct={bool(row[6])} "
                f"local_grads_distinct={bool(row[7])} "
                f"native_dp_synced={bool(row[8])} "
                f"dp_grad_equal={bool(row[9])} "
                f"sp_norm_tp_synced={bool(row[10])} "
                f"optimizer_step={bool(row[11])} "
                f"dp_params_equal={bool(row[12])} "
                f"changed_elements={int(row[13])} "
                f"native_grad_norm={row[14]:.7g} "
                f"gate_pass={bool(row[15])}",
                flush=True,
            )
    dist.barrier()
    failures = [(rank, row) for rank, row in enumerate(observed) if row[-1] != 1.0]
    assert not failures, f"native Megatron TP2(+SP)xDP2 sync/step gate failed: {failures}"
