# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Isolated native Megatron TP2+SP x DP2 module correctness, real Qwen3-1.7B.

Run with 4 Ascend NPUs and pretrained Qwen3-1.7B:
  TPR_RUN_TP_SP_DP=1 TPR_MODULE_SP=1 TPR_MODULE_TP_SIZE=2 TPR_MODULE_DP_SIZE=2 \
    torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29537 \
      -m pytest -vv -s -x \
      tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_dp2_npu.py

Two DP replicas intentionally use DIFFERENT physical tokens, while each TP
pair executes the same tree. This is a controlled isolated module test with
authentic TP-sharded pretrained weights, not real-TQ heterogeneous training,
production VERL trainer dispatch, DP gradient sync or optimizer integration.
The full-sequence oracle uses the existing controlled CANN square-causal
reference, not unmodified MindSpeed Flash Attention. SP is Megatron-native.
"""

from __future__ import annotations

import hashlib
import math
import os

import pytest
import torch
import torch.distributed as dist

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_tpr_dp_tpdp_module_npu import (
    _QWEN,
    _reference,
    _tpr,
    parallel_runtime,
)


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_TP_SP_DP") != "1",
    reason="set TPR_RUN_TP_SP_DP=1 for isolated four-NPU TP2+SPxDP2",
)


def _distinct_dp_trajectories(dp_rank):
    """Two sibling branches in each DP replica, with independent token data."""
    assert dp_rank in (0, 1)
    offset = 300 * dp_rank
    prefix = torch.arange(17 + offset, 81 + offset, dtype=torch.long)
    suffix1 = torch.arange(80101 + offset, 80133 + offset, dtype=torch.long)
    suffix2 = torch.arange(100201 + offset, 100233 + offset, dtype=torch.long)
    assert prefix.numel() == 64 and suffix1.numel() == suffix2.numel() == 32
    signature = hashlib.sha256(
        torch.cat((prefix, suffix1, suffix2)).numpy().tobytes()
    ).hexdigest()
    return prefix, suffix1, suffix2, int(signature[:15], 16)


def _first(value):
    return value[0] if isinstance(value, tuple) else value


def _observe_model(model):
    """Observe the true Megatron SP boundary without changing tensor contents."""
    first_attention = model.decoder.layers[0].self_attention
    shapes = {"attention_input": [], "qkv_output": [], "proj_output": []}
    logits_tail = []

    def on_input(module, args):
        shapes["attention_input"].append(tuple(args[0].shape))

    def on_qkv(module, args, output):
        shapes["qkv_output"].append(tuple(_first(output).shape))

    def on_proj(module, args, output):
        shapes["proj_output"].append(tuple(_first(output).shape))

    def on_model(module, args, output):
        logits = _first(output)
        if logits.ndim != 3 or logits.shape[-1] != 75968:
            raise AssertionError(f"unexpected TP2 vocab-shard logits: {logits.shape}")
        logits_tail.append(logits[0, -8:, :].detach().float().cpu())

    hooks = (
        first_attention.register_forward_pre_hook(on_input),
        first_attention.linear_qkv.register_forward_hook(on_qkv),
        first_attention.linear_proj.register_forward_hook(on_proj),
        model.register_forward_hook(on_model),
    )
    return shapes, logits_tail, hooks


def _has_layout(shapes, lengths):
    """Length vectors are the *full* logical sequences before SP sharding."""
    return (
        shapes["attention_input"]
        == [(n // 2, 1, 2048) for n in lengths]
        and shapes["qkv_output"] == [(n, 1, 2048) for n in lengths]
        and shapes["proj_output"] == [(n // 2, 1, 2048) for n in lengths]
    )


def test_native_tp2_sp_dp2_local_tree_and_gradients(parallel_runtime):
    runtime = parallel_runtime
    assert (runtime.tp_size, runtime.dp_size) == (2, 2)
    assert runtime.sequence_parallel is True, (
        "TP2+SPxDP2 gate requires TPR_MODULE_SP=1; "
        "do not silently run the no-SP topology"
    )
    assert dist.get_world_size(group=runtime.tp_group) == 2
    assert dist.get_world_size(group=runtime.dp_group) == 2
    assert dist.get_world_size(group=runtime.cp_group) == 1
    prefix, suffix1, suffix2, signature = _distinct_dp_trajectories(runtime.dp_rank)

    torch.manual_seed(20261011)
    reference, hf = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN, tp_size=2, sequence_parallel=True,
    )
    assert hf.vocab_size == 151936
    torch.manual_seed(20261011)
    candidate, _ = make_real_qwen3_tp_model(
        runtime, model_path=_QWEN, tp_size=2, load_weights=False,
        sequence_parallel=True,
    )
    candidate.load_state_dict(reference.state_dict(), strict=True)

    native_shapes, native_tail, native_hooks = _observe_model(reference)
    tree_shapes, tree_tail, tree_hooks = _observe_model(candidate)
    dist.barrier()
    try:
        native_loss, native_grad = _reference(
            reference, prefix, suffix1, suffix2, runtime.tp_size,
        )
    finally:
        for hook in native_hooks:
            hook.remove()
    try:
        tpr_loss, tpr_grad, plan_digest = _tpr(
            candidate, runtime, prefix, suffix1, suffix2,
        )
    finally:
        for hook in tree_hooks:
            hook.remove()

    # Native: two full 96-token trajectories; TPR: Push(P), Visit(S1),
    # Visit(S2), Pop(P). Megatron SP shards attention input, QKV all-gathers
    # the full logical segment and output projection reduces/scatters it.
    layout_ok = _has_layout(native_shapes, (96, 96)) and _has_layout(
        tree_shapes, (64, 32, 32, 64)
    )
    captures_ok = len(native_tail) == 2 and len(tree_tail) == 4
    logit_diff_sq = logit_ref_sq = 0.0
    if captures_ok:
        for full, leaf in zip(native_tail, tree_tail[1:3], strict=True):
            logit_diff_sq += (full - leaf).square().sum().item()
            logit_ref_sq += full.square().sum().item()
    logit_rel_l2 = (
        (logit_diff_sq / max(logit_ref_sq, 1e-24)) ** 0.5
        if captures_ok else float("inf")
    )

    same_names = native_grad.keys() == tpr_grad.keys()
    common_names = set(native_grad) & set(tpr_grad)
    grad_ref_sq = sum(native_grad[name].square().sum().item() for name in common_names)
    grad_delta_sq = sum(
        (native_grad[name] - tpr_grad[name]).square().sum().item()
        for name in common_names
    )
    grad_rel_l2 = (grad_delta_sq / max(grad_ref_sq, 1e-24)) ** 0.5
    tp_norms = torch.tensor(
        [grad_delta_sq, grad_ref_sq], dtype=torch.float32, device=runtime.device
    )
    dist.all_reduce(tp_norms, group=runtime.tp_group)
    tp_grad_rel_l2 = (
        float(tp_norms[0].item()) / max(float(tp_norms[1].item()), 1e-24)
    ) ** 0.5

    # This is a tiny control-plane ownership check, not a DP gradient sync.
    signatures = torch.tensor([signature], device=runtime.device, dtype=torch.int64)
    signatures_from_dp = [torch.empty_like(signatures) for _ in range(2)]
    dist.all_gather(signatures_from_dp, signatures, group=runtime.dp_group)
    physical_dp_separation = signatures_from_dp[0].item() != signatures_from_dp[1].item()

    ce_gap = abs(native_loss - tpr_loss)
    numeric_ok = (
        math.isfinite(native_loss) and math.isfinite(tpr_loss)
        and math.isfinite(grad_rel_l2) and math.isfinite(logit_rel_l2)
        and ce_gap < 0.02 and logit_rel_l2 < 0.05 and grad_rel_l2 < 0.05
    )
    # Keep the existing TP2 gates unchanged; a failing rank must not strand
    # peers in a collective. Collect every rank's diagnostics before failing.
    local_ok = layout_ok and captures_ok and same_names and bool(common_names)
    local_ok = local_ok and physical_dp_separation and numeric_ok
    local_metrics = torch.tensor(
        [
            float(runtime.dp_rank), float(runtime.rank), native_loss, tpr_loss,
            ce_gap, logit_rel_l2, grad_rel_l2, tp_grad_rel_l2,
            float(layout_ok), float(physical_dp_separation), float(local_ok),
        ],
        device=runtime.device, dtype=torch.float32,
    )
    world_metrics = [
        torch.empty_like(local_metrics) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(world_metrics, local_metrics)
    world_rows = [tensor.cpu().tolist() for tensor in world_metrics]
    if dist.get_rank() == 0:
        for global_rank, row in enumerate(world_rows):
            print(
                f"TPR_TP2_SP_DP2 world_rank={global_rank} "
                f"dp_rank={int(row[0])} tp_rank={int(row[1])} "
                f"native_ce={row[2]:.7f} tpr_ce={row[3]:.7f} "
                f"loss_gap={row[4]:.7f} logit_rel_l2={row[5]:.7f} "
                f"grad_rel_l2={row[6]:.7f} tp_grad_rel_l2={row[7]:.7f} "
                f"layout_ok={bool(row[8])} dp_workloads_distinct={bool(row[9])} "
                f"gate_pass={bool(row[10])}",
                flush=True,
            )
    if not local_ok:
        print(
            f"TPR_TP2_SP_DP2_DIAG global_rank={dist.get_rank()} "
            f"native_shapes={native_shapes} tree_shapes={tree_shapes} "
            f"native_captures={len(native_tail)} tree_captures={len(tree_tail)} "
            f"param_names_equal={same_names} plan_digest={plan_digest} "
            f"grad_rel_l2={grad_rel_l2:.7f} logit_rel_l2={logit_rel_l2:.7f}",
            flush=True,
        )
    dist.barrier()
    failures = [
        (index, row[0], row[1], row[-1])
        for index, row in enumerate(world_rows) if row[-1] != 1.0
    ]
    assert not failures, f"TP2+SPxDP2 module acceptance failed: {failures}"
