# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Isolated CP1 dataset #1 oracle for a Ring CP2 gradient-drift investigation.

This is a diagnostic *not* a replacement for TPR's production Ring CP.
It evaluates the SAME physical tokens as the failing DP1 dataset via
TP1, CP1 (controlled CANN square vs rectangular), DP1, SP=False.
Reference: two independent full causal trajectories, no Prefix reuse.
Candidate: one shared-Prefix TPR tree with its ordinary CP1 attention.

  TPR_RUN_CP1_DATASET_CONTROL=1 torchrun --nproc_per_node=1 \
    -m pytest -vv -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_cp1_dataset_control_npu.py

No custom DP/CP communication, modified kernels or threshold relaxation.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_tpr_dp_tpdp_module_npu import _reference, _tpr
from .test_tpr_tp2_sp_dp2_npu import _distinct_dp_trajectories


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_CP1_DATASET_CONTROL") != "1",
    reason="set TPR_RUN_CP1_DATASET_CONTROL=1 for 1-NPU controlled CP1 diagnostic",
)

_QWEN = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))


def test_cp1_same_dataset_as_failing_ringcp2():
    assert int(os.getenv("WORLD_SIZE", "1")) == 1, (
        "CP1 differential control requires torchrun --nproc_per_node=1"
    )
    import torch_npu  # noqa: F401

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
        "tensor_model_parallel_size": 1,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(261011)

    runtime = SimpleNamespace(
        rank=0, tp_rank=0, cp_rank=0, dp_rank=0,
        tp_size=1, cp_size=1, dp_size=1, sequence_parallel=False,
        device=torch.device("npu", local_rank),
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        cp_group=parallel_state.get_context_parallel_group(),
        dp_group=parallel_state.get_data_parallel_group(),
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
    )
    dataset_id = int(os.getenv("TPR_CP1_CONTROL_DATASET", "1"))
    if dataset_id not in (0, 1):
        raise ValueError("TPR_CP1_CONTROL_DATASET must be 0 or 1")
    prefix, first, second, _ = _distinct_dp_trajectories(dataset_id)
    try:
        torch.manual_seed(20261011)
        reference, hf = make_real_qwen3_tp_model(
            runtime, model_path=_QWEN, tp_size=1, cp_size=1,
            sequence_parallel=False,
        )
        torch.manual_seed(20261011)
        tpr, _ = make_real_qwen3_tp_model(
            runtime, model_path=_QWEN, tp_size=1, cp_size=1,
            sequence_parallel=False, load_weights=False,
        )
        tpr.load_state_dict(reference.state_dict(), strict=True)
        assert hf.vocab_size == 151936

        reference_loss, reference_grads = _reference(
            reference, prefix, first, second, tp_size=1,
        )
        tpr_loss, tpr_grads, _ = _tpr(tpr, runtime, prefix, first, second)

        names_equal = reference_grads.keys() == tpr_grads.keys()
        names = set(reference_grads) & set(tpr_grads)
        ref_sq = sum(reference_grads[n].square().sum().item() for n in names)
        delta_sq = sum(
            (reference_grads[n] - tpr_grads[n]).square().sum().item()
            for n in names
        )
        grad_rel_l2 = (delta_sq / max(ref_sq, 1e-24)) ** 0.5
        loss_gap = abs(reference_loss - tpr_loss)
        contributors = sorted((
            (
                n,
                ((reference_grads[n] - tpr_grads[n]).square().sum().item()
                 / max(ref_sq, 1e-24)) ** 0.5,
            )
            for n in names
        ), key=lambda pair: pair[1], reverse=True)[:8]
        passed = (
            names_equal and len(names) >= 200
            and all(math.isfinite(x) for x in (
                reference_loss, tpr_loss, grad_rel_l2,
            ))
            and loss_gap < 0.02 and grad_rel_l2 < 0.05
        )
        print(
            f"TPR_CP1_DATASET_CONTROL dataset_id={dataset_id} "
            f"reference_ce={reference_loss:.7f} tpr_ce={tpr_loss:.7f} "
            f"loss_gap={loss_gap:.8f} grad_rel_l2={grad_rel_l2:.8f} "
            f"reference_grad_norm={ref_sq ** 0.5:.8f} "
            f"grad_names_equal={names_equal} grad_count={len(names)} "
            f"top_contributors={contributors} "
            f"gate_pass={bool(passed)}",
            flush=True,
        )
        assert passed, (
            "CP1 numerical control failed; preserve this as diagnostic "
            f"evidence, not a reason to relax 5% threshold: {grad_rel_l2}"
        )
    finally:
        dist.barrier()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
