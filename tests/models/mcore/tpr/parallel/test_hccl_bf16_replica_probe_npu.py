"""Isolated 8-NPU HCCL BF16 all-reduce identity test: no model or PPO data.

Probe exactly the current DP×CP gradient-finalize communication path:
BF16 SUM over four ranks, BF16 division by four, then raw BF16
AllGather to test whether all four ranks really hold equal results.
Large and small tensor sizes are both covered, independently from
TPR autograd and parameter storage.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from .test_tpr_tp2cp2dp2_one_step_npu import _runtime

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_HCCL_BF16_GRAD_PROBE") != "1",
    reason="explicitly opt in to isolated 8-NPU BF16 gradient collective test",
)


def test_bf16_gradient_finalize_collectives_match_across_cpdp_replicas():
    runtime = _runtime()
    try:
        cpdp = runtime.dp_cp_group
        group_members = tuple(dist.get_process_group_ranks(cpdp))
        rank = dist.get_rank()
        for n in (2560, 1048576):
            base = torch.linspace(
                -0.025, 0.025, steps=n, dtype=torch.float32,
                device=runtime.device,
            )
            # All ranks contribute a rank-specific BF16 gradient. This
            # requires a true SUM, not a copied/broadcast input.
            values = (base + (rank - 3.5) * 0.000244140625).to(torch.bfloat16)
            dist.all_reduce(values, op=dist.ReduceOp.SUM, group=cpdp)
            values.div_(dist.get_world_size(cpdp))
            replicas = [torch.empty_like(values) for _ in group_members]
            dist.all_gather(replicas, values, group=cpdp)
            # Compare original BF16 bit patterns on CPU, not MIN/MAX reduce.
            stack = torch.stack(replicas, dim=0).cpu()
            equal = all(torch.equal(stack[0], x) for x in stack[1:])
            span = (stack.float().max(dim=0).values -
                    stack.float().min(dim=0).values)
            max_abs = float(span.abs().max().item())
            differing = int((span != 0).sum().item())
            # Collective world MAX uses supported float32, not FP64.
            status = torch.tensor(
                [max_abs, float(differing)], dtype=torch.float32,
                device=runtime.device,
            )
            dist.all_reduce(status, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
            if rank == 0:
                print(
                    f"TPR_BF16_HCCL_PROBE elements={n} cpdp_group={group_members} "
                    f"world_max_abs={float(status[0].item()):.9g} "
                    f"world_max_differing_elements={int(status[1].item())} "
                    f"all_replicas_exact={int(float(status[1].item()) == 0.0)}",
                    flush=True,
                )
            if float(status[1].item()) != 0.0 or not equal:
                raise AssertionError(
                    f"BF16 HCCL AllReduce did not produce identical gradients "
                    f"after SUM/4 at elements={n} group={group_members}: "
                    f"world_max_abs={status[0].item()} "
                    f"world_max_differing_elements={status[1].item()}"
                )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
