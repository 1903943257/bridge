"""Experimental offline ONLY: precision control for weight gradient collectives.

Ascend 910B BF16 SUM/4 produced different post-finalize BF16 gradients
on weight replicas in the isolated HCCL probe. Promoting each existing
BF16 parameter gradient to FP32 for the SUM/average can independently
test/fix the consistency issue without altering TPR tree execution.
This helper is NOT installed into production VERL / Megatron paths.

The extra FP32 per-parameter staging memory and FP32 bandwidth must
be included in measured F/B+finalize performance. Do not mix FP32
gradient-sync results with the native BF16 baseline.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def offline_grad_sync_dtype() -> str:
    precision = os.getenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", "bf16").lower()
    if precision not in ("bf16", "fp32"):
        raise ValueError(
            "TPR_OFFLINE_GRAD_SYNC_DTYPE must be 'bf16' or 'fp32'; "
            f"got {precision!r}"
        )
    return precision


@torch.no_grad()
def reduce_average_weight_grad(param, group, *, name: str,
                               reduce_dtype: str) -> None:
    """SUM/average one already accumulated TP-sharded weight gradient.

    Preserve the original tensor/dtype/parameter identity. Never apply
    this helper to ancestor Prefix KV gradients or autograd buffers.
    """
    if param.grad is None:
        raise RuntimeError(f"Missing gradient before finalize: {name}")
    if reduce_dtype not in ("bf16", "fp32"):
        raise ValueError(f"Unexpected weight gradient reduce_dtype={reduce_dtype!r}")
    size = dist.get_world_size(group)
    if size < 1:
        raise AssertionError("Gradient group is empty")
    if reduce_dtype == "fp32":
        # Only one tensor is staged at a time. FP32 accumulation is a
        # controlled alternative to the original in-place BF16 reduction.
        workspace = param.grad.detach().to(dtype=torch.float32)
        dist.all_reduce(workspace, op=dist.ReduceOp.SUM, group=group)
        workspace.div_(size)
        param.grad.copy_(workspace.to(dtype=param.grad.dtype))
    else:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
        param.grad.div_(size)
