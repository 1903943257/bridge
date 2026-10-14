"""CPU checks for isolated BF16 vs FP32 weight-gradient reductions."""
from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from ..parallel.grad_sync_precision import (
    offline_grad_sync_dtype,
    reduce_average_weight_grad,
)


def test_offline_grad_precision_flag_defaults_to_native_bf16(monkeypatch):
    monkeypatch.delenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", raising=False)
    assert offline_grad_sync_dtype() == "bf16"
    monkeypatch.setenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", "fp32")
    assert offline_grad_sync_dtype() == "fp32"
    monkeypatch.setenv("TPR_OFFLINE_GRAD_SYNC_DTYPE", "bad")
    with pytest.raises(ValueError, match="TPR_OFFLINE_GRAD_SYNC_DTYPE"):
        offline_grad_sync_dtype()


@pytest.mark.parametrize("reduce_dtype", ["bf16", "fp32"])
def test_reduce_average_does_not_change_original_grad_dtype(
    monkeypatch, reduce_dtype,
):
    original = torch.tensor([0.25, -0.5, 0.75], dtype=torch.bfloat16)
    param = torch.nn.Parameter(torch.ones_like(original))
    param.grad = original.clone()
    calls = []

    def fake_collective(tensor, *, op, group):
        calls.append((tensor.dtype, op, group))
        tensor.mul_(4)

    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "all_reduce", fake_collective)
    handle = object()
    reduce_average_weight_grad(
        param, handle, name="w", reduce_dtype=reduce_dtype,
    )
    assert len(calls) == 1
    assert calls[0][0] == (
        torch.float32 if reduce_dtype == "fp32" else torch.bfloat16
    )
    assert calls[0][1] == dist.ReduceOp.SUM
    assert calls[0][2] is handle
    assert param.grad.dtype == torch.bfloat16
    torch.testing.assert_close(param.grad, original, rtol=0, atol=0)


def test_reduce_average_rejects_missing_gradient(monkeypatch):
    param = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="Missing gradient"):
        reduce_average_weight_grad(param, object(), name="missing",
                                   reduce_dtype="fp32")
