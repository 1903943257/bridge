"""Isolated autograd capability probe for Ascend grouped GEMM.

This ONLY tests whether torch_npu.npu_grouped_matmul propagates dX/dW when
all group weights refer to ONE trainable tensor. It does not patch production
TPR or claim equivalent PPO/weight-gradient accumulation. Explicit opt-in.
"""
from __future__ import annotations

import os

import pytest
import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_GROUPED_BACKWARD") != "1",
    reason="Set TPR_RUN_QWEN17_GROUPED_BACKWARD=1",
)


def _diff(label, actual, reference):
    if actual is None or reference is None:
        raise AssertionError(f"{label}: missing gradient")
    a, b = actual.detach().float(), reference.detach().float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all()):
        raise AssertionError(f"{label}: invalid shape/nonfinite gradient")
    rel_l2 = float(
        torch.linalg.vector_norm(a-b)
        / torch.linalg.vector_norm(b).clamp_min(1e-12)
    )
    cosine = float(
        (a*b).sum() / (
            torch.linalg.vector_norm(a)
            * torch.linalg.vector_norm(b)
        ).clamp_min(1e-12)
    )
    exact = torch.equal(actual, reference)
    print(
        f"QWEN17 GROUPED BACKWARD {label} rel_l2={rel_l2:.9g} "
        f"cosine={cosine:.9g} bitwise={exact} "
        f"actual_dtype={actual.dtype} expected_dtype={reference.dtype}",
        flush=True,
    )


def test_grouped_shared_weight_backward_vs_tiled_gemm():
    import torch_npu

    grouped_api = getattr(torch_npu, "npu_grouped_matmul", None)
    if grouped_api is None:
        pytest.skip("torch_npu.npu_grouped_matmul unavailable")
    tile = int(os.getenv("TPR_QWEN17_GROUPED_BACKWARD_TILE", "128"))
    n_groups = int(os.getenv("TPR_QWEN17_GROUPED_BACKWARD_N", "2"))
    k = int(os.getenv("TPR_QWEN17_GROUPED_BACKWARD_K", "2048"))
    n = int(os.getenv("TPR_QWEN17_GROUPED_BACKWARD_OUT", "2048"))
    if tile <= 0 or n_groups < 2 or min(k, n) <= 0:
        raise AssertionError("invalid shape")
    m = tile * n_groups
    dev = torch.device("npu")
    torch.manual_seed(29)
    x_base = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w_base = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    dy = torch.randn(m, n, device=dev, dtype=torch.bfloat16)

    x_ref = x_base.detach().clone().requires_grad_(True)
    w_ref = w_base.detach().clone().requires_grad_(True)
    y_ref = torch.cat(
        [torch.matmul(part.contiguous(), w_ref)
         for part in x_ref.split(tile, dim=0)],
        dim=0,
    )
    ref_dx, ref_dw = torch.autograd.grad(
        y_ref, (x_ref, w_ref), grad_outputs=dy,
    )
    print(
        f"QWEN17 GROUPED BACKWARD reference="
        f"M={m} K={k} N={n} tile={tile} n_groups={n_groups} "
        f"dx_dtype={ref_dx.dtype} dw_dtype={ref_dw.dtype}",
        flush=True,
    )

    x_grouped = x_base.detach().clone().requires_grad_(True)
    w_grouped = w_base.detach().clone().requires_grad_(True)
    boundaries = torch.arange(
        tile, m + 1, tile, device=dev, dtype=torch.int64,
    )
    try:
        result = grouped_api(
            [x_grouped], [w_grouped] * n_groups,
            group_list=boundaries, group_type=0,
            group_list_type=0, split_item=2,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 1:
            raise AssertionError(f"unexpected grouped output {type(result)}")
        y_grouped = result[0]
        _diff("forward", y_grouped, y_ref)
        if not y_grouped.requires_grad:
            print(
                "QWEN17 GROUPED BACKWARD status=NO_AUTOGRAD "
                "output_requires_grad=False", flush=True,
            )
            return
        dx, dw = torch.autograd.grad(
            y_grouped, (x_grouped, w_grouped),
            grad_outputs=dy, allow_unused=True,
        )
        if dx is None or dw is None:
            print(
                "QWEN17 GROUPED BACKWARD status=NO_AUTOGRAD "
                f"dx_present={dx is not None} dw_present={dw is not None}",
                flush=True,
            )
            return
        _diff("dX", dx, ref_dx)
        _diff("shared_dW", dw, ref_dw)
        print(
            "QWEN17 GROUPED BACKWARD status=SUPPORTED_AUTOGRAD "
            "numerical_equivalence_reported_separately=True",
            flush=True,
        )
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        print(
            f"QWEN17 GROUPED BACKWARD status=UNSUPPORTED "
            f"exception={type(exc).__name__}: {exc}",
            flush=True,
        )
    torch.npu.synchronize()
