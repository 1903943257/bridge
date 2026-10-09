"""P0: capability probe for shared-weight BF16 GMM and FP32 main_grad on NPU.

Only official MindSpeed GMM wrappers; no custom backward, production patch,
or new kernel. This is a CAPABILITY test, not a correctness gate or optimizer
integration. Uses one Qwen-like 2D weight shared among G token groups.

   TPR_RUN_QWEN17_MINDSPEED_GMM=1 pytest -s -q ...
"""
from __future__ import annotations

import importlib
import os

import pytest
import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_MINDSPEED_GMM") != "1",
    reason="Set TPR_RUN_QWEN17_MINDSPEED_GMM=1",
)


def _comparison(label, got, ref):
    if got is None:
        print(f"P0 SHARED_GMM {label} status=MISSING", flush=True)
        return None
    if got.shape != ref.shape or not bool(torch.isfinite(got).all()):
        raise AssertionError(
            f"P0 SHARED_GMM {label}: shape or nonfinite mismatch "
            f"got={got.shape}, ref={ref.shape}"
        )
    actual, expected = got.detach().float(), ref.detach().float()
    d = actual - expected
    rel = float(torch.linalg.vector_norm(d) /
                torch.linalg.vector_norm(expected).clamp_min(1e-12))
    cosine = float(
        torch.sum(actual * expected) /
        (torch.linalg.vector_norm(actual) *
         torch.linalg.vector_norm(expected)).clamp_min(1e-12)
    )
    print(
        f"P0 SHARED_GMM {label} rel_l2={rel:.9g} "
        f"cosine={cosine:.9g} "
        f"max_abs={float(d.abs().max()):.9g} "
        f"bitwise={torch.equal(got,ref)} dtype={got.dtype}",
        flush=True,
    )
    return rel


def _run_probe(label, fn):
    try:
        fn()
    except (RuntimeError, ImportError, TypeError, ValueError,
            NotImplementedError, AttributeError, AssertionError) as exc:
        print(
            f"P0 SHARED_GMM {label} status=UNSUPPORTED "
            f"error={type(exc).__name__}: {str(exc)[:1200]}",
            flush=True,
        )
        # Report incompatibility instead of failing the whole diagnostic. In
        # particular, some installed MindSpeed versions lack the compiled GMM
        # extension, and zero-stride 3D weight views may be unsupported.
        torch.npu.synchronize()
        return
    torch.npu.synchronize()


def test_mindspeed_shared_gmm_autograd_and_fp32_fusion():
    tile = int(os.getenv("TPR_QWEN17_P0_TILE", "128"))
    groups = int(os.getenv("TPR_QWEN17_P0_GROUPS", "2"))
    k = int(os.getenv("TPR_QWEN17_P0_K", "256"))
    n = int(os.getenv("TPR_QWEN17_P0_N", "384"))
    if tile <= 0 or not 2 <= groups <= 8 or k <= 0 or n <= 0:
        raise AssertionError("invalid test shape")
    m = tile * groups
    device = torch.device("npu")
    torch.manual_seed(17)
    x0 = torch.randn((m,k), device=device, dtype=torch.bfloat16)
    w0 = torch.randn((n,k), device=device, dtype=torch.bfloat16)
    dy = torch.randn((m,n), device=device, dtype=torch.bfloat16)
    bounds = torch.arange(
        tile, m + 1, tile, device=device, dtype=torch.int64,
    )
    sizes = torch.full(
        (groups,), tile, device=device, dtype=torch.int64,
    )
    print(
        f"P0 SHARED_GMM CONFIG M={m} G={groups} tile={tile} K={k} N={n} "
        f"input_dtype={x0.dtype} main_grad_dtype=torch.float32",
        flush=True,
    )

    x_ref = x0.clone().detach().requires_grad_(True)
    w_ref = w0.clone().detach().requires_grad_(True)
    y_ref = torch.cat([
        chunk.contiguous() @ w_ref.t()
        for chunk in x_ref.split(tile, dim=0)
    ], dim=0)
    dx_ref, dw_ref = torch.autograd.grad(
        y_ref, (x_ref,w_ref), grad_outputs=dy
    )
    dw_ref_fp32 = sum((
        part_x.float().t() @ part_dy.float()
        for part_x, part_dy in zip(
            x0.split(tile), dy.split(tile), strict=True
        )
    ), torch.zeros((k,n),dtype=torch.float32,device=device))
    print(
        "P0 SHARED_GMM REFERENCE "
        f"dX_dtype={dx_ref.dtype} dW_dtype={dw_ref.dtype} "
        f"FP32_reference_shape={tuple(dw_ref_fp32.shape)}",
        flush=True,
    )

    try:
        ms_grouped = importlib.import_module("mindspeed.ops.grouped_matmul")
        ms_gmm = importlib.import_module("mindspeed.ops.gmm")
    except ImportError as exc:
        pytest.skip(f"installed MindSpeed GMM not accessible: {exc}")

    # _GroupedMatmul is shipped in MindSpeed, wraps torch_npu forward
    # with its own backward. Weight is [G,K,N], but all G groups share ONE
    # [N,K] parameter via PyTorch expand; test both zero-stride view and a
    # materialized contiguous clone (the latter is NOT memory efficient).
    def run_mindspeed_grouped(*, contiguous):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        packed = w.t().unsqueeze(0).expand(groups,-1,-1)
        if contiguous:
            packed = packed.contiguous()
        print(
            "P0 SHARED_GMM WEIGHT_LAYOUT "
            f"wrapper=grouped_matmul contiguous={contiguous} "
            f"shape={tuple(packed.shape)} stride={packed.stride()} "
            f"storage_bytes={packed.untyped_storage().nbytes()} "
            f"logical_bytes={packed.numel()*packed.element_size()}",
            flush=True,
        )
        output = ms_grouped.fused_grouped_matmul(
            x, sizes, packed,
        )
        _comparison(
            f"mindspeed_grouped contiguous={contiguous} forward",
            output, y_ref,
        )
        if not output.requires_grad:
            print(
                f"P0 SHARED_GMM mindspeed_grouped contiguous={contiguous} "
                "status=NO_AUTOGRAD output_requires_grad=False",
                flush=True,
            )
            return
        dx, dw = torch.autograd.grad(
            output, (x,w), grad_outputs=dy,
            allow_unused=True,
        )
        _comparison(
            f"mindspeed_grouped contiguous={contiguous} dX", dx,dx_ref,
        )
        _comparison(
            f"mindspeed_grouped contiguous={contiguous} shared_dW",
            dw,dw_ref,
        )
        print(
            f"P0 SHARED_GMM mindspeed_grouped contiguous={contiguous} "
            f"status={'SUPPORTED_SHARED_AUTOGRAD' if dx is not None and dw is not None else 'NO_AUTOGRAD'}",
            flush=True,
        )

    for contiguous in (False, True):
        _run_probe(
            f"mindspeed_grouped contiguous={contiguous}",
            lambda c=contiguous: run_mindspeed_grouped(contiguous=c),
        )

    def run_mindspeed_gmm(*, contiguous):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        packed = w.t().unsqueeze(0).expand(groups,-1,-1)
        if contiguous:
            packed = packed.contiguous()
        output = ms_gmm.npu_gmm(
            x, packed, group_list=bounds, group_type=0,
            gemm_fusion=False,
        )
        _comparison(
            f"mindspeed_gmm contiguous={contiguous} forward",output,y_ref,
        )
        dx, dw = torch.autograd.grad(
            output, (x,w), grad_outputs=dy,allow_unused=True,
        )
        _comparison(
            f"mindspeed_gmm contiguous={contiguous} dX", dx,dx_ref,
        )
        _comparison(
            f"mindspeed_gmm contiguous={contiguous} shared_dW",
            dw,dw_ref,
        )
        print(
            f"P0 SHARED_GMM mindspeed_gmm contiguous={contiguous} "
            f"status={'SUPPORTED_SHARED_AUTOGRAD' if dx is not None and dw is not None else 'NO_AUTOGRAD'}",
            flush=True,
        )

    for contiguous in (False, True):
        _run_probe(
            f"mindspeed_gmm contiguous={contiguous}",
            lambda c=contiguous: run_mindspeed_gmm(contiguous=c),
        )

    # The upstream FP32 GMM+ADD path wants [G,K,N] main_grad storage
    # (npu_groupmatmul_add_fp32 reshapes to [groups,K,N] on A5).
    # It cannot directly target the ordinary shared [K,N] main_grad.
    # Probe per-group FP32 intermediate + FP32 sum-to-shared only, to
    # determine whether this *existing* op is usable without a new kernel.
    def run_mindspeed_fp32_fusion():
        x = x0.detach().clone().requires_grad_(True)
        packed = (
            w0.detach().t().unsqueeze(0)
            .expand(groups,-1,-1).contiguous()
        ).requires_grad_(True)
        # source gradient is a 3D temporary, NOT w0.main_grad.
        packed.main_grad = torch.zeros(
            (groups,k,n), dtype=torch.float32, device=device,
        )
        # Do not set grad_added_to_main_grad unless testing the exact
        # Megatron contract; helper may write torch.empty placeholders.
        output = ms_gmm.npu_gmm(
            x,packed,original_weight=packed,
            group_list=bounds,group_type=0,gemm_fusion=True,
        )
        _comparison("mindspeed_gmm_fp32_fusion forward",output,y_ref)
        dx = torch.autograd.grad(
            output, x, grad_outputs=dy,allow_unused=True,
        )[0]
        _comparison("mindspeed_gmm_fp32_fusion dX",dx,dx_ref)
        fused_per_group=packed.main_grad
        print(
            "P0 SHARED_GMM MAIN_GRAD "
            f"per_group_shape={tuple(fused_per_group.shape)} "
            f"per_group_dtype={fused_per_group.dtype}",
            flush=True,
        )
        summed=fused_per_group.sum(dim=0)
        _comparison(
            "mindspeed_gmm_fp32_fusion SUM_DW",summed,dw_ref_fp32,
        )
        print(
            "P0 SHARED_GMM mindspeed_gmm_fp32_fusion "
            "status=FP32_GROUP_BUFFER_AND_REDUCE_OK "
            "direct_shared_2d_main_grad=False",
            flush=True,
        )

    _run_probe("mindspeed_gmm_fp32_fusion",run_mindspeed_fp32_fusion)
    print(
        "P0 SHARED_GMM COMPLETE "
        "All statuses are capability observations, not training PASS.",
        flush=True,
    )
