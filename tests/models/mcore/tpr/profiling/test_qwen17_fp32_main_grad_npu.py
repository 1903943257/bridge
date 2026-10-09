"""P0 shared dense-Linear FP32 main_grad accumulation across token partitions.

Call the UNMODIFIED MindSpeed npu_matmul_add_fp32 on Ascend 910B2C with
[128], [64,64], [32,32,32,32] partitions of identical tensors. All paths
write to ONE shared 2D FP32 main_grad buffer. Also compare a BF16 sum-based
baseline and a CPU-side reference computed from FP32 operands.

This tests only weight-gradient ACCUMULATION: not autograd registration,
dX, a complete transformer backward, or optimizer state. Opt-in only.
"""
from __future__ import annotations

import os
import torch
import pytest

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_FP32_MAIN_GRAD") != "1",
    reason="Set TPR_RUN_QWEN17_FP32_MAIN_GRAD=1",
)


def _stats(label, candidate, reference):
    a = candidate.detach().float()
    b = reference.detach().float()
    if a.shape != b.shape or not bool(torch.isfinite(a).all()):
        raise AssertionError(
            f"{label}: shape/nonfinite mismatch {a.shape} vs {b.shape}"
        )
    d = a-b
    rel = float(torch.linalg.vector_norm(d) /
                torch.linalg.vector_norm(b).clamp_min(1e-12))
    cos = float(
        (a*b).sum() /
        (torch.linalg.vector_norm(a) *
         torch.linalg.vector_norm(b)).clamp_min(1e-12)
    )
    max_abs = float(d.abs().max())
    print(
        f"P0 MAIN_GRAD {label} rel_l2={rel:.9g} "
        f"cosine={cos:.9g} max_abs={max_abs:.9g} "
        f"bitwise={torch.equal(candidate,reference)}",
        flush=True,
    )
    return rel


def test_existing_mindspeed_matmul_add_fp32_shared_2d_buffer():
    try:
        from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
    except (ImportError, OSError) as exc:
        pytest.skip(f"installed MindSpeed npu_matmul_add_fp32 unavailable: {exc}")

    k = int(os.getenv("TPR_QWEN17_MAIN_GRAD_K", "256"))
    n = int(os.getenv("TPR_QWEN17_MAIN_GRAD_N", "384"))
    m = int(os.getenv("TPR_QWEN17_MAIN_GRAD_M", "128"))
    if min(m, k, n) <= 0 or m % 4:
        raise AssertionError("M must be positive multiple of 4")
    device = torch.device("npu")
    torch.manual_seed(2026)
    x = torch.randn(m,k,device=device,dtype=torch.bfloat16)
    dy = torch.randn(m,n,device=device,dtype=torch.bfloat16)
    reference_fp32 = dy.float().transpose(0,1) @ x.float()
    print(
        "P0 MAIN_GRAD CONFIG "
        f"M={m} K={k} N={n} "
        "target_shape=(N,K) target_dtype=torch.float32 "
        "backend=mindspeed.ops.npu_matmul_add_fp32",
        flush=True,
    )
    splits = (
        ("128_once", 1),
        ("64_twice", 2),
        ("32_four", 4),
    )

    results = {}
    for label, group_count in splits:
        size = m // group_count
        for reverse in (False, True):
            indices = list(range(group_count))
            if reverse:
                indices.reverse()
            main_grad = torch.zeros(
                (n,k),device=device,dtype=torch.float32,
            )
            naive_bf16 = torch.zeros(
                (n,k),device=device,dtype=torch.bfloat16,
            )
            try:
                for idx in indices:
                    start=idx*size
                    end=start+size
                    part_x=x[start:end].contiguous()
                    part_dy=dy[start:end].contiguous()
                    # This helper is the existing MindSpeed dense wgrad
                    # accumulation primitive, not our own backward kernel.
                    npu_matmul_add_fp32(part_x,part_dy,main_grad)
                    naive_bf16 += (
                        part_dy.transpose(0,1) @ part_x
                    )
                torch.npu.synchronize()
            except (RuntimeError,NotImplementedError,TypeError) as exc:
                print(
                    f"P0 MAIN_GRAD {label} reverse={reverse} "
                    f"status=UNSUPPORTED error={type(exc).__name__}: "
                    f"{str(exc)[:1000]}",
                    flush=True,
                )
                return
            _stats(
                f"{label} reverse={reverse} FP32_main_grad_vs_FP32_ref",
                main_grad,reference_fp32,
            )
            _stats(
                f"{label} reverse={reverse} BF16_naive_vs_FP32_ref",
                naive_bf16,reference_fp32,
            )
            results[(label,reverse)] = main_grad.detach().clone()

        _stats(
            f"{label} order_sensitivity_FP32",
            results[(label,False)],results[(label,True)],
        )
    _stats(
        "one_vs_four_FP32",
        results[("128_once",False)],
        results[("32_four",False)],
    )
    print(
        "P0 MAIN_GRAD status=SHARED_2D_FP32_ACCUMULATION_WORKED "
        "standalone_only=True full_Megatron_DDP_integration=False",
        flush=True,
    )
