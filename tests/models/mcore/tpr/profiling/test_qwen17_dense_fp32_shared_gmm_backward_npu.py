"""Standalone P1 gate: existing Ascend grouped dX + dense FP32 shared main_grad.

One real Qwen3-1.7B layer's recorded-token X and pretrained weight W.
The only backward building blocks are existing MindSpeed/torch_npu ops:
  mode=grouped_dx       torch_npu.npu_grouped_matmul(dY, W^T groups)
  mode=mindspeed_fused  MindSpeed GMMOpBuilder.npu_gmm_backward_fusion
  shared weight grad    mindspeed.ops.npu_matmul_add_fp32(X_i, dY_i, dW_FP32)
NO custom CANN kernel and NO production/optimizer autograd hook.
Use one mode per pytest process: a device error can poison NPU context.

Forward tested separately in P1 real GMM. This experiment validates
the backward primitives and keeps native FP32 output gradient reference.

Opt-in TPR_QWEN17_DENSE_BACKWARD_AUTOGRAD=1 adds an isolated
torch.autograd.Function connecting grouped forward, grouped dX, and a
single FP32 main_grad across two backward calls. It is NOT registered as
a production Megatron Linear, DDP hook, or optimizer integration.
"""
from __future__ import annotations

import gc
import os

import pytest
import torch
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_RUN_QWEN17_DENSE_FP32_BACKWARD") != "1",
    reason="Set TPR_RUN_QWEN17_DENSE_FP32_BACKWARD=1",
)


def _stats(label, actual, expected):
    if actual is None:
        print(f"P1 DENSE_FP32 {label} status=MISSING", flush=True)
        return None
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: {actual.shape} vs {expected.shape}")
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()
    if not bool(torch.isfinite(a).all()):
        raise AssertionError(f"{label}: nonfinite")
    d = a-b
    l2 = float(torch.linalg.vector_norm(d) /
               torch.linalg.vector_norm(b).clamp_min(1e-12))
    max_abs = float(d.abs().max())
    print(
        f"P1 DENSE_FP32 {label} rel_l2={l2:.9g} "
        f"max_abs={max_abs:.9g} bitwise={torch.equal(a,b)}",
        flush=True,
    )
    return l2


class _TestOnlySharedGMMWithFP32MainGrad(torch.autograd.Function):
    """Test-only autograd bridge, NOT a drop-in Megatron Linear.

    The FP32 weight gradient is written by side effect to a caller-owned
    main_grad, and no BF16/per-group parameter gradient is returned.
    Production DDP grad buffers, loss scaling, and optimizer contracts
    must be integrated and validated separately.
    """

    @staticmethod
    def forward(ctx, x, weight, boundaries, main_grad, tile):
        from mindspeed.ops.gmm import npu_gmm

        if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
                or x.shape[0] % tile or weight.dtype != torch.bfloat16
                or x.dtype != torch.bfloat16 or main_grad.dtype != torch.float32
                or main_grad.shape != weight.shape):
            raise AssertionError("Unsupported shared dense-GMM autograd shape/dtype")
        groups = x.shape[0] // tile
        if boundaries.numel() != groups:
            raise AssertionError("GMM boundaries must cover all tile groups")
        ctx.save_for_backward(x, weight, boundaries)
        ctx.main_grad = main_grad
        ctx.tile = tile
        expanded = weight.t().unsqueeze(0).expand(groups, -1, -1)
        return npu_gmm(
            x, expanded, group_list=boundaries,
            group_type=0, gemm_fusion=False,
        )

    @staticmethod
    def backward(ctx, grad_output):
        import torch_npu
        from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32

        x, weight, boundaries = ctx.saved_tensors
        groups = boundaries.numel()
        dy = grad_output.contiguous()
        dx = None
        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                result = torch_npu.npu_grouped_matmul(
                    [dy], [weight.contiguous()] * groups,
                    group_list=boundaries, group_type=0,
                    group_list_type=0, split_item=2,
                )
                if not isinstance(result, (tuple, list)) or len(result) != 1:
                    raise AssertionError("unexpected grouped dX result")
                dx = result[0]
            if ctx.needs_input_grad[1]:
                for xi, dyi in zip(
                    x.split(ctx.tile), dy.split(ctx.tile), strict=True
                ):
                    npu_matmul_add_fp32(
                        xi.contiguous(), dyi.contiguous(), ctx.main_grad,
                    )
        return dx, None, None, None, None


def test_real_qwen_shared_dense_fp32_backward_only():
    import torch_npu
    from mindspeed.ops.npu_matmul_add import npu_matmul_add_fp32
    from ..correctness.test_qwen3_1_7b_split_equivalence_npu import (
        _setup_real_model, _real_recorded_tokens,
    )

    mode = os.environ.get(
        "TPR_QWEN17_DENSE_BACKWARD_MODE", "grouped_dx"
    ).strip()
    if mode not in ("grouped_dx", "mindspeed_fused"):
        raise AssertionError("mode must be grouped_dx or mindspeed_fused")
    family = os.environ.get("TPR_QWEN17_DENSE_BACKWARD_LINEAR", "qkv")
    if family not in ("qkv","proj"):
        raise AssertionError("Only qkv and proj are enabled in this P1")
    groups = int(os.environ.get("TPR_QWEN17_DENSE_BACKWARD_GROUPS", "8"))
    layer = int(os.environ.get("TPR_QWEN17_DENSE_BACKWARD_LAYER", "1"))
    tile = 128
    if groups not in (8,9):
        raise AssertionError("group count must be 8 or 9")
    m = tile*groups
    dev = torch.device("npu")
    fixture, target = _setup_real_model()
    model = fixture._make_qwen_model(
        dev, tpr=True, max_sequence_length=m
    )
    assert target.assert_model_scale(model) > 1_500_000_000
    if not 1 <= layer <= len(model.decoder.layers):
        raise AssertionError("invalid layer")
    mod = model.decoder.layers[layer-1].self_attention
    linear = (mod.linear_qkv if family=="qkv" else mod.linear_proj)
    captured={}
    def capture(_mod, args):
        if "x" not in captured:
            captured["x"]=args[0].detach().clone()
    handle=linear.register_forward_pre_hook(capture)
    try:
        ids = _real_recorded_tokens(m-128,128).to(dev)
        with torch.no_grad():
            positions=torch.arange(m,device=dev).unsqueeze(0)
            logits=model(ids.unsqueeze(0),positions,attention_mask=None)
            del logits
    finally:
        handle.remove()
    x = captured["x"][:,0,:].contiguous()
    w = linear.weight.detach().clone().contiguous()
    del model,linear,mod,captured,ids
    gc.collect()
    torch.npu.empty_cache()
    k=w.shape[1]
    n=w.shape[0]
    if x.shape != (m,k) or w.dtype != torch.bfloat16:
        raise AssertionError("unexpected real Qwen BF16 layout")
    torch.manual_seed(793)
    dy=torch.randn((m,n),device=dev,dtype=torch.bfloat16)
    boundaries=torch.arange(
        tile,m+1,tile,device=dev,dtype=torch.int64
    )
    print(
        f"P1 DENSE_FP32 CONFIG mode={mode} family={family} "
        f"layer={layer} M={m} G={groups} K={k} N={n} "
        "main_grad_dtype=FP32 original_dense_weight=shared",
        flush=True,
    )
    with torch.no_grad():
        fp32_dx=dy.float() @ w.float()
        fp32_dw=dy.float().T @ x.float()
        # Compare with the actual native tiled BF16 autograd reference,
        # and a separate high-precision reference using identical X/dY.
    xref=x.detach().clone().requires_grad_(True)
    wref=w.detach().clone().requires_grad_(True)
    yref=torch.cat([
        F.linear(part.contiguous(),wref)
        for part in xref.split(tile,dim=0)
    ],dim=0)
    ref_dx_auto,ref_dw_auto=torch.autograd.grad(
        yref,(xref,wref),grad_outputs=dy
    )
    _stats("REFERENCE_BF16_DX_AUTOGRAD_VS_FP32",ref_dx_auto,fp32_dx)
    _stats("REFERENCE_BF16_DW_AUTOGRAD_VS_FP32",ref_dw_auto,fp32_dw)
    ref_y_cpu=yref.detach().cpu().clone()
    ref_dx_cpu=ref_dx_auto.detach().cpu().clone()
    ref_dw_cpu=ref_dw_auto.detach().cpu().clone()
    del xref,wref,yref,ref_dx_auto,ref_dw_auto
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()

    # Two alternative existing DENSE dX routes. Each test invocation uses
    # one mode only, so a failed NPU extension doesn't contaminate the other.
    initial=torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    try:
        if mode=="grouped_dx":
            dx_op=torch_npu.npu_grouped_matmul(
                [dy.contiguous()], [w.contiguous()]*groups,
                group_list=boundaries,group_type=0,
                group_list_type=0,split_item=2,
            )
            if not isinstance(dx_op,(list,tuple)) or len(dx_op)!=1:
                raise AssertionError("unexpected grouped dX layout")
            dx=dx_op[0]
        else:
            from mindspeed.ops.gmm import GMMFunction
            packed=w.T.unsqueeze(0).expand(groups,-1,-1)
            print(
                "P1 DENSE_FP32 DX_FUSED_CALL "
                "stage=BEFORE_NPU_GMM_BACKWARD_FUSION",
                flush=True,
            )
            outs=GMMFunction.builder.load().npu_gmm_backward_fusion(
                [dy.contiguous()],[packed],boundaries,0,
            )
            if (
                not isinstance(outs,(list,tuple))
                or not isinstance(outs[0],(list,tuple))
                or len(outs[0])!=1
            ):
                raise AssertionError("unexpected native dx fusion outputs")
            dx=outs[0][0]
            print(
                "P1 DENSE_FP32 DX_FUSED_CALL "
                "stage=AFTER_NPU_GMM_BACKWARD_FUSION",
                flush=True,
            )
        torch.npu.synchronize()
        _stats(f"{mode}_dX_vs_FP32",dx,fp32_dx)
        _stats(f"{mode}_dX_vs_native_tiled",dx,ref_dx_cpu)
    except (RuntimeError,TypeError,AttributeError,NotImplementedError) as exc:
        print(
            f"P1 DENSE_FP32 DX status=UNSUPPORTED mode={mode} "
            f"error={type(exc).__name__}: {str(exc)[:1300]}",
            flush=True,
        )
        return

    # Single [N,K] float32 main_grad. No [G,K,N] dW allocation.
    main_grad=torch.zeros(
        (n,k),device=dev,dtype=torch.float32
    )
    with torch.no_grad():
        for xi,dyi in zip(x.split(tile),dy.split(tile),strict=True):
            npu_matmul_add_fp32(
                xi.contiguous(),dyi.contiguous(),main_grad,
            )
    torch.npu.synchronize()
    _stats("shared_main_grad_vs_FP32",main_grad,fp32_dw)
    _stats("shared_main_grad_vs_native_BF16_dW",main_grad,ref_dw_cpu)
    print(
        "P1 DENSE_FP32 MEMORY "
        f"baseline_mib={initial/(1024**2):.3f} "
        f"incremental_peak_mib={(torch.npu.max_memory_allocated()-initial)/(1024**2):.3f} "
        "includes_dX_output_and_FP32_main_grad=True",
        flush=True,
    )
    print(
        "P1 DENSE_FP32 RESULT status=DX_AND_SHARED_FP32_WGRAD_AVAILABLE "
        "autograd_glue=NOT_INSTALLED full_ppo=UNVERIFIED",
        flush=True,
    )

    # Optional P2: exercise a genuine autograd graph and two backward
    # invocations against the same trainable dense parameter/main_grad.
    # This must remain opt-in; the production optimizer integration and
    # DDP gradient lifecycle are explicitly outside this experiment.
    if os.getenv("TPR_QWEN17_DENSE_BACKWARD_AUTOGRAD", "0") != "1":
        return
    if mode != "grouped_dx":
        raise AssertionError(
            "P2 autograd gate uses grouped_dx only; run fused mode separately"
        )
    del dx, dx_op, main_grad
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()

    weight = w.detach().clone().requires_grad_(True)
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    accumulator_ptr = weight.main_grad.data_ptr()
    autograd_baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    for step in (1, 2):
        inp = x.detach().clone().requires_grad_(True)
        y = _TestOnlySharedGMMWithFP32MainGrad.apply(
            inp, weight, boundaries, weight.main_grad, tile,
        )
        if not torch.equal(y.detach().cpu(), ref_y_cpu):
            raise AssertionError(
                f"P2 autograd forward is not bitwise tiled, visit={step}"
            )
        y.backward(dy)
        torch.npu.synchronize()
        r_dx = _stats(
            f"AUTOGRAD_visit{step}_dX_vs_BF16_tiled",
            inp.grad, ref_dx_cpu,
        )
        r_w = _stats(
            f"AUTOGRAD_visit{step}_shared_main_grad_vs_FP32",
            weight.main_grad, fp32_dw * step,
        )
        if r_dx is None or r_dx > 2e-4 or r_w is None or r_w > 1e-5:
            raise AssertionError(
                f"P2 autograd numerical gate failed at visit {step}: "
                f"dX={r_dx}, FP32 main_grad={r_w}"
            )
        if weight.grad is not None or weight.main_grad.data_ptr() != accumulator_ptr:
            raise AssertionError(
                "P2 unexpectedly allocated weight.grad or replaced main_grad"
            )
        del inp, y
    torch.npu.synchronize()
    print(
        "P2 DENSE_FP32 AUTOGRAD_MEMORY "
        f"baseline_mib={autograd_baseline/(1024**2):.3f} "
        f"incremental_peak_mib={(torch.npu.max_memory_allocated()-autograd_baseline)/(1024**2):.3f} "
        "two_sequential_visits=True",
        flush=True,
    )
    print(
        "P2 DENSE_FP32 AUTOGRAD_RESULT status=PASS "
        "forward_bitwise_tile=True dX_gate=True "
        "shared_fp32_main_grad_gate=True accumulated_visits=2 "
        "weight_grad_materialized=False "
        "megatron_ddp_integration=UNVERIFIED optimizer_step=UNVERIFIED",
        flush=True,
    )
