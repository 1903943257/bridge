"""P1 diagnostic: shared-weight MindSpeed GMM autograd on pretrained Qwen3-1.7B.

Captures REAL recorded-TQ hidden activations at a selected Qwen decoder Linear.
Tests M=128*G (G=8 or 9), one original BF16 trainable [N,K] weight expanded
to [G,K,N] WITHOUT extra forward weight storage, and compares with repeated
native BF16 token-tile F.linear in forward, dX and shared dW.

Does not patch production, perform training PPO, or establish main_grad
integration. Potential temporary GMM backward group gradients are monitored.
Set TPR_RUN_QWEN17_MINDSPEED_REAL_GMM=1 to opt in.
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
    os.getenv("TPR_RUN_QWEN17_MINDSPEED_REAL_GMM") != "1",
    reason="Set TPR_RUN_QWEN17_MINDSPEED_REAL_GMM=1",
)


def _metric(name, actual, reference):
    if actual is None:
        raise AssertionError(f"{name} missing gradient/output")
    if actual.shape != reference.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {reference.shape}")
    aa = actual.detach().float()
    bb = reference.detach().float()
    if not bool(torch.isfinite(aa).all()):
        raise AssertionError(f"{name}: nonfinite values")
    dd = aa - bb
    rel = float(torch.linalg.vector_norm(dd) /
                torch.linalg.vector_norm(bb).clamp_min(1e-12))
    mx = float(dd.abs().max())
    eq = torch.equal(actual, reference)
    print(
        f"P1 REAL_SHARED_GMM {name} rel_l2={rel:.9g} "
        f"max_abs={mx:.9g} bitwise={eq} "
        f"actual_dtype={actual.dtype}",
        flush=True,
    )
    return rel


def _print_memory(prefix):
    try:
        current = torch.npu.memory_allocated()
        peak = torch.npu.max_memory_allocated()
        print(
            f"P1 REAL_SHARED_GMM MEMORY {prefix} "
            f"current_mib={current / (1024**2):.3f} "
            f"peak_mib={peak / (1024**2):.3f}",
            flush=True,
        )
    except (AttributeError, RuntimeError) as exc:
        print(
            f"P1 REAL_SHARED_GMM MEMORY {prefix} "
            f"status=UNAVAILABLE reason={type(exc).__name__}: {exc}",
            flush=True,
        )


def test_real_qwen17_shared_gmm_autograd_8_or_9_groups():
    from mindspeed.ops.gmm import npu_gmm
    from ..correctness.test_qwen3_1_7b_split_equivalence_npu import (
        _real_recorded_tokens, _setup_real_model,
    )

    tile = int(os.getenv("TPR_QWEN17_REAL_GMM_TILE", "128"))
    groups = int(os.getenv("TPR_QWEN17_REAL_GMM_GROUPS", "8"))
    name = os.getenv("TPR_QWEN17_REAL_GMM_LINEAR", "proj").strip().lower()
    layer = int(os.getenv("TPR_QWEN17_REAL_GMM_LAYER", "1"))
    if tile != 128 or groups not in (8, 9):
        raise AssertionError("Supported test scope: tile=128 groups=8 or 9")
    if name not in ("qkv", "proj", "fc1", "fc2"):
        raise AssertionError("Invalid Linear family")
    m = tile * groups
    device = torch.device("npu")
    fixture, target = _setup_real_model()
    model = fixture._make_qwen_model(
        device, tpr=True, max_sequence_length=m,
    )
    assert target.assert_model_scale(model) > 1_500_000_000
    if not 1 <= layer <= len(model.decoder.layers):
        raise AssertionError(f"invalid layer {layer}")
    layer_mod = model.decoder.layers[layer-1]
    modules = {
        "qkv": layer_mod.self_attention.linear_qkv,
        "proj": layer_mod.self_attention.linear_proj,
        "fc1": layer_mod.mlp.linear_fc1,
        "fc2": layer_mod.mlp.linear_fc2,
    }
    module = modules[name]
    activation = {}
    def capture(_mod, args):
        if "input" not in activation:
            activation["input"] = args[0].detach().contiguous().clone()
    hook = module.register_forward_pre_hook(capture)
    try:
        tokens = _real_recorded_tokens(m-128, 128).to(device)
        with torch.no_grad():
            positions = torch.arange(m, device=device).unsqueeze(0)
            output = model(
                tokens.unsqueeze(0), positions, attention_mask=None,
            )
            del output
    finally:
        hook.remove()

    x0 = activation["input"][:,0,:].detach().contiguous()
    w0 = module.weight.detach().contiguous().clone()
    if (x0.shape[0] != m or x0.shape[1] != w0.shape[1]
            or x0.dtype != torch.bfloat16 or w0.dtype != torch.bfloat16):
        raise AssertionError(
            f"Unexpected Qwen data input={x0.shape}/{x0.dtype} "
            f"weight={w0.shape}/{w0.dtype}"
        )
    # Verify that F.linear reference really matches the existing Megatron
    # physical-M=128 execution for these exact pretrained inputs/weights.
    with torch.no_grad():
        true_tile_outputs = []
        for part in x0.split(tile, dim=0):
            result = module(part.unsqueeze(1).contiguous())
            if isinstance(result, tuple):
                if len(result)!=2 or result[1] is not None:
                    raise AssertionError("Unexpected Megatron Linear bias")
                result = result[0]
            true_tile_outputs.append(result[:,0,:].contiguous())
        megatron_tile = torch.cat(true_tile_outputs, dim=0)
        del true_tile_outputs
    del model, module, layer_mod, modules, activation, tokens
    gc.collect()
    torch.npu.empty_cache()

    k = w0.shape[1]
    n = w0.shape[0]
    print(
        f"P1 REAL_SHARED_GMM CONFIG layer={layer} linear={name} "
        f"M={m} groups={groups} tile={tile} K={k} N={n} "
        f"stack=mindspeed.ops.gmm.npu_gmm",
        flush=True,
    )
    group_boundaries = torch.arange(
        tile, m+1, tile, device=device, dtype=torch.int64,
    )
    torch.manual_seed(3407)
    dy = torch.randn((m,n), device=device, dtype=torch.bfloat16)

    # Same original weight and activation bytes for both paths.
    xref = x0.clone().detach().requires_grad_(True)
    wref = w0.clone().detach().requires_grad_(True)
    try:
        torch.npu.reset_peak_memory_stats()
    except (RuntimeError, AttributeError):
        pass
    ref = torch.cat(
        [
            F.linear(chunk.contiguous(), wref)
            for chunk in xref.split(tile, dim=0)
        ], dim=0
    )
    _metric("F_linear_vs_Megatron_tile",ref,megatron_tile)
    dxref, dwref = torch.autograd.grad(
        ref, (xref,wref), grad_outputs=dy,
    )
    del megatron_tile
    torch.npu.synchronize()
    _print_memory("native_tile_reference")
    del xref, wref

    x = x0.clone().detach().requires_grad_(True)
    w = w0.clone().detach().requires_grad_(True)
    grouped_weight = w.t().unsqueeze(0).expand(groups, -1, -1)
    print(
        "P1 REAL_SHARED_GMM WEIGHT_VIEW "
        f"shape={tuple(grouped_weight.shape)} "
        f"strides={grouped_weight.stride()} "
        f"view_storage_mib={grouped_weight.untyped_storage().nbytes() / (1024**2):.3f} "
        f"logical_mib={grouped_weight.numel()*grouped_weight.element_size()/(1024**2):.3f}",
        flush=True,
    )
    try:
        torch.npu.reset_peak_memory_stats()
    except (RuntimeError, AttributeError):
        pass
    y = npu_gmm(
        x,grouped_weight,group_list=group_boundaries,
        group_type=0,gemm_fusion=False,
    )
    _metric("forward_gmm_vs_tile", y, ref)
    dx, dw = torch.autograd.grad(
        y, (x,w), grad_outputs=dy, allow_unused=True,
    )
    torch.npu.synchronize()
    _print_memory("mindspeed_gmm_autograd")
    r_dx = _metric("dX_gmm_vs_tile", dx, dxref)
    r_dw = _metric("shared_dW_gmm_vs_tile", dw, dwref)
    print(
        "P1 REAL_SHARED_GMM RESULT "
        f"status={'SUPPORTED_SHARED_AUTOGRAD' if r_dx is not None and r_dw is not None else 'MISSING_GRAD'} "
        "shared_weight_forward_view=True "
        "full_model_training=UNVERIFIED",
        flush=True,
    )
