"""P0-2A: faithful native Megatron GEMM physical-M replay on real Qwen.

Capture the actual BF16 module input X and backward output-gradient dY from
the CP1 pretrained Qwen graph, then call the SAME original Megatron Linear
module/weights with M=1024 (full), M=512 (CP2 zigzag) and M=256 (CP4 zigzag).
This test does not patch native GEMM or alter official CP training behavior.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from verl.utils.device import is_torch_npu_available
from .test_native_cp1_cp2_cp4_drift_npu import SEQ_LEN, _expected_positions, _model, _runtime
from .test_native_cp_multistep_npu import _load_real_tq_rows, _loss

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_P0_GEMM_REPLAY") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="P0-2A is an opt-in single-NPU real-Qwen GEMM diagnostic",
)


def _unwrap(output):
    result = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(result, torch.Tensor):
        raise AssertionError("Native Megatron Linear did not return a Tensor / Tensor-first tuple")
    return result


def _metric(reference, candidate):
    a, b = reference.detach().float(), candidate.detach().float()
    if a.shape != b.shape:
        raise AssertionError(f"GEMM replay shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("nonfinite GEMM replay witness")
    diff = (a - b).abs()
    return {
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "rel_l2": float(
            (diff.square().sum().sqrt() / a.square().sum().sqrt().clamp_min(1e-30)).item()
        ),
        "changed": int(torch.count_nonzero(diff).item()),
        "count": int(a.numel()),
    }


def _modules(model):
    layers = os.getenv("TPR_P0_2A_LAYERS", "1")
    indices = [int(x.strip()) for x in layers.split(",")]
    if not indices or len(set(indices)) != len(indices) or not all(0 <= x < 28 for x in indices):
        raise AssertionError(f"Invalid 0-based Qwen3 layer numbers: {layers}")
    modules = {}
    for i in indices:
        layer = model.decoder.layers[i]
        modules[f"layer{i}.qkv"] = layer.self_attention.linear_qkv
        modules[f"layer{i}.proj"] = layer.self_attention.linear_proj
        modules[f"layer{i}.fc1"] = layer.mlp.linear_fc1
        modules[f"layer{i}.fc2"] = layer.mlp.linear_fc2
    return modules


def _capture_real_operands(model, runtime, row):
    """Capture X/Y/dY at true native Linear calls (not synthetic tensors)."""
    modules = _modules(model)
    records, hooks = {}, []
    for name, module in modules.items():
        records[name] = {"calls": 0}
        def before(_module, args, *, label=name):
            rec = records[label]
            rec["calls"] += 1
            if rec["calls"] != 1 or not args or not isinstance(args[0], torch.Tensor):
                raise AssertionError(f"Unexpected native Linear invocation {label}")
            rec["x"] = args[0].detach().clone()
        def after(_module, _args, output, *, label=name):
            y = _unwrap(output)
            records[label]["y"] = y.detach().clone()
            if not y.requires_grad:
                raise AssertionError(f"Native {label} output unexpectedly detached")
            def save_dy(grad):
                records[label]["dy"] = grad.detach().clone()
            y.register_hook(save_dy)
        hooks.append(module.register_forward_pre_hook(before))
        hooks.append(module.register_forward_hook(after))
    try:
        model.zero_grad(set_to_none=True)
        actual_loss, supervised = _loss(model, runtime, row, training=True)
    finally:
        for hook in hooks:
            hook.remove()
    for name, rec in records.items():
        if rec["calls"] != 1 or not all(k in rec for k in ("x", "y", "dy")):
            raise AssertionError(f"Incomplete real native forward/backward {name}: {list(rec)}")
        if rec["x"].shape[0] != SEQ_LEN or rec["dy"].shape[0] != SEQ_LEN:
            raise AssertionError(f"Unexpected full-M capture at {name}")
    return modules, records, actual_loss, supervised


def _native_module_replay(module, x, dy):
    """Preserve native Megatron module, weight and BF16 backward kernel."""
    input_ = x.detach().clone().requires_grad_(True)
    y = _unwrap(module(input_))
    w = getattr(module, "weight", None)
    if not isinstance(w, torch.Tensor) or not w.requires_grad:
        raise AssertionError(f"Native Linear has no trainable native weight: {type(module)}")
    dx, dw = torch.autograd.grad(
        outputs=y, inputs=(input_, w), grad_outputs=dy,
        retain_graph=False, allow_unused=False,
    )
    return y.detach(), dx.detach(), dw.detach()


def _replay(module, record, cp):
    x, dy = record["x"], record["dy"]
    if x.shape[0] != SEQ_LEN or dy.shape[0] != SEQ_LEN:
        raise AssertionError("GEMM input/output-gradient M must be globally 1024")
    # Full and chunked forwards BOTH use this very same original module.
    full_y, full_dx, full_dw = _native_module_replay(module, x, dy)
    if full_y.shape != record["y"].shape:
        raise AssertionError("Native forward replay changed output shape")
    replay_check = _metric(record["y"], full_y)
    if cp not in (2, 4):
        raise AssertionError("P0-2A shard count must be 2 or 4")
    y_restored = torch.empty_like(full_y)
    dx_restored = torch.empty_like(full_dx)
    dw_sum_fp32 = torch.zeros_like(full_dw, dtype=torch.float32)
    dw_sum_bf16 = torch.zeros_like(full_dw)
    ownership = torch.zeros(SEQ_LEN, dtype=torch.int32, device=x.device)
    for rank in range(cp):
        idx = _expected_positions(cp, rank).to(x.device)
        local_y, local_dx, local_dw = _native_module_replay(
            module, x.index_select(0, idx), dy.index_select(0, idx),
        )
        y_restored.index_copy_(0, idx, local_y)
        dx_restored.index_copy_(0, idx, local_dx)
        dw_sum_fp32 += local_dw.float()
        dw_sum_bf16 += local_dw
        ownership.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int32))
    if not bool((ownership == 1).all().item()):
        raise AssertionError("P0-2A output/dX reconstruction lost or duplicated token rows")
    return {
        "cp": cp, "full_M": int(x.shape[0]),
        "local_M": int(x.shape[0] // cp),
        "module_type": type(module).__module__ + "." + type(module).__qualname__,
        "input_shape": list(x.shape), "weight_shape": list(module.weight.shape),
        "output_shape": list(full_y.shape),
        "full_module_replay_vs_actual": replay_check,
        "Y": _metric(full_y, y_restored),
        "dX": _metric(full_dx, dx_restored),
        "dW_fp32_SUM": _metric(full_dw, dw_sum_fp32),
        "dW_bf16_SUM": _metric(full_dw, dw_sum_bf16),
    }


def test_real_qwen_native_gemm_physical_m():
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("P0-2A is a single-NPU real-operands replay")
    path = Path(os.environ["TPR_REAL_TQ_BATCH"])
    ckpt = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
    output_dir = Path(os.getenv("TPR_P0_2A_OUTPUT_DIR", "/tmp/tpr_p0_gemm_replay"))
    with _runtime(1) as runtime:
        model = _model(runtime, ckpt)
        train, val, skipped, data_sha = _load_real_tq_rows(path, model.vocab_size)
        # The same chosen real row is used for ALL CP shapes.
        row = train[0]
        modules, records, true_loss, n_targets = _capture_real_operands(model, runtime, row)
        outputs = []
        for name, module in modules.items():
            for cp in (2, 4):
                record = _replay(module, records[name], cp)
                record["name"] = name
                outputs.append(record)
                print(
                    f"P0_2A {name} M={record['full_M']}->{record['local_M']} "
                    f"Y_rel_l2={record['Y']['rel_l2']:.8g} "
                    f"dX_rel_l2={record['dX']['rel_l2']:.8g} "
                    f"dW_fp32_rel_l2={record['dW_fp32_SUM']['rel_l2']:.8g} "
                    f"dW_bf16_rel_l2={record['dW_bf16_SUM']['rel_l2']:.8g}",
                    flush=True,
                )
        output_dir.mkdir(parents=True, exist_ok=True)
        source_sha = hashlib.sha256(row["tokens"].numpy().tobytes()).hexdigest()
        payload = {
            "tag": "P0_2A_REAL_X_W_DY_NATIVE_MEGATRON_GEMM_REPLAY",
            "model": "Qwen3-1.7B", "dtype": "bf16",
            "checkpoint": str(ckpt), "data_path": str(path),
            "data_sha": data_sha, "selected_row": row["id"],
            "token_sha": source_sha, "true_native_cp1_nll": true_loss,
            "supervised_tokens": n_targets, "torch": str(torch.__version__),
            "controls": "SAME_REAL_X_W_DY_NO_ATTENTION_REPLAY",
            "reports": outputs,
        }
        destination = output_dir / "p0_2a_gemm_replay.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"P0_2A saved={destination} probes={len(outputs)} "
              "status=DIAGNOSTIC_ONLY", flush=True)
