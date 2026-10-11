"""P0-2B: real-Qwen native full attention vs official MindSpeed CP Ring.

First run CP1: capture the REAL Q/K/V and dO from pretrained Qwen's native
core_attention during a masked real-TQ backward pass; independently replay
the same core with frozen inputs and dO to establish a full-attention oracle.

Then launch fresh CP2 / CP4 processes: reinitialize the native MindSpeed CP
stack and replay its unmodified core_attention with the EXACT CP1 Q/K/V/dO,
sharded by official zigzag indices. Restore O/dQ/dK/dV globally with a
disjoint-owner all_reduce (never a parameter-gradient reduction).

This removes QKV GEMM/MLP/optimizer differences. It does NOT claim the native
Ring internals are bitwise parity or that this proves end-to-end CP correctness.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available

from .test_native_cp1_cp2_cp4_drift_npu import (
    SEQ_LEN, _expected_positions, _model, _runtime,
)
from .test_native_cp_multistep_npu import _load_real_tq_rows, _loss

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_P0_NATIVE_ATTN_REPLAY") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="P0-2B requires TPR_P0_NATIVE_ATTN_REPLAY=1 and an Ascend NPU",
)


def _layer_indices():
    raw = os.getenv("TPR_P0_2B_LAYERS", "1")
    result = tuple(int(value.strip()) for value in raw.split(","))
    if not result or len(set(result)) != len(result) or not all(0 <= x < 28 for x in result):
        raise AssertionError(f"Invalid zero-based Qwen3 layer list: {raw}")
    return result


def _get_attention(model, layer):
    native = model.decoder.layers[layer].self_attention.core_attention
    if not isinstance(native, torch.nn.Module):
        raise AssertionError(
            f"Expected unmodified native Megatron core_attention nn.Module; "
            f"got {type(native)} in layer {layer}"
        )
    if "tpr" in type(native).__module__.lower():
        raise AssertionError("TPR attention accidentally entered the native CP oracle")
    return native


def _metric(reference, actual):
    # CP1 artifacts live on CPU, whereas newly replayed native Attention
    # outputs and gradients live on NPU. Normalize device BEFORE arithmetic.
    # Metrics are diagnostic only: CPU float64 is not part of any model
    # forward/backward or MindSpeed Ring communication.
    if reference.shape != actual.shape:
        raise AssertionError(
            f"native Attention shape mismatch: {tuple(reference.shape)} "
            f"vs {tuple(actual.shape)}"
        )
    ref = reference.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    val = actual.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(val).all()):
        raise AssertionError("nonfinite native Attention replay witness")
    delta = (val - ref).double()
    r, v = ref.double(), val.double()
    nr, nv = r.norm(), v.norm()
    return {
        "count": int(ref.numel()),
        "changed": int(torch.count_nonzero(delta).item()),
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "p95_abs": float(torch.quantile(delta.abs().float(), 0.95).item()),
        "rel_l2": float((delta.norm() / nr.clamp_min(1e-30)).item()),
        "cosine": float(((r * v).sum() / (nr * nv).clamp_min(1e-30)).item()),
    }


def _replay_attention(attention, full_inputs, indices=None):
    names = ("q", "k", "v")
    local = [
        (
            full_inputs[key].index_select(0, indices).contiguous()
            if indices is not None else full_inputs[key].contiguous()
        ).detach().clone().requires_grad_(True)
        for key in names
    ]
    dy = (
        full_inputs["do"].index_select(0, indices).contiguous()
        if indices is not None else full_inputs["do"].contiguous()
    )
    # Invoke the original Megatron/MindSpeed attention forward with actual
    # Qwen Q/K/V shape [S,B,H,D], no custom attention backend.
    output = attention(*local, None)
    if not isinstance(output, torch.Tensor) or output.shape != dy.shape:
        raise AssertionError(f"Native attention output/dO mismatch: "
                             f"{getattr(output, 'shape', None)} vs {dy.shape}")
    dqs = torch.autograd.grad(
        output, local, grad_outputs=dy,
        allow_unused=False, retain_graph=False,
    )
    return {
        "o": output.detach(), "dq": dqs[0].detach(),
        "dk": dqs[1].detach(), "dv": dqs[2].detach(),
    }


def _capture_native_cp1(model, runtime, real_row, layer_numbers):
    real = {}
    hooks = []
    for layer in layer_numbers:
        attention = _get_attention(model, layer)
        record = {"calls": 0}
        real[str(layer)] = record

        def pre(_module, inputs, *, key=str(layer)):
            item = real[key]
            item["calls"] += 1
            if item["calls"] != 1 or len(inputs) < 3:
                raise AssertionError(f"Unexpected native Attention call layer{key}")
            for k, value in zip(("q", "k", "v"), inputs[:3], strict=True):
                if not isinstance(value, torch.Tensor) or value.shape[0] != SEQ_LEN:
                    raise AssertionError(f"Bad real {k} at layer{key}: "
                                         f"{getattr(value, 'shape', None)}")
                item[k] = value.detach().cpu().contiguous().clone()
                if not value.requires_grad:
                    raise AssertionError(f"Real core Attention {k} has no grad")
                def save(d, *, name=k, target=item):
                    target["d" + name] = d.detach().cpu().contiguous().clone()
                value.register_hook(save)

        def post(_module, _inputs, output, *, key=str(layer)):
            item = real[key]
            if not isinstance(output, torch.Tensor) or not output.requires_grad:
                raise AssertionError(f"Unexpected real native core output layer{key}")
            item["o"] = output.detach().cpu().contiguous().clone()
            def save(d):
                item["do"] = d.detach().cpu().contiguous().clone()
            output.register_hook(save)

        hooks.append(attention.register_forward_pre_hook(pre))
        hooks.append(attention.register_forward_hook(post))
    try:
        model.zero_grad(set_to_none=True)
        loss, targets = _loss(model, runtime, real_row, training=True)
    finally:
        for hook in hooks:
            hook.remove()
    for key, record in real.items():
        if record["calls"] != 1:
            raise AssertionError(f"Native layer{key} was called {record['calls']} times")
        for k in ("q", "k", "v", "o", "do", "dq", "dk", "dv"):
            if k not in record:
                raise AssertionError(f"Missing native CP1 capture {k} at layer{key}")
    return real, loss, targets


def _disjoint_global(local, index, runtime):
    if local.shape[0] != SEQ_LEN // runtime.cp:
        raise AssertionError(f"Wrong local row count: {local.shape}")
    output = torch.zeros((SEQ_LEN, *local.shape[1:]), device=local.device, dtype=local.dtype)
    owner = torch.zeros(SEQ_LEN, device=local.device, dtype=torch.int32)
    output.index_copy_(0, index, local)
    owner.index_fill_(0, index, 1)
    # No overlapping owner entries: all_reduce SUM is *only* a tensor-gather
    # method here; never sum an already-reduced Ring dK/dV twice.
    if runtime.cp > 1:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=runtime.group)
        dist.all_reduce(owner, op=dist.ReduceOp.SUM, group=runtime.group)
    if not bool((owner == 1).all().item()):
        raise AssertionError("Native CP Attention replay lost/duplicated positions")
    return output


def _save_cp1_oracle(runtime, model, row, data_sha, outdir, ckpt, layers):
    captured, loss, count = _capture_native_cp1(model, runtime, row, layers)
    records = {}
    for layer in layers:
        rec = captured[str(layer)]
        native = _get_attention(model, layer)
        inputs = {k: rec[k].to(runtime.device) for k in ("q", "k", "v", "do")}
        fresh = _replay_attention(native, inputs)
        replay_metrics = {k: _metric(rec[k], fresh[k]) for k in ("o", "dq", "dk", "dv")}
        print(f"P0_2B CP1 layer={layer} actual_vs_replay="
              f"{ {k: v['rel_l2'] for k,v in replay_metrics.items()} }", flush=True)
        if any(v["rel_l2"] > 5e-3 for v in replay_metrics.values()):
            raise AssertionError(
                f"CP1 replay differs too much from true native Attention layer{layer}"
            )
        records[str(layer)] = {
            "raw": {k: rec[k] for k in ("q", "k", "v", "do")},
            "cp1_reference": {k: fresh[k].cpu().clone() for k in ("o", "dq", "dk", "dv")},
            "actual_vs_replay": replay_metrics,
            "shapes": {k: tuple(rec[k].shape) for k in ("q", "k", "v", "do")},
            "native_module": type(native).__module__ + "." + type(native).__qualname__,
            "softmax_scale": float(native.softmax_scale),
        }
    stamp = hashlib.sha256(row["tokens"].numpy().tobytes()).hexdigest()
    payload = {
        "tag": "P0_2B_REAL_QWEN_CORE_ATTENTION_CP1_ORACLE",
        "torch": str(torch.__version__), "checkpoint": str(ckpt),
        "token_sha256": stamp, "source_data_sha256": data_sha,
        "sample_id": row["id"], "layer_indices": list(layers),
        "actual_native_nll": loss, "supervised_targets": count,
        "records": records,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / "native_cp1_attention_oracle.pt"
    torch.save(payload, output)
    print(f"P0_2B CP1 saved={output} native_nll={loss:.9f} "
          f"layers={list(layers)} status=DIAGNOSTIC_ONLY", flush=True)


def _run_native_ring_replay(model, runtime, outdir, ckpt, layers):
    path = outdir / "native_cp1_attention_oracle.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CP1 real-Qwen reference: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["tag"] != "P0_2B_REAL_QWEN_CORE_ATTENTION_CP1_ORACLE":
        raise AssertionError("P0-2B wrong CP1 oracle artifact")
    if payload["layer_indices"] != list(layers) or payload["checkpoint"] != str(ckpt):
        raise AssertionError("P0-2B CP1 layer configuration / checkpoint mismatch")
    if payload["torch"] != str(torch.__version__):
        raise AssertionError("P0-2B CP1 and CP>1 torch versions differ")
    from mindspeed.core.context_parallel.dot_product_attention import CPDotProductAttentionImpl
    # CP1 and CP2/4 are independently instantiated official Megatron modules.
    # CP-specific communication groups come from the standard MindSpeed runtime.
    if runtime.cp == 1 or not CPDotProductAttentionImpl:
        raise AssertionError("P0-2B requires native context parallel CP>1")
    module = importlib.import_module("mindspeed.core.context_parallel.dot_product_attention")
    native_entry = getattr(module, "ringattn_context_parallel")
    count = [0]

    def counted(*args, **kwargs):
        count[0] += 1
        return native_entry(*args, **kwargs)

    reports = []
    with patch.object(module, "ringattn_context_parallel", counted):
        for layer in layers:
            real = payload["records"][str(layer)]
            core = _get_attention(model, layer)
            if abs(float(core.softmax_scale) - real["softmax_scale"]) > 1e-12:
                raise AssertionError(f"Native CP layer{layer} scale mismatch")
            index = _expected_positions(runtime.cp, runtime.rank).to(runtime.device)
            inputs = {k: real["raw"][k].to(runtime.device)
                      for k in ("q", "k", "v", "do")}
            for k in inputs:
                if inputs[k].shape[0] != SEQ_LEN or inputs[k].dtype != torch.bfloat16:
                    raise AssertionError(f"Bad CP1 BF16 {k} artifact: {inputs[k].shape}")
            local = _replay_attention(core, inputs, index)
            restored = {k: _disjoint_global(value, index, runtime)
                        for k, value in local.items()}
            if runtime.rank == 0:
                metrics = {k: _metric(real["cp1_reference"][k], restored[k])
                           for k in ("o", "dq", "dk", "dv")}
                reports.append({
                    "layer": layer, "native_module":
                        type(core).__module__ + "." + type(core).__qualname__,
                    "physical_local_M": SEQ_LEN // runtime.cp,
                    "metrics": metrics,
                })
                print(f"P0_2B CP1_VS_CP{runtime.cp} layer={layer} "
                      + " ".join(f"{k}_rel_l2={v['rel_l2']:.9g} "
                                 f"{k}_max_abs={v['max_abs']:.9g}"
                                 for k,v in metrics.items()),
                      flush=True)
    if count[0] != len(layers):
        raise AssertionError(f"CP{runtime.cp} native Ring dispatch count "
                             f"{count[0]} != expected {len(layers)}")
    if runtime.rank == 0:
        path = outdir / f"native_cp{runtime.cp}_attention_replay.json"
        path.write_text(json.dumps({
            "tag": "P0_2B_REAL_QWEN_NATIVE_RING_ATTENTION_REPLAY",
            "cp": runtime.cp, "torch": str(torch.__version__),
            "source_cp1": str(outdir / "native_cp1_attention_oracle.pt"),
            "native_ring_calls": count[0],
            "checks": "SAME_QKV_AND_UPSTREAM_DO_NATIVE_MEGATRON_CP_CORE",
            "reports": reports,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"P0_2B CP{runtime.cp} saved={path} "
              f"native_ring_calls={count[0]} status=DIAGNOSTIC_ONLY",
              flush=True)


def test_real_qwen_native_attention_replay():
    cp = int(os.environ["WORLD_SIZE"])
    if cp not in (1, 2, 4):
        raise AssertionError("P0-2B runs CP1, CP2 or CP4, world_size==CP")
    ckpt = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
    outdir = Path(os.getenv("TPR_P0_2B_OUTPUT_DIR", "/tmp/tpr_p0_attn_replay"))
    layers = _layer_indices()
    with _runtime(cp) as runtime:
        model = _model(runtime, ckpt)
        if cp == 1:
            real_tq = Path(os.environ["TPR_REAL_TQ_BATCH"])
            train, _val, _skipped, data_sha = _load_real_tq_rows(
                real_tq, model.vocab_size,
            )
            _save_cp1_oracle(runtime, model, train[0], data_sha, outdir, ckpt, layers)
        else:
            _run_native_ring_replay(model, runtime, outdir, ckpt, layers)
        dist.barrier(group=runtime.group)
