#!/usr/bin/env python3
"""Standalone BF16 GEMM M-shape reproduction on CPU, CUDA or Ascend NPU.

Only PyTorch is required; torch_npu is needed only for --device npu.
No VERL, Megatron, MindSpeed, Qwen, tokenizer or TQ dump is needed.

To reproduce the *actual* Qwen3 layer-1 QKV inputs, export with
TPR_QWEN17_GPT_EXPORT_L1_QKV=/tmp/qwen_l1_qkv.pt in the existing split
test, then copy that .pt to the target machine and use --input.

Compare (A) one large GEMM (M=192), (B) token-axis chunks (M=64),
(C) Prefix/Suffix GEMMs (M=128+64), and optional exported original
Megatron QKV outputs, holding the same BF16 X and W fixed. FP64 CPU
dot products on selected output elements provide an independent oracle.

WARNING: a different GEMM launch shape may change low-precision rounding.
This script diagnoses that behavior; it does not assert that either
BF16 path is an incorrect kernel or that either matches production PPO.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def metrics(name: str, got: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    a = got.detach().float().cpu()
    b = reference.detach().float().cpu()
    if a.shape != b.shape:
        raise ValueError(f"{name}: shape mismatch {tuple(a.shape)} != {tuple(b.shape)}")
    diff = (a - b).abs()
    denom = torch.linalg.vector_norm(b).clamp_min(1e-20)
    print(
        f"GEMM_COMPARE {name} "
        f"mean_abs={diff.mean().item():.9g} "
        f"max_abs={diff.max().item():.9g} "
        f"rel_l2={(torch.linalg.vector_norm(a-b)/denom).item():.9g} "
        f"nonbitwise={int((a!=b).sum())}/{a.numel()} "
        f"bitwise={bool(torch.equal(got.cpu(), reference.cpu()))}",
        flush=True,
    )
    return diff


def load_operands(args: argparse.Namespace):
    if args.input:
        obj = torch.load(Path(args.input), map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or not {"x", "w"}.issubset(obj):
            raise ValueError("Replay file must contain x and w tensors")
        x = obj["x"]
        w = obj["w"]
        expected = obj.get("y_native")
        prefix = int(obj.get("p", args.prefix))
        print(f"GEMM_INPUT mode=recorded file={args.input}", flush=True)
    else:
        torch.manual_seed(args.seed)
        x = (torch.randn(args.m, args.k) * args.x_scale).bfloat16()
        w = (torch.randn(args.n, args.k) * args.w_scale).bfloat16()
        expected = None
        prefix = args.prefix
        print(f"GEMM_INPUT mode=synthetic seed={args.seed}", flush=True)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if w.ndim != 2 or x.ndim != 2:
        raise ValueError(f"Require x=[M,K], w=[N,K], got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[1] or not 0 < prefix < x.shape[0]:
        raise ValueError("Invalid input dimensions or prefix split")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("Inputs must be stored as BF16; do not cast FP32 weights implicitly")
    if expected is not None:
        if expected.ndim == 3 and expected.shape[1] == 1:
            expected = expected[:, 0, :]
        if expected.shape != (x.shape[0], w.shape[0]):
            raise ValueError(f"Recorded QKV output shape invalid: {expected.shape}")
    return x.contiguous(), w.contiguous(), expected, prefix


def gemm(x: torch.Tensor, w: torch.Tensor, *, op: str, layout: str):
    # Actual Megatron input layout is [T,1,H]. Also test the flattened
    # [T,H] GEMM because dispatch may differ even for identical operands.
    z = x.unsqueeze(1) if layout == "3d" else x
    if op == "linear":
        y = F.linear(z, w)
    else:
        y = torch.matmul(z, w.transpose(0, 1))
    return y[:, 0, :] if layout == "3d" else y


def tiled(x: torch.Tensor, w: torch.Tensor, *, tile: int, op: str, layout: str):
    pieces = []
    for start in range(0, x.shape[0], tile):
        part = x[start:start+tile].contiguous()
        valid = part.shape[0]
        if valid < tile:
            part = torch.cat(
                [part, part.new_zeros((tile-valid, part.shape[1]))], dim=0
            )
        pieces.append(gemm(part, w, op=op, layout=layout)[:valid])
    return torch.cat(pieces, dim=0)


def fp64_oracle(x: torch.Tensor, w: torch.Tensor, native: torch.Tensor,
                chunked: torch.Tensor, *, audit: int):
    # Compare the SAME BF16 operands, not the original FP32 checkpoint.
    # Prefer entries where large and tiled GEMM differ.
    a = native.float().cpu()
    b = chunked.float().cpu()
    errors = (a-b).abs()
    mismatches = (errors > 0).nonzero(as_tuple=False)
    coords = []
    if errors.max() > 0:
        max_index = int(errors.reshape(-1).argmax().item())
        coords.append((max_index // errors.shape[1], max_index % errors.shape[1]))
    for row, col in mismatches[:max(audit, 1)*3].tolist():
        coord = (row, col)
        if coord not in coords:
            coords.append(coord)
        if len(coords) >= audit:
            break
    if not coords:
        coords = [(0, 0), (x.shape[0]-1, w.shape[0]-1)][:audit]

    native_wins = tile_wins = ties = native_correct_round = tile_correct_round = 0
    for row, col in coords:
        truth = float(torch.dot(x[row].double(), w[col].double()).item())
        ideal_bf16 = float(torch.tensor(truth, dtype=torch.float64).bfloat16().float().item())
        av, bv = float(a[row, col]), float(b[row, col])
        ea, eb = abs(av-truth), abs(bv-truth)
        native_wins += ea < eb
        tile_wins += eb < ea
        ties += ea == eb
        native_correct_round += av == ideal_bf16
        tile_correct_round += bv == ideal_bf16
        print(
            f"GEMM_FP64 row={row} out={col} "
            f"fp64={truth:.12g} ideal_bf16={ideal_bf16:.9g} "
            f"full={av:.9g} tile={bv:.9g} "
            f"full_abs_err={ea:.9g} tile_abs_err={eb:.9g}",
            flush=True,
        )
    print(
        f"GEMM_FP64_SUMMARY samples={len(coords)} full_closer={native_wins} "
        f"tile_closer={tile_wins} ties={ties} "
        f"full_correct_round={native_correct_round} "
        f"tile_correct_round={tile_correct_round}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "npu"], default="auto")
    parser.add_argument("--input", help="Optional exported real Qwen layer-1 QKV .pt")
    parser.add_argument("--m", type=int, default=192)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--prefix", type=int, default=128)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--x-scale", type=float, default=1.0)
    parser.add_argument("--w-scale", type=float, default=0.02)
    parser.add_argument("--op", choices=["matmul", "linear", "both"], default="both")
    parser.add_argument("--layout", choices=["2d", "3d", "both"], default="both")
    parser.add_argument("--audit", type=int, default=12,
                        help="Number of FP64 CPU dot products per operator variant")
    args = parser.parse_args()
    if args.tile < 1 or args.audit < 1:
        parser.error("--tile and --audit must be positive")
    if args.device == "npu":
        import torch_npu  # noqa: F401 - registers torch.npu
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment")
    if device == "npu" and (not hasattr(torch, "npu") or not torch.npu.is_available()):
        parser.error("NPU is not available in this PyTorch environment")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    if device == "npu" and hasattr(torch.npu, "matmul"):
        if hasattr(torch.npu.matmul, "allow_hf32"):
            torch.npu.matmul.allow_hf32 = False

    x_cpu, w_cpu, recorded_y, prefix = load_operands(args)
    x, w = x_cpu.to(device), w_cpu.to(device)
    print(
        f"GEMM_CONFIG device={device} torch={torch.__version__} "
        f"M={x.shape[0]} K={x.shape[1]} N={w.shape[0]} "
        f"p={prefix} s={x.shape[0]-prefix} tile={args.tile} "
        f"dtype=BF16 native_reference=unmodified_GEMM "
        "fp64_oracle=CPU_selected_elements",
        flush=True,
    )
    ops = ("matmul", "linear") if args.op == "both" else (args.op,)
    layouts = ("3d", "2d") if args.layout == "both" else (args.layout,)
    with torch.no_grad():
        for op in ops:
            for layout in layouts:
                print(f"GEMM_VARIANT op={op} layout={layout}", flush=True)
                full = gemm(x, w, op=op, layout=layout).cpu()
                split = torch.cat([
                    gemm(x[:prefix].contiguous(), w, op=op, layout=layout),
                    gemm(x[prefix:].contiguous(), w, op=op, layout=layout),
                ], dim=0).cpu()
                tile = tiled(x, w, tile=args.tile, op=op, layout=layout).cpu()
                metrics("SPLIT_VS_FULL", split, full)
                metrics("TILED_FULL_VS_FULL", tile, full)
                metrics("SPLIT_VS_TILED_FULL", split, tile)
                if recorded_y is not None:
                    metrics("REPLAY_FULL_VS_RECORDED_MEGATRON_QKV", full, recorded_y)
                    metrics("REPLAY_TILED_VS_RECORDED_MEGATRON_QKV", tile, recorded_y)
                fp64_oracle(x_cpu, w_cpu, full, tile, audit=args.audit)


if __name__ == "__main__":
    main()
