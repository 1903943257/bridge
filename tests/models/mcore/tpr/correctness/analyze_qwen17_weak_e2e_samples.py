#!/usr/bin/env python3
"""Offline detailed diagnosis of real-TQ Native vs TPR AdamW samples.

Uses ONLY already-captured sampled tensors and torch CPU. Does NOT launch
NPU, re-train the model, or claim all-parameter correctness.

Usage:
  python tests/models/mcore/tpr/correctness/analyze_qwen17_weak_e2e_samples.py \\
    /tmp/tpr_qwen17_weak_e2e.XXXX/native_optimizer_sample.pt \\
    /tmp/tpr_qwen17_weak_e2e.XXXX/tpr_optimizer_sample.pt
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import torch


def _diagnostics(native: torch.Tensor, tpr: torch.Tensor) -> dict[str, float]:
    a, b = native.detach().reshape(-1).double(), tpr.detach().reshape(-1).double()
    if a.shape != b.shape or a.numel() == 0:
        raise ValueError("sample shapes must match and must not be empty")
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("nonfinite sampled values")
    ref2 = float((a * a).sum())
    cand2 = float((b * b).sum())
    err2 = float(((a - b) ** 2).sum())
    dot = float((a * b).sum())
    both_nonzero = (a != 0) & (b != 0)
    flips = int(((a * b < 0) & both_nonzero).sum())
    # Determine whether flipped coordinates carry meaningful gradient energy:
    # this is much more useful than raw count for BF16 tiny-gradient noise.
    flip_energy = float((a[a * b < 0] ** 2).sum())
    # Fraction of reference energy in near-zero coordinates.
    weak = a.abs() <= 1e-6
    return {
        "n": float(a.numel()), "r2": ref2, "c2": cand2,
        "e2": err2, "dot": dot,
        "flips": float(flips), "both_nonzero": float(int(both_nonzero.sum())),
        "flip_ref_energy": flip_energy,
        "weak_ref_energy": float((a[weak] ** 2).sum()),
        "weak_count": float(int(weak.sum())),
    }


def _merge(items) -> dict[str, float]:
    names = ("n", "r2", "c2", "e2", "dot", "flips",
             "both_nonzero", "flip_ref_energy", "weak_ref_energy", "weak_count")
    result = {name: 0.0 for name in names}
    for item in items:
        for name in names:
            result[name] += item[name]
    return result


def _summary(data: dict[str, float]) -> str:
    r2 = data["r2"]
    denom = max(r2, 1e-24)
    cosine = data["dot"] / max(math.sqrt(r2 * data["c2"]), 1e-24)
    return (
        f"rel_l2={math.sqrt(data['e2'] / denom):.6f} "
        f"cosine={cosine:.6f} "
        f"ref_energy={r2:.5g} err_energy={data['e2']:.5g} "
        f"flips={int(data['flips'])}/{int(data['both_nonzero'])} "
        f"flipped_ref_energy_share={data['flip_ref_energy']/denom:.6f} "
        f"near_zero_ref_energy_share={data['weak_ref_energy']/denom:.6f} "
        f"near_zero_entries={int(data['weak_count'])}/{int(data['n'])}"
    )


def _group_name(name: str) -> str:
    layer = re.search(r"(?:^|\.)decoder\.layers\.(\d+)\.", name)
    if layer:
        return f"decoder.layers.{int(layer.group(1)):02d}"
    if "embedding" in name or "output_layer" in name:
        return "embedding_or_lm_head"
    return "other"


def analyze_samples(native: dict, tpr: dict, *, top: int = 20) -> list[str]:
    """Return report lines. Inputs are dicts as saved by weak-E2E."""
    if set(native) != set(tpr) or not native:
        raise ValueError("Native/TPR parameter sets are empty or mismatched")
    report = [
        "DIAGNOSTIC ONLY: samples are stratified per tensor, not uniform over "
        "all 1.7B elements; grads were captured AFTER global clip_norm=1",
    ]
    for field in ("grad", "update"):
        per_name = {}
        per_group = defaultdict(list)
        for name in sorted(native):
            left, right = native[name], tpr[name]
            if not torch.equal(left["indices"], right["indices"]):
                raise ValueError(f"{name}: sampling locations mismatch")
            d = _diagnostics(left[field], right[field])
            per_name[name] = d
            per_group[_group_name(name)].append(d)
        agg = _merge(per_name.values())
        report.append(f"[{field.upper()}] ALL SAMPLED {_summary(agg)}")
        report.append(f"[{field.upper()}] TOP {top} PARAMETER ERROR ENERGIES:")
        sorted_items = sorted(
            per_name.items(), key=lambda x: x[1]["e2"], reverse=True
        )
        for name, d in sorted_items[:top]:
            report.append(
                f"  {name}: {_summary(d)} "
                f"err_share={d['e2']/max(agg['e2'], 1e-24):.4f} "
                f"ref_share={d['r2']/max(agg['r2'], 1e-24):.4f}"
            )
        report.append(f"[{field.upper()}] GROUP BREAKDOWN:")
        for name, items in sorted(per_group.items()):
            d = _merge(items)
            report.append(
                f"  {name}: {_summary(d)} "
                f"err_share={d['e2']/max(agg['e2'], 1e-24):.4f}"
            )
    report.append(
        "Interpretation: if high error share concentrates in particular "
        "layers/MLP/attention, debug those operations; if most sign flips "
        "carry near-zero reference energy, the first-step AdamW update "
        "cosine can be poor even when significant gradients align."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("tpr", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    # Only read torch.save files produced by this test / trusted environment.
    native = torch.load(args.native, weights_only=True, map_location="cpu")
    tpr = torch.load(args.tpr, weights_only=True, map_location="cpu")
    for line in analyze_samples(native, tpr, top=args.top):
        print(line)


if __name__ == "__main__":
    main()
