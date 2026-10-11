"""Compare P0 native CP1/CP2/CP4 artifacts; NO synthetic pass thresholds.

After CP=1/2/4 have run from /workspace/uni-agent/verl:
    python /workspace/bridge/tools/compare_native_cp_drift.py /tmp/tpr_p0_native_cp

Source is pulled into /workspace/bridge; only tests/models/mcore/tpr/ is
rsynced into the old VERL checkout. The launcher in that synced tests tree
invokes this comparator by absolute bridge path.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def _summary(reference, actual):
    ref = reference.detach().float().reshape(-1)
    val = actual.detach().float().reshape(-1)
    if ref.shape != val.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {val.shape}")
    if not torch.isfinite(ref).all() or not torch.isfinite(val).all():
        raise ValueError("nonfinite witness")
    delta = (val - ref).double()
    norm = ref.double().norm().clamp_min(1e-24)
    cos = float(torch.dot(ref.double(), val.double()) / (
        ref.double().norm() * val.double().norm()).clamp_min(1e-24))
    return {
        "count": ref.numel(),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "p95_abs": float(torch.quantile(delta.abs(), 0.95)),
        "rel_l2": float(delta.norm() / norm),
        "cosine": cos,
        "changed": int((delta != 0).sum()),
    }


def compare(reference, candidate):
    for field in ("token_sha256", "model", "dtype", "source", "objective",
                  "denominator", "sequence_length", "response_length",
                  "sampled_weight_sha256_rank0"):
        if reference[field] != candidate[field]:
            raise ValueError(f"P0 invalid cross-CP comparison: {field} mismatch: "
                             f"{reference[field]} != {candidate[field]}")
    if reference["cp"] != 1 or candidate["cp"] not in (2, 4):
        raise ValueError("comparison must be native CP1 vs CP2/4")
    if reference["checkpoint"] != candidate["checkpoint"]:
        raise ValueError("checkpoint path differs; use the same checkpoint")
    for field in ("torch", "versions"):
        if reference[field] != candidate[field]:
            raise ValueError(f"software version mismatch for {field}; rerun in same env")
    if candidate["native_ring_calls_rank0"] < 28:
        raise ValueError("native Ring was not verified in candidate")

    cp = candidate["cp"]
    print(f"CP1_VS_CP{cp} LOSS cp1={reference['loss']:.9f} "
          f"cp{cp}={candidate['loss']:.9f} abs={abs(reference['loss']-candidate['loss']):.8g}")
    log_ref = reference["logprobs"]
    log_val = candidate["logprobs"]
    response_begin = reference["sequence_length"] - reference["response_length"] - 1
    for title, a, b in (
        ("ALL_LOGPROB", log_ref, log_val),
        ("RESPONSE_LOGPROB", log_ref[response_begin:], log_val[response_begin:]),
    ):
        print(f"CP1_VS_CP{cp} {title} {_summary(a, b)}")
    ga, gb = reference["gradients"], candidate["gradients"]
    if ga.keys() != gb.keys():
        raise ValueError(f"parameter name mismatch: {set(ga) ^ set(gb)}")
    pairs = []
    total_ref_sq = total_diff_sq = total_val_sq = total_dot = 0.0
    for name in sorted(ga):
        a, b = ga[name], gb[name]
        if a["shape"] != b["shape"] or not torch.equal(a["indices"], b["indices"]):
            raise ValueError(f"sample index / shape mismatch: {name}")
        use_full = "full" in a and "full" in b
        if use_full:
            left, right = a["full"].float().flatten(), b["full"].float().flatten()
        else:
            left, right = a["values"], b["values"]
        stats = _summary(left, right)
        pairs.append((stats["rel_l2"], name, stats))
        lhs, rhs = left.double(), right.double()
        total_ref_sq += float((lhs * lhs).sum())
        total_val_sq += float((rhs * rhs).sum())
        total_diff_sq += float(((rhs-lhs)**2).sum())
        total_dot += float((lhs * rhs).sum())
    full_mode = all("full" in ga[n] and "full" in gb[n] for n in ga)
    print(f"CP1_VS_CP{cp} GRAD mode={'FULL_ALL_PARAMETERS' if full_mode else 'STRIDED_SAMPLED_ONLY'} "
          f"rel_l2={math.sqrt(total_diff_sq/max(total_ref_sq,1e-48)):.8g} "
          f"cosine={total_dot/max(math.sqrt(total_ref_sq*total_val_sq),1e-48):.9f} "
          f"parameter_count={len(ga)}")
    for rel, name, stats in sorted(pairs, reverse=True)[:12]:
        print(f"CP1_VS_CP{cp} GRAD_WORST name={name} {stats}")
    print(f"CP1_VS_CP{cp} status=DIAGNOSTIC_ONLY no_tpr_no_padding_no_fp32_gemm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    files = [args.directory / f"native_cp{cp}.pt" for cp in (1, 2, 4)]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        p.error(f"Missing CP evidence, refusing partial comparison: {missing}")
    data = [torch.load(path, map_location="cpu", weights_only=True) for path in files]
    if [x["cp"] for x in data] != [1, 2, 4]:
        raise ValueError("wrong CP metadata")
    for other in data[1:]:
        compare(data[0], other)


if __name__ == "__main__":
    main()
