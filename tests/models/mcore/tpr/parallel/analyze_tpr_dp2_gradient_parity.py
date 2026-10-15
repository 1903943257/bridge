"""Postprocess saved TP2 Full/TPR gradient witnesses without GPUs.

A scalar PPO loss can match closely even if gradients differ significantly.
Report which modules, layers, and TP shards contribute most to the squared
sampled-gradient error; do not silently treat sampled parity as tensor parity.

Usage:
 python -m tests.models.mcore.tpr.parallel.analyze_tpr_dp2_gradient_parity \\
    /path/full/step6_full_fixed_shape.json \\
    /path/tpr/step6_tpr_tp2cp2dp2.json [--top 25] [--output report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

_LAYER_RE = re.compile(r"(?:^|\\.)layers\\.(\\d+)\\.")


def _kind(name: str) -> str:
    if "linear_qkv" in name or "qkv" in name:
        return "attention_qkv"
    if "linear_proj" in name or "self_attention" in name or "attention" in name:
        return "attention_other"
    if ".mlp." in name or ".linear_fc" in name:
        return "mlp"
    if "layernorm" in name or "norm" in name:
        return "norm"
    if "embedding" in name or "output_layer" in name:
        return "embedding_output"
    return "other"


def _metrics(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right):
        raise AssertionError("Mismatched gradient sample vectors")
    fn2 = math.fsum(x * x for x in left)
    tn2 = math.fsum(y * y for y in right)
    dot = math.fsum(x * y for x, y in zip(left, right, strict=True))
    err2 = math.fsum((x-y)**2 for x, y in zip(left, right, strict=True))
    best_scale = dot / fn2 if fn2 else None  # TPR ~= scale * Full
    rescaled_err2 = (
        math.fsum((y-best_scale*x)**2 for x,y in zip(left,right,strict=True))
        if best_scale is not None else None
    )
    return {
        "samples": len(left),
        "full_l2": math.sqrt(fn2),
        "tpr_l2": math.sqrt(tn2),
        "error_l2": math.sqrt(err2),
        "error_sq": err2,
        "full_sq": fn2,
        "relative_l2": math.sqrt(err2 / fn2) if fn2 else (0.0 if err2 == 0 else math.inf),
        "cosine": dot / math.sqrt(fn2*tn2) if fn2*tn2 else None,
        "best_tpr_over_full_scale": best_scale,
        "relative_l2_after_best_scalar_rescale": (
            math.sqrt(rescaled_err2 / fn2)
            if fn2 and rescaled_err2 is not None else None
        ),
        "max_abs_error": max((abs(x-y) for x,y in zip(left,right,strict=True)), default=0.0),
        "nonzero_full": sum(x!=0 for x in left),
        "nonzero_tpr": sum(y!=0 for y in right),
        "opposite_sign_nonzero": sum(
            x*y < 0 for x,y in zip(left,right,strict=True)
        ),
    }


def analyze_gradient_witness(full: dict, tpr: dict, *, top: int = 25) -> dict:
    left, right = full.get("gradient_witness"), tpr.get("gradient_witness")
    if left is None or right is None:
        raise ValueError("Both Full and TPR must contain gradient_witness")
    for key in ("names", "shapes", "samples_per_name", "samples_per_parameter"):
        if left[key] != right[key]:
            raise AssertionError(f"Full/TPR sampled witness layout differs: {key}")
    names = left["names"]
    counts = left["samples_per_name"]
    if len(names) != len(counts):
        raise AssertionError("Invalid parameter witness layout")
    if len(left["tp_shards"]) != len(right["tp_shards"]) or len(left["tp_shards"]) != 2:
        raise AssertionError("Expected two TP gradient shards")
    total_count = sum(counts)
    if any(len(values) != total_count for values in
           (*left["tp_shards"], *right["tp_shards"])):
        raise AssertionError("Invalid flattened sampled-gradient lengths")

    by_param = []
    by_kind = defaultdict(lambda: [[], []])
    by_layer = defaultdict(lambda: [[], []])
    non_norm_full, non_norm_tpr = [], []
    by_tp = {}
    all_full, all_tpr = [], []
    for tp in (0,1):
        fvec = [float(x) for x in left["tp_shards"][tp]]
        tvec = [float(y) for y in right["tp_shards"][tp]]
        if not all(math.isfinite(z) for z in fvec + tvec):
            raise ValueError("Nonfinite gradient witness value")
        by_tp[str(tp)] = _metrics(fvec, tvec)
        all_full.extend(fvec)
        all_tpr.extend(tvec)
        offset=0
        for name, count in zip(names, counts, strict=True):
            f = fvec[offset:offset+count]
            t = tvec[offset:offset+count]
            m = _metrics(f,t)
            match = _LAYER_RE.search(name)
            kind = _kind(name)
            layer = match.group(1) if match else "(non-layer)"
            by_param.append({
                "tp_rank": tp, "name": name, "kind": kind, "layer": layer,
                "sample_offset_in_param_witness": offset,
                **m,
            })
            by_kind[kind][0].extend(f)
            by_kind[kind][1].extend(t)
            if kind != "norm":
                non_norm_full.extend(f)
                non_norm_tpr.extend(t)
            by_layer[layer][0].extend(f)
            by_layer[layer][1].extend(t)
            offset += count
    global_metrics = _metrics(all_full, all_tpr)
    non_norm_metrics = _metrics(non_norm_full, non_norm_tpr)
    denom = global_metrics["error_sq"]
    for item in by_param:
        item["share_of_total_error_sq"] = item["error_sq"] / denom if denom else 0.0
    by_param.sort(key=lambda m:m["error_sq"], reverse=True)
    kinds = {
        key: {**_metrics(pair[0],pair[1]),
              "share_of_total_error_sq": _metrics(pair[0],pair[1])["error_sq"] / denom if denom else 0.0}
        for key,pair in by_kind.items()
    }
    layers = {
        key: {**_metrics(pair[0],pair[1]),
              "share_of_total_error_sq": _metrics(pair[0],pair[1])["error_sq"] / denom if denom else 0.0}
        for key,pair in by_layer.items()
    }
    # Identical strided TP0/TP1 Norm samples provide no independent
    # evidence. Do not silently count this duplication as broader coverage.
    repeated_norm_params = []
    off = 0
    for name, count in zip(names, counts, strict=True):
        if _kind(name) == "norm":
            a0 = left["tp_shards"][0][off:off+count]
            a1 = left["tp_shards"][1][off:off+count]
            b0 = right["tp_shards"][0][off:off+count]
            b1 = right["tp_shards"][1][off:off+count]
            if a0 == a1 and b0 == b1:
                repeated_norm_params.append(name)
        off += count
    norm_metrics = _metrics(by_kind["norm"][0], by_kind["norm"][1])
    return {
        "source_step": tpr.get("source_step"),
        "source_sha_match": full.get("input_sha256") == tpr.get("input_sha256"),
        "full_loss": full["runs"][-1]["loss"],
        "tpr_loss": tpr["runs"][-1]["loss"],
        "full_replica_exact": left.get("replica_exact"),
        "tpr_replica_exact": right.get("replica_exact"),
        "is_full_tensor_gradient_parity": False,
        "global_sampled": global_metrics,
        "sampled_non_norm": non_norm_metrics,
        "norm_share_of_full_sampled_gradient_energy": (
            norm_metrics["full_sq"] / global_metrics["full_sq"]
            if global_metrics["full_sq"] else None
        ),
        "norm_share_of_sampled_squared_error": (
            norm_metrics["error_sq"] / denom if denom else 0.0
        ),
        "identical_norm_sample_pairs_across_tp": {
            "count": len(repeated_norm_params),
            "names": repeated_norm_params,
            "warning": (
                "These TP0/TP1 sampled values are identical, so the "
                "unweighted 8-samples-per-parameter metric counts "
                "replicated evidence twice; this does not establish "
                "tensor-wide equality or the parameter sharding policy."
            ),
        },
        "sampling_warning": (
            "Eight deterministic, evenly spaced samples per named parameter "
            "bias the global metric toward small parameters such as LayerNorm. "
            "These numbers are NOT full-gradient rel-L2 or full-model error energy."
        ),
        "per_tp": by_tp,
        "per_kind": dict(sorted(kinds.items(), key=lambda p:p[1]["error_sq"],reverse=True)),
        "per_layer": dict(sorted(layers.items(), key=lambda p:p[1]["error_sq"],reverse=True)),
        "top_parameters": by_param[:top],
        "all_parameters_ranked": by_param,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("full", type=Path)
    p.add_argument("tpr", type=Path)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--output", type=Path)
    args=p.parse_args()
    if args.top < 1:
        p.error("--top must be positive")
    report = analyze_gradient_witness(
        json.loads(args.full.read_text()),
        json.loads(args.tpr.read_text()), top=args.top,
    )
    if not report["source_sha_match"]:
        raise AssertionError("Full and TPR compared different source captures")
    g=report["global_sampled"]
    print(
        "TPR_GRAD_PARITY_BREAKDOWN "
        f"rel_l2={g['relative_l2']:.9g} cosine={g['cosine']:.9g} "
        f"best_scale={g['best_tpr_over_full_scale']:.9g} "
        f"residual_rel_l2={g['relative_l2_after_best_scalar_rescale']:.9g}"
    )
    nn=report["sampled_non_norm"]
    print(
        "TPR_GRAD_PARITY_NON_NORM "
        f"rel_l2={nn['relative_l2']:.9g} cosine={nn['cosine']} "
        f"norm_full_sample_energy_share="
        f"{report['norm_share_of_full_sampled_gradient_energy']:.2%} "
        f"duplicated_norm_tp_sample_pairs="
        f"{report['identical_norm_sample_pairs_across_tp']['count']}"
    )
    print("TPR_GRAD_SAMPLING_CAVEAT " + report["sampling_warning"])
    for tp,m in report["per_tp"].items():
        print(
            f"  TP{tp}: rel_l2={m['relative_l2']:.6g} "
            f"cos={m['cosine']:.6g} best_scale={m['best_tpr_over_full_scale']:.6g}"
        )
    for kind,m in report["per_kind"].items():
        print(
            f"  KIND {kind}: error_share={m['share_of_total_error_sq']:.2%} "
            f"rel_l2={m['relative_l2']:.6g} cos={m['cosine']}"
        )
    print("  TOP_PARAMS_BY_SQUARED_ERROR:")
    for item in report["top_parameters"]:
        print(
            f"    tp={item['tp_rank']} {item['name']} "
            f"error_share={item['share_of_total_error_sq']:.2%} "
            f"rel_l2={item['relative_l2']:.6g} "
            f"cos={item['cosine']} "
            f"full_l2={item['full_l2']:.6g} tpr_l2={item['tpr_l2']:.6g}"
        )
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(report, indent=2))
        print("TPR_GRAD_PARITY_BREAKDOWN_JSON="+str(args.output))


if __name__ == "__main__":
    main()
