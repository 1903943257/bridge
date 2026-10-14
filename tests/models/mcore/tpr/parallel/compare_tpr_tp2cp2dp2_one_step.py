"""Validate comparable source data and score TP2CP4 Full vs TP2CP2DP2 TPR.

Perf only: both are fixed-weight F/B+finalize replays; no optimizer,
rollouts or multiple distinct training steps are claimed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def compare_sampled_gradients(full: dict, tpr: dict):
    left, right = full.get("gradient_witness"), tpr.get("gradient_witness")
    if left is None and right is None:
        return None
    if left is None or right is None:
        raise AssertionError("Full/TPR sampled gradient witness must be present on both")
    for key in ("names", "shapes", "samples_per_name", "samples_per_parameter"):
        if left[key] != right[key]:
            raise AssertionError(f"Full/TPR sampled gradient layout differs: {key}")
    # Full CP4's native Megatron gradient path is the reference; if it
    # exhibits small per-CP replica drift, report it as a limitation of
    # sampled Full-vs-TPR parity instead of misdiagnosing a DP2 bug.
    # TPR DP2 must have exactly equal finalized gradients on all DP+CP ranks.
    if right.get("replica_sample_max_abs") != 0:
        raise AssertionError(
            "TPR DP2 replica gradients are not identical after finalize: "
            f"sample_max_abs={right['replica_sample_max_abs']}"
        )
    full_replica_max_abs = float(left.get("replica_sample_max_abs", 0.0))
    fvalues = [float(v) for shard in left["tp_shards"] for v in shard]
    tvalues = [float(v) for shard in right["tp_shards"] for v in shard]
    if not fvalues or len(fvalues) != len(tvalues):
        raise AssertionError("Empty or mismatched TP2 gradient witness")
    if not all(math.isfinite(v) for v in fvalues + tvalues):
        raise AssertionError("Non-finite Full/TPR gradient witness")
    dot = sum(a*b for a, b in zip(fvalues, tvalues))
    fnorm2 = sum(a*a for a in fvalues)
    tnorm2 = sum(b*b for b in tvalues)
    error2 = sum((a-b)**2 for a, b in zip(fvalues, tvalues))
    return {
        "full_reference_replica_exact": full_replica_max_abs == 0,
        "full_reference_replica_sample_max_abs": full_replica_max_abs,
        "tpr_dp2_replica_exact": True,
        "sampled_gradient_parity_caveat": (
            "Full CP4 reference gradients have sampled replica disagreement; "
            "Full-vs-TPR gradient similarity uses its CP0 representative."
            if full_replica_max_abs else None
        ),
        "sampled_parameter_count": len(left["names"]),
        "sampled_gradient_element_count": len(fvalues),
        "sampled_gradient_rel_l2": math.sqrt(error2 / fnorm2) if fnorm2 else (
            0.0 if error2 == 0.0 else math.inf
        ),
        "sampled_gradient_cosine": (
            dot / math.sqrt(fnorm2*tnorm2)
            if fnorm2 > 0 and tnorm2 > 0 else
            (1.0 if fnorm2 == tnorm2 == 0 else 0.0)
        ),
        "sampled_gradient_max_abs": max(abs(a-b) for a, b in zip(fvalues, tvalues)),
        "full_sample_l2": math.sqrt(fnorm2),
        "tpr_sample_l2": math.sqrt(tnorm2),
        "note": "Deterministic sparse gradient samples; not full-parameter rel-L2",
    }


def compare(full_path: Path, tpr_path: Path) -> dict:
    full = json.loads(full_path.read_text())
    tpr = json.loads(tpr_path.read_text())
    if full.get("mode") != "full" or tpr.get("mode") != "tpr":
        raise AssertionError("Expected Native Full and TPR respectively")
    if tpr.get("topology") != {"tp": 2, "cp": 2, "dp": 2, "world": 8}:
        raise AssertionError("TPR result is not TP2 CP2 DP2")
    for field in ("source_step", "source_capture", "input_sha256",
                  "checkpoint", "warmups", "repeats",
                  "capture_logprobs", "profiler_enabled"):
        if full[field] != tpr[field]:
            raise AssertionError(f"Full vs TPR comparison rejected: {field} differs")
    full_cap = int(full.get("effective_sequence_cap", 40960))
    tpr_cap = int(tpr.get("effective_sequence_cap", 40960))
    if full_cap != tpr_cap:
        raise AssertionError(
            f"Full vs TPR sequence cap mismatch: {full_cap} vs {tpr_cap}"
        )
    if full.get("updated_weights") is not False or tpr.get("updated_weights") is not False:
        raise AssertionError("Both experiments must be fixed-weight F/B without AdamW")
    if full["runs"][0]["ppo_tokens"] != tpr["global_ppo_tokens"]:
        raise AssertionError("Global PPO-token count changed between Full and TPR")
    sampled_gradients = compare_sampled_gradients(full, tpr)
    if tpr_cap < 40960 and sampled_gradients is None:
        raise AssertionError("30K DP correctness experiment requires sampled gradient witness")
    if (len(full["runs"]) != full["warmups"] + full["repeats"]
            or len(tpr["runs"]) != tpr["warmups"] + tpr["repeats"]):
        raise AssertionError("Missing warmup or measured F/B iteration")
    if sorted(x for p in tpr["dp_partitions"] for x in p) != list(range(8)):
        raise AssertionError("TPR DP replicas did not cover eight Sessions exactly once")
    for record in tpr["runs"]:
        ranks = record["ranks"]
        if len(ranks) != 8:
            raise AssertionError("Missing distributed TPR rank record")
        for dp in (0, 1):
            subset = [r for r in ranks if r["dp_rank"] == dp]
            if len(subset) != 4 or any(
                r["rows"] != tpr["dp_partitions"][dp] for r in subset
            ):
                raise AssertionError("Inconsistent DP row ownership across TP/CP peers")
    f = float(full["median_fwd_bwd_s"])
    t = float(tpr["median_fwd_bwd_s"])
    if not (math.isfinite(f) and math.isfinite(t) and f > 0 and t > 0):
        raise ValueError("Invalid Full/TPR performance data")

    # Full and TPR peak memory are worst rank for each measured replay.
    fpeak = statistics.mean(
        r["peak_allocated_gib"] for r in full["runs"] if not r["warmup"]
    )
    tpeak = statistics.mean(
        r["peak_allocated_gib"] for r in tpr["runs"] if not r["warmup"]
    )
    fl = full["runs"][-1]["loss"]
    tl = tpr["runs"][-1]["loss"]
    dp2_loss_report = tpr["runs"][-1].get("loss_report")
    result = {
        "source_step": tpr["source_step"],
        "effective_sequence_cap": tpr_cap,
        "source_capture": tpr["source_capture"],
        "input_sha256": tpr["input_sha256"],
        "full_topology": "TP2_CP4_DP1",
        "tpr_topology": "TP2_CP2_DP2",
        "full_median_fwd_bwd_s": f,
        "tpr_median_fwd_bwd_s": t,
        "speedup": f / t,
        "full_mean_peak_allocated_gib_worst_rank": fpeak,
        "tpr_mean_peak_allocated_gib_worst_rank": tpeak,
        "mean_peak_memory_reduction_ratio": (fpeak-tpeak)/fpeak,
        "full_loss": fl, "tpr_global_loss": tl,
        "ppo_loss_absolute_gap": abs(fl-tl),
        "tpr_loss_consistency": dp2_loss_report,
        "global_ppo_tokens": tpr["global_ppo_tokens"],
        "sampled_gradient_parity": sampled_gradients,
        "dp_rows": tpr["dp_partitions"],
        "dp_tree_token_costs": tpr["dp_tree_token_costs"],
        "duplicated_tree_tokens": tpr["duplicated_tree_tokens"],
        "timed_scope": "forward_backward_plus_gradient_finalize_only",
        "updated_weights": False,
    }
    grad_text = (
        f"sampled_grad_rel_l2={sampled_gradients['sampled_gradient_rel_l2']:.6g} "
        f"sampled_grad_cosine={sampled_gradients['sampled_gradient_cosine']:.6g} "
        f"full_replica_max_abs={sampled_gradients['full_reference_replica_sample_max_abs']:.9g} "
        if sampled_gradients is not None else
        "sampled_grad_rel_l2=not_available "
    )
    loss_consistency_text = (
        f"tpr_tp_shard_loss_gap={dp2_loss_report['tp_shard_loss_gap']:.9g} "
        f"tpr_cp_replica_loss_gap={dp2_loss_report['cp_replica_max_gap']:.9g} "
        f"tpr_dp_local_contribution_gap={dp2_loss_report['dp_local_contribution_gap']:.9g} "
        if dp2_loss_report is not None else ""
    )
    print(
        f"TPR_DP2_VS_FULL step={result['source_step']} "
        f"full_median_s={f:.6f} tpr_median_s={t:.6f} "
        f"speedup={f/t:.6f} "
        f"full_peak_gib={fpeak:.3f} tpr_peak_gib={tpeak:.3f} "
        f"ppo_loss_gap={result['ppo_loss_absolute_gap']:.9g} "
        f"{loss_consistency_text}"
        f"{grad_text}"
        f"dp_rows={result['dp_rows']} "
        f"duplicated_tree_tokens={result['duplicated_tree_tokens']}",
        flush=True,
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("full_json", type=Path)
    p.add_argument("tpr_json", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()
    result = compare(args.full_json, args.tpr_json)
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.write_text(json.dumps(result, indent=2))
    print("TPR_DP2_COMPARISON=" + str(args.output_json), flush=True)


if __name__ == "__main__":
    main()
