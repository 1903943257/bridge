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
        "global_ppo_tokens": tpr["global_ppo_tokens"],
        "dp_rows": tpr["dp_partitions"],
        "dp_tree_token_costs": tpr["dp_tree_token_costs"],
        "duplicated_tree_tokens": tpr["duplicated_tree_tokens"],
        "timed_scope": "forward_backward_plus_gradient_finalize_only",
        "updated_weights": False,
    }
    print(
        f"TPR_DP2_VS_FULL step={result['source_step']} "
        f"full_median_s={f:.6f} tpr_median_s={t:.6f} "
        f"speedup={f/t:.6f} "
        f"full_peak_gib={fpeak:.3f} tpr_peak_gib={tpeak:.3f} "
        f"ppo_loss_gap={result['ppo_loss_absolute_gap']:.9g} "
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
