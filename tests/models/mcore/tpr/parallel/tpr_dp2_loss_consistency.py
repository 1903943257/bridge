"""Validate one TP2/CP2/DP2 recorded PPO replay's loss-report reductions.

The TPR PPO adapter already CP-SUMs a local forest's loss. The benchmark
then DP-averages the two replicas, leaving ONE reporting scalar per TP
shard. TP shards can differ slightly numerically, even though their
forward's vocab-parallel logprob value should approximately match.

This module checks strict *within-TP* CP×DP reporting consistency,
separately records pre-TP scalar drift, and checks the final TP-averaged
reporting value. It never changes the training loss or gradients.
"""
from __future__ import annotations

import math


def analyze_dp2_loss_report(ranks: list[dict], *, cpdp_atol: float = 1e-6) -> dict:
    if len(ranks) != 8:
        raise AssertionError(f"Expected eight loss reports, got {len(ranks)}")
    by_tp: dict[int, list[dict]] = {0: [], 1: []}
    for rank in ranks:
        tp, cp, dp = (int(rank[k]) for k in ("tp_rank", "cp_rank", "dp_rank"))
        if tp not in by_tp or cp not in (0, 1) or dp not in (0, 1):
            raise AssertionError(f"Unexpected TP/CP/DP rank coordinates: {rank}")
        by_tp[tp].append(rank)
    for tp, peers in by_tp.items():
        if {(int(p["cp_rank"]), int(p["dp_rank"])) for p in peers} != {
            (0, 0), (0, 1), (1, 0), (1, 1)
        }:
            raise AssertionError(f"TP={tp} does not cover CP2×DP2 exactly once")

    values = [
        float(p["global_loss"]) for p in ranks
    ]
    raw_by_tp = [
        [float(p["global_loss_pre_tp"]) for p in by_tp[tp]]
        for tp in (0, 1)
    ]
    if not all(math.isfinite(x) for x in values + raw_by_tp[0] + raw_by_tp[1]):
        raise AssertionError("Non-finite PPO global loss reported by rank")
    cpdp_spread = {
        str(tp): max(raw_by_tp[tp]) - min(raw_by_tp[tp])
        for tp in (0, 1)
    }
    # Each CP/DP rank of the same TP shard performed identical CP and DP
    # AllReduce; disagreement in that replica group is a real bug.
    if any(gap > cpdp_atol for gap in cpdp_spread.values()):
        raise AssertionError(
            "CP/DP replicas disagree on pre-TP PPO loss: "
            f"cpdp_spread={cpdp_spread}, "
            f"rank_values={[ (r['rank'], r['tp_rank'], r['cp_rank'], r['dp_rank'], r['global_loss_pre_tp']) for r in ranks ]}"
        )
    consensus_spread = max(values) - min(values)
    if consensus_spread > cpdp_atol:
        raise AssertionError(
            "TP-averaged global PPO reporting loss still differs across ranks: "
            f"spread={consensus_spread:.9g}, "
            f"rank_values={[ (r['rank'], r['global_loss']) for r in ranks ]}"
        )
    means = [sum(raw_by_tp[tp]) / len(raw_by_tp[tp]) for tp in (0, 1)]
    reconstructed = (means[0] + means[1]) / 2
    if abs(sum(values)/len(values) - reconstructed) > cpdp_atol:
        raise AssertionError(
            f"TP loss aggregation is not a mean of the two TP shards: "
            f"global={sum(values)/len(values)} expected={reconstructed}"
        )
    return {
        "global_reporting_loss": sum(values) / len(values),
        "tp0_pre_tp_loss": means[0],
        "tp1_pre_tp_loss": means[1],
        "tp_shard_loss_gap": abs(means[0] - means[1]),
        "tp_shard_loss_gap_exceeds_1e_4": abs(means[0] - means[1]) > 1e-4,
        "cpdp_pre_tp_max_gap": max(cpdp_spread.values()),
        "global_reporting_max_gap": consensus_spread,
        "tp_reporting_reduce": "SUM(tp_group)/2 after SUM(dp_group)/2 after CP SUM",
        "training_objective_modified": False,
    }
