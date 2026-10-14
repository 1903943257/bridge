"""Strictly audit reporting losses in one TP2/CP2/DP2 recorded PPO replay.

Each DP replica handles a DIFFERENT subset of the original eight Sessions.
The engine's returned scalar has ALREADY been summed across CP, but is
a LOCAL DP contribution (scaled by VERL's dp_size=2 policy). Thus DP0
and DP1 loss contributions MUST NOT be expected to match.

The benchmark's world-wide scalar mean sums the two DP contributions
while averaging over duplicated TP/CP views. It is reporting-only.
The production PPO objective and weight-gradient collectives are untouched.
"""
from __future__ import annotations

import math


def analyze_dp2_loss_report(ranks: list[dict], *, cp_atol: float = 1e-6) -> dict:
    if len(ranks) != 8:
        raise AssertionError(f"Expected eight rank reports, got {len(ranks)}")
    by_coordinate = {}
    for row in ranks:
        key = tuple(int(row[k]) for k in ("tp_rank", "cp_rank", "dp_rank"))
        if any(x not in (0, 1) for x in key) or key in by_coordinate:
            raise AssertionError(f"Invalid or repeated TP/CP/DP coordinates: {key}")
        by_coordinate[key] = row
    if len(by_coordinate) != 8:
        raise AssertionError("TP2/CP2/DP2 coordinates do not cover all eight ranks")

    local = {
        key: float(row["local_loss_dp_scaled"])
        for key, row in by_coordinate.items()
    }
    global_values = [float(row["global_loss"]) for row in ranks]
    if not all(math.isfinite(x) for x in (*local.values(), *global_values)):
        raise AssertionError("Non-finite PPO loss in distributed report")

    # CP-SUMmed losses should be equal on both CP ranks of one TP shard
    # and one DP replica. Do NOT check equality across DP replicas:
    # their input forests and token contributions are intentionally different.
    cp_gaps = {
        (tp, dp): abs(local[(tp, 0, dp)] - local[(tp, 1, dp)])
        for tp in (0, 1) for dp in (0, 1)
    }
    if max(cp_gaps.values()) > cp_atol:
        raise AssertionError(
            "CP replicas disagree on CP-SUMmed PPO local loss: "
            f"gaps={cp_gaps}, "
            f"rank_values={[(r['rank'], r['tp_rank'], r['cp_rank'], r['dp_rank'], r['local_loss_dp_scaled']) for r in ranks]}"
        )

    # The TP shards each represent the same logical result but may drift
    # numerically. Preserve the discrepancy, do not silently claim TP parity.
    tp_gaps = {
        (cp, dp): abs(local[(0, cp, dp)] - local[(1, cp, dp)])
        for cp in (0, 1) for dp in (0, 1)
    }
    dp_means = {
        dp: sum(local[(tp, cp, dp)] for tp in (0, 1) for cp in (0, 1)) / 4
        for dp in (0, 1)
    }
    tp_means = {
        tp: sum(local[(tp, cp, dp)] for cp in (0, 1) for dp in (0, 1)) / 4
        for tp in (0, 1)
    }
    expected_global = sum(local.values()) / 8
    global_gap = max(global_values) - min(global_values)
    if global_gap > cp_atol:
        raise AssertionError(
            "World-reduced PPO reporting loss differs across ranks: "
            f"max_gap={global_gap:.9g}, values={global_values}"
        )
    if abs(sum(global_values) / 8 - expected_global) > cp_atol:
        raise AssertionError(
            "Global PPO reporting mean does not match DP local contributions: "
            f"reported={sum(global_values)/8:.9g} expected={expected_global:.9g}"
        )

    return {
        "global_reporting_loss": sum(global_values) / 8,
        "dp0_local_loss": dp_means[0],
        "dp1_local_loss": dp_means[1],
        "dp_local_contribution_gap": abs(dp_means[0] - dp_means[1]),
        "tp0_pre_tp_loss": tp_means[0],
        "tp1_pre_tp_loss": tp_means[1],
        "tp_shard_loss_gap": abs(tp_means[0] - tp_means[1]),
        "tp_local_max_gap": max(tp_gaps.values()),
        "cp_replica_max_gap": max(cp_gaps.values()),
        "global_reporting_max_gap": global_gap,
        "expected_global_from_raw_rank_losses": expected_global,
        "reporting_reduce": "SUM(all_world_local_CP_aggregated_DP_contributions)/8",
        "training_objective_modified": False,
    }
