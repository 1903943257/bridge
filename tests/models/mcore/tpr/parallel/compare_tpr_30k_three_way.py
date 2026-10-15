"""Compare one Step-6 30K PPO capture in three eight-NPU topologies.

A: Native Full TP2 CP4 DP1 (previously recorded).
B: TPR TP2 CP4 DP1 (the ONE new NPU run).
C: TPR TP2 CP2 DP2 (previously recorded).

Compare only identical checkpoint, derived-capture SHA, 30K cap, original PPO
labels, warmups, repeats, gradient-sync precision and benchmark scope.
No AdamW, rollout or production-DDP bucket-overlap speedups are claimed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .compare_tpr_tp2cp2dp2_one_step import compare_sampled_gradients


CP4 = {"tp": 2, "cp": 4, "dp": 1, "world": 8}
DP2 = {"tp": 2, "cp": 2, "dp": 2, "world": 8}


def _guard_equal(*values, key: str):
    if any(value != values[0] for value in values[1:]):
        raise AssertionError(f"Three-way mismatched {key}: {values!r}")
    return values[0]


def _mean_peak(item: dict) -> float:
    measured = [r for r in item["runs"] if not r["warmup"]]
    if len(measured) != item["repeats"]:
        raise AssertionError("Missing measured iterations")
    return statistics.mean(float(r["peak_allocated_gib"]) for r in measured)


def _sampled_pair(left: dict, right: dict) -> dict:
    comparison = compare_sampled_gradients(left, right)
    if comparison is None:
        raise AssertionError("Both compared modes require gradient witnesses")
    return {
        "sampled_rel_l2": comparison["sampled_gradient_rel_l2"],
        "sampled_cosine": comparison["sampled_gradient_cosine"],
        "sampled_max_abs": comparison["sampled_gradient_max_abs"],
        "sampled_elements": comparison["sampled_gradient_element_count"],
        "left_replica_max_abs": comparison["full_reference_replica_sample_max_abs"],
        "right_replica_max_abs": comparison["tpr_dp2_replica_sample_max_abs"],
        "note": "Sparse, fixed eight indices per parameter; NOT full-model grad rel-L2",
    }


def compare_three(full: dict, cp4: dict, dp2: dict) -> dict:
    if (full.get("mode"), cp4.get("mode"), dp2.get("mode")) != (
        "full", "tpr", "tpr",
    ):
        raise AssertionError("Expected Full CP4, TPR CP4, TPR CP2DP2")
    if cp4.get("topology") != CP4 or dp2.get("topology") != DP2:
        raise AssertionError("Expected CP4 topology and CP2DP2 topology in new records")
    # Old Full reference predates an explicit 'topology' metadata field.
    if full.get("topology", CP4) != CP4:
        raise AssertionError("Reference Full topology must be TP2CP4DP1")
    for field in (
        "source_step", "source_capture", "input_sha256", "checkpoint",
        "warmups", "repeats", "capture_logprobs", "profiler_enabled",
        "effective_sequence_cap", "gradient_sync_reduce_dtype",
        "hccl_stats_reduce_dtype",
    ):
        _guard_equal(full[field], cp4[field], dp2[field], key=field)
    if int(full["effective_sequence_cap"]) != 30720:
        raise AssertionError("This experiment requires the existing 30,720-token capture")
    if full["gradient_sync_reduce_dtype"] not in ("bf16", "fp32"):
        raise AssertionError("Unsupported gradient sync dtype")
    if full["hccl_stats_reduce_dtype"] != "float32":
        raise AssertionError("All experiments must use FP32 HCCL timing reductions")
    if cp4.get("hccl_deterministic") != "true":
        raise AssertionError("New CP4 TPR must run with HCCL_DETERMINISTIC=true")
    # Older Full/DP2 artifacts do not stamp this variable. They must be
    # supplied from the already completed deterministic experiment and are
    # explicitly marked unverified by the archived JSON alone.
    for name, item in (("full", full), ("tpr_cp2dp2", dp2)):
        flag = item.get("hccl_deterministic")
        if flag is not None and flag != "true":
            raise AssertionError(
                f"{name} used a different HCCL_DETERMINISTIC setting: {flag}"
            )
    if any(x.get("updated_weights") is not False for x in (full, cp4, dp2)):
        raise AssertionError("Weights must remain frozen across repeated F/B")
    if any(len(item["runs"]) != item["warmups"]+item["repeats"]
           for item in (full, cp4, dp2)):
        raise AssertionError("Missing F/B warmup or benchmark iteration")
    ppo_count = _guard_equal(
        int(full["runs"][0]["ppo_tokens"]),
        int(cp4["runs"][0]["ppo_tokens"]),
        int(dp2["global_ppo_tokens"]),
        key="global_ppo_tokens",
    )
    if ppo_count <= 0:
        raise AssertionError("Missing supervised PPO tokens")
    if sorted(row for part in dp2["dp_partitions"] for row in part) != list(range(8)):
        raise AssertionError("DP2 forest partitions must cover eight original trajectories")
    witness = {}
    for name, item in (("Full CP4", full), ("TPR CP4", cp4), ("TPR CP2DP2", dp2)):
        w = item.get("gradient_witness")
        if w is None:
            raise AssertionError(f"{name} missing sampled gradient witness")
        if w.get("replica_sample_max_abs") != 0:
            raise AssertionError(
                f"{name} gradients do not match across replicated CP/DP weights"
            )
    witness["full_vs_tpr_cp4"] = _sampled_pair(full, cp4)
    witness["full_vs_tpr_cp2dp2"] = _sampled_pair(full, dp2)
    witness["tpr_cp4_vs_tpr_cp2dp2"] = _sampled_pair(cp4, dp2)

    times = {
        "full_cp4": float(full["median_fwd_bwd_s"]),
        "tpr_cp4": float(cp4["median_fwd_bwd_s"]),
        "tpr_cp2dp2": float(dp2["median_fwd_bwd_s"]),
    }
    if not all(math.isfinite(t) and t > 0 for t in times.values()):
        raise ValueError(f"Non-finite or non-positive times: {times}")
    peak = {name: _mean_peak(item) for name,item in
            (("full_cp4", full), ("tpr_cp4", cp4), ("tpr_cp2dp2", dp2))}
    losses = {name: float(item["runs"][-1]["loss"]) for name,item in
              (("full_cp4", full), ("tpr_cp4", cp4), ("tpr_cp2dp2", dp2))}
    if not all(math.isfinite(z) for z in losses.values()):
        raise ValueError("Non-finite PPO loss")
    return {
        "source_step": full["source_step"],
        "source_capture": full["source_capture"],
        "input_sha256": full["input_sha256"],
        "effective_sequence_cap": full["effective_sequence_cap"],
        "global_ppo_tokens": ppo_count,
        "checkpoint": full["checkpoint"],
        "gradient_sync_reduce_dtype": full["gradient_sync_reduce_dtype"],
        "hccl_deterministic_new_cp4": "true",
        "hccl_deterministic_old_baselines_metadata": {
            "full": full.get("hccl_deterministic", "not_recorded"),
            "tpr_cp2dp2": dp2.get("hccl_deterministic", "not_recorded"),
        },
        "full_cp4_median_s": times["full_cp4"],
        "tpr_cp4_median_s": times["tpr_cp4"],
        "tpr_cp2dp2_median_s": times["tpr_cp2dp2"],
        "full_vs_tpr_cp4_speedup": times["full_cp4"] / times["tpr_cp4"],
        "tpr_cp4_vs_tpr_cp2dp2_dp_gain": (
            times["tpr_cp4"] / times["tpr_cp2dp2"]
        ),
        "full_vs_tpr_cp2dp2_speedup": (
            times["full_cp4"] / times["tpr_cp2dp2"]
        ),
        "mean_peak_allocated_gib_worst_rank": peak,
        "ppo_loss": losses,
        "ppo_loss_abs_gap_full_vs_tpr_cp4": (
            abs(losses["full_cp4"]-losses["tpr_cp4"])
        ),
        "ppo_loss_abs_gap_full_vs_tpr_cp2dp2": (
            abs(losses["full_cp4"]-losses["tpr_cp2dp2"])
        ),
        "ppo_loss_abs_gap_tpr_cp4_vs_cp2dp2": (
            abs(losses["tpr_cp4"]-losses["tpr_cp2dp2"])
        ),
        "sampled_gradient_parity": witness,
        "dp_partitions": dp2["dp_partitions"],
        "dp_tree_token_costs": dp2["dp_tree_token_costs"],
        "duplicated_tree_tokens": dp2["duplicated_tree_tokens"],
        "timed_scope": "forward_backward_plus_gradient_finalize_only",
        "updated_weights": False,
        "numerical_correctness_full_vs_tpr": (
            "NOT established: sampled rel-L2 and cosine must be independently assessed"
        ),
        "scope_caveat": (
            "No rollout/optimizer; Full CP4 and TPR both use manual per-parameter "
            "finalize, not production Megatron DDP bucket overlap"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("full_json", type=Path)
    parser.add_argument("tpr_cp4_json", type=Path)
    parser.add_argument("tpr_cp2dp2_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    report = compare_three(
        json.loads(args.full_json.read_text()),
        json.loads(args.tpr_cp4_json.read_text()),
        json.loads(args.tpr_cp2dp2_json.read_text()),
    )
    args.output_json.write_text(json.dumps(report, indent=2))
    print(
        f"TPR_30K_THREE_WAY step={report['source_step']} "
        f"full_s={report['full_cp4_median_s']:.6f} "
        f"tpr_cp4_s={report['tpr_cp4_median_s']:.6f} "
        f"tpr_cp2dp2_s={report['tpr_cp2dp2_median_s']:.6f} "
        f"tpr_cp4_speedup={report['full_vs_tpr_cp4_speedup']:.6f} "
        f"dp2_incremental_gain={report['tpr_cp4_vs_tpr_cp2dp2_dp_gain']:.6f} "
        f"tpr_dp2_speedup={report['full_vs_tpr_cp2dp2_speedup']:.6f} "
        f"full_vs_cp4_grad_rel_l2="
        f"{report['sampled_gradient_parity']['full_vs_tpr_cp4']['sampled_rel_l2']:.6g} "
        f"full_vs_dp2_grad_rel_l2="
        f"{report['sampled_gradient_parity']['full_vs_tpr_cp2dp2']['sampled_rel_l2']:.6g} "
        f"cp4_vs_dp2_grad_rel_l2="
        f"{report['sampled_gradient_parity']['tpr_cp4_vs_tpr_cp2dp2']['sampled_rel_l2']:.6g}",
        flush=True,
    )
    print("TPR_30K_THREE_WAY_JSON="+str(args.output_json), flush=True)


if __name__ == "__main__":
    main()
