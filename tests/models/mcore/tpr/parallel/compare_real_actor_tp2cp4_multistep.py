"""CPU-only milestone comparison of 1-50 true-PPO fixed-trajectory replay.

Full and TPR load independently from the same real 4B checkpoint. Evaluate
training PPO, *sampled FP32-master cumulative update* drift, sampled gradient
drift, and per-step memory/time. NOT an on-policy RL convergence proof.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from compare_real_actor_tp2cp4_offline_ppo import (
    _l2_pair, _logprobs, _complete_gradient_metrics,
)


def _master_update(a, b):
    aa, bb = 0., 0.
    dot, delta = 0., 0.
    old_sq = 0.
    for tp_rank in (0, 1):
        f = next(x for x in a if x["tp_rank"] == tp_rank and x["cp_rank"] == 0)
        t = next(x for x in b if x["tp_rank"] == tp_rank and x["cp_rank"] == 0)
        if f["master_initial"].keys() != t["master_initial"].keys():
            raise AssertionError("Different FP32 master parameters")
        for name in f["master_initial"]:
            initial = f["master_initial"][name]
            initial_t = t["master_initial"][name]
            end_f, end_t = f["master_after"][name], t["master_after"][name]
            if (initial["shape"] != initial_t["shape"] or
                not torch.equal(initial["indices"], initial_t["indices"]) or
                not torch.equal(initial["values"], initial_t["values"])):
                raise AssertionError(f"Initial FP32 master mismatch: TP{tp_rank} {name}")
            if not torch.equal(end_f["indices"], end_t["indices"]):
                raise AssertionError(f"Sampled master indices mismatch: {name}")
            u = end_f["values"].double() - initial["values"].double()
            v = end_t["values"].double() - initial["values"].double()
            aa += float((u*u).sum())
            bb += float((v*v).sum())
            delta += float(((u-v)**2).sum())
            dot += float((u*v).sum())
            old_sq += float((initial["values"].double()**2).sum())
    return {
        "master_full_update_norm": math.sqrt(aa),
        "master_tpr_update_norm": math.sqrt(bb),
        "master_update_rel_l2": math.sqrt(delta/max(aa, 1e-30)),
        "master_update_cosine": dot/max(math.sqrt(aa*bb), 1e-30),
        "master_weight_rel_l2": math.sqrt(delta/max(old_sq, 1e-30)),
    }


def _lp_compare(full, tpr):
    a = _logprobs(full)
    b = _logprobs(tpr)
    if a.keys() != b.keys():
        raise AssertionError("Missing or duplicated logical PPO tokens")
    if len(a) != full[0]["tokens"]:
        raise AssertionError("Incomplete logical PPO logprob coverage")
    order = sorted(a)
    x = torch.tensor([a[i] for i in order], dtype=torch.float64)
    y = torch.tensor([b[i] for i in order], dtype=torch.float64)
    return {
        "tokens": len(order),
        "rel_l2": float(torch.linalg.vector_norm(x-y) /
                        torch.linalg.vector_norm(x).clamp_min(1e-30)),
        "max_abs": float((x-y).abs().max()),
        "mean_abs": float((x-y).abs().mean()),
    }



def _profile_summary(rank_snapshots):
    """Per-segment envelope: max-rank event time and allocator peak.

    Stream NPU events are NOT summed as exclusive wall time; separate
    end-to-end critical-path time is measured after torch.npu.synchronize().
    """
    groups = {}
    stage_ms = {}
    for rank_snap in rank_snapshots:
        for key, value in rank_snap.get("stage_profile", {}).items():
            stage_ms[key] = max(stage_ms.get(key,0.0), float(value))
        for event in rank_snap.get("segment_events", ()):
            key = (event["tree_index"],event["segment_id"],event["kind"])
            group = groups.setdefault(key, {
                "tree":event["tree_index"],"segment_id":event["segment_id"],
                "kind":event["kind"],"length":event["length"],
                "prefix":event["prefix_length"],"max_rank_npu_event_ms":0.,
                "max_rank_cpu_enqueue_ms":0.,"peak_allocated_gib":0.,
                "peak_reserved_gib":0.,
                "max_rank_live_alloc_before_gib":0.,
                "max_rank_live_alloc_after_gib":0.,
                "max_rank_live_alloc_delta_gib":float("-inf"),
            })
            if group["length"] != event["length"] or group["prefix"] != event["prefix_length"]:
                raise AssertionError(f"segment plan different across ranks {key}")
            group["max_rank_npu_event_ms"] = max(
                group["max_rank_npu_event_ms"], float(event["npu_event_ms"])
            )
            group["max_rank_cpu_enqueue_ms"] = max(
                group["max_rank_cpu_enqueue_ms"], float(event["cpu_enqueue_ms"])
            )
            group["peak_allocated_gib"] = max(
                group["peak_allocated_gib"],event["peak_allocated_bytes"]/2**30
            )
            group["peak_reserved_gib"] = max(
                group["peak_reserved_gib"],event["peak_reserved_bytes"]/2**30
            )
            group["max_rank_live_alloc_before_gib"] = max(
                group["max_rank_live_alloc_before_gib"],event["alloc_before_bytes"]/2**30
            )
            group["max_rank_live_alloc_after_gib"] = max(
                group["max_rank_live_alloc_after_gib"],event["alloc_after_bytes"]/2**30
            )
            group["max_rank_live_alloc_delta_gib"] = max(
                group["max_rank_live_alloc_delta_gib"],
                (event["alloc_after_bytes"]-event["alloc_before_bytes"])/2**30
            )
    return {"segment_events":list(groups.values()),"cpu_stage_ms_worst_rank":stage_ms}

def compare(outdir: Path, steps: int):
    first = outdir / "full_step1_rank0.pt"
    if not first.exists():
        raise FileNotFoundError(first)
    first_snap = torch.load(first, map_location="cpu", weights_only=False)
    if first_snap.get("capture_manifest"):
        milestones = tuple(range(1,steps+1))
        run_protocol = (
            "SESSION_FIRST_SEQUENTIAL_CAPTURE_UPDATES"
            if first_snap.get("source_capture_index") is not None
            else "SEQUENTIAL_20_DISTINCT_REAL_CAPTURES"
        )
    else:
        milestones = tuple(n for n in (1,2,5,10,20,30,40,50) if n <= steps)
        run_protocol = "REPEATED_FIXED_PPO_DATA_NOT_ON_POLICY"
    del first_snap
    observed = []
    for step in milestones:
        mode = {}
        for label in ("full", "tpr"):
            expected = [outdir / f"{label}_step{step}_rank{r}.pt" for r in range(8)]
            missing = [str(path) for path in expected if not path.exists()]
            if missing:
                raise FileNotFoundError(f"step {step} {label} snapshots missing: {missing}")
            mode[label] = [
                torch.load(path, map_location="cpu", weights_only=False)
                for path in expected
            ]
        f, t = mode["full"], mode["tpr"]
        if any(x.get("execution_backend") != "native_megatron_schedule" for x in f):
            raise AssertionError(
                "Full baseline must be the genuine Megatron schedule; old "
                "unshared TPR snapshots are not interchangeable"
            )
        if any(
            x.get("native_loss_scaling") != "megatron_cp_once_global_token_mean_v2"
            for x in f
        ):
            raise AssertionError(
                "Native Full snapshots use obsolete/unknown CP loss scaling. "
                "Earlier runs multiplied CP=4 twice and must be replayed "
                "from the original checkpoint before comparison."
            )
        if any(x.get("execution_backend", "tpr_segment_executor") != "tpr_segment_executor" for x in t):
            raise AssertionError("TPR baseline must remain the original segment executor")
        native_recompute = {x.get("native_recompute_layers") for x in f}
        if native_recompute != {0}:
            raise AssertionError(
                "Fair Native Full vs TPR replay requires recompute OFF on both "
                f"paths, found Native Full recompute settings {native_recompute}. "
                "Do not compare a capacity-only recompute run as parity."
            )
        for a, b in zip(f, t):
            for field in (
                "step", "steps", "source_capture", "source_rows", "source_keys",
                "data_sha256", "checkpoint", "optimizer", "lr", "clip_norm",
                "tokens", "run_kind", "rank", "tp_rank", "cp_rank",
                "source_online_global_step", "source_capture_index"
            ):
                if a[field] != b[field]:
                    raise AssertionError(f"Full vs TPR step={step} mismatch in {field}")
        if f[0]["steps"] != steps:
            raise AssertionError("Replay output has unexpected total step count")
        for label, items in mode.items():
            if max(abs(x["loss"] - items[0]["loss"]) for x in items) >= 1e-4:
                raise AssertionError(f"{label} loss differs across ranks")
            if max(abs(x["global_grad_norm_preclip"] -
                       items[0]["global_grad_norm_preclip"]) for x in items) >= 1e-5:
                raise AssertionError(f"{label} global norm differs across ranks")
        for label, items in mode.items():
            if len({x.get("ring_attention_impl","legacy_custom") for x in items}) != 1:
                raise AssertionError(f"inconsistent {label} Ring backend among ranks")
        master = _master_update(f,t)
        lps = _lp_compare(f,t)
        sampled_grad = _l2_pair(f,t)
        complete_grad = _complete_gradient_metrics(f,t)
        f_profile = _profile_summary(f)
        t_profile = _profile_summary(t)
        full_norm = f[0]["global_grad_norm_preclip"]
        tpr_norm = t[0]["global_grad_norm_preclip"]
        record = {
            "step":step,
            "source_online_global_step":f[0].get("source_online_global_step"),
            "source_capture_index":f[0].get("source_capture_index"),
            "full_attention_impl":f[0].get("ring_attention_impl","legacy_custom"),
            "full_execution_backend":f[0]["execution_backend"],
            "tpr_execution_backend":t[0].get("execution_backend", "tpr_segment_executor"),
            "full_native_recompute_layers":f[0]["native_recompute_layers"],
            "full_native_loss_scaling":f[0]["native_loss_scaling"],
            "tpr_attention_impl":t[0].get("ring_attention_impl","legacy_custom"),
            "full_loss":f[0]["loss"], "tpr_loss":t[0]["loss"],
            "loss_gap":t[0]["loss"]-f[0]["loss"],
            "full_grad_norm":f[0]["global_grad_norm_preclip"],
            "tpr_grad_norm":tpr_norm,
            "grad_norm_relative_gap":abs(tpr_norm-full_norm)/max(full_norm,1e-30),
            "full_finalize_seconds_critical":max(x.get("finalize_seconds",0.) for x in f),
            "tpr_finalize_seconds_critical":max(x.get("finalize_seconds",0.) for x in t),
            "full_segment_profile":f_profile,
            "tpr_segment_profile":t_profile,
            "full_grad_clip_factor":f[0]["grad_clip_factor"],
            "tpr_grad_clip_factor":t[0]["grad_clip_factor"],
            "sampled_grad_relative_l2":sampled_grad,
            "complete_grad":complete_grad,
            "logprob":lps,
            "master":master,
            "full_fwd_bwd_seconds_critical":max(x["fwd_bwd_seconds"] for x in f),
            "tpr_fwd_bwd_seconds_critical":max(x["fwd_bwd_seconds"] for x in t),
            "fwd_bwd_speedup": (
                max(x["fwd_bwd_seconds"] for x in f) /
                max(x["fwd_bwd_seconds"] for x in t)
            ),
            "full_cpu_adamw_seconds_critical":max(x["cpu_optimizer_seconds"] for x in f),
            "tpr_cpu_adamw_seconds_critical":max(x["cpu_optimizer_seconds"] for x in t),
            "full_peak_allocated_gib":max(x["peak_allocated_bytes"] for x in f)/2**30,
            "tpr_peak_allocated_gib":max(x["peak_allocated_bytes"] for x in t)/2**30,
            "peak_allocated_saving_fraction": (
                1 - max(x["peak_allocated_bytes"] for x in t) /
                max(x["peak_allocated_bytes"] for x in f)
            ),
        }
        observed.append(record)
        if f_profile["segment_events"] or t_profile["segment_events"]:
            for label, prof in (("full",f_profile),("tpr",t_profile)):
                top_events = sorted(prof["segment_events"], key=lambda x:x["max_rank_npu_event_ms"], reverse=True)
                print(f"TPR_20STEP_PROFILE step={step} mode={label} "
                      f"overhead_ms={prof['cpu_stage_ms_worst_rank']} "
                      f"segment_event_count={len(top_events)} "
                      f"top_events={top_events[:12]}",flush=True)
        print(
            f"TPR_50STEP step={step} loss_full={record['full_loss']:.9g} "
            f"loss_tpr={record['tpr_loss']:.9g} gap={record['loss_gap']:.9g} "
            f"lp_rel_l2={lps['rel_l2']:.6g} "
            f"sampled_grad_rel_l2={sampled_grad:.6g} "
            f"full_gradnorm={record['full_grad_norm']:.6g} "
            f"tpr_gradnorm={record['tpr_grad_norm']:.6g} "
            f"gradnorm_rel_gap={record['grad_norm_relative_gap']:.6g} "
            f"FP32_master_update_rel_l2={master['master_update_rel_l2']:.6g} "
            f"FP32_master_update_cos={master['master_update_cosine']:.6g} "
            f"full_peak_gib={record['full_peak_allocated_gib']:.3f} "
            f"tpr_peak_gib={record['tpr_peak_allocated_gib']:.3f} "
            f"speedup={record['fwd_bwd_speedup']:.4g} "
            f"allocated_saving={record['peak_allocated_saving_fraction']:.2%}", flush=True
        )
    result = {
        "status": "DIAGNOSTIC_SEQUENTIAL_OFFLINE_PPO_REPLAY",
        "run_protocol":run_protocol,
        "steps_completed":steps,
        "milestones":observed,
        "interpretation":(
            "Original actor-captured PPO tokens, advantages and old logprobs; "
            "if run_protocol is SESSION_FIRST_SEQUENTIAL_CAPTURE_UPDATES, "
            "multiple actor captures from the same online global step can be "
            "replayed as separate optimizer updates; session-first filtering "
            "uses the explicit original trajectory ordinal only when present, "
            "otherwise earliest capture-order row (not guaranteed original first). "
            "No historical reward/advantage value is modified. "
            "The replay policies are separately trained from the same initial "
            "weights and are NOT resampled online at each replay step. "
            "CPU FP32-master AdamW is diagnostic and not production optimizer. "
            "Native Full uses the original Megatron forward/backward schedule, "
            "unmodified MindSpeed model attention, and VERL THD/PPO helpers. "
            "TPR uses its custom SegmentExecutor and TPR Ring backend. "
            "Native Full recompute is separately reported: its performance "
            "must not be labeled an equal-recompute ablation when enabled."
        ),
    }
    target = outdir / "multistep_comparison.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"TPR_50STEP_SUMMARY status={result['status']} saved={target}", flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()
    compare(args.output_dir, args.steps)
