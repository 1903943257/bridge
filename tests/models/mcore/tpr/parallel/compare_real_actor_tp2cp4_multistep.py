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
from .compare_real_actor_tp2cp4_offline_ppo import (
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


def compare(outdir: Path, steps: int):
    milestones = tuple(n for n in (1,2,5,10,20,30,40,50) if n <= steps)
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
        for a, b in zip(f, t):
            for field in (
                "step", "steps", "source_capture", "source_rows", "source_keys",
                "data_sha256", "checkpoint", "optimizer", "lr", "clip_norm",
                "tokens", "run_kind", "rank", "tp_rank", "cp_rank"
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
        master = _master_update(f,t)
        lps = _lp_compare(f,t)
        sampled_grad = _l2_pair(f,t)
        complete_grad = _complete_gradient_metrics(f,t)
        record = {
            "step":step,
            "full_loss":f[0]["loss"], "tpr_loss":t[0]["loss"],
            "loss_gap":t[0]["loss"]-f[0]["loss"],
            "full_grad_norm":f[0]["global_grad_norm_preclip"],
            "tpr_grad_norm":t[0]["global_grad_norm_preclip"],
            "full_grad_clip_factor":f[0]["grad_clip_factor"],
            "tpr_grad_clip_factor":t[0]["grad_clip_factor"],
            "sampled_grad_relative_l2":sampled_grad,
            "complete_grad":complete_grad,
            "logprob":lps,
            "master":master,
            "full_fwd_bwd_seconds_critical":max(x["fwd_bwd_seconds"] for x in f),
            "tpr_fwd_bwd_seconds_critical":max(x["fwd_bwd_seconds"] for x in t),
            "full_cpu_adamw_seconds_critical":max(x["cpu_optimizer_seconds"] for x in f),
            "tpr_cpu_adamw_seconds_critical":max(x["cpu_optimizer_seconds"] for x in t),
            "full_peak_allocated_gib":max(x["peak_allocated_bytes"] for x in f)/2**30,
            "tpr_peak_allocated_gib":max(x["peak_allocated_bytes"] for x in t)/2**30,
        }
        observed.append(record)
        print(
            f"TPR_50STEP step={step} loss_full={record['full_loss']:.9g} "
            f"loss_tpr={record['tpr_loss']:.9g} gap={record['loss_gap']:.9g} "
            f"lp_rel_l2={lps['rel_l2']:.6g} "
            f"sampled_grad_rel_l2={sampled_grad:.6g} "
            f"full_gradnorm={record['full_grad_norm']:.6g} "
            f"tpr_gradnorm={record['tpr_grad_norm']:.6g} "
            f"FP32_master_update_rel_l2={master['master_update_rel_l2']:.6g} "
            f"FP32_master_update_cos={master['master_update_cosine']:.6g} "
            f"full_peak_gib={record['full_peak_allocated_gib']:.3f} "
            f"tpr_peak_gib={record['tpr_peak_allocated_gib']:.3f}", flush=True
        )
    result = {
        "status": "DIAGNOSTIC_FIXED_DATA_PPO_NOT_ON_POLICY",
        "steps_completed":steps,
        "milestones":observed,
        "interpretation":(
            "Repeated identical actor-captured PPO tokens, advantages, old "
            "logprobs and CPU FP32-master AdamW; no fresh rollouts, so not "
            "an RL generalization or on-policy convergence comparison. "
            "Full is unshared RingCP and not native MindSpeed attention."
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
