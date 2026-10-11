"""P0-5: compare real-TQ FULL independent trajectories vs shared-Prefix TPR.

Read only saved independently executed Ascend NPU evidence.
No invented accuracy thresholds. Model update evidence uses strided samples.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import torch
from torch.torch_version import TorchVersion


def _load(p):
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(p, map_location="cpu", weights_only=True)


def _metric(ref, value):
    a = ref.detach().float().flatten().double()
    b = value.detach().float().flatten().double()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError(f"Invalid compared tensors: {a.shape}, {b.shape}")
    err = b - a
    norm = a.norm().clamp_min(1e-24)
    dot = (a.norm() * b.norm()).clamp_min(1e-24)
    return {
        "numel": int(a.numel()),
        "mean_abs": float(err.abs().mean()),
        "max_abs": float(err.abs().max()),
        "p95_abs": float(torch.quantile(err.abs().float(), 0.95)),
        "rel_l2": float(err.norm()/norm),
        "cosine": float(torch.dot(a,b)/dot),
    }


def _sample_vector(a, b, field):
    if set(a) != set(b):
        raise AssertionError("Parameter names differ")
    x, y = [], []
    for name in sorted(a):
        left, right = a[name], b[name]
        if left["shape"] != right["shape"] or not torch.equal(left["indices"], right["indices"]):
            raise AssertionError(f"Strided samples not aligned for {name}")
        x.append(left[field].float().flatten())
        y.append(right[field].float().flatten())
    return _metric(torch.cat(x), torch.cat(y))


def _update_vector(full, tpr, step):
    start1 = full["checkpoints"]["0"]["sampled_weights"]
    start2 = tpr["checkpoints"]["0"]["sampled_weights"]
    f = full["checkpoints"][str(step)]["sampled_weights"]
    t = tpr["checkpoints"][str(step)]["sampled_weights"]
    left, right = [], []
    for name in sorted(start1):
        if not torch.equal(start1[name]["indices"], start2[name]["indices"]):
            raise AssertionError(f"Initial strided weight indices differ at {name}")
        left.append(f[name]["master"].float() - start1[name]["master"].float())
        right.append(t[name]["master"].float() - start2[name]["master"].float())
    return _metric(torch.cat(left), torch.cat(right))


def _masked_lps(a, b, masks):
    one, two = [], []
    if set(a) != set(b) or set(a) != set(masks):
        raise AssertionError("Real sibling response logprobs keys do not match")
    for key in sorted(masks, key=int):
        m = masks[key].bool()
        if m.numel() != 512:
            raise AssertionError("Original TQ response mask must be 512")
        one.append(a[key][m])
        two.append(b[key][m])
    return _metric(torch.cat(one), torch.cat(two))


def _validate(full, tpr, steps):
    if (full["mode"], tpr["mode"]) != ("full", "tpr"):
        raise AssertionError("Wrong Full/TPR experiment modes")
    for field in (
        "tag","objective","ppo_clip","checkpoint","initial_weight_sig",
        "torch","steps","num_siblings","lr","grad_clip","optimizer",
        "same_evaluation_backend","source",
    ):
        if full[field] != tpr[field]:
            raise AssertionError(f"Unmatched Full/TPR sibling controls: {field}")
    if full["steps"] != steps or len(full["events"]) != steps or len(tpr["events"]) != steps:
        raise AssertionError("Full/TPR did not complete requested optimizer steps")
    if full["num_siblings"] < 2 or len(full["source"]["selected_original_rows"]) != full["num_siblings"]:
        raise AssertionError("No proven real sibling reuse")
    if full["objective"] == "ppo":
        if not full["source"]["raw_has_old_logprobs"] or not full["source"]["raw_has_advantages"]:
            raise AssertionError("PPO-labelled results lack actual recorded fields")
    for item in (full,tpr):
        for ev in item["events"]:
            if ev["supervised"] != sum(item["source"]["supervised_per_sibling"]):
                raise AssertionError("Real masked target count changed during training")
            if not math.isfinite(ev["training_loss"]):
                raise AssertionError("Nonfinite step objective")
    for step in ("0",str(steps)):
        if step not in full["checkpoints"] or step not in tpr["checkpoints"]:
            raise AssertionError(f"Missing saved common-oracle step {step}")
    for field in ("bf16","master"):
        initial=_sample_vector(
            full["checkpoints"]["0"]["sampled_weights"],
            tpr["checkpoints"]["0"]["sampled_weights"], field)
        if initial["max_abs"] != 0:
            raise AssertionError(f"Initial Full vs TPR model/optimizer values mismatch: {field}")
    for key, tensor in full["checkpoints"]["0"]["common_eval_logprobs"].items():
        other=tpr["checkpoints"]["0"]["common_eval_logprobs"][key]
        if not torch.equal(tensor, other):
            raise AssertionError(f"Step0 common Full validation differs for sibling {key}")
    first = tpr["events"][0]["tpr_execution"]
    if first.get("root_forward_count") != 1 or first.get("root_backward_count") != 1:
        raise AssertionError("TPR did not really reuse one shared Prefix")
    if first.get("leaf_backward_count") != tpr["num_siblings"]:
        raise AssertionError("TPR did not visit every sibling")
    if first.get("prefix_relay_layers") != 28:
        raise AssertionError("Root gradient relay did not reach every Qwen layer")
    if first.get("audited_prefix_kv_sibling_accumulations") != 28 * tpr["num_siblings"]:
        raise AssertionError("Missing/duplicate sibling dKV accumulation")
    if first.get("audited_prefix_kv_sample_exact") is not True:
        raise AssertionError("Prefix dKV sample-add witness failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--siblings", type=int, default=2)
    p.add_argument("--objective", choices=("nll","ppo"), default="nll")
    args = p.parse_args()
    files = {
        k: args.directory / f"{args.objective}_{k}_siblings{args.siblings}_steps{args.steps}.pt"
        for k in ("full","tpr")
    }
    for f in files.values():
        if not f.is_file():
            p.error(f"Missing saved NPU result: {f}")
    full,tpr = (_load(files["full"]),_load(files["tpr"]))
    _validate(full,tpr,args.steps)
    m=full["source"]
    print(
        f"P0_5 CONTROL status=PASS "
        f"source=ACTUAL_TQ_ORIGINAL_SIBLINGS={m['selected_original_rows']} "
        f"distinct_real_response=True root_P512_sha={m['prefix_sha256']} "
        f"first_divergent_response_offset={m['first_different_response_offset']} "
        f"objective={args.objective} "
        f"has_actual_advantages={m['raw_has_advantages']} "
        f"has_recorded_old_logprobs={m['raw_has_old_logprobs']} "
        "actor_provenance_not_verified=True"
    )
    for step in range(1,args.steps+1):
        a,b=full["events"][step-1],tpr["events"][step-1]
        grads=_sample_vector(a["training_grad_samples"],b["training_grad_samples"],"values")
        forwards=_masked_lps(
            a["response_logprobs"],b["response_logprobs"],
            full["checkpoints"]["0"]["common_eval_masks"],
        )
        print(
            f"P0_5 TRAIN step={step} full_loss={a['training_loss']:.9g} "
            f"tpr_loss={b['training_loss']:.9g} "
            f"loss_gap={b['training_loss']-a['training_loss']:.9g} "
            f"masked_logprobs={forwards} sampled_preclip_grad={grads} "
            f"prefix_kv_sum_verified={b['tpr_execution']['audited_prefix_kv_sample_exact']} "
            f"ratio_outside_clip_frac_full={a['ratio_outside_clip_frac']:.9g} "
            f"ratio_outside_clip_frac_tpr={b['ratio_outside_clip_frac']:.9g} "
            "ratio_status=OUTSIDE_CLIP_RANGE_NOT_POLICY_SURROGATE_BRANCH_FRACTION"
        )
        if str(step) in full["checkpoints"] and str(step) in tpr["checkpoints"]:
            sa=full["checkpoints"][str(step)]
            sb=tpr["checkpoints"][str(step)]
            oracle=_masked_lps(sa["common_eval_logprobs"],sb["common_eval_logprobs"],
                               sa["common_eval_masks"])
            update=_update_vector(full,tpr,step)
            print(
                f"P0_5 SHARED_FULL_ORACLE step={step} "
                f"nll_full={sa['common_eval_nll']:.9g} "
                f"nll_tpr={sb['common_eval_nll']:.9g} "
                f"nll_gap={sb['common_eval_nll']-sa['common_eval_nll']:.9g} "
                f"logprob={oracle} sampled_FP32_master_update={update}"
            )
    print(
        f"P0_5 status=DIAGNOSTIC_ONLY objective={args.objective} "
        "real_siblings=True multi_leaf_kv_relay=True no_synthetic_advantages=True "
        "no_claim_on_policy_PPO_or_reward_convergence"
    )


if __name__ == "__main__":
    main()
