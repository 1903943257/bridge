"""Compare actual Full and TPR P512/S512 50-step real-TQ optimizer training.

Both modes are evaluated with the SAME Full-M1024 backend on identical
held-out real trajectories. Parameter metrics are fixed strided SAMPLE ONLY.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.torch_version import TorchVersion


def _load(path):
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _metrics(left, right):
    a = left.detach().float().reshape(-1).double()
    b = right.detach().float().reshape(-1).double()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError(f"Nonfinite or mismatched tensor: {a.shape}, {b.shape}")
    diff = b - a
    return {
        "count": int(a.numel()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "p95_abs": float(torch.quantile(diff.abs().float(), 0.95).item()),
        "rel_l2": float((diff.norm() / a.norm().clamp_min(1e-24)).item()),
        "cosine": float(
            torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-24)
        ),
    }


def _verify(full, tpr, steps):
    if (full["mode"], tpr["mode"]) != ("full", "tpr"):
        raise AssertionError("Expected full_trainN.pt and tpr_trainN.pt")
    common_fields = (
        "tag", "steps_requested", "checkpoint", "model", "dtype", "torch",
        "data_path", "data_sha", "stream_sha", "schedule", "train_row_ids",
        "val_row_ids", "train_trajectory_ids", "val_trajectory_ids",
        "skipped_rows", "initial_sampled_weight_sha", "objective",
        "seq_len", "prefix_len", "suffix_len", "lr", "clip_norm",
        "betas", "eps", "weight_decay", "optimizer", "gradient_reduce",
        "eval_backend",
    )
    for key in common_fields:
        if full[key] != tpr[key]:
            raise AssertionError(f"Full and TPR have different comparison metadata: {key}")
    if full["steps_completed"] != steps or tpr["steps_completed"] != steps:
        raise AssertionError("A training run did not reach the requested step count")
    for item in (full, tpr):
        if len(item["events"]) != steps:
            raise AssertionError(f"{item['mode']} incomplete event records")
        for i, e in enumerate(item["events"]):
            if e["step"] != i + 1 or e["row"] != item["schedule"][i]:
                raise AssertionError(f"{item['mode']} consumed wrong real-TQ row")
            if e["supervised"] <= 0 or not (
                math.isfinite(e["loss"]) and math.isfinite(e["preclip_grad_norm"])
            ):
                raise AssertionError(f"{item['mode']} nonfinite or empty step={i + 1}")
        if set(item["train_trajectory_ids"]) & set(item["val_trajectory_ids"]):
            raise AssertionError(f"{item['mode']} train/val trajectory leak")
        for step in (0, steps):
            if str(step) not in item["checkpoints"]:
                raise AssertionError(f"{item['mode']} missing snapshot step {step}")
    a, b = full["checkpoints"]["0"]["sampled_weights"], tpr["checkpoints"]["0"]["sampled_weights"]
    if set(a) != set(b):
        raise AssertionError("Initial parameter name mismatch")
    for name in a:
        if (not torch.equal(a[name]["indices"], b[name]["indices"])
                or a[name]["shape"] != b[name]["shape"]
                or not torch.equal(a[name]["master"], b[name]["master"])
                or not torch.equal(a[name]["bf16"], b[name]["bf16"])):
            raise AssertionError(f"Initial optimizer/model weight mismatch: {name}")
    for a0, b0 in zip(
        full["checkpoints"]["0"]["validation_by_row"],
        tpr["checkpoints"]["0"]["validation_by_row"],
        strict=True,
    ):
        if a0["row"] != b0["row"] or not torch.equal(a0["response_mask"], b0["response_mask"]):
            raise AssertionError("Initial validation rows/masks are not identical")
        if not torch.equal(a0["response_logprobs"], b0["response_logprobs"]):
            raise AssertionError(
                "Step0 common-Full evaluation differs despite identical weights: "
                f"row={a0['row']} — investigate nondeterminism/backend before "
                "comparing learned-parameter drift"
            )
    print(
        "P0_4 CONTROL status=PASS identical_initial_weights=True "
        f"train_windows={len(full['train_row_ids'])} "
        f"train_trajectories={full['train_trajectory_ids']} "
        f"heldout_trajectories={full['val_trajectory_ids']} "
        "real_response_mask=True cp=1 common_evaluation=FULL_M1024",
    )


def _sample_weight_updates(full, tpr, step):
    base_f = full["checkpoints"]["0"]["sampled_weights"]
    base_t = tpr["checkpoints"]["0"]["sampled_weights"]
    now_f = full["checkpoints"][str(step)]["sampled_weights"]
    now_t = tpr["checkpoints"][str(step)]["sampled_weights"]
    if not (base_f.keys() == base_t.keys() == now_f.keys() == now_t.keys()):
        raise AssertionError("Full/TPR snapshot parameter name mismatch")
    ref_sq = cand_sq = diff_sq = dot = 0.0
    param_diff_sq = 0.0
    for name in base_f:
        f0, t0, f, t = (x[name] for x in (base_f, base_t, now_f, now_t))
        if not all(torch.equal(f0["indices"], entry["indices"])
                   for entry in (t0, f, t)):
            raise AssertionError(f"Different sampled offsets at {name}")
        if f0["shape"] != f["shape"] or f0["shape"] != t["shape"]:
            raise AssertionError(f"Parameter shape mismatch at {name}")
        lhs = (f["master"].double() - f0["master"].double())
        rhs = (t["master"].double() - t0["master"].double())
        delta = rhs - lhs
        ref_sq += float(lhs.square().sum())
        cand_sq += float(rhs.square().sum())
        diff_sq += float(delta.square().sum())
        dot += float((lhs * rhs).sum())
        param_diff_sq += float((t["master"].double() - f["master"].double()).square().sum())
    return {
        "sampled_master_update_rel_l2": math.sqrt(diff_sq / max(ref_sq, 1e-48)),
        "sampled_master_update_cosine": dot / max(math.sqrt(ref_sq * cand_sq), 1e-48),
        "sampled_master_weight_diff_l2": math.sqrt(param_diff_sq),
        "sampled_master_full_update_norm": math.sqrt(ref_sq),
        "sampled_master_tpr_update_norm": math.sqrt(cand_sq),
        "mode": "STRIDED_SAMPLED_ONLY",
    }


def _common_oracle_logprobs(full_snap, tpr_snap):
    left, right = full_snap["validation_by_row"], tpr_snap["validation_by_row"]
    if len(left) != len(right):
        raise AssertionError("Heldout evaluation row count differs")
    lf, lt = [], []
    for a, b in zip(left, right, strict=True):
        if (a["row"] != b["row"] or a["supervised"] != b["supervised"]
                or not torch.equal(a["response_mask"], b["response_mask"])):
            raise AssertionError("Mismatched real masked heldout row")
        mask = a["response_mask"].bool()
        if int(mask.sum()) != a["supervised"]:
            raise AssertionError("Incorrect real token ownership in common eval")
        lf.append(a["response_logprobs"][mask])
        lt.append(b["response_logprobs"][mask])
    return _metrics(torch.cat(lf), torch.cat(lt))


def _sampled_grads_at_step(full, tpr, step):
    # Step1 uses identical pretrained weights. Later steps also capture
    # parameter/optimizer trajectory differences, not just kernel drift.
    a = full["preclip_grad_samples_by_step"].get(str(step))
    b = tpr["preclip_grad_samples_by_step"].get(str(step))
    if a is None or b is None:
        return None
    if a.keys() != b.keys():
        raise AssertionError("First-step parameter gradient name mismatch")
    da = db = ds = dot = 0.0
    for name in a:
        l, r = a[name], b[name]
        if l["shape"] != r["shape"] or not torch.equal(l["indices"], r["indices"]):
            raise AssertionError(f"Gradient sample layout mismatch: {name}")
        x, y = l["values"].double(), r["values"].double()
        da += float(x.square().sum())
        db += float(y.square().sum())
        ds += float((y-x).square().sum())
        dot += float((x*y).sum())
    return {
        "sampled_preclip_grad_rel_l2": math.sqrt(ds / max(da, 1e-48)),
        "sampled_preclip_grad_cosine": dot / max(math.sqrt(da*db), 1e-48),
        "mode": "STRIDED_SAMPLED_ONLY",
        "parameter_count": len(a),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    files = [args.directory / f"{mode}_train{args.steps}.pt" for mode in ("full", "tpr")]
    if any(not file.is_file() for file in files):
        parser.error(f"Missing completed Full/TPR evidence: {files}")
    full, tpr = map(_load, files)
    _verify(full, tpr, args.steps)
    milestones = sorted(
        {int(k) for k in full["checkpoints"] if k.isdigit()}
        & {int(k) for k in tpr["checkpoints"] if k.isdigit()}
    )
    for step in milestones:
        if step > 0:
            grad_comparison = _sampled_grads_at_step(full, tpr, step)
            if grad_comparison is None:
                raise AssertionError(f"Missing preclip gradient witnesses step={step}")
            print(f"P0_4 STEP={step} PRECLIP_GRAD {grad_comparison}")
        snap_f = full["checkpoints"][str(step)]
        snap_t = tpr["checkpoints"][str(step)]
        v_f, v_t = snap_f["validation_loss_mean"], snap_t["validation_loss_mean"]
        p_f, p_t = snap_f["train_probe_loss"], snap_t["train_probe_loss"]
        val_metrics = _common_oracle_logprobs(snap_f, snap_t)
        print(
            f"P0_4 STEP={step} COMMON_FULL_VALIDATION "
            f"full={v_f:.9f} tpr={v_t:.9f} gap={v_t-v_f:.9g} "
            f"full_improvement={full['checkpoints']['0']['validation_loss_mean']-v_f:.9g} "
            f"tpr_improvement={tpr['checkpoints']['0']['validation_loss_mean']-v_t:.9g} "
            f"train_probe_full={p_f:.9f} train_probe_tpr={p_t:.9f} "
            f"probe_gap={p_t-p_f:.9g}"
        )
        print(f"P0_4 STEP={step} HELDOUT_RESPONSE_LOGPROB {val_metrics}")
        if step:
            print(f"P0_4 STEP={step} MASTER_UPDATE {_sample_weight_updates(full, tpr, step)}")
    for j in (0, 1, 4, 9, 19, 29, 39, 49):
        if j >= args.steps:
            continue
        a, b = full["events"][j], tpr["events"][j]
        print(
            f"P0_4 TRAIN_STEP={j+1} row={a['row']} "
            f"nll_full={a['loss']:.9f} nll_tpr={b['loss']:.9f} "
            f"nll_gap={b['loss']-a['loss']:.9g} "
            f"grad_norm_full={a['preclip_grad_norm']:.9g} "
            f"grad_norm_tpr={b['preclip_grad_norm']:.9g}"
        )
    print(
        f"P0_4 status=DIAGNOSTIC_ONLY steps={args.steps} "
        "true_optimizer_updates=True common_oracle=True "
        "not_PPO_not_multi_sibling_not_production_distributed_optimizer"
    )


if __name__ == "__main__":
    main()
