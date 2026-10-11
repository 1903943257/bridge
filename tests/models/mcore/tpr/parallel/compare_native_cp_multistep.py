"""Compare native MindSpeed CP=1/2/4 deterministic 50-step REAL-TQ NLL trajectories.

Results use per-CP native attention for validation; loss gaps include backend
numerical differences. Sampled weight deltas are NOT whole-model L2.
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


def _metrics(lhs, rhs):
    a, b = lhs.float().flatten().double(), rhs.float().flatten().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("sampled parameter shape or finite check failed")
    delta = b - a
    na, nb = a.norm().item(), b.norm().item()
    return na ** 2, nb ** 2, delta.square().sum().item(), (a * b).sum().item()


def _compare_weights(ref, cand, milestone):
    init_r = ref["checkpoints"]["0"]["sampled_weights"]
    init_c = cand["checkpoints"]["0"]["sampled_weights"]
    current_r = ref["checkpoints"][str(milestone)]["sampled_weights"]
    current_c = cand["checkpoints"][str(milestone)]["sampled_weights"]
    if not (init_r.keys() == init_c.keys() == current_r.keys() == current_c.keys()):
        raise ValueError("parameter name mismatch")
    # Compare UPDATE vectors Δθ_t=θ_t-θ_0, rather than norm of raw pretrained θ.
    sums = [0.0] * 4
    raw_diff_sq = 0.0
    for name in sorted(init_r):
        r0, c0, r, c = init_r[name], init_c[name], current_r[name], current_c[name]
        if (r0["shape"] != c0["shape"] or r0["shape"] != r["shape"]
                or r0["shape"] != c["shape"]):
            raise ValueError(f"parameter shape mismatch: {name}")
        for key in ("indices",):
            if not (torch.equal(r0[key], c0[key]) and torch.equal(r0[key], r[key])
                    and torch.equal(r0[key], c[key])):
                raise ValueError(f"gradient sample indices differ: {name}")
        ref_update = r["master"].float() - r0["master"].float()
        cp_update = c["master"].float() - c0["master"].float()
        pieces = _metrics(ref_update, cp_update)
        sums = [x + y for x, y in zip(sums, pieces, strict=True)]
        raw_diff_sq += float((r["master"].float() - c["master"].float()).double()
                             .square().sum())
    nref, ncp, ndelta, dot = sums
    rel = math.sqrt(ndelta / max(nref, 1e-48))
    cos = dot / max(math.sqrt(nref * ncp), 1e-48)
    print(f"P0_B CP1_VS_CP{cand['cp']} step={milestone} sampled_master_update_rel_l2="
          f"{rel:.9g} sampled_master_update_cosine={cos:.9g} "
          f"sampled_master_weight_abs_l2={math.sqrt(raw_diff_sq):.9g} "
          f"mode=STRIDED_SAMPLED_ONLY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    files = [args.directory / f"native_cp{cp}_train{args.steps}.pt" for cp in (1, 2, 4)]
    if any(not f.exists() for f in files):
        parser.error(f"Missing P0-B experiment file(s): {[str(f) for f in files if not f.exists()]}")
    data = [_load(f) for f in files]
    if [x["cp"] for x in data] != [1, 2, 4]:
        raise ValueError("wrong CP labels")
    ref = data[0]
    fields = ("tag", "model", "checkpoint", "data_sha", "data_path", "stream_sha",
              "schedule", "train_row_ids", "val_row_ids", "train_trajectory_ids",
              "val_trajectory_ids", "steps_requested",
              "lr", "clip_norm", "grad_reduce", "optimizer", "betas", "eps",
              "weight_decay", "seq_len", "response_len", "initial_weights_sha",
              "torch", "gradient_aggregation", "validation_backend")
    for x in data:
        if x["steps_completed"] != args.steps or len(x["events"]) != args.steps:
            raise ValueError(f"CP{x['cp']} incomplete; steps={x['steps_completed']}")
        for j, event in enumerate(x["events"], start=1):
            if event["step"] != j or event["row"] != x["schedule"][j - 1]:
                raise ValueError(f"CP{x['cp']} wrong step / row scheduling")
            for field in ("train_loss", "grad_norm", "estimated_master_update_norm"):
                if not math.isfinite(event[field]):
                    raise ValueError(f"CP{x['cp']} nonfinite {field} at step {j}")
        if x is not ref:
            for field in fields:
                if ref[field] != x[field]:
                    raise ValueError(f"CP{x['cp']} invalid cross-CP comparison: {field} mismatch")
        start = x["checkpoints"]["0"]["validation_loss_mean"]
        final = x["checkpoints"][str(args.steps)]["validation_loss_mean"]
        print(f"P0_B CP{x['cp']} train_windows={len(x['train_row_ids'])} "
              f"val_windows={len(x['val_row_ids'])} "
              f"train_trajectories={x['train_trajectory_ids']} "
              f"val_trajectories={x['val_trajectory_ids']} "
              f"steps={args.steps} train_loss_first={x['events'][0]['train_loss']:.9f} "
              f"train_loss_last={x['events'][-1]['train_loss']:.9f} "
              f"val_initial={start:.9f} val_final={final:.9f} "
              f"val_delta={final-start:.9f} "
              f"grad_reduce={x['grad_reduce']} optimizer={x['optimizer']} "
              f"low_independent_trajectory_count={len(x['train_trajectory_ids'])<64}")
    common = sorted(set.intersection(*(set(x["checkpoints"]) for x in data)), key=int)
    for cand in data[1:]:
        for milestone in common:
            j = int(milestone)
            ra = ref["checkpoints"][milestone]["validation_loss_mean"]
            ca = cand["checkpoints"][milestone]["validation_loss_mean"]
            rb = ref["checkpoints"]["0"]["validation_loss_mean"]
            cb = cand["checkpoints"]["0"]["validation_loss_mean"]
            print(f"P0_B CP1_VS_CP{cand['cp']} step={j} val_loss_gap={ca-ra:.9g} "
                  f"val_improvement_gap={(cb-ca)-(rb-ra):.9g} "
                  "evaluation=OWN_NATIVE_CP_NOT_COMMON_ORACLE")
            if j > 0:
                _compare_weights(ref, cand, j)
    print("P0_B status=DIAGNOSTIC_ONLY fixed_real_tq_nll_"
          "not_ppo_not_generalization_not_megatron_distributed_optimizer")


if __name__ == "__main__":
    main()
