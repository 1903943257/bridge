"""CPU-only: compare true VERL vanilla-PPO d(loss)/d(new_logprob).

Uses original recorded old_log_probs/advantages/response_mask and the Full/TPR
per-token logprobs from the completed single-step snapshot. No NPU or rollout.
This isolates PPO upstream derivatives from model Jacobian/KV backward error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _row_snapshots(root: Path, mode: str):
    logical = {}
    shards = set()
    for rank in range(8):
        snap = torch.load(root / f"{mode}_rank{rank}.pt",
                          map_location="cpu", weights_only=False)
        if snap["tp_rank"] != 0:
            continue
        if snap["cp_rank"] in shards:
            raise AssertionError(f"Duplicate CP rank {snap['cp_rank']}")
        shards.add(snap["cp_rank"])
        for key, value in snap["logprobs"].items():
            pair = tuple(key)
            if pair in logical:
                raise AssertionError(f"duplicate logical token {pair}")
            logical[pair] = float(value)
    if len(shards) != 4:
        raise AssertionError(f"expected four CP shards, got {shards}")
    return logical


def _ppo_loss_and_adjoint(new_logprob, old_logprob, advantage, denom, weights=None):
    """Exactly the vanilla clip/dual-clip shape; torch autograd handles ties.

    Upstream derivative is computed in FP32 from recorded logprob values.
    Numerically exact backend-dtype replay is not claimed.
    """
    x = new_logprob.detach().to(torch.float32).clone().requires_grad_()
    old = old_logprob.to(torch.float32)
    adv = advantage.to(torch.float32)
    ratio = torch.exp(torch.clamp(x - old, -20.0, 20.0))
    l1 = -adv * ratio
    l2 = -adv * torch.clamp(ratio, min=0.8, max=1.2)
    primary = torch.maximum(l1, l2)
    loss = torch.where(adv < 0, torch.minimum(primary, -adv * 3.0), primary)
    if weights is not None:
        loss = loss * weights.to(torch.float32)
    loss = loss.sum() / float(denom)
    upstream = torch.autograd.grad(loss, x)[0].detach()
    clip = torch.logical_or(l2 > l1, (adv < 0) & (primary > -adv * 3.0))
    return loss.detach(), upstream, clip.detach(), ratio.detach()


def _stats(f, t):
    f = f.to(torch.float64)
    t = t.to(torch.float64)
    diff = t - f
    fnorm = torch.linalg.vector_norm(f)
    tnorm = torch.linalg.vector_norm(t)
    return {
        "full_norm": float(fnorm),
        "tpr_norm": float(tnorm),
        "norm_gap": float((tnorm - fnorm).abs() / fnorm.clamp_min(1e-30)),
        "rel_l2": float(torch.linalg.vector_norm(diff) / fnorm.clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(
            f.reshape(1, -1), t.reshape(1, -1), dim=1
        )[0]),
    }


def analyze(root: Path, capture: Path):
    source = torch.load(capture, map_location="cpu", weights_only=False)
    if source.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected original captured PPO actor minibatch")
    full, tpr = _row_snapshots(root, "full"), _row_snapshots(root, "tpr")
    if full.keys() != tpr.keys():
        raise AssertionError("Full/TPR logical token identities disagree")
    batch = source["tensordict"]
    first = torch.load(root / "full_rank0.pt", map_location="cpu", weights_only=False)
    rows = tuple(first["source_rows"])
    if len(batch) < max(rows) + 1:
        raise AssertionError("Captured PPO rows are missing")

    total = len(full)
    if total != first["logical_ppo_tokens"]:
        raise AssertionError("PPO target count mismatches original capture")
    grad_f, grad_t, lp_diffs, all_flips, positions = [], [], [], [], []
    positive_support, negative_support = [], []
    large_errors_zero = 0
    for sample_row, original_row in enumerate(rows):
        mask = batch["response_mask"][original_row].bool()
        ids = mask.nonzero().reshape(-1).tolist()
        old = batch["old_log_probs"][original_row][mask].float().cpu()
        adv = batch["advantages"][original_row][mask].float().cpu()
        f_lp = torch.tensor([full[(sample_row, i)] for i in ids], dtype=torch.float32)
        t_lp = torch.tensor([tpr[(sample_row, i)] for i in ids], dtype=torch.float32)
        weights = (batch["rollout_is_weights"][original_row][mask].float().cpu()
                   if "rollout_is_weights" in batch else None)
        f_loss, f_grad, f_clip, f_ratio = _ppo_loss_and_adjoint(
            f_lp, old, adv, total, weights
        )
        t_loss, t_grad, t_clip, t_ratio = _ppo_loss_and_adjoint(
            t_lp, old, adv, total, weights
        )
        delta = (f_grad - t_grad).abs()
        flip = f_clip != t_clip
        lp_diff = (f_lp - t_lp).abs()
        large_errors_zero += int(((lp_diff >= 0.5) & (f_grad.abs() < 1e-8)
                                   & (t_grad.abs() < 1e-8)).sum())
        grad_f.append(f_grad)
        grad_t.append(t_grad)
        lp_diffs.append(lp_diff)
        all_flips.append(flip)
        positive_support.extend([float(x) for x in f_grad[adv > 0]])
        negative_support.extend([float(x) for x in f_grad[adv < 0]])
        positions.extend([(sample_row, original_row, i,
                           float(lp_diff[j]), float(delta[j]),
                           float(f_grad[j]), float(t_grad[j]),
                           float(f_ratio[j]), float(t_ratio[j]),
                           bool(flip[j]))
                          for j, i in enumerate(ids)])
        print(
            f"TPR_PPO_ADJOINT row={sample_row} original_row={original_row} "
            f"tokens={len(ids)} loss_full={float(f_loss):.9g} "
            f"loss_tpr={float(t_loss):.9g} "
            f"clip_flips={int(flip.sum())} "
            f"effective_nonzero_full={int((f_grad.abs()>1e-12).sum())} "
            f"effective_nonzero_tpr={int((t_grad.abs()>1e-12).sum())} "
            f"grad_error={_stats(f_grad,t_grad)}", flush=True
        )
    f = torch.cat(grad_f)
    t = torch.cat(grad_t)
    all_flip = torch.cat(all_flips)
    lp_err = torch.cat(lp_diffs)
    print(
        f"TPR_PPO_ADJOINT_TOTAL tokens={total} clipping_branch_flips={int(all_flip.sum())} "
        f"lp_max_abs={float(lp_err.max()):.6g} "
        f"num_large_lp_errors_with_near_zero_adjoint={large_errors_zero} "
        f"effective_upstream={_stats(f,t)}", flush=True
    )
    print("TPR_PPO_ADJOINT_TOP_DELTA")
    for p in sorted(positions, key=lambda x: x[4], reverse=True)[:25]:
        print(
            f"row={p[0]} original={p[1]} offset={p[2]} "
            f"lp_abs_error={p[3]:.6g} upstream_abs_error={p[4]:.6g} "
            f"up_full={p[5]:.6g} up_tpr={p[6]:.6g} "
            f"ratio_full={p[7]:.6g} ratio_tpr={p[8]:.6g} flip={p[9]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("actor_capture", type=Path)
    args = parser.parse_args()
    analyze(args.snapshot_dir, args.actor_capture)
