"""Diagnostic acceptance metrics for real Qwen3-1.7B Native vs TPR PPO.

Pure CPU/torch functions. They do NOT change optimizer/model behavior or
replace the production VERL PPO loss. The AdamW check is a sampled first-step
probe with fresh zero moments, not a full Megatron optimizer integration.
"""
from __future__ import annotations

import torch


def report_ppo_clip_agreement(native_new, tpr_new, old, advantages, *, clip_ratio=0.2, label="native_vs_tpr"):
    """Compare actual PPO objectives and advantage-aware clipped branches.

    All inputs refer to the SAME ordered, valid response tokens. The old
    policy is held fixed between Native and TPR. Counts exclude zero-advantage
    tokens because their clipping branch does not influence policy gradient.
    """
    tensors = [v.detach().float().cpu().flatten() for v in (
        native_new, tpr_new, old, advantages
    )]
    native, tpr, old_lp, adv = tensors
    if not all(v.shape == native.shape for v in tensors) or native.numel() == 0:
        raise AssertionError("PPO acceptance needs nonempty, equally sized vectors")
    if not all(bool(torch.isfinite(v).all()) for v in tensors):
        raise AssertionError("nonfinite PPO acceptance inputs")
    active = adv != 0
    if not bool(active.any()):
        raise AssertionError("all PPO advantages are zero")
    ratio_native = torch.exp(native - old_lp)
    ratio_tpr = torch.exp(tpr - old_lp)
    lo, hi = 1.0 - clip_ratio, 1.0 + clip_ratio

    def objective(ratio):
        unclipped = ratio * adv
        clipped = ratio.clamp(lo, hi) * adv
        # Effective PPO policy branch: upper for positive advantages,
        # lower for negative advantages. Mere outside-window is NOT equivalent.
        clipped_branch = active & torch.where(
            adv > 0, ratio > hi, ratio < lo
        )
        return -torch.minimum(unclipped, clipped), clipped_branch

    loss_native, branch_native = objective(ratio_native)
    loss_tpr, branch_tpr = objective(ratio_tpr)
    mismatch = active & (branch_native != branch_tpr)
    print(
        "QWEN17 PPO ACCEPTANCE CLIP "
        f"label={label} "
        f"valid={int(active.sum())} "
        f"native_clipped={int(branch_native.sum())} "
        f"tpr_clipped={int(branch_tpr.sum())} "
        f"branch_disagree={int(mismatch.sum())} "
        f"ratio_max_abs={float((ratio_tpr-ratio_native).abs().max()):.9g} "
        f"native_policy_loss={float(loss_native[active].mean()):.9g} "
        f"tpr_policy_loss={float(loss_tpr[active].mean()):.9g} "
        f"loss_delta={float((loss_tpr[active]-loss_native[active]).mean()):.9g}",
        flush=True,
    )
    return {
        "branch_disagree": int(mismatch.sum()),
        "valid": int(active.sum()),
        "native_policy_loss": float(loss_native[active].mean()),
        "tpr_policy_loss": float(loss_tpr[active].mean()),
    }


def report_sampled_fresh_adamw(native_weights, tpr_weights,
                               native_grads, tpr_grads, *,
                               lr=1e-5, weight_decay=0.01, eps=1e-8):
    """Compare FIRST AdamW steps on sampled parameter entries, CPU FP32.

    Fresh zero-moment state: m_hat = g and v_hat = g**2 for step 1. No
    optimizer state, global gradient clipping, sharding or real step updates
    are implied. The full gradient comparison is still the primary gate.
    """
    if native_weights.keys() != tpr_weights.keys():
        raise AssertionError("sampled AdamW weight names mismatch")
    sum_diff2 = sum_ref2 = 0.0
    step_diff2 = step_ref2 = step_actual2 = step_dot = 0.0
    grad_sign_flips = grad_common_nonzero = 0
    maximum = 0.0
    count = 0
    for name, before in native_weights.items():
        other = tpr_weights[name]
        if not torch.equal(before, other):
            raise AssertionError(f"initial weight mismatch for {name}")
        n = before.numel()
        gn = native_grads[name].reshape(-1)[:n].float()
        gt = tpr_grads[name].reshape(-1)[:n].float()
        w = before.float()
        # Bias-corrected AdamW with initially zero moments, no clipping.
        step_n = -lr * (weight_decay*w + gn/(gn.abs()+eps))
        step_t = -lr * (weight_decay*w + gt/(gt.abs()+eps))
        wn = w + step_n
        wt = w + step_t
        diff = wt-wn
        step_diff = step_t-step_n
        sum_diff2 += float(diff.square().sum())
        sum_ref2 += float(wn.square().sum())
        step_diff2 += float(step_diff.square().sum())
        step_ref2 += float(step_n.square().sum())
        step_actual2 += float(step_t.square().sum())
        step_dot += float((step_n*step_t).sum())
        common_nonzero = (gn != 0) & (gt != 0)
        grad_common_nonzero += int(common_nonzero.sum())
        grad_sign_flips += int(((gn*gt < 0) & common_nonzero).sum())
        maximum = max(maximum, float(diff.abs().max()))
        count += n
    rel_l2 = (sum_diff2/max(sum_ref2, 1e-24)) ** 0.5
    step_rel_l2 = (step_diff2/max(step_ref2, 1e-24)) ** 0.5
    step_cosine = step_dot / max((step_ref2*step_actual2)**0.5, 1e-24)
    print(
        "QWEN17 PPO ACCEPTANCE ADAMW_SAMPLED_FIRST_STEP "
        f"entries={count} lr={lr} weight_decay={weight_decay} "
        f"max_abs={maximum:.9g} rel_l2={rel_l2:.9g} "
        f"update_rel_l2={step_rel_l2:.9g} "
        f"update_cosine={step_cosine:.9g} "
        f"gradient_sign_flips={grad_sign_flips}/{grad_common_nonzero} "
        "fresh_zero_moments=True real_optimizer_step=False",
        flush=True,
    )
    return maximum, rel_l2
