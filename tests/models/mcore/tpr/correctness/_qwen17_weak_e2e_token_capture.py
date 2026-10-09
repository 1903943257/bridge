"""Optional real-token PPO comparison for the weak Qwen TQ train-step.

The data is diagnostic (cropped real TQ, recomputed old policy, possibly
synthetic signed advantages), and this tool does not change the loss.
"""
from __future__ import annotations

from pathlib import Path

import torch


def _rows(batch, name):
    value = batch[name]
    return list(value) if isinstance(value, (tuple, list)) else list(value.unbind())


def save_ppo_tokens(batch, new_by_token: dict, path: str | Path) -> None:
    """Save aligned per-logical-token new/old logprobs and advantages."""
    masks = _rows(batch, "response_mask")
    old = _rows(batch, "old_log_probs")
    advantages = _rows(batch, "advantages")
    identities = tuple(
        (r, int(offset))
        for r, mask in enumerate(masks)
        for offset in torch.nonzero(mask.bool(), as_tuple=False).flatten().tolist()
    )
    if not identities or set(new_by_token) != set(identities):
        missing = set(identities) - set(new_by_token)
        extra = set(new_by_token) - set(identities)
        raise ValueError(
            f"token capture key mismatch: expected={len(identities)} "
            f"actual={len(new_by_token)} missing={len(missing)} extra={len(extra)}"
        )
    payload = {
        "logical_row_offset": torch.tensor(identities, dtype=torch.int64),
        "new": torch.tensor(
            [float(new_by_token[key]) for key in identities], dtype=torch.float32
        ),
        "old": torch.tensor(
            [float(old[r][i]) for r, i in identities], dtype=torch.float32
        ),
        "advantage": torch.tensor(
            [float(advantages[r][i]) for r, i in identities], dtype=torch.float32
        ),
    }
    if any(not bool(torch.isfinite(v.float()).all()) for v in payload.values()):
        raise ValueError("PPO token capture includes nonfinite values")
    file = Path(path)
    if file.exists():
        raise FileExistsError(f"refusing to overwrite prior PPO token capture: {file}")
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, file)
    print(f"P0 WEAK_TQ TOKEN_CAPTURE path={file} tokens={len(identities)}", flush=True)


def compare_ppo_tokens(native: dict, tpr: dict, *, clip_ratio: float = 0.2):
    """Return diagnostic statistics of full logical 512-token PPO decisions."""
    if not torch.equal(native["logical_row_offset"], tpr["logical_row_offset"]):
        raise ValueError("Native and TPR logical PPO token identities differ")
    old, other_old = native["old"].float(), tpr["old"].float()
    adv, other_adv = native["advantage"].float(), tpr["advantage"].float()
    # Different processes may recompute the same frozen BF16 old policy
    # with a few floating-point ulps of noise. Flag any *material* mismatch,
    # but do not reject harmless sub-1e-5 differences as different PPO data.
    if not torch.allclose(old, other_old, atol=1e-5, rtol=0) or not torch.equal(adv, other_adv):
        raise ValueError("Native/TPR old policy or advantages differ")
    a, b = native["new"].float(), tpr["new"].float()
    if not a.numel() or a.shape != b.shape:
        raise ValueError("nonmatching empty PPO logits")
    if any(not bool(torch.isfinite(v).all()) for v in (a, b, old, adv)):
        raise ValueError("nonfinite PPO log-probs or advantages")
    ra, rb = torch.exp(a - old), torch.exp(b - old)
    if not bool(torch.isfinite(ra).all()) or not bool(torch.isfinite(rb).all()):
        raise ValueError("nonfinite PPO importance ratios")
    active = adv != 0
    lo, hi = 1.0 - clip_ratio, 1.0 + clip_ratio
    clipped_a = active & torch.where(adv > 0, ra > hi, ra < lo)
    clipped_b = active & torch.where(adv > 0, rb > hi, rb < lo)
    differ = active & (clipped_a != clipped_b)
    loss_a = -torch.minimum(ra * adv, ra.clamp(lo, hi) * adv)
    loss_b = -torch.minimum(rb * adv, rb.clamp(lo, hi) * adv)
    diff = (b - a).abs()
    result = {
        "logical_tokens": int(a.numel()),
        "recomputed_old_max_abs": float((old - other_old).abs().max()),
        "clip_native": int(clipped_a.sum()),
        "clip_tpr": int(clipped_b.sum()),
        "clip_disagreement": int(differ.sum()),
        "logprob_max_abs": float(diff.max()),
        "logprob_mean_abs": float(diff.mean()),
        "ratio_max_abs": float((ra - rb).abs().max()),
        "ppo_loss_delta": float((loss_b - loss_a).mean()),
        "native_self_repeat_max_abs": float((a - old).abs().max()),
        "worst_row_offset": tuple(
            native["logical_row_offset"][int(diff.argmax())].tolist()
        ),
    }
    return result


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Compare exact real-TQ PPO clipping")
    cli.add_argument("native")
    cli.add_argument("tpr")
    args = cli.parse_args()
    n = torch.load(args.native, map_location="cpu", weights_only=True)
    t = torch.load(args.tpr, map_location="cpu", weights_only=True)
    result = compare_ppo_tokens(n, t)
    print(
        "P0 WEAK_TQ PPO_TOKEN_COMPARISON "
        + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
