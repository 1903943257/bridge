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



def compare_ppo_threeway(native: dict, cutoff: dict, tpr: dict,
                         segment_owners: dict) -> list[str]:
    """Attribute full Native/TPR logprob drift to native cutoff vs Forest."""
    ids = native["logical_row_offset"]
    if not torch.equal(ids, cutoff["logical_row_offset"]) or not torch.equal(
        ids, tpr["logical_row_offset"]
    ) or not torch.equal(ids, segment_owners["logical_row_offset"]):
        raise ValueError("Native/Cutoff/TPR logical PPO identities differ")
    descriptors = segment_owners["segment_id_start_end"]
    if descriptors.ndim != 2 or descriptors.shape != (len(ids), 3):
        raise ValueError("cutoff owner metadata has invalid [tokens, 3] shape")
    comparisons = (
        ("NATIVE_FULL_TO_NATIVE_CUTOFF", native, cutoff),
        ("NATIVE_CUTOFF_TO_TPR", cutoff, tpr),
        ("NATIVE_FULL_TO_TPR", native, tpr),
    )
    result = []
    for label, a, b in comparisons:
        metrics = compare_ppo_tokens(a, b)
        result.append(
            f"P0 WEAK_TQ CUTOFF_ORACLE {label} "
            + " ".join(f"{key}={value}" for key, value in metrics.items())
        )
    # Map logical refs back to physical owner Segments. This separates
    # errors due to changing the native *total input length* from errors
    # due to the external KV / rectangular attention / segment M.
    full_d = (native["new"].float() - cutoff["new"].float()).abs()
    forest_d = (tpr["new"].float() - cutoff["new"].float()).abs()
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for i, seg in enumerate(descriptors.tolist()):
        grouped.setdefault(tuple(map(int, seg)), []).append(i)
    for (seg_id, start, end), indices in sorted(
        grouped.items(), key=lambda pair: pair[0][1:]
    ):
        idx = torch.tensor(indices, dtype=torch.long)
        a = full_d.index_select(0, idx)
        b = forest_d.index_select(0, idx)
        result.append(
            "P0 WEAK_TQ CUTOFF_SEGMENT "
            f"segment={seg_id}[{start}:{end}] tokens={len(indices)} "
            f"native_full_to_cutoff_max={float(a.max()):.9g} "
            f"native_full_to_cutoff_mean={float(a.mean()):.9g} "
            f"forest_to_cutoff_max={float(b.max()):.9g} "
            f"forest_to_cutoff_mean={float(b.mean()):.9g}"
        )
    worst = torch.argsort(
        (native["new"].float() - tpr["new"].float()).abs(),
        descending=True,
    )[:12]
    for ix in worst.tolist():
        row, offset = ids[ix].tolist()
        seg_id, start, end = descriptors[ix].tolist()
        result.append(
            "P0 WEAK_TQ CUTOFF_WORST "
            f"row={row} response={offset} segment={seg_id}[{start}:{end}] "
            f"full={float(native['new'][ix]):.9g} "
            f"cutoff={float(cutoff['new'][ix]):.9g} "
            f"forest={float(tpr['new'][ix]):.9g}"
        )
    return result


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Compare exact real-TQ PPO clipping")
    cli.add_argument("native")
    cli.add_argument("tpr")
    cli.add_argument(
        "--cutoff", help="Optional native per-physical-Segment PPO logprobs"
    )
    cli.add_argument(
        "--owners", help="cutoff_segment_owners.pt for per-Segment attribution"
    )
    args = cli.parse_args()
    n = torch.load(args.native, map_location="cpu", weights_only=True)
    t = torch.load(args.tpr, map_location="cpu", weights_only=True)
    result = compare_ppo_tokens(n, t)
    print(
        "P0 WEAK_TQ PPO_TOKEN_COMPARISON "
        + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
    if bool(args.cutoff) != bool(args.owners):
        cli.error("--cutoff and --owners must be supplied together")
    if args.cutoff:
        c = torch.load(args.cutoff, map_location="cpu", weights_only=True)
        owners = torch.load(args.owners, map_location="cpu", weights_only=True)
        for row in compare_ppo_threeway(n, c, t, owners):
            print(row, flush=True)
