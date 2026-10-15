"""Controlled <=10,240-token Qwen3-4B real-label PPO parity capture.

Inputs are the *already matched* 30K Step6 capture, not rollout. If the
original prompt is too long, retain its last 4,096 tokens and retain the
leading response. That necessarily changes the policy conditioning: labels
are original fixed PPO fields but this is NOT the historical online objective.
All topology arms see identical cropped bytes and original retained labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu

from .prepare_tpr_30k_one_step import (
    _FULL_FIELDS, _PROMPT_FIELDS, _RESPONSE_FIELDS,
)

CAP = 10240
PROMPT_WINDOW = 4096


def _retained_indices(name, item, *, p: int, r: int, start_p: int,
                      start_r: int, keep_p: int, keep_r: int):
    n = int(item.shape[0])
    if name in _PROMPT_FIELDS:
        if n != p:
            raise AssertionError(f"{name}: expected prompt len {p}, got {n}")
        return item[start_p:start_p+keep_p].clone()
    if name in _RESPONSE_FIELDS:
        if n != r:
            raise AssertionError(f"{name}: expected response len {r}, got {n}")
        return item[start_r:start_r+keep_r].clone()
    if name in _FULL_FIELDS:
        if n != p+r:
            raise AssertionError(f"{name}: expected full len {p+r}, got {n}")
        pieces = torch.cat((
            item[start_p:start_p+keep_p],
            item[p+start_r:p+start_r+keep_r],
        ), dim=0)
        if name == "position_ids":
            # New truncated causal context starts at position zero.
            if pieces.ndim != 1:
                raise NotImplementedError("Only 1D captured position_ids supported")
            pieces = torch.arange(len(pieces), dtype=item.dtype, device=item.device)
        return pieces.clone()
    possible = []
    if n == p:
        possible.append("prompt")
    if n == r:
        possible.append("response")
    if n == p+r:
        possible.append("full")
    if len(possible) != 1:
        raise ValueError(
            f"Unknown nested field {name} has ambiguous alignment "
            f"{n} among prompt={p}, response={r}, full={p+r}"
        )
    return _retained_indices(
        {"prompt":"prompts", "response":"responses", "full":"input_ids"}[possible[0]],
        item, p=p, r=r, start_p=start_p, start_r=start_r,
        keep_p=keep_p, keep_r=keep_r,
    )


def derive_10k(source: Path, target: Path, *, cap: int = CAP) -> dict:
    if cap != CAP:
        raise ValueError("This experiment fixes cap=10240")
    original = torch.load(source, map_location="cpu", weights_only=False)
    if original.get("capture_stage") != "actor_update_mini_batch":
        raise ValueError("Expected real actor_update_mini_batch capture")
    data = original["tensordict"]
    keys = tuple(original["keys"])
    if len(keys) != 8:
        raise ValueError("Expected same eight original trajectories")
    required = ("input_ids", "prompts", "responses", "response_mask",
                "old_log_probs", "advantages", "loss_mask")
    if any(key not in data for key in required):
        raise ValueError("Original PPO fields missing")
    starts, audit_rows = [], []
    for row in range(8):
        full, prompt, response = (
            data["input_ids"][row], data["prompts"][row], data["responses"][row],
        )
        p, r = int(prompt.numel()), int(response.numel())
        if not torch.equal(full, torch.cat((prompt, response))):
            raise AssertionError(f"Original trajectory {row} does not match prompt+response")
        if r < 1:
            raise ValueError(f"Row {row}: zero response length")
        keep_p = p if p+1 <= cap else min(p, PROMPT_WINDOW)
        keep_r = min(r, cap-keep_p)
        if keep_r < 1:
            raise ValueError(f"Row {row}: no response capacity")
        mask = data["response_mask"][row].bool()
        if mask.numel() != r:
            raise AssertionError("response_mask length mismatches original response")
        # Select the earliest contiguous response window with supervised tokens.
        hits = torch.nonzero(mask, as_tuple=False).flatten()
        if not hits.numel():
            raise ValueError(f"Row {row}: no supervised PPO response token")
        first_hit = int(hits[0].item())
        start_r = max(0, first_hit-keep_r+1)
        start_p = p-keep_p
        starts.append((p,r,start_p,start_r,keep_p,keep_r))
        audit_rows.append({
            "row": row, "original_length": p+r, "original_prompt_len": p,
            "retained_prompt_start": start_p, "retained_prompt_len": keep_p,
            "retained_response_start": start_r, "retained_response_len": keep_r,
            "new_length": keep_p+keep_r,
            "prompt_left_truncated": start_p > 0,
            "response_window_shifted": start_r > 0,
        })
    cropped = data.clone(recurse=False)
    for name in data.keys():
        value = data[name]
        if isinstance(value, torch.Tensor) and value.is_nested:
            values = []
            for row, (p,r,sp,sr,kp,kr) in enumerate(starts):
                values.append(_retained_indices(
                    name, value[row], p=p, r=r, start_p=sp,
                    start_r=sr, keep_p=kp, keep_r=kr,
                ))
            cropped.set(
                name,
                tu.nested_tensor_from_tensor_list(
                    values, ragged_idx=getattr(value, "_ragged_idx", value.dim()-1),
                ),
            )
        elif isinstance(value, torch.Tensor) and name in (
            _FULL_FIELDS | _PROMPT_FIELDS | _RESPONSE_FIELDS
        ):
            raise TypeError(f"Sequence field {name} unexpectedly non-nested")
    if set(cropped.keys()) != set(data.keys()):
        raise AssertionError("Truncated dataset must retain all PPO fields")
    ppo_tokens = 0
    for row, (_,_,sp,sr,kp,kr) in enumerate(starts):
        a = cropped["input_ids"][row]
        p = cropped["prompts"][row]
        response = cropped["responses"][row]
        if not torch.equal(a, torch.cat((p,response))) or len(a) > cap:
            raise AssertionError(f"Invalid new input_ids alignment row={row}")
        for name in ("old_log_probs","advantages","response_mask","loss_mask"):
            if not torch.equal(
                cropped[name][row], data[name][row][sr:sr+kr]
            ):
                raise AssertionError(f"Original PPO labels modified row={row} field={name}")
        ppo_tokens += int(cropped["response_mask"][row].bool().sum().item())
    if ppo_tokens <= 0:
        raise ValueError("No supervised response tokens in 10K parity window")
    payload = dict(original)
    payload["tensordict"] = cropped
    payload["derived_from_capture"] = str(source.resolve())
    payload["offline_sequence_cap"] = cap
    payload["parity_window_context_changed"] = any(
        x["prompt_left_truncated"] or x["response_window_shifted"] for x in audit_rows
    )
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,target)
    audit = {
        "source_capture": str(source.resolve()),
        "capture": str(target.resolve()),
        "cap": cap,
        "ppo_tokens": ppo_tokens,
        "source_row_count": 8,
        "rows": audit_rows,
        "changes_original_conditioning": payload["parity_window_context_changed"],
        "labels": "Original retained response old_log_probs/advantages/masks unchanged",
        "note": (
            "Offline numerical parity only. Truncating left prompt changes model "
            "conditioning and keeps historical PPO labels; not an on-policy PPO replay."
        ),
    }
    return audit


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("existing_30k_pair_root", type=Path)
    parser.add_argument("new_10k_dir", type=Path)
    parser.add_argument("--step", type=int, default=6)
    args=parser.parse_args()
    if args.new_10k_dir.exists():
        raise FileExistsError(args.new_10k_dir)
    source=(args.existing_30k_pair_root /
            f"step{args.step}_cap30720.pt").resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audit=derive_10k(
        source, args.new_10k_dir / f"step{args.step}_cap10240_parity.pt"
    )
    (args.new_10k_dir / "parity_capture_audit.json").write_text(
        json.dumps(audit,indent=2)
    )
    print(
        f"TPR_10K_PARITY_CAPTURE source={source} "
        f"new={audit['capture']} ppo_tokens={audit['ppo_tokens']} "
        f"conditioning_changed={audit['changes_original_conditioning']}",
        flush=True,
    )


if __name__=="__main__":
    main()
