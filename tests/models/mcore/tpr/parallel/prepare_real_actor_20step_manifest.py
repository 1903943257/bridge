"""CPU preflight: select bounded FULL trajectories from 20 real actor captures.

Never truncate trajectories or fabricate PPO fields. The result is consumed
identically by independent Full and Turn-TPR 20-step offline replay jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

REQUIRED = ("input_ids", "prompts", "responses", "response_mask",
            "loss_mask", "old_log_probs", "advantages", "temperature")
CAPTURE_NAME = re.compile(
    r"actor_update_step(?P<step>\d+)_rank0_capture(?P<ordinal>\d+)_batch(?P<batch>\d+)\.pt$"
)


def _all_capture_steps(directory: Path) -> dict[int, Path]:
    found = {}
    for path in sorted(directory.glob("actor_update_step*_rank0_capture*_batch*.pt")):
        match = CAPTURE_NAME.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if step in found:
            raise RuntimeError(
                f"step {step} has multiple rank0 actor mini-batches; "
                "need explicit per-mini-batch scheduling, refusing to discard one"
            )
        found[step] = path.resolve()
    return found


def prepare(directory: Path, output: Path, *, steps: int = 20,
            max_seq_len: int = 49152, max_rows: int = 3,
            max_total_tokens: int = 130000) -> dict:
    if steps <= 0 or max_seq_len <= 0 or max_rows < 2 or max_total_tokens <= 0:
        raise ValueError("invalid selection limits")
    found = _all_capture_steps(directory)
    missing = [step for step in range(1, steps + 1) if step not in found]
    if missing:
        raise FileNotFoundError(
            f"expected captured actor update for EVERY online step 1..{steps}; "
            f"missing={missing}, present={sorted(found)}. "
            "Do not silently replay fewer steps."
        )
    items = []
    for step in range(1, steps + 1):
        path = found[step]
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("capture_stage") != "actor_update_mini_batch":
            raise AssertionError(f"not a genuine actor mini-batch: {path}")
        if int(raw.get("capture_global_step", -1)) != step:
            raise AssertionError(f"captured step number does not match filename: {path}")
        batch = raw["tensordict"]
        if any(k not in batch for k in REQUIRED):
            raise AssertionError(f"missing original PPO fields at step {step}")
        if len(raw["keys"]) != len(batch["input_ids"]):
            raise AssertionError(f"row identifiers are misaligned at step {step}")
        candidates = []
        for row, (ids, mask, old, adv) in enumerate(zip(
                batch["input_ids"].unbind(0),
                batch["response_mask"].unbind(0),
                batch["old_log_probs"].unbind(0),
                batch["advantages"].unbind(0), strict=True)):
            length = int(ids.numel())
            count = int(mask.bool().sum())
            if mask.numel() != old.numel() or mask.numel() != adv.numel():
                raise AssertionError(f"row {row} PPO alignment broken at step={step}")
            if length > max_seq_len or count == 0:
                continue
            valid_adv = adv[mask.bool()]
            candidates.append((
                length, row, count,
                bool((valid_adv > 0).any().item()),
                bool((valid_adv < 0).any().item()),
                float(valid_adv.abs().max().item()),
            ))
        # Greedy with smallest full-sequence length first; fixed ordering
        # across both independent jobs and no synthetic class-balanced rows.
        candidates.sort()
        selected = []
        total = 0
        for length, row, count, positive, negative, max_abs_adv in candidates:
            if len(selected) >= max_rows or total + length > max_total_tokens:
                continue
            selected.append((length, row, count, positive, negative, max_abs_adv))
            total += length
        if len(selected) < 2:
            raise RuntimeError(
                f"step={step} has only {len(selected)} eligible FULL trajectories "
                f"with cap={max_seq_len}, budget={max_total_tokens}; "
                "no truncation or invented batch allowed."
            )
        indices = sorted(x[1] for x in selected)
        ppo_tokens = sum(x[2] for x in selected)
        item = {
            "step": step, "capture": str(path), "rows": indices,
            "original_rows": len(raw["keys"]),
            "logical_full_tokens": total, "ppo_tokens": ppo_tokens,
            "positive_advantage_present": any(x[3] for x in selected),
            "negative_advantage_present": any(x[4] for x in selected),
            "max_abs_advantage": max(x[5] for x in selected),
            "max_sequence_length": max(x[0] for x in selected),
        }
        print(
            f"TPR_20STEP_MANIFEST step={step} rows={indices} "
            f"total_tokens={total} PPO_tokens={ppo_tokens} "
            f"positive={item['positive_advantage_present']} "
            f"negative={item['negative_advantage_present']}", flush=True
        )
        items.append(item)
        del raw
    result = {
        "protocol": "real_actor_20_distinct_captures_v1",
        "steps": steps, "max_seq_len": max_seq_len,
        "max_rows": max_rows, "max_total_tokens": max_total_tokens,
        "items": items,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=49152)
    p.add_argument("--max-rows", type=int, default=3)
    p.add_argument("--max-total-tokens", type=int, default=130000)
    x = p.parse_args()
    prepare(x.capture_dir, x.output, steps=x.steps,
            max_seq_len=x.max_seq_len, max_rows=x.max_rows,
            max_total_tokens=x.max_total_tokens)
