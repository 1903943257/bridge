"""CPU-only: build 8 original whole trajectories per online step for replay.

The source captures are immutable. A session is identified by (uid, session_id).
Across ALL captures for that online global step choose its first original chain
(when trajectory ordinal is available), else earliest captured row (explicitly
audited as an approximation). Assemble all selected sessions into one
TensorDict-backed capture per step, so Full and TPR can see the SAME eight-way
tree, instead of separately replaying accidental Actor micro-batches.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu

try:
    from .offline_session_selection import first_session_indices, session_rows
except ImportError:
    from offline_session_selection import first_session_indices, session_rows

REQUIRED = (
    "input_ids", "prompts", "responses", "response_mask",
    "loss_mask", "old_log_probs", "advantages", "temperature",
)
def _concat_original_rows(parts):
    """Losslessly combine per-row TensorDicts across Actor captures.

    torch.cat(TensorDict) dispatches to aten.cat for nested jagged tensors,
    unsupported on the pinned PyTorch version. Rebuild each jagged field from
    its ORIGINAL unbound tensors, preserving dtype and exact values.
    """
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    keys = tuple(parts[0].keys())
    if any(set(part.keys()) != set(keys) for part in parts):
        raise ValueError("Actor captures have incompatible TensorDict fields")
    combined = {}
    for key in keys:
        values = [part.get(key) for part in parts]
        first = values[0]
        if isinstance(first, torch.Tensor):
            if any(not isinstance(v, torch.Tensor) for v in values):
                raise TypeError(f"mixed Tensor field for {key}")
            if first.is_nested:
                if any(not v.is_nested for v in values):
                    raise TypeError(f"incompatible nested field for {key}")
                original_rows = [v.unbind(0)[0] for v in values]
                ragged_idx = getattr(first, "_ragged_idx", first.dim() - 1)
                combined[key] = tu.nested_tensor_from_tensor_list(
                    original_rows, ragged_idx=ragged_idx
                )
            else:
                combined[key] = torch.cat(values, dim=0)
        elif isinstance(first, NonTensorStack):
            combined[key] = NonTensorStack.from_list([
                NonTensorData(tu.unwrap_non_tensor_data(v[0]))
                for v in values
            ])
        elif isinstance(first, NonTensorData):
            originals = [tu.unwrap_non_tensor_data(v) for v in values]
            if any(v != originals[0] for v in originals[1:]):
                # A scalar NonTensorData cannot hold per-row differences.
                # Promote only this field to row-wise stack.
                combined[key] = NonTensorStack.from_list([
                    NonTensorData(value) for value in originals
                ])
            else:
                combined[key] = first
        else:
            raise TypeError(f"unsupported TensorDict field {key}: {type(first)}")
    merged = TensorDict(combined, batch_size=[len(parts)])
    for i, part in enumerate(parts):
        for key in keys:
            original = tu.get(part,key)
            actual = tu.get(merged,key)
            if isinstance(original, torch.Tensor):
                if not torch.equal(actual[i],original[0]):
                    raise AssertionError(f"lossless field reconstruction failed: {key}[{i}]")
    return merged


CAPTURE_NAME = re.compile(
    r"actor_update_step(?P<step>\d+)_rank0_capture(?P<ordinal>\d+)_batch(?P<batch>\d+)\.pt$"
)


def _all_capture_steps(directory: Path) -> dict[int, list[Path]]:
    found: dict[int, list[tuple[int,int,Path]]] = {}
    seen_ordinals = set()
    for path in directory.glob("actor_update_step*_rank0_capture*_batch*.pt"):
        match = CAPTURE_NAME.fullmatch(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        ordinal = int(match.group("ordinal"))
        batch_idx = int(match.group("batch"))
        if (ordinal, batch_idx) in seen_ordinals:
            raise RuntimeError(f"duplicate Actor capture ordinal/batch: {path}")
        seen_ordinals.add((ordinal, batch_idx))
        found.setdefault(step, []).append((ordinal, batch_idx, path.resolve()))
    return {
        step: [path for _,_,path in sorted(entries)]
        for step,entries in found.items()
    }


def prepare(directory: Path, output: Path, *, steps: int = 20,
            max_seq_len: int = 49152, expected_sessions: int = 8):
    if steps < 1 or max_seq_len < 1 or expected_sessions < 1:
        raise ValueError("steps, max_seq_len and expected_sessions must be positive")
    if output.exists():
        raise FileExistsError(output)
    found = _all_capture_steps(directory)
    missing = [step for step in range(1, steps+1) if step not in found]
    if missing:
        raise FileNotFoundError(
            f"expected Actor captures at EVERY online step 1..{steps}, "
            f"missing={missing}, present={sorted(found)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged_dir = output.parent / "session_first_batches"
    merged_dir.mkdir(parents=True, exist_ok=True)
    items, audits = [], []

    for global_step in range(1, steps+1):
        paths = found[global_step]
        sources = []
        for path in paths:
            raw = torch.load(path, map_location="cpu", weights_only=False)
            if raw.get("capture_stage") != "actor_update_mini_batch":
                raise AssertionError(f"not an Actor PPO update capture: {path}")
            if int(raw.get("capture_global_step", -1)) != global_step:
                raise AssertionError(f"global step mismatch: {path}")
            batch = raw["tensordict"]
            missing_fields = [key for key in REQUIRED if key not in batch]
            if missing_fields:
                raise AssertionError(f"{path}: missing PPO fields {missing_fields}")
            if len(raw["keys"]) != len(batch["input_ids"]):
                raise AssertionError(f"{path}: keys / TensorDict rows mismatch")
            sources.append(raw)

        keep, audit = first_session_indices([
            (str(path), raw["tensordict"])
            for path, raw in zip(paths, sources)
        ])
        if audit["sessions_seen"] != expected_sessions:
            raise AssertionError(
                f"online step={global_step} has "
                f"{audit['sessions_seen']} distinct (uid,session_id), "
                f"expected exactly {expected_sessions} for a complete N=8 batch"
            )
        if sum(len(row_indices) for row_indices in keep) != expected_sessions:
            raise AssertionError("session dedup did not retain exactly one row per Session")

        selected_parts, provenance, identities = [], [], []
        lengths, ppo_counts, advantages = [], [], []
        for path, raw, row_indices in zip(paths, sources, keep):
            if not row_indices:
                continue
            batch = raw["tensordict"]
            row_sessions, _, _ = session_rows(batch)
            for row in row_indices:
                input_ids = batch["input_ids"][row]
                prompt = batch["prompts"][row]
                response = batch["responses"][row]
                mask = batch["response_mask"][row].bool()
                loss_mask = batch["loss_mask"][row]
                old = batch["old_log_probs"][row]
                adv = batch["advantages"][row]
                size = int(input_ids.numel())
                if size > max_seq_len:
                    raise AssertionError(
                        f"online step={global_step} session={row_sessions[row]} "
                        f"first trajectory length {size} exceeds {max_seq_len}. "
                        "Do not silently replace first trajectory with a shorter sibling."
                    )
                if (not torch.equal(input_ids, torch.cat((prompt, response)))
                        or mask.numel() != response.numel()
                        or old.numel() != response.numel()
                        or adv.numel() != response.numel()
                        or loss_mask.numel() != response.numel()):
                    raise AssertionError(
                        f"online step={global_step} session={row_sessions[row]}: "
                        "original PPO token/label alignment mismatch"
                    )
                selected_parts.append(tu.index_select_tensor_dict(batch, [row]))
                identities.append(row_sessions[row])
                provenance.append({
                    "source_capture": str(path),
                    "source_capture_index": int(raw["capture_index"]),
                    "source_mini_batch_idx": int(raw["capture_mini_batch_idx"]),
                    "source_row": row,
                    "uid": row_sessions[row][0],
                    "session_id": row_sessions[row][1],
                })
                lengths.append(size)
                ppo_counts.append(int(mask.sum().item()))
                advantages.append(adv[mask].float())

        if len(selected_parts) != expected_sessions or len(set(identities)) != expected_sessions:
            raise AssertionError("missing or duplicate selected original Sessions")
        if sum(ppo_counts) == 0:
            raise AssertionError(f"online step={global_step}: no supervised PPO tokens")

        # IMPORTANT: concatenate the ORIGINAL selected TensorDict rows, never
        # reconstruct logits, advantages, masks, temperatures or rewards.
        # A failed merge is a hard error; silently losing TensorDict fields is
        # unacceptable for a real PPO correctness comparison.
        try:
            merged = _concat_original_rows(selected_parts)
        except Exception as exc:
            raise RuntimeError(
                f"step={global_step}: cannot concatenate original captured "
                "TensorDict rows losslessly; inspect non-tensor fields"
            ) from exc
        if merged.batch_size[0] != expected_sessions:
            raise AssertionError("merged Actor batch does not have all eight rows")
        # The raw capture's per-mini-batch '_tpr_<row>' suffix may repeat across
        # source mini-batches, so regenerate only trajectory KEYS from original
        # UIDs and new stable row ordinals. Input tensors remain untouched.
        keys = tuple(f"{uid}_tpr_{row}" for row,(uid,_) in enumerate(identities))
        target = merged_dir / f"actor_update_step{global_step:06d}_session_first8.pt"
        if target.exists():
            raise FileExistsError(target)
        payload = {
            "capture_stage": "actor_update_mini_batch",
            "tensordict": merged,
            "keys": keys,
            "capture_global_step": global_step,
            "capture_index": 0,
            "capture_mini_batch_idx": 0,
            "offline_session_first_merge": True,
            "source_row_provenance": provenance,
        }
        tmp = target.with_suffix(".pt.tmp")
        if tmp.exists():
            raise FileExistsError(tmp)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

        replay_step = len(items)+1
        item = {
            "step": replay_step,  # exactly one replay update per online step
            "source_global_step": global_step,
            "source_capture_index": 0,
            "source_capture_indices": sorted(set(
                p["source_capture_index"] for p in provenance
            )),
            "capture": str(target.resolve()),
            "rows": list(range(expected_sessions)),
            "original_capture_count": len(paths),
            "original_rows": sum(len(raw["keys"]) for raw in sources),
            "session_count": len(identities),
            "source_row_provenance": provenance,
            "logical_full_tokens": sum(lengths),
            "ppo_tokens": sum(ppo_counts),
            "positive_advantage_present": any(bool((adv>0).any()) for adv in advantages),
            "negative_advantage_present": any(bool((adv<0).any()) for adv in advantages),
            "max_abs_advantage": max(
                (float(adv.abs().max()) for adv in advantages if adv.numel()), default=0
            ),
            "max_sequence_length": max(lengths),
            "min_sequence_length": min(lengths),
        }
        items.append(item)
        audit.update({
            "online_global_step": global_step,
            "original_capture_count": len(paths),
            "original_rows": item["original_rows"],
            "chosen_rows": len(identities),
            "chosen_total_tokens": sum(lengths),
            "merged_capture": str(target),
        })
        if audit["capture_first_fallback_sessions"]:
            print(
                f"TPR_SESSION_ORDER_WARNING online_step={global_step}: "
                f"{audit['capture_first_fallback_sessions']} Sessions lack "
                "original TQ trajectory ordinal; deterministic first CAPTURE "
                "encounter is not proof of Gateway first chain",
                flush=True,
            )
        audits.append(audit)
        print(
            f"TPR_20STEP_FULL_N8 step={global_step} sessions={len(identities)} "
            f"capture_files={len(paths)} discarded_sibling_rows="
            f"{audit['discarded_trajectory_rows']} total_tokens={sum(lengths)} "
            f"longest={max(lengths)} ppo_tokens={sum(ppo_counts)} "
            f"merged={target}", flush=True,
        )
        del sources, selected_parts, merged

    result = {
        "protocol": "real_actor_session_first_8merged_v3",
        "online_steps": steps,
        "steps": len(items),
        "expected_sessions": expected_sessions,
        "max_seq_len": max_seq_len,
        "max_rows": None,
        "max_total_tokens": None,
        "session_selection": "one_per_uid_session_id_across_all_captures",
        "historical_ppo_labels_modified": False,
        "items": items,
        "session_audits": audits,
    }
    output.write_text(json.dumps(result, indent=2))
    print(
        f"TPR_20STEP_FULL_N8_READY online_steps={steps} "
        f"replay_steps={len(items)} sessions_per_step={expected_sessions} "
        f"manifest={output}", flush=True,
    )
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("capture_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=49152)
    p.add_argument("--expected-sessions", type=int, default=8)
    args=p.parse_args()
    prepare(args.capture_dir, args.output,
            steps=args.steps, max_seq_len=args.max_seq_len,
            expected_sessions=args.expected_sessions)
