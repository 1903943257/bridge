#!/usr/bin/env python3
"""CPU-only original length/reward audit + a *separate* 8-row 45Ki Full capacity fixture.

Original capture/manifest are NEVER written. Truncation changes the dataset:
the resulting fixture must NOT be used for original-PPO parity, rewards, or
Full-vs-TPR speedup claims without creating the SAME trimmed input for TPR.

python tests/models/mcore/tpr/parallel/probe_full_45k.py audit MANIFEST
python tests/models/mcore/tpr/parallel/probe_full_45k.py prepare MANIFEST NEW_DIR --step 1
python tests/models/mcore/tpr/parallel/probe_full_45k.py prepare-all MANIFEST NEW_ROOT --cap 40960
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from verl.utils import tensordict_utils as tu

TOTAL = {"input_ids", "position_ids", "attention_mask"}
RESPONSE = {
    "responses", "response_mask", "loss_mask", "old_log_probs", "advantages",
    "ref_log_prob", "rollout_is_weights", "rm_scores", "log_probs",
    "rollout_log_probs", "token_level_scores", "token_level_rewards",
    "values", "returns",
}


def load_item(manifest, step):
    if manifest.get("protocol") != "real_actor_session_first_8merged_v3":
        raise ValueError("Require original v3 8-Session manifest")
    if manifest.get("expected_sessions") != 8:
        raise ValueError("Missing original eight-Session contract")
    if not 1 <= step <= len(manifest["items"]):
        raise ValueError("Invalid online step")
    item = manifest["items"][step - 1]
    if item["rows"] != list(range(8)):
        raise ValueError("Must preserve all eight original Session rows")
    raw = torch.load(item["capture"], map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch" or len(raw["keys"]) != 8:
        raise ValueError("Not an eight-row original Actor capture")
    return item, raw, raw["tensordict"]


def scalar_value(v):
    v = tu.unwrap_non_tensor_data(v)
    if isinstance(v, torch.Tensor):
        return float(v.float().sum().item())
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def reward_for_row(batch, row):
    # Do NOT confuse positive GRPO advantage with task correctness.
    for key in ("reward_metrics", "reward_info"):
        if key not in batch:
            continue
        value = tu.unwrap_non_tensor_data(batch.get(key)[row])
        if isinstance(value, dict):
            value = value.get("reward_metrics", value)
            if isinstance(value, dict) and "acc" in value:
                v = scalar_value(value["acc"])
                if v is not None:
                    return ("SUCCESS" if v >= 1.0 else "FAIL", key + ".acc", v)
    for key in ("acc", "accuracy"):
        if key in batch:
            v = scalar_value(batch.get(key)[row])
            if v is not None:
                return ("SUCCESS" if v >= 1.0 else "FAIL", key, v)
    # Reward-score proxy only. It is not proof of SWE-bench success when
    # shaped rewards or multiple trajectories per Session are possible.
    for key in ("reward_score", "rm_scores"):
        if key in batch:
            v = scalar_value(batch.get(key)[row])
            if v is not None:
                return ("POSITIVE_REWARD_PROXY" if v > 0 else "NONPOSITIVE_REWARD_PROXY",
                        key, v)
    return "UNKNOWN", "no_success_metric_in_actor_capture", None


def audit(manifest, cap):
    total = long_count = proven_success = proven_fail = proxy_pos = unknown = 0
    for step in range(1, len(manifest["items"]) + 1):
        item, _, batch = load_item(manifest, step)
        for row in range(8):
            total += 1
            ids = batch["input_ids"][row]
            p = batch["prompts"][row]
            r = batch["responses"][row]
            if ids.numel() != p.numel() + r.numel():
                raise AssertionError("Original prompt/response alignment mismatch")
            length = int(ids.numel())
            if length <= cap:
                continue
            long_count += 1
            cls, field, val = reward_for_row(batch, row)
            if cls == "SUCCESS":
                proven_success += 1
            elif cls == "FAIL":
                proven_fail += 1
            elif cls == "POSITIVE_REWARD_PROXY":
                proxy_pos += 1
            else:
                unknown += 1
            adv = batch["advantages"][row]
            mask = batch["response_mask"][row].bool()
            if adv.numel() != mask.numel():
                raise AssertionError("Original PPO advantages/mask misaligned")
            print("FULL45K_LONG " + json.dumps({
                "step": step, "row": row, "tokens": length,
                "prompt_tokens": int(p.numel()), "response_tokens": int(r.numel()),
                "positive_advantage": bool((adv[mask].float() > 0).any().item()),
                "reward_class": cls, "reward_source": field, "reward_value": val,
                "session_id": item["source_row_provenance"][row].get("session_id"),
            }, ensure_ascii=False), flush=True)
    print("FULL45K_AUDIT " + json.dumps({
        "cap": cap, "total_rows": total, "over_cap": long_count,
        "proven_correct_over_cap": proven_success,
        "proven_incorrect_over_cap": proven_fail,
        "positive_reward_proxy_over_cap": proxy_pos,
        "no_success_evidence_over_cap": unknown,
        "definitive_yes_no_available": (proven_success + proven_fail == long_count),
    }), flush=True)
    if proven_success + proven_fail != long_count:
        print("FULL45K_WARNING original Actor capture lacks definitive acc on some "
              "rows: inspect original rollout reward_metrics.acc before claiming "
              "no successful trajectories exceed 45Ki", flush=True)


def slice_row(x, key, old_len, old_resp, new_len, new_resp):
    if x.ndim == 0:
        return x
    n = x.shape[0]
    if key in TOTAL:
        if n != old_len:
            raise AssertionError(f"{key} expected total-length {old_len}, got {n}")
        return x[:new_len]
    if key in RESPONSE:
        if n != old_resp:
            raise AssertionError(f"{key} expected response-length {old_resp}, got {n}")
        return x[:new_resp]
    if n == old_len:
        return x[:new_len]
    if n == old_resp:
        return x[:new_resp]
    return x


def trim_eight(batch, cap):
    old_len = [int(batch["input_ids"][i].numel()) for i in range(8)]
    prompt_len = [int(batch["prompts"][i].numel()) for i in range(8)]
    old_resp = [int(batch["responses"][i].numel()) for i in range(8)]
    new_len = [min(cap, n) for n in old_len]
    new_resp = [new_len[i] - prompt_len[i] for i in range(8)]
    if any(n <= 1 for n in new_resp):
        raise ValueError("Cannot truncate away all supervised response tokens")
    for key in list(batch.keys()):
        col = batch.get(key)
        if not isinstance(col, torch.Tensor):
            continue
        if col.is_nested:
            rows = [
                slice_row(x, key, old_len[i], old_resp[i], new_len[i], new_resp[i])
                .contiguous()
                for i, x in enumerate(col.unbind(0))
            ]
            ragged = getattr(col, "_ragged_idx", col.dim() - 1)
            batch.set(key, tu.nested_tensor_from_tensor_list(rows, ragged_idx=ragged))
        elif key in TOTAL | RESPONSE and col.ndim >= 2:
            before = old_len if key in TOTAL else old_resp
            after = new_len if key in TOTAL else new_resp
            if len(set(before)) == len(set(after)) == 1 and col.shape[1] == before[0]:
                batch.set(key, col[:, :after[0]].contiguous())
            elif any(a != b for a, b in zip(before, after)):
                raise ValueError(f"Cannot safely trim dense variable-length field: {key}")
    for i in range(8):
        ids, p, r = (batch[k][i] for k in ("input_ids", "prompts", "responses"))
        if not torch.equal(ids, torch.cat((p, r))):
            raise AssertionError(f"Truncated input_ids differ from prompt+response: row {i}")
        for key in ("loss_mask", "response_mask", "old_log_probs", "advantages"):
            if batch[key][i].numel() != r.numel():
                raise AssertionError(f"Truncated {key} misaligned at row {i}")
    ppo_tokens = sum(int(batch["response_mask"][i].bool().sum().item()) for i in range(8))
    if ppo_tokens == 0:
        raise ValueError("No PPO tokens after truncation")
    return old_len, new_len, ppo_tokens


def prepare(manifest, original_step, cap, output):
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    item, raw, batch = load_item(manifest, original_step)
    old, new, ppo = trim_eight(batch, cap)
    output.mkdir(parents=True)
    merged = output / "session_first_batches"
    merged.mkdir()
    capture = merged / "actor_update_step000001_session_first8.pt"
    raw["tensordict"] = batch
    raw["capture_global_step"] = 1
    raw["offline_capacity_probe"] = {
        "original_capture": str(item["capture"]), "original_step": original_step,
        "original_lengths": old, "trimmed_lengths": new,
        "max_tokens": cap, "original_ppo_tokens": item["ppo_tokens"],
        "retained_ppo_tokens": ppo,
        "purpose": "OOM capacity only; truncated label sequences are not full PPO parity",
    }
    torch.save(raw, capture)
    new_item = copy.deepcopy(item)
    new_item.update({
        "step": 1, "source_global_step": 1,
        "original_source_global_step": original_step,
        "capture": str(capture.resolve()), "ppo_tokens": ppo,
        "logical_full_tokens": sum(new), "max_sequence_length": max(new),
        "min_sequence_length": min(new),
    })
    result = {
        "protocol": "real_actor_session_first_8merged_v3", "online_steps": 1,
        "steps": 1, "expected_sessions": 8, "max_seq_len": cap,
        "max_rows": None, "max_total_tokens": None,
        "historical_ppo_labels_modified": False,
        "memory_capacity_probe_only": True,
        "warning": "Original token labels retained ONLY in their prefixes. "
                   "Do not use for untruncated PPO correctness or speed comparison.",
        "items": [new_item], "session_audits": [],
    }
    (output / "capture_manifest.json").write_text(json.dumps(result, indent=2))
    print("FULL45K_PROBE_PREPARED " + json.dumps({
        "step": original_step, "cap": cap, "original_lengths": old,
        "new_lengths": new, "ppo_tokens": ppo,
        "manifest": str(output / "capture_manifest.json"),
    }), flush=True)



def prepare_all(manifest, cap, output):
    """Truncate existing first-of-Session v3 rows for a 20-step speed/memory run.

    This does NOT reselect a chain: all 160 keys, row identities, and original
    first-of-Session selections come exclusively from the existing manifest.
    Copies preserve token-aligned PPO arrays on retained response positions.
    No source capture or original manifest is changed.
    """
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite benchmark directory: {output}")
    if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
            or manifest.get("online_steps") != 20
            or manifest.get("steps") != 20
            or manifest.get("expected_sessions") != 8
            or len(manifest.get("items", [])) != 20
            or any(item.get("rows") != list(range(8))
                   or item.get("session_count") != 8
                   for item in manifest["items"])):
        raise ValueError("Requires the EXISTING 20-step, N=8 session-first v3 manifest")

    # Preflight all original prompt lengths BEFORE creating output files.
    # A long second/third Session chain must not be picked as a replacement.
    prompt_lengths = []
    for step in range(1, 21):
        item, raw, batch = load_item(manifest, step)
        if item["step"] != step or item["source_global_step"] != step:
            raise AssertionError(f"Original session-first step order changed: {step}")
        if int(raw["capture_global_step"]) != step:
            raise AssertionError(f"Original merged capture step mismatch: {step}")
        if tuple(raw["keys"]) != tuple(
            f"{p['uid']}_tpr_{row}"
            for row, p in enumerate(item["source_row_provenance"])
        ):
            raise AssertionError(f"Original first-of-session trajectory keys differ step={step}")
        for row in range(8):
            p = int(batch["prompts"][row].numel())
            prompt_lengths.append(p)
            if p >= cap - 1:
                raise ValueError(
                    f"step={step} row={row} original selected prompt length={p} "
                    f"cannot be truncated to total cap={cap} without changing prompt "
                    "or removing response. Do not silently select another chain."
                )
        del raw, batch
    print(
        f"TPR_40K_PREFLIGHT steps=20 sessions=160 first_chain_only=true "
        f"prompt_min={min(prompt_lengths)} prompt_max={max(prompt_lengths)} "
        f"cap={cap} status=PASS", flush=True,
    )

    merged_dir = output / "session_first_batches"
    merged_dir.mkdir(parents=True)
    new_manifest = copy.deepcopy(manifest)
    new_manifest["max_seq_len"] = cap
    new_manifest["performance_benchmark"] = {
        "kind": "OFFLINE_FIRST_SESSION_8_PER_STEP_TRUNCATED_FOR_CAPACITY",
        "total_token_cap": cap,
        "original_source_manifest": "v3 session-first, passed as prepare-all input",
        "original_capture_files_mutated": False,
        "first_session_selection_changed": False,
        "rows_per_step": 8,
        "recompute": "OFF",
        "original_ppo_values_recomputed": False,
        "warning": "Retained old_log_probs/advantages remain original values. "
                   "Shortened trajectories are not valid original SWE rewards or "
                   "untruncated 48K PPO correctness evidence. The identical "
                   "derived manifest MUST be used by both Full and TPR.",
    }
    total_original = total_benchmark = total_ppo_original = total_ppo_benchmark = 0
    truncated_rows = 0
    for step in range(1, 21):
        item, raw, batch = load_item(manifest, step)
        old, new, ppo = trim_eight(batch, cap)
        truncated = sum(a != b for a, b in zip(old, new))
        truncated_rows += truncated
        total_original += sum(old)
        total_benchmark += sum(new)
        total_ppo_original += int(item["ppo_tokens"])
        total_ppo_benchmark += ppo
        # These rows and keys are ALREADY the preserved first Session trajectories.
        # Never call the source first_session_indices selector again.
        raw["tensordict"] = batch
        raw["offline_benchmark_truncation"] = {
            "original_capture": item["capture"],
            "source_global_step": step,
            "original_lengths": old,
            "benchmark_lengths": new,
            "cap": cap,
            "original_ppo_tokens": item["ppo_tokens"],
            "benchmark_ppo_tokens": ppo,
            "first_session_selection_changed": False,
        }
        capture = merged_dir / f"actor_update_step{step:06d}_session_first8.pt"
        if capture.exists():
            raise FileExistsError(capture)
        temporary = capture.with_suffix(".pt.tmp")
        try:
            torch.save(raw, temporary)
            temporary.replace(capture)
        finally:
            if temporary.exists():
                temporary.unlink()
        entry = new_manifest["items"][step - 1]
        entry["capture"] = str(capture.resolve())
        entry["ppo_tokens"] = ppo
        entry["logical_full_tokens"] = sum(new)
        entry["max_sequence_length"] = max(new)
        entry["min_sequence_length"] = min(new)
        entry["benchmark_original_lengths"] = old
        entry["benchmark_lengths"] = new
        entry["benchmark_original_ppo_tokens"] = item["ppo_tokens"]
        if entry["rows"] != list(range(8)) or len(raw["keys"]) != 8:
            raise AssertionError("Selected eight Session rows unexpectedly changed")
        print(
            f"TPR_40K_PREPARED step={step} sessions=8 truncated_rows={truncated} "
            f"original_tokens={sum(old)} benchmark_tokens={sum(new)} "
            f"original_ppo={item['ppo_tokens']} benchmark_ppo={ppo} "
            f"max_len={max(new)}", flush=True,
        )
        del item, raw, batch
    summary = {
        "steps": 20, "sessions_per_step": 8, "cap_tokens": cap,
        "truncated_rows": truncated_rows,
        "original_total_tokens": total_original,
        "benchmark_total_tokens": total_benchmark,
        "original_ppo_tokens": total_ppo_original,
        "benchmark_ppo_tokens": total_ppo_benchmark,
    }
    new_manifest["performance_benchmark"]["summary"] = summary
    target = output / "capture_manifest.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False))
    print("TPR_40K_20STEP_READY " + json.dumps({
        **summary, "manifest": str(target.resolve())
    }, ensure_ascii=False), flush=True)
    print("TPR_40K_COMMAND: bash tests/models/mcore/tpr/parallel/"
          "run_existing_20step_offline_full_tpr.sh "
          "/workspace/tpr_actor_capture_4b_n8_run5 "
          f"{output} full", flush=True)
    return new_manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare", "prepare-all"):
        p = subs.add_parser(name)
        p.add_argument("manifest", type=Path)
        p.add_argument("--cap", type=int, default=45 * 1024)
        if name in ("prepare", "prepare-all"):
            p.add_argument("output", type=Path)
            p.add_argument("--step", type=int, default=1)
    args = ap.parse_args()
    if args.cap <= 0 or args.cap % 16:
        raise ValueError("Token cap must be divisible by 16 for TP2/CP4 zigzag alignment")
    loaded = json.loads(args.manifest.read_text())
    if args.command == "audit":
        audit(loaded, args.cap)
    elif args.command == "prepare":
        prepare(loaded, args.step, args.cap, args.output)
    else:
        prepare_all(loaded, args.cap, args.output)
