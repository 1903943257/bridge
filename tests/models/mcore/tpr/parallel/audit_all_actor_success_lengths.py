#!/usr/bin/env python3
"""CPU-only audit of EVERY original Actor capture trajectory in a rollout run.

All actor_update_step*_rank0_capture*_batch*.pt files are inspected, not only
the 8 rows retained in the session-first replay manifest. Results distinguish
explicit success/accuracy fields from positive reward proxies and UNKNOWN.
No trajectory, reward, advantage or capture file is modified.

Example:
  python tests/models/mcore/tpr/parallel/audit_all_actor_success_lengths.py \\
      /workspace/tpr_actor_capture_4b_n8_run5 --cap 40960
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import torch
from verl.utils import tensordict_utils as tu

PATTERN = re.compile(
    r"actor_update_step(\d+)_rank0_capture(\d+)_batch(\d+)\.pt$"
)
SUCCESS_FIELDS = ("is_correct", "correct", "success", "is_success", "acc", "accuracy")
PROXY_FIELDS = ("reward_score", "rm_scores", "reward", "rewards")
STATUS_PRIORITY = {
    "TRUE": 5, "FALSE": 5, "POSITIVE_REWARD": 3,
    "NONPOSITIVE_REWARD": 3, "UNKNOWN": 0,
}


def unwrap(x):
    return tu.unwrap_non_tensor_data(x)


def field_at(batch, key, row, count):
    if key not in batch:
        return None
    x = batch.get(key)
    try:
        if isinstance(x, torch.Tensor):
            return x if x.ndim == 0 else x[row]
        if hasattr(x, "__len__") and len(x) == count:
            return unwrap(x[row])
    except (IndexError, KeyError, TypeError, RuntimeError):
        pass
    return unwrap(x)


def number(x, *, reduce: bool = False):
    x = unwrap(x)
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        if x.numel() == 1:
            return float(x.item())
        return float(x.float().sum().item()) if reduce else None
    return None


def status_for_row(batch, row, count):
    # Explicit correctness evidence wins. A positive advantage is NOT success.
    for key in ("reward_metrics", "reward_info"):
        value = field_at(batch, key, row, count)
        if isinstance(value, dict):
            candidates = [value, value.get("reward_metrics"), value.get("metrics")]
            for obj in candidates:
                if not isinstance(obj, dict):
                    continue
                for leaf in SUCCESS_FIELDS:
                    if leaf in obj:
                        v = number(obj[leaf])
                        if v is not None:
                            return ("TRUE" if v >= 1.0 else "FALSE", key + "." + leaf, v)
    for key in SUCCESS_FIELDS:
        v = number(field_at(batch, key, row, count))
        if v is not None:
            return ("TRUE" if v >= 1.0 else "FALSE", key, v)
    for key in PROXY_FIELDS:
        v = number(field_at(batch, key, row, count), reduce=True)
        if v is not None:
            return ("POSITIVE_REWARD" if v > 0 else "NONPOSITIVE_REWARD", key, v)
    return ("UNKNOWN", "no_accuracy_or_reward_field", None)


def norm_identity(v):
    value = unwrap(v)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.item()
    return str(value) if value is not None else ""


def classify_length(length, cap):
    if length <= cap:
        return f"<= {cap}"
    if length <= 45056:
        return "(40960,45056]"
    if length <= 46080:
        return "(45056,46080]"
    if length <= 49152:
        return "(46080,49152]"
    return ">49152"



def selected_manifest_identities(manifest_path):
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "real_actor_session_first_8merged_v3":
        raise ValueError("Selection comparison requires original session-first v3 manifest")
    identities = set()
    for item in manifest["items"]:
        step = int(item["source_global_step"])
        raw = torch.load(item["capture"], map_location="cpu", weights_only=False)
        batch = raw["tensordict"]
        provenance = item["source_row_provenance"]
        if len(provenance) != len(raw["keys"]):
            raise AssertionError("Manifest provenance row count mismatch")
        for row, info in enumerate(provenance):
            ids = batch["input_ids"][row]
            sha = hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
            identities.add((step, str(info["uid"]), str(info["session_id"]), sha))
    return identities

def scan(directory, cap, show_top, output, manifest_path=None):
    files = []
    for path in directory.glob("actor_update_step*_rank0_capture*_batch*.pt"):
        match = PATTERN.fullmatch(path.name)
        if match:
            files.append((int(match[1]), int(match[2]), int(match[3]), path))
    files.sort(key=lambda x: x[:3])
    if not files:
        raise FileNotFoundError(f"No original rank0 Actor capture files in {directory}")

    # There can be duplicate captures/rows from the same PPO trajectory.
    # Keep one record per (step, uid, session_id, input_id_bytes), retaining
    # the strongest available evidence. No dedup across different steps.
    records = {}
    raw_rows = 0
    observed_fields = Counter()
    for step, ordinal, batch_idx, path in files:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("capture_stage") != "actor_update_mini_batch":
            raise AssertionError(f"Not original Actor Capture: {path}")
        if int(raw["capture_global_step"]) != step:
            raise AssertionError(f"Step mismatch: {path}")
        data = raw["tensordict"]
        n = len(raw["keys"])
        if n != len(data["input_ids"]):
            raise AssertionError(f"Row mismatch: {path}")
        observed_fields.update(str(k) for k in data.keys())
        for row in range(n):
            uid = norm_identity(field_at(data, "uid", row, n))
            session = norm_identity(field_at(data, "session_id", row, n))
            if uid.startswith("pad"):
                continue
            ids = data["input_ids"][row]
            prompt = data["prompts"][row]
            response = data["responses"][row]
            if not torch.equal(ids, torch.cat((prompt, response))):
                raise AssertionError(f"Original tokens disagree: {path} row={row}")
            raw_rows += 1
            # Input bytes avoid incorrectly dropping two distinct same-length
            # trajectories from the same Session. SHA is only a dedup key.
            sha = hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()
            identity = (step, uid, session, sha)
            status, source, value = status_for_row(data, row, n)
            entry = {
                "step": step, "length": int(ids.numel()),
                "prompt_tokens": int(prompt.numel()),
                "response_tokens": int(response.numel()),
                "status": status, "evidence_field": source,
                "evidence_value": value, "capture": path.name,
                "row": row, "session_id": session,
            }
            previous = records.get(identity)
            if previous is None or STATUS_PRIORITY[status] > STATUS_PRIORITY[previous["status"]]:
                records[identity] = entry

    values = list(records.values())
    selected = selected_manifest_identities(manifest_path)
    selected_values = ([records[key] for key in selected if key in records]
                       if selected is not None else None)
    counts = Counter(r["status"] for r in values)
    overcap = Counter(r["status"] for r in values if r["length"] > cap)
    buckets = {
        status: dict(Counter(classify_length(r["length"], cap) for r in values
                             if r["status"] == status))
        for status in ("TRUE", "FALSE", "POSITIVE_REWARD", "NONPOSITIVE_REWARD", "UNKNOWN")
    }

    def maximal(label):
        xs = sorted((x for x in values if x["status"] == label),
                    key=lambda x: (-x["length"], x["step"], x["row"]))
        return xs[:show_top]

    summary = {
        "capture_dir": str(directory.resolve()),
        "capture_files": len(files), "raw_real_rows": raw_rows,
        "deduplicated_original_trajectories": len(values),
        "cap_tokens": cap, "statuses": dict(counts),
        "over_cap_statuses": dict(overcap),
        "maximum_confirmed_true_tokens": max(
            (r["length"] for r in values if r["status"] == "TRUE"), default=None
        ),
        "maximum_positive_reward_proxy_tokens": max(
            (r["length"] for r in values if r["status"] == "POSITIVE_REWARD"), default=None
        ),
        "positive_reward_is_proven_success": False,
        "unclassified_rows": counts["UNKNOWN"],
        "length_buckets": buckets,
        "observed_fields": sorted(observed_fields),
        "manifest_selected": (None if selected_values is None else {
            "matched_unique_trajectories": len(selected_values),
            "expected_manifest_trajectories": len(selected),
            "maximum_confirmed_true_tokens": max(
                (r["length"] for r in selected_values if r["status"] == "TRUE"), default=None),
            "maximum_positive_reward_proxy_tokens": max(
                (r["length"] for r in selected_values
                 if r["status"] == "POSITIVE_REWARD"), default=None),
            "statuses": dict(Counter(r["status"] for r in selected_values)),
        }),
        "longest_confirmed_true": maximal("TRUE"),
        "longest_positive_reward_proxy": maximal("POSITIVE_REWARD"),
        "longest_unknown": maximal("UNKNOWN"),
    }
    print("ACTOR_SUCCESS_LENGTH_SUMMARY " + json.dumps({
        k: summary[k] for k in (
            "capture_files", "raw_real_rows", "deduplicated_original_trajectories",
            "cap_tokens", "statuses", "over_cap_statuses",
            "maximum_confirmed_true_tokens", "maximum_positive_reward_proxy_tokens",
            "unclassified_rows", "length_buckets", "manifest_selected",
        )}, ensure_ascii=False), flush=True)
    for title, key in (("CONFIRMED_TRUE", "longest_confirmed_true"),
                       ("REWARD_PROXY", "longest_positive_reward_proxy"),
                       ("UNKNOWN", "longest_unknown")):
        for value in summary[key]:
            print("ACTOR_SUCCESS_LENGTH_TOP " +
                  json.dumps({"category": title, **value}, ensure_ascii=False),
                  flush=True)
    if counts["TRUE"] == 0:
        print("ACTOR_SUCCESS_LENGTH_WARNING No explicit success=true/acc=1 "
              "was found. Positive rm_scores/reward_score is ONLY a proxy. "
              "Use original rollout/TQ reward_metrics.acc to establish the "
              "true maximum; do NOT report proxy maximum as confirmed success.",
              flush=True)
    if output:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"ACTOR_SUCCESS_LENGTH_REPORT={output}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--cap", type=int, default=40960)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Also report the original 8-per-step v3 replay selection")
    args = parser.parse_args()
    if args.cap <= 0 or args.top <= 0:
        parser.error("--cap and --top must be positive")
    scan(args.capture_dir, args.cap, args.top, args.output, args.manifest)
