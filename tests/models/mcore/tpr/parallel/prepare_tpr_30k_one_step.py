"""Build a ONE-step, 30-Ki-token derived PPO capture from the existing 40K replay.

Never overwrite the historical capture or change PPO labels on *retained*
response tokens. A response is truncated from the right, keeping the entire
original prompt. An original prompt longer than the cap is rejected rather
than silently shifting its absolute context. Full and TPR must use the very
same derived manifest, not the original 40K Full JSON.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
from verl.utils import tensordict_utils as tu

_RESPONSE_FIELDS = {
    "responses", "response_mask", "loss_mask", "old_log_probs", "advantages",
    "ref_log_prob", "rollout_is_weights",
}
_FULL_FIELDS = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
_PROMPT_FIELDS = {"prompts"}


def _truncate_nested_field(name, value, original_lengths, cap_lengths, prompt_lengths,
                           response_lengths, retained_response_lengths):
    originals = value.unbind(0)
    result = []
    for row, item in enumerate(originals):
        n = int(item.shape[0])
        if name in _FULL_FIELDS:
            if n != original_lengths[row]:
                raise AssertionError(f"{name}[{row}] not aligned with input_ids")
            keep = cap_lengths[row]
        elif name in _RESPONSE_FIELDS:
            if n != response_lengths[row]:
                raise AssertionError(f"{name}[{row}] not aligned with responses")
            keep = retained_response_lengths[row]
        elif name in _PROMPT_FIELDS:
            if n != prompt_lengths[row]:
                raise AssertionError(f"{name}[{row}] not aligned with prompts")
            keep = n
        else:
            possibilities = {
                typ for typ, size in (
                    ("input", original_lengths[row]),
                    ("response", response_lengths[row]),
                    ("prompt", prompt_lengths[row]),
                ) if n == size
            }
            if len(possibilities) != 1:
                raise ValueError(
                    f"Cannot safely truncate unknown nested field {name}[{row}]: "
                    f"length={n} possible_alignments={sorted(possibilities)}"
                )
            kind = next(iter(possibilities))
            keep = {
                "input": cap_lengths[row],
                "response": retained_response_lengths[row],
                "prompt": n,
            }[kind]
        result.append(item[:keep].clone())
    return tu.nested_tensor_from_tensor_list(
        result, ragged_idx=getattr(value, "_ragged_idx", value.dim()-1)
    )


def derive_capture(source: Path, target: Path, cap: int):
    original = torch.load(source, map_location="cpu", weights_only=False)
    if original.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected captured real Actor PPO minibatch")
    data = original["tensordict"]
    keys = tuple(original["keys"])
    if len(keys) != 8 or len(data["input_ids"]) != 8:
        raise AssertionError("Only one original eight-Session source step is supported")
    required = ("input_ids", "prompts", "responses", "response_mask",
                "loss_mask", "old_log_probs", "advantages")
    if any(key not in data for key in required):
        raise ValueError("Missing original PPO fields")

    ids = data["input_ids"].unbind(0)
    prompts = data["prompts"].unbind(0)
    responses = data["responses"].unbind(0)
    masks = data["response_mask"].unbind(0)
    original_lengths = [int(x.shape[0]) for x in ids]
    prompt_lengths = [int(x.shape[0]) for x in prompts]
    response_lengths = [int(x.shape[0]) for x in responses]
    retained = []
    for row in range(8):
        if not torch.equal(ids[row], torch.cat((prompts[row], responses[row]))):
            raise AssertionError(f"Original row {row} violates input_ids=prompt+response")
        p, r = prompt_lengths[row], response_lengths[row]
        if p >= cap:
            raise ValueError(
                f"Row {row}: prompt alone is {p} tokens >= cap {cap}; "
                "refusing to change the original prompt"
            )
        n = min(r, cap-p)
        if n < 1 or not bool(masks[row][:n].bool().any()):
            raise ValueError(
                f"Row {row}: cropping to {cap} retains no supervised PPO tokens"
            )
        retained.append(n)
    cap_lengths = [p + r for p, r in zip(prompt_lengths, retained)]
    if not any(new < old for new, old in zip(cap_lengths, original_lengths)):
        raise ValueError(f"No trajectory exceeds cap {cap}; refusing a misleading 30K run")

    # Never reconstruct an existing Actor TensorDict from a plain dict.
    # The captured batch can contain scalar/zero-dimensional Tensor values
    # and NonTensorData metadata already accepted by its own TensorDict
    # representation. Passing those unchanged values back through
    # TensorDict(..., batch_size=[8]) spuriously rejects shape=[] fields.
    # Shallow clone retains the original batch semantics, then replace ONLY
    # the jagged time-aligned tensors. No original capture is mutated.
    cropped = data.clone(recurse=False)
    if tuple(cropped.batch_size) != (8,):
        raise AssertionError("Derived TensorDict must keep original batch_size=[8]")
    for name in data.keys():
        value = data[name]
        if isinstance(value, torch.Tensor) and value.is_nested:
            cropped.set(name, _truncate_nested_field(
                name, value, original_lengths, cap_lengths, prompt_lengths,
                response_lengths, retained
            ))
        elif isinstance(value, torch.Tensor) and name in (
            _FULL_FIELDS | _RESPONSE_FIELDS | _PROMPT_FIELDS
        ):
            raise TypeError(
                f"Sequence field {name} is unexpectedly non-nested: "
                f"shape={tuple(value.shape)}"
            )
    if set(cropped.keys()) != set(data.keys()):
        raise AssertionError("Truncating a capture must preserve all PPO fields")
    actual_tokens = 0
    kept_advantages = []
    for i in range(8):
        new_ids = cropped["input_ids"][i]
        new_prompt = cropped["prompts"][i]
        new_response = cropped["responses"][i]
        if (len(new_ids) > cap or
                not torch.equal(new_ids, torch.cat((new_prompt, new_response)))):
            raise AssertionError(f"Derived PPO row {i} has invalid sequence alignment")
        for name in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            if int(cropped[name][i].numel()) != len(new_response):
                raise AssertionError(f"Derived {name} length mismatch on row {i}")
            if not torch.equal(cropped[name][i], data[name][i][:retained[i]]):
                raise AssertionError(f"Original retained PPO labels changed: {name}[{i}]")
        mask = cropped["response_mask"][i].bool()
        actual_tokens += int(mask.sum().item())
        kept_advantages.append(cropped["advantages"][i][mask].float())
    if actual_tokens <= 0:
        raise ValueError("No retained PPO tokens")
    payload = dict(original)
    payload["tensordict"] = cropped
    payload["derived_from_capture"] = str(source.resolve())
    payload["offline_sequence_cap"] = cap
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    stats = {
        "source_capture": str(source.resolve()),
        "capture": str(target.resolve()),
        "cap": cap, "row_keys": keys,
        "old_lengths": original_lengths, "new_lengths": cap_lengths,
        "prompt_lengths": prompt_lengths,
        "old_response_lengths": response_lengths,
        "new_response_lengths": retained,
        "ppo_tokens": actual_tokens,
        "logical_full_tokens": sum(cap_lengths),
        "max_sequence_length": max(cap_lengths),
        "min_sequence_length": min(cap_lengths),
        "positive_advantage_present": any(bool((x > 0).any()) for x in kept_advantages),
        "negative_advantage_present": any(bool((x < 0).any()) for x in kept_advantages),
        "max_abs_advantage": max(float(x.abs().max()) for x in kept_advantages),
    }
    return stats


def prepare(manifest_path: Path, output_dir: Path, *, step: int = 6, cap: int = 30720):
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
            or manifest.get("steps") != 20
            or manifest.get("max_seq_len") != 40960
            or manifest.get("expected_sessions") != 8
            or manifest.get("historical_ppo_labels_modified") is not False):
        raise ValueError("Expected original immutable 20-step/8-session 40K manifest")
    if not 1 <= step <= 20 or not 1 <= cap < 40960:
        raise ValueError("Require source step 1..20 and cap 1..40959")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    item = manifest["items"][step-1]
    if item["step"] != step or item["rows"] != list(range(8)):
        raise ValueError("Source step does not preserve eight original rows")
    source = Path(item["capture"])
    # derive_capture creates the directory only after all input and
    # TensorDict validations pass. A failed preparation must not leave an
    # empty output directory that prevents a clean retry.
    stats = derive_capture(source, output_dir / f"step{step}_cap{cap}.pt", cap)
    derived = copy.deepcopy(manifest)
    derived["experimental_sequence_cap"] = cap
    derived["only_step_derived"] = step
    derived["derived_from_manifest"] = str(manifest_path.resolve())
    # All other manifest steps still refer to the original 40K capture.
    # The original max_seq_len remains 40960 and must never be reported
    # as the new per-step cap; only this one step is used for comparison.
    changed = derived["items"][step-1]
    for k in ("capture", "ppo_tokens", "logical_full_tokens",
              "max_sequence_length", "min_sequence_length",
              "positive_advantage_present", "negative_advantage_present",
              "max_abs_advantage"):
        changed[k] = stats[k]
    changed["original_40k_capture"] = str(source.resolve())
    changed["effective_sequence_cap"] = cap
    dest = output_dir / "capture_manifest.json"
    dest.write_text(json.dumps(derived, indent=2))
    (output_dir / "truncation_audit.json").write_text(json.dumps(stats, indent=2))
    print(
        f"TPR_30K_DERIVED step={step} cap={cap} "
        f"old_lengths={stats['old_lengths']} new_lengths={stats['new_lengths']} "
        f"ppo_tokens={stats['ppo_tokens']} manifest={dest}", flush=True,
    )
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--cap", type=int, default=30720)
    args = parser.parse_args()
    prepare(args.source_manifest, args.output_dir, step=args.step, cap=args.cap)
