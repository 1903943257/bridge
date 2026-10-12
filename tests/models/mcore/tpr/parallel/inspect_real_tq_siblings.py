"""CPU-only TQ preflight: verify authentic 512-token shared-prefix siblings.

Does not import Megatron, instantiate Qwen or synthesize advantages.
Run inside the trusted user's NPU Docker on the original captured TQ file.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import torch


def _rows(data, key):
    x = data[key]
    return list(x) if isinstance(x, (list, tuple)) else list(x.unbind())


def inspect(path, *, siblings=2, require_ppo=False):
    if not path.is_file():
        raise FileNotFoundError(f"Original TQ data not found: {path}")
    # This is the original trusted local actor checkpoint/tensordict dump.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    data = raw["tensordict"]
    source_stage = raw.get("capture_stage", "tq_rollout")
    keys = tuple(raw["keys"])
    fields = tuple(sorted(str(x) for x in data.keys()))
    print(f"P0_5 DATA source={path} original_trajectories={len(keys)} fields={fields}", flush=True)
    old_field = next((k for k in ("old_log_probs", "old_logprobs") if k in data), None)
    has_adv = "advantages" in data
    if not old_field or not has_adv:
        print("P0_5 PPO_FIELDS status=MISSING_REAL_ACTOR_FIELDS "
              f"original_old_logprobs={bool(old_field)} "
              f"original_advantages={has_adv} "
              "next=RUN_REAL_SIBLING_NLL_NOT_REAL_PPO", flush=True)
    else:
        print(f"P0_5 PPO_FIELDS status=RECORDED_VALUES_PRESENT "
              f"old_field={old_field} advantages=True "
              "on_policy_actor_provenance=NOT_VERIFIED", flush=True)

    if require_ppo:
        if source_stage != "actor_update_mini_batch":
            raise AssertionError(
                "ACTUAL_PPO_REQUIRED: this is a rollout TQ dump, not an "
                "actor_update_mini_batch capture; capture after advantages "
                "and old_log_probs are ready, before train_batch"
            )
        if not old_field or not has_adv or "loss_mask" not in data or "temperature" not in data:
            raise AssertionError(
                "ACTUAL_PPO_REQUIRED: original actor capture must contain "
                "old_log_probs, advantages, loss_mask and temperature"
            )
        if len(keys) != len(set(keys)) or not all(isinstance(k, str) and k for k in keys):
            raise AssertionError("Actor capture requires exact unique trajectory keys")
        temperature = torch.as_tensor(data["temperature"], dtype=torch.float32).detach().cpu()
        if temperature.numel() not in (1, len(keys)) or not bool(torch.isfinite(temperature).all() and (temperature > 0).all()):
            raise AssertionError("P0-5 PPO temperature must be finite, positive and scalar or per-row")
        print(f"P0_5 PPO_TEMPERATURE values={temperature.flatten().tolist()} mode=preserve_original", flush=True)
    rows = {k: _rows(data,k) for k in ("prompts","responses","input_ids","response_mask")}
    if require_ppo:
        for name in ("old_log_probs" if old_field == "old_log_probs" else "old_logprobs",
                     "advantages", "loss_mask"):
            fields = _rows(data, name)
            if len(fields) != len(keys):
                raise AssertionError(f"Actor PPO field {name} row count mismatch")
            for i, tensor in enumerate(fields):
                tensor = tensor.detach().cpu()
                response_len = int(rows["responses"][i].numel())
                full_len = int(rows["input_ids"][i].numel())
                if name == "loss_mask":
                    mask = tensor.bool()
                    if mask.numel() == full_len:
                        if bool(mask[:full_len-response_len].any()):
                            raise AssertionError(f"Actor loss_mask covers prompt row={i}")
                        mask = mask[-response_len:]
                    if not torch.equal(mask, rows["response_mask"][i].detach().cpu().bool()):
                        raise AssertionError(f"Actor response/loss mask mismatch row={i}")
                elif tensor.numel() != response_len or not bool(torch.isfinite(tensor.float()).all()):
                    raise AssertionError(f"Actor PPO field invalid: {name} row={i}")
    groups = {}
    for idx in range(len(keys)):
        p=rows["prompts"][idx].detach().cpu().long()
        r=rows["responses"][idx].detach().cpu().long()
        entire=rows["input_ids"][idx].detach().cpu().long()
        m=rows["response_mask"][idx].detach().cpu().bool()
        if not torch.equal(entire,torch.cat((p,r))) or m.numel()!=r.numel():
            raise AssertionError(f"Original TQ trajectory {idx} data integrity mismatch")
        if (p.numel()<512 or r.numel()<512
            or not bool(m[0]) or not bool(m[1:512].any())):
            continue
        h=hashlib.sha256(p[-512:].numpy().tobytes()).hexdigest()
        groups.setdefault(h,[]).append((idx,r[:512].contiguous(),int(m[:512].sum())))
    eligible=[]
    for h, rows_for_prefix in sorted(groups.items()):
        unique=[]
        for item in rows_for_prefix:
            if not any(torch.equal(item[1],v[1]) for v in unique):
                unique.append(item)
        if len(unique)>=siblings:
            first=next(
                i for i in range(512)
                if len({int(z[1][i]) for z in unique[:siblings]})>1
            )
            eligible.append((h,unique,first))
        print(f"P0_5 GROUP prefix_sha={h[:16]} "
              f"original_rows={[z[0] for z in rows_for_prefix]} "
              f"distinct_512_response_suffixes={len(unique)}",flush=True)
    if not eligible:
        raise AssertionError(f"No group with {siblings} distinct true original siblings sharing last512 real prompt tokens")
    eligible.sort(key=lambda it:(it[2]!=0,it[2]))
    best=eligible[0]
    print(f"P0_5 ELIGIBLE status=PASS "
          f"shared_prefix_sha={best[0]} "
          f"candidate_row_ids={[z[0] for z in best[1]]} "
          f"first_diff_offset={best[2]} "
          f"supervised={[z[2] for z in best[1]]} "
          "crop=LAST512_PROMPT_PLUS_FIRST512_REAL_RESPONSE "
          "PPO_STATUS=NOT_ESTABLISHED_BY_TOKENS",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path",type=Path)
    p.add_argument("--siblings",type=int,default=2)
    p.add_argument("--require-ppo",action="store_true",
                   help="Reject rollout dump and incomplete actor-update PPO fields")
    args=p.parse_args()
    inspect(args.path,siblings=args.siblings,require_ppo=args.require_ppo)
