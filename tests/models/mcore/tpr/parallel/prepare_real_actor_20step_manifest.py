"""CPU-only session-first selection manifest for offline Full vs Turn TPR.

Reads EVERY original rank0 Actor mini-batch captured at each online step.
Keeps the earliest available row from each (uid,session_id) across captures;
never changes old_log_probs, advantages, rewards or any original .pt file.
A new offline optimizer update is scheduled for each capture with >=1 selected
valid row. Offline optimizer updates may outnumber online global steps.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

try:
    from .offline_session_selection import first_session_indices
except ImportError:
    from offline_session_selection import first_session_indices

REQUIRED = ("input_ids", "prompts", "responses", "response_mask",
            "loss_mask", "old_log_probs", "advantages", "temperature")
CAPTURE_NAME = re.compile(
    r"actor_update_step(?P<step>\d+)_rank0_capture(?P<ordinal>\d+)_batch(?P<batch>\d+)\.pt$"
)


def _all_capture_steps(directory: Path) -> dict[int, list[Path]]:
    """All actual actor updates, not only the first capture in a global step."""
    found: dict[int,list[tuple[int,int,Path]]] = {}
    seen_ordinals=set()
    for path in directory.glob("actor_update_step*_rank0_capture*_batch*.pt"):
        match = CAPTURE_NAME.fullmatch(path.name)
        if match is None:
            continue
        step=int(match.group("step"))
        order=int(match.group("ordinal"))
        batch_idx=int(match.group("batch"))
        if (order, batch_idx) in seen_ordinals:
            raise RuntimeError(f"duplicate Actor capture ordinal/batch: {path}")
        seen_ordinals.add((order,batch_idx))
        found.setdefault(step,[]).append((order,batch_idx,path.resolve()))
    return {
        step:[path for _,_,path in sorted(items)]
        for step,items in found.items()
    }


def prepare(directory: Path, output: Path, *, steps: int = 20,
            max_seq_len: int = 49152, max_rows: int = 3,
            max_total_tokens: int = 130000):
    if steps <= 0 or max_seq_len <= 0 or max_rows < 1 or max_total_tokens <= 0:
        raise ValueError("invalid selection limits")
    found = _all_capture_steps(directory)
    missing=[step for step in range(1,steps+1) if step not in found]
    if missing:
        raise FileNotFoundError(
            f"expected Actor captures at EVERY global step 1..{steps}; "
            f"missing={missing}; present={sorted(found)}"
        )
    items=[]
    audits=[]
    for global_step in range(1,steps+1):
        paths=found[global_step]
        originals=[]
        for path in paths:
            raw=torch.load(path,map_location="cpu",weights_only=False)
            if raw.get("capture_stage")!="actor_update_mini_batch":
                raise AssertionError(f"not real Actor PPO capture: {path}")
            if int(raw.get("capture_global_step",-1))!=global_step:
                raise AssertionError(f"global step mismatch: {path}")
            batch=raw["tensordict"]
            if any(k not in batch for k in REQUIRED):
                raise AssertionError(f"missing PPO fields: {path}")
            if len(raw["keys"])!=len(batch["input_ids"]):
                raise AssertionError(f"bad row identities: {path}")
            originals.append(raw)
        rows_by_capture,audit=first_session_indices([
            (str(path),raw["tensordict"])
            for path,raw in zip(paths,originals)
        ])
        audit["online_global_step"]=global_step
        audit["original_capture_count"]=len(paths)
        audit["original_rows"]=sum(len(raw["keys"]) for raw in originals)
        audit["eligible_capture_count"]=0
        audit["chosen_rows"]=0
        for path,raw,kept in zip(paths,originals,rows_by_capture):
            batch=raw["tensordict"]
            candidates=[]
            for row in kept:
                ids=batch["input_ids"][row]
                mask=batch["response_mask"][row].bool()
                old=batch["old_log_probs"][row]
                adv=batch["advantages"][row]
                size=int(ids.numel())
                nt=int(mask.sum().item())
                if mask.numel()!=old.numel() or mask.numel()!=adv.numel():
                    raise AssertionError(
                        f"bad old_lp/advantages alignment step={global_step} row={row}"
                    )
                if size>max_seq_len or nt==0:
                    continue
                valid_adv=adv[mask].float()
                candidates.append((
                    size,row,nt,
                    bool((valid_adv>0).any().item()),
                    bool((valid_adv<0).any().item()),
                    float(valid_adv.abs().max().item()),
                ))
            candidates.sort()
            selected=[]
            logical_tokens=0
            def choose(candidate):
                nonlocal logical_tokens
                if (candidate is None or candidate in selected
                        or len(selected)>=max_rows
                        or logical_tokens+candidate[0]>max_total_tokens):
                    return False
                selected.append(candidate)
                logical_tokens+=candidate[0]
                return True
            pos=[x for x in candidates if x[3]]
            neg=[x for x in candidates if x[4]]
            pairs=[
                (a,b) for a in pos for b in neg
                if a[1]!=b[1] and a[0]+b[0]<=max_total_tokens
            ]
            if pairs:
                a,b=min(pairs,key=lambda pair:(
                    pair[0][0]+pair[1][0],pair[0][1],pair[1][1]
                ))
                choose(a)
                choose(b)
            for candidate in candidates:
                choose(candidate)
            if not selected:
                # This Actor mini-batch has no first-of-session eligible
                # trajectory (e.g. all its rows are discarded continuation
                # chains); don't fabricate a replay update for it.
                print(f"TPR_OFFLINE_CAPTURE_SKIPPED global_step={global_step} "
                      f"file={path.name} original_rows={len(raw['keys'])} "
                      f"session_first_candidates={len(kept)}",flush=True)
                continue
            selected_rows=sorted(x[1] for x in selected)
            ntokens=sum(x[2] for x in selected)
            replay_step=len(items)+1
            item={
                "step":replay_step, # sequential OFFLINE optimizer update
                "source_global_step":global_step,
                "source_capture_index":int(raw["capture_index"]),
                "source_mini_batch_idx":int(raw["capture_mini_batch_idx"]),
                "capture":str(path),
                "rows":selected_rows,
                "original_rows":len(raw["keys"]),
                "session_first_candidates":len(kept),
                "logical_full_tokens":logical_tokens,
                "ppo_tokens":ntokens,
                "positive_advantage_present":any(x[3] for x in selected),
                "negative_advantage_present":any(x[4] for x in selected),
                "max_abs_advantage":max(x[5] for x in selected),
                "max_sequence_length":max(x[0] for x in selected),
            }
            items.append(item)
            audit["eligible_capture_count"]+=1
            audit["chosen_rows"]+=len(selected_rows)
            print(f"TPR_20STEP_MANIFEST replay_update={replay_step} "
                  f"online_step={global_step} capture={path.name} "
                  f"rows={selected_rows} logical_tokens={logical_tokens} "
                  f"ppo_tokens={ntokens}",flush=True)
        if audit["eligible_capture_count"]==0:
            raise RuntimeError(
                f"online global_step={global_step} has no valid FIRST-session "
                "trajectory in any captured Actor mini-batch; refusing "
                "to silently omit an online step."
            )
        if audit["capture_first_fallback_sessions"]:
            print(
                f"TPR_SESSION_ORDER_WARNING step={global_step} "
                f"sessions_without_original_trajectory_ordinal="
                f"{audit['capture_first_fallback_sessions']}; "
                "using first encounter in captured mini-batch order "
                "(not provably the original Gateway first chain)",
                flush=True,
            )
        audits.append(audit)
        del originals
    if not items:
        raise AssertionError("No valid offline PPO updates")
    result={
        "protocol":"real_actor_session_first_multicapture_v2",
        "online_steps":steps,
        "steps":len(items), # total OFFLINE optimizer updates
        "max_seq_len":max_seq_len,"max_rows":max_rows,
        "max_total_tokens":max_total_tokens,
        "session_selection":"one_first_available_per_uid_session_id_across_capture_files",
        "historical_ppo_labels_modified":False,
        "items":items,
        "session_audits":audits,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(f"TPR_20STEP_SESSION_FIRST_SUMMARY online_steps={steps} "
          f"captured_optimizer_updates={sum(len(x) for x in found.values())} "
          f"selected_replay_updates={len(items)} "
          f"selected_full_trajectories={sum(len(x['rows']) for x in items)} "
          f"manifest={output}",flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("capture_dir",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--steps",type=int,default=20)
    p.add_argument("--max-seq-len",type=int,default=49152)
    p.add_argument("--max-rows",type=int,default=3)
    p.add_argument("--max-total-tokens",type=int,default=130000)
    a=p.parse_args()
    prepare(a.capture_dir,a.output,steps=a.steps,
            max_seq_len=a.max_seq_len,max_rows=a.max_rows,
            max_total_tokens=a.max_total_tokens)
