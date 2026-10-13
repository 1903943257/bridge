"""Compare independent eight-rank recorded Full vs shared-TPR PPO snapshots.

No GPU; this intentionally reports diagnostics AND fails provisional gates.
Full here is independent Ring attention on complete sequences, not the native
MindSpeed production training backend.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import torch

def _l2_pair(left, right):
    num, den = 0.0, 0.0
    for rank in range(8):
        la, rb = left[rank]["gradient_samples"], right[rank]["gradient_samples"]
        if la.keys() != rb.keys():
            raise AssertionError(f"rank {rank}: parameter names differ")
        for name in la:
            a, b = la[name], rb[name]
            if a["shape"] != b["shape"] or not torch.equal(a["indices"], b["indices"]):
                raise AssertionError(f"rank {rank} parameter witness mismatch: {name}")
            v = a["values"].double().flatten()
            w = b["values"].double().flatten()
            num += float(((v-w)**2).sum())
            den += float((v**2).sum())
    return math.sqrt(num/max(den, 1e-30))

def _weight_diff(left, right, key):
    delta, base = 0.0, 0.0
    for rank in range(8):
        a, b = left[rank][key], right[rank][key]
        if a.keys() != b.keys():
            raise AssertionError(f"rank {rank} parameter name mismatch")
        for name in a:
            va, vb = a[name], b[name]
            if va["shape"] != vb["shape"] or not torch.equal(va["indices"], vb["indices"]):
                raise AssertionError(f"rank {rank} witness identity mismatch: {name}")
            x, y = va["values"].double(), vb["values"].double()
            delta += float(((x-y)**2).sum())
            base += float((x**2).sum())
    return math.sqrt(delta/max(base, 1e-30))

def _logprobs(rows):
    # TP peers reproduce each logical token; use ONLY TP-rank zero.
    output = {}
    for item in rows:
        if item["tp_rank"] != 0:
            continue
        for key, value in item["logprobs"].items():
            key = tuple(key)
            if key in output:
                raise AssertionError(f"duplicate logical response-token owner: {key}")
            output[key] = float(value)
    return output


def _complete_gradient_metrics(full, tpr):
    """Exact relative L2, cosine, and per-parameter contributors.

    Streams one saved original-BF16 gradient tensor at a time. Only CP0
    replicas participate. TP-sharded parameters count both TP shards;
    TP-replicated parameters count once.
    """
    import math
    dot = full_sq = tpr_sq = diff_sq = 0.0
    contributions = []
    enabled = any(x.get("full_grad_dir") for x in full + tpr)
    if not enabled:
        return None
    for tp_rank in (0, 1):
        a = next(x for x in full if x["tp_rank"] == tp_rank and x["cp_rank"] == 0)
        b = next(x for x in tpr if x["tp_rank"] == tp_rank and x["cp_rank"] == 0)
        if not a.get("full_grad_dir") or not b.get("full_grad_dir"):
            raise AssertionError("Both modes require complete BF16 grads on all TP shards")
        left, right = Path(a["full_grad_dir"]), Path(b["full_grad_dir"])
        la = torch.load(left/"manifest.pt", map_location="cpu", weights_only=False)
        lb = torch.load(right/"manifest.pt", map_location="cpu", weights_only=False)
        if la != lb:
            raise AssertionError(f"TP{tp_rank} gradient manifest mismatch")
        for entry in la:
            if not entry["tp_sharded"] and tp_rank != 0:
                continue
            x = torch.load(left/entry["filename"], map_location="cpu", weights_only=True)
            y = torch.load(right/entry["filename"], map_location="cpu", weights_only=True)
            if x.shape != y.shape or x.dtype != y.dtype:
                raise AssertionError(f"Full/TPR exact gradient mismatch {entry['name']}")
            # Stream tensor chunks: a 4B shard may be hundreds of millions
            # of BF16 entries; avoid materializing whole FP64 copies on CPU.
            aa = bb = xy = dd = 0.0
            xf = x.reshape(-1)
            yf = y.reshape(-1)
            for start in range(0, xf.numel(), 1_048_576):
                v = xf[start:start+1_048_576].double()
                w = yf[start:start+1_048_576].double()
                d = v - w
                aa += float((v*v).sum())
                bb += float((w*w).sum())
                xy += float((v*w).sum())
                dd += float((d*d).sum())
            full_sq += aa
            tpr_sq += bb
            dot += xy
            diff_sq += dd
            contributions.append((entry["name"], tp_rank, math.sqrt(dd)))
            del x, y, xf, yf
    return {
        "full_norm": math.sqrt(full_sq), "tpr_norm": math.sqrt(tpr_sq),
        "norm_relative_gap": abs(math.sqrt(full_sq)-math.sqrt(tpr_sq))/max(math.sqrt(full_sq), 1e-30),
        "relative_l2": math.sqrt(diff_sq/max(full_sq, 1e-30)),
        "cosine": dot / max(math.sqrt(full_sq*tpr_sq), 1e-30),
        "largest_absolute_gradient_differences": sorted(
            contributions, key=lambda z: z[2], reverse=True
        )[:12],
    }


def compare(root):
    snapshots = {}
    for mode in ("full", "tpr"):
        files = [root / f"{mode}_rank{i}.pt" for i in range(8)]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {mode} ranks: {missing}")
        snapshots[mode] = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    full, tpr = snapshots["full"], snapshots["tpr"]
    invariant = (
        "source_capture", "source_rows", "source_keys", "data_sha256",
        "checkpoint", "objective", "optimizer", "logical_ppo_tokens",
        "tp", "cp", "dp",
    )
    for rank in range(8):
        a, b = full[rank], tpr[rank]
        if a["rank"] != b["rank"] or a["rank"] != rank:
            raise AssertionError("rank identity mismatch")
        for key in invariant:
            if a[key] != b[key]:
                raise AssertionError(f"rank={rank} original data mismatch: {key}")
    if len({x["data_sha256"] for x in full+tpr}) != 1:
        raise AssertionError("All ranks and modes must see the same original PPO inputs")
    before_gap = _weight_diff(full, tpr, "param_before")
    if before_gap != 0:
        raise AssertionError(f"Initial weight mismatch; replay invalid: {before_gap}")

    f_lp, t_lp = _logprobs(full), _logprobs(tpr)
    if f_lp.keys() != t_lp.keys():
        raise AssertionError(
            f"Different logical PPO supervision keys: full={len(f_lp)} tpr={len(t_lp)}"
        )
    if len(f_lp) != full[0]["logical_ppo_tokens"]:
        raise AssertionError(
            f"Missing logical logprobs: {len(f_lp)} vs {full[0]['logical_ppo_tokens']}"
        )
    keys = sorted(f_lp)
    a = torch.tensor([f_lp[k] for k in keys], dtype=torch.float64)
    b = torch.tensor([t_lp[k] for k in keys], dtype=torch.float64)
    logprob_rel = float(torch.linalg.vector_norm(a-b) /
                        torch.linalg.vector_norm(a).clamp_min(1e-30))
    logprob_max = float((a-b).abs().max())

    losses = [(x["loss"], y["loss"]) for x, y in zip(full, tpr)]
    if any(not math.isfinite(v) for pair in losses for v in pair):
        raise AssertionError("non-finite PPO loss")
    if max(abs(v-losses[0][i]) for i in (0,1) for v in [pair[i] for pair in losses]) > 1e-4:
        raise AssertionError(f"PPO scalar not synchronized across CP ranks: {losses}")
    loss_gap = abs(losses[0][0]-losses[0][1])
    grad_rel = _l2_pair(full, tpr)
    complete_grad = _complete_gradient_metrics(full, tpr)
    grad_norms = None
    if all("global_bf16_grad_norm" in x for x in full + tpr):
        f_norm = full[0]["global_bf16_grad_norm"]
        t_norm = tpr[0]["global_bf16_grad_norm"]
        grad_norms = {
            "full": f_norm, "tpr": t_norm,
            "relative_gap": abs(t_norm-f_norm)/max(f_norm,1e-30),
        }
        for item in full:
            if abs(item["global_bf16_grad_norm"]-f_norm) > 1e-6:
                raise AssertionError("inconsistent distributed Full gradient norm")
        for item in tpr:
            if abs(item["global_bf16_grad_norm"]-t_norm) > 1e-6:
                raise AssertionError("inconsistent distributed TPR gradient norm")
    step_rel = _weight_diff(full, tpr, "param_after")

    def perf(mode):
        xs = snapshots[mode]
        return {
            "fwd_bwd_seconds_critical": max(x["time_forward_backward_finalize_s"] for x in xs),
            "finalize_seconds_critical": max(x["time_finalize_s"] for x in xs),
            "sgd_seconds_critical": max(x["time_sgd_s"] for x in xs),
            "peak_allocated_gib_worst": max(x["peak_allocated_bytes"] for x in xs)/2**30,
            "peak_reserved_gib_worst": max(x["peak_reserved_bytes"] for x in xs)/2**30,
            "allocated_before_gib_worst": max(x["allocated_before_bytes"] for x in xs)/2**30,
            "forest_trees": xs[0]["forest_trees"],
            "forest_segments": xs[0]["forest_segments"],
        }
    fp, tp = perf("full"), perf("tpr")
    ratio = fp["fwd_bwd_seconds_critical"]/max(tp["fwd_bwd_seconds_critical"], 1e-12)
    # Provisional 1.7B-level thresholds, not already-passed 4B gates.
    gates = {
        "initial_weights_exact": before_gap == 0,
        "ppo_loss_gap_lt_0.02": loss_gap < 0.02,
        "logical_logprob_rel_l2_lt_0.05": logprob_rel < 0.05,
        "sampled_grad_rel_l2_lt_0.05": grad_rel < 0.05,
        **({"complete_grad_rel_l2_lt_0.05": complete_grad["relative_l2"] < 0.05}
           if complete_grad is not None else {}),
        "finite": all(math.isfinite(v) for v in
                      (loss_gap, logprob_rel, grad_rel, step_rel, ratio)),
    }
    print(
        f"TPR_OFFLINE_COMPARISON rows={full[0]['source_rows']} "
        f"ppo_tokens={len(keys)} loss_full={losses[0][0]:.9g} "
        f"loss_tpr={losses[0][1]:.9g} loss_gap={loss_gap:.9g} "
        f"logprob_rel_l2={logprob_rel:.9g} logprob_max_abs={logprob_max:.9g} "
        f"sampled_grad_rel_l2={grad_rel:.9g} "
        f"sampled_post_sgd_weight_rel_l2={step_rel:.9g} "
        f"fwd_bwd_speedup={ratio:.3f}x gates={gates} "
        f"gate_pass={all(gates.values())}",
        flush=True,
    )
    print(f"TPR_OFFLINE_PERF full={fp} tpr={tp}", flush=True)
    print(f"TPR_OFFLINE_GRAD_NORM {grad_norms}", flush=True)
    print(f"TPR_OFFLINE_COMPLETE_GRAD {complete_grad}", flush=True)
    if not all(gates.values()):
        raise AssertionError("Recorded PPO Full/TPR one-step provisional gate failed")
    return gates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    compare(args.output_dir)
