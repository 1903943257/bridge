"""P0-B: 50-step native MindSpeed CP=1/2/4 real-TQ controlled NLL training.

Use torchrun with explicit localhost address/port (via run_native_cp_multistep.sh).
This is a deterministic low-diversity replay of REAL TQ tokens; not PPO, not a
production DDP/ZeRO optimizer and not a generalization/convergence benchmark.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available

from .test_native_cp1_cp2_cp4_drift_npu import (
    RESPONSE_LEN, SEQ_LEN, _model, _native_batch, _runtime,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_P0_CP_MULTISTEP") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="P0-B requires TPR_P0_CP_MULTISTEP=1 and an Ascend NPU",
)

_MILESTONES = (0, 1, 2, 5, 10, 20, 30, 40, 50)


def _load_real_tq_rows(path: Path, vocab_size: int):
    """Make contiguous 1024-token REAL windows with disjoint trajectory IDs.

    Every window is [512 previous REAL context tokens, 512 next REAL response
    tokens], with response_mask inherited exactly. No fabricated examples,
    no rolling labels over a Zigzag partition, and no train/val trajectory leak.
    """
    if not path.is_file():
        raise FileNotFoundError(f"real TQ batch required; no synthetic fallback: {path}")
    from ..correctness.test_qwen3_1_7b_real_tq_ppo_npu import _rows
    dump = torch.load(path, map_location="cpu", weights_only=False)
    data = dump["tensordict"]
    prompts, responses = _rows(data, "prompts"), _rows(data, "responses")
    masks, fulls = _rows(data, "response_mask"), _rows(data, "input_ids")
    if not (len(prompts) == len(responses) == len(masks) == len(fulls)):
        raise AssertionError("TQ prompt/response/mask/input_ids row counts disagree")
    max_windows = int(os.getenv("TPR_P0_B_MAX_WINDOWS_PER_TRAJECTORY", "16"))
    if not 1 <= max_windows <= 64:
        raise AssertionError("max real windows per trajectory must be within [1,64]")
    grouped, skipped = [], []
    for i, (p, r, m, full) in enumerate(zip(prompts, responses, masks, fulls, strict=True)):
        p = p.detach().cpu().long()
        r = r.detach().cpu().long()
        m = m.detach().cpu().bool()
        entire = full.detach().cpu().long()
        if not torch.equal(entire, torch.cat((p, r))) or m.numel() != r.numel():
            raise AssertionError(f"TQ row {i} prompt/response/full/mask disagreement")
        if p.numel() < 512 or r.numel() < RESPONSE_LEN:
            skipped.append(i)
            continue
        windows = []
        for w in range(min(max_windows, r.numel() // RESPONSE_LEN)):
            offset = w * RESPONSE_LEN
            mask = m[offset:offset + RESPONSE_LEN].contiguous()
            if not mask.any():
                continue
            first_response_global_pos = p.numel() + offset
            tokens = entire[first_response_global_pos - 512:
                            first_response_global_pos + RESPONSE_LEN].contiguous()
            if tokens.numel() != SEQ_LEN:
                raise AssertionError(f"TQ row {i} window {w} length is not 1024")
            if int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size:
                raise AssertionError(f"TQ row {i} window {w} contains out-of-vocab tokens")
            windows.append({
                "id": f"{i}:w{w}", "trajectory_id": i, "tokens": tokens,
                "mask": mask,
            })
        if windows:
            grouped.append((i, windows))
        else:
            skipped.append(i)
    if len(grouped) < 2:
        raise AssertionError(
            f"Need >=2 separate real TQ trajectories for disjoint "
            f"train/validation; usable={len(grouped)} skipped={skipped}"
        )
    # Trajectory-disjoint holdout: never split windows of the same rollout
    # between train and validation. Validation samples stay fixed throughout.
    val_trajectories = max(1, len(grouped) // 4)
    training_groups, heldout_groups = grouped[:-val_trajectories], grouped[-val_trajectories:]
    training = [window for _, windows in training_groups for window in windows]
    val_all = [window for _, windows in heldout_groups for window in windows]
    # Cap validation at 8 windows to avoid spending most of the 50-step
    # compute on repeated validation forward calls.
    if len(val_all) > 8:
        selection = torch.linspace(0, len(val_all) - 1, steps=8).round().long().tolist()
        validation = [val_all[i] for i in selection]
    else:
        validation = val_all
    # Fixed random permutation of REAL windows: identical across CP sizes and
    # spread 50 updates across different trajectories rather than grouped rows.
    gen = torch.Generator(device="cpu").manual_seed(20261011)
    perm = torch.randperm(len(training), generator=gen).tolist()
    training = [training[i] for i in perm]
    sha = hashlib.sha256()
    for idx, windows in grouped:
        sha.update(str(idx).encode("ascii"))
        for row in windows:
            sha.update(row["id"].encode("ascii"))
            sha.update(row["tokens"].numpy().tobytes())
            sha.update(row["mask"].to(torch.uint8).numpy().tobytes())
    return training, validation, skipped, sha.hexdigest()


def _loss(model, runtime, row, *, training: bool):
    """Same global supervised target indices for all causal Zigzag ranks."""
    batch, positions, _ = _native_batch(row["tokens"], runtime)
    selected = positions < SEQ_LEN - 1
    # Query at global position 511 predicts response token 0.
    queries_cpu = positions[selected]
    response_offsets = queries_cpu - (SEQ_LEN - RESPONSE_LEN - 1)
    supervised_cpu = torch.zeros_like(response_offsets, dtype=torch.bool)
    relevant = (response_offsets >= 0) & (response_offsets < RESPONSE_LEN)
    supervised_cpu[relevant] = row["mask"][response_offsets[relevant]]
    count = torch.tensor(int(supervised_cpu.sum()), device=runtime.device, dtype=torch.int32)
    if runtime.cp > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=runtime.group)
    denom = int(count.item())
    if denom < 1:
        raise AssertionError("No supervised target tokens in the real TQ row")
    labels = row["tokens"].index_select(0, queries_cpu + 1).to(runtime.device)
    logits = model(batch["tokens"], batch["position_ids"], None)
    if logits.shape != (1, SEQ_LEN // runtime.cp, model.vocab_size):
        raise AssertionError(f"unexpected model logits shape: {tuple(logits.shape)}")
    selected_gpu = torch.nonzero(batch["position_ids"][0] < SEQ_LEN - 1).flatten()
    local_lp = F.log_softmax(logits[0].index_select(0, selected_gpu).float(), dim=-1)
    token_lp = local_lp.gather(1, labels[:, None]).squeeze(1)
    local_loss = -token_lp[supervised_cpu.to(runtime.device)].sum() / denom
    # IMPORTANT: backward local contribution before scalar SUM; never backprop
    # through all_reduce of detached scalar.
    if training:
        local_loss.backward()
    summed = local_loss.detach().clone()
    if runtime.cp > 1:
        dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=runtime.group)
    return float(summed.item()), denom


def _make_optimizer(model, lr):
    """Explicit FP32 master AdamW; forward/backward remains native BF16.

    This is a diagnostic single-replica (DP=1) optimizer, NOT native
    Megatron DistributedOptimizer. Each CP rank owns equal FP32 masters
    and performs the identical update after cross-CP gradient SUM.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    masters = [torch.nn.Parameter(p.detach().float().clone(), requires_grad=True)
               for _, p in named]
    optimizer = torch.optim.AdamW(
        masters, lr=lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=0.0, foreach=False,
    )
    return named, masters, optimizer


def _sample_weights(named, masters):
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
    weights = {}
    for (name, p), master in zip(named, masters, strict=True):
        index = torch.tensor(exact_sample_indices(p.numel(), 512),
                             device=p.device, dtype=torch.long)
        weights[name] = {
            "shape": tuple(p.shape),
            "indices": index.cpu(),
            "master": master.detach().flatten().index_select(0, index).cpu().clone(),
            "bf16": p.detach().flatten().index_select(0, index).float().cpu().clone(),
        }
    return weights


def _rank_consensus(named, runtime):
    """Check that all CP rank model replicas really carry identical weights."""
    h = hashlib.sha256()
    for name, p in named:
        h.update(name.encode())
        # Bounded exact BF16 bits at predetermined endpoints.
        flat = p.detach().flatten()
        v = flat.index_select(0, torch.tensor(
            sorted({0, flat.numel() // 2, flat.numel() - 1}),
            device=p.device, dtype=torch.long)).contiguous().cpu()
        h.update(v.view(torch.int16).numpy().tobytes())
    value = torch.tensor(list(h.digest()), device=runtime.device, dtype=torch.int32)
    highest, lowest = value.clone(), value.clone()
    if runtime.cp > 1:
        dist.all_reduce(highest, op=dist.ReduceOp.MAX, group=runtime.group)
        dist.all_reduce(lowest, op=dist.ReduceOp.MIN, group=runtime.group)
    if not torch.equal(highest, lowest):
        raise AssertionError("Model replicas have diverged between CP ranks")
    return h.hexdigest()


def _step(model, runtime, row, named, masters, optimizer, grad_reduce, clip_norm):
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    loss, supervised = _loss(model, runtime, row, training=True)
    # The local loss already has the GLOBAL denominator. SUM CP contributions
    # exactly once, without division by CP. Explicit reducer rather than DDP.
    for (_, p), m in zip(named, masters, strict=True):
        if p.grad is None:
            raise AssertionError("Missing gradient during native CP training")
        g = p.grad.detach()
        if grad_reduce == "fp32":
            reduced = g.float()
        elif grad_reduce == "bf16":
            reduced = g.clone()
        else:
            raise AssertionError(f"unknown gradient reduction dtype {grad_reduce}")
        if runtime.cp > 1:
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=runtime.group)
        m.grad = reduced.float()
    preclip_norm = torch.nn.utils.clip_grad_norm_(masters, max_norm=clip_norm, foreach=False)
    if not bool(torch.isfinite(preclip_norm).item()):
        raise AssertionError(f"nonfinite native CP FP32 master grad norm: {preclip_norm}")
    # Avoid a full ~7 GiB clone of all FP32 master weights on every step:
    # estimate AdamW's FP32 update from exp_avg/exp_avg_sq after optimizer.step.
    # Decoupled weight decay is deliberately zero in this experiment.
    optimizer.step()
    total_update_sq = torch.zeros((), dtype=torch.float32, device=runtime.device)
    with torch.no_grad():
        for (_, p), m in zip(named, masters, strict=True):
            state = optimizer.state[m]
            step = int(state["step"].item())
            bias1 = 1.0 - 0.9 ** step
            bias2 = 1.0 - 0.95 ** step
            denom = state["exp_avg_sq"].sqrt() / (bias2 ** 0.5) + 1e-8
            lr_now = optimizer.param_groups[0]["lr"]
            estimated_delta = (lr_now / bias1) * (state["exp_avg"] / denom)
            total_update_sq += estimated_delta.square().sum()
            p.copy_(m.to(dtype=p.dtype))
    model.zero_grad(set_to_none=True)
    return {
        "train_loss": loss, "supervised": supervised,
        "grad_norm": float(preclip_norm.item()),
        "estimated_master_update_norm": float(total_update_sq.sqrt().item()),
    }


def test_native_cp_multistep_real_tq_nll():
    cp = int(os.environ["WORLD_SIZE"])
    if cp not in (1, 2, 4):
        raise AssertionError("P0-B requires world size / CP in {1,2,4}")
    steps = int(os.getenv("TPR_P0_B_STEPS", "50"))
    if not 1 <= steps <= 100:
        raise AssertionError("P0-B steps must be within [1,100]")
    lr = float(os.getenv("TPR_P0_B_LR", "2e-5"))
    clip_norm = float(os.getenv("TPR_P0_B_CLIP_NORM", "1"))
    grad_reduce = os.getenv("TPR_P0_B_GRAD_REDUCE", "fp32").lower()
    if grad_reduce not in {"fp32", "bf16"} or not (lr > 0 and clip_norm > 0):
        raise AssertionError("P0-B invalid optimizer configuration")
    path = Path(os.environ["TPR_REAL_TQ_BATCH"])
    ckpt = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
    outdir = Path(os.getenv("TPR_P0_B_OUTPUT_DIR", "/tmp/tpr_p0_native_cp_b"))

    with _runtime(cp) as runtime:
        model = _model(runtime, ckpt)
        train, val, skipped, data_sha = _load_real_tq_rows(path, model.vocab_size)
        named, masters, optimizer = _make_optimizer(model, lr)
        initial_sha = _rank_consensus(named, runtime)
        train_trajectory_ids = sorted({x["trajectory_id"] for x in train})
        val_trajectory_ids = sorted({x["trajectory_id"] for x in val})
        if set(train_trajectory_ids) & set(val_trajectory_ids):
            raise AssertionError("Real TQ trajectory leaked between training and validation")
        checkpoints = {}
        schedule = [train[i % len(train)]["id"] for i in range(steps)]
        stream_sha = hashlib.sha256(
            (data_sha + ":" + str(schedule)).encode("ascii")).hexdigest()
        milestones = sorted({s for s in _MILESTONES if s <= steps} | {steps})
        events = []

        def snapshot(step):
            _rank_consensus(named, runtime)
            model.eval()
            with torch.no_grad():
                # Same held-out REAL rows for every CP; these validation losses
                # are still evaluated using each run's native CP backend.
                evaluations = [_loss(model, runtime, item, training=False)[0]
                               for item in val]
            model.train()
            if runtime.rank == 0:
                checkpoints[str(step)] = {
                    "validation_loss_mean": sum(evaluations) / len(evaluations),
                    "validation_loss_by_row": evaluations,
                    "sampled_weights": _sample_weights(named, masters),
                }

        snapshot(0)
        if runtime.rank == 0:
            print(f"P0_B cp={cp} start steps={steps} train_rows={[x['id'] for x in train]} "
                  f"val_rows={[x['id'] for x in val]} skipped={skipped} "
                  f"grad_reduce={grad_reduce} lr={lr} "
                  f"train_trajectories={train_trajectory_ids} "
                  f"val_trajectories={val_trajectory_ids} "
                  f"low_independent_trajectory_count={len(train_trajectory_ids)<64}",
                  flush=True)
        for step in range(1, steps + 1):
            data = train[(step - 1) % len(train)]
            stat = _step(model, runtime, data, named, masters, optimizer,
                         grad_reduce, clip_norm)
            stat.update(step=step, row=data["id"])
            if runtime.rank == 0:
                events.append(stat)
                print(f"P0_B cp={cp} step={step}/{steps} row={data['id']} "
                      f"loss={stat['train_loss']:.8f} grad_norm={stat['grad_norm']:.5f} "
                      f"update_norm={stat['estimated_master_update_norm']:.5f}", flush=True)
            if step in milestones:
                snapshot(step)
                # Recover results from interruption without rerunning steps.
                if runtime.rank == 0:
                    outdir.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "tag": "P0_B_NATIVE_CP_REAL_TQ_NLL_LOW_DIVERSITY",
                        "cp": cp, "steps_requested": steps,
                        "steps_completed": step, "model": "Qwen3-1.7B",
                        "checkpoint": str(ckpt), "data_path": str(path),
                        "torch": str(torch.__version__), "lr": lr,
                        "grad_reduce": grad_reduce, "clip_norm": clip_norm,
                        "dtype": "bf16", "optimizer": "FP32_MASTER_ADAMW",
                        "weight_decay": 0.0, "betas": (0.9, 0.95), "eps": 1e-8,
                        "seq_len": SEQ_LEN, "response_len": RESPONSE_LEN,
                        "train_row_ids": [x["id"] for x in train],
                        "val_row_ids": [x["id"] for x in val],
                        "train_trajectory_ids": train_trajectory_ids,
                        "val_trajectory_ids": val_trajectory_ids,
                        "skipped_rows": skipped, "data_sha": data_sha,
                        "stream_sha": stream_sha, "schedule": schedule,
                        "initial_weights_sha": initial_sha,
                        "events": events, "checkpoints": checkpoints,
                        "validation_backend": "OWN_NATIVE_CP_NOT_SHARED_CP1",
                        "gradient_aggregation": "CP_SUM_ONCE_NO_DIV_CP",
                    }
                    output = outdir / f"native_cp{cp}_train{steps}.pt"
                    temporary = output.with_suffix(".pt.tmp")
                    torch.save(payload, temporary)
                    os.replace(temporary, output)
                    print(f"P0_B cp={cp} saved step={step} file={output}", flush=True)
        dist.barrier(group=runtime.group)
