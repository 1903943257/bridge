"""P0-4: actual 50-step Full-vs-TPR Qwen3-1.7B optimizer drift, CP=1.

Two INDEPENDENT training runs consume the same real TQ 1024-token windows:
  mode=full: one GPTModel 1024-token forward/backward, no TPR context;
  mode=tpr: SegmentExecutor Push(P512)->Visit(S512)->Pop(P512).

Both use the same pretrained TPR-capable Megatron model, BF16 compute,
masked response NLL with the same global denominator, and FP32-master AdamW.
At identical steps, evaluate BOTH updated models using the SAME Full-M1024
forward (shared evaluation oracle); compare saved FP32-master update vectors.

This is real optimizer training, NOT simulated steps, NOT production PPO,
and NOT a multi-sibling tree reuse benchmark. Never infer numerical PASS from
the pytest completion status; compare actual validation/parameter metrics.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available

from .test_native_cp1_cp2_cp4_drift_npu import (
    SEQ_LEN, RESPONSE_LEN, _runtime, _weight_signature,
)
from .test_native_cp_multistep_npu import (
    _load_real_tq_rows, _make_optimizer, _sample_weights,
)
from .test_real_tq_tpr_p512s512_triplet_npu import _sample_grads

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_P0_TPR50") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="Set TPR_P0_TPR50=1 in the Ascend NPU Docker torchrun runtime",
)

MILESTONES = (0, 1, 2, 5, 10, 20, 30, 40, 50)


def _masked_lp(logits, row):
    if tuple(logits.shape[:2]) != (1, SEQ_LEN):
        raise AssertionError(f"Expected complete 1024-token logits, got {logits.shape}")
    response = row["tokens"][SEQ_LEN - RESPONSE_LEN:].to(device=logits.device)
    queries = logits[0, SEQ_LEN - RESPONSE_LEN - 1:SEQ_LEN - 1].float()
    return F.log_softmax(queries, dim=-1).gather(1, response[:, None]).squeeze(1)


def _full_loss(model, row, *, backward):
    device = next(model.parameters()).device
    tokens = row["tokens"].to(device)
    positions = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    logits = model(tokens.unsqueeze(0), positions, None)
    lp = _masked_lp(logits, row)
    mask = row["mask"].to(device=device, dtype=torch.bool)
    denominator = int(mask.sum().item())
    if denominator <= 0:
        raise AssertionError(f"No valid response targets in {row['id']}")
    loss = -lp[mask].sum() / denominator
    if backward:
        loss.backward()
    return float(loss.detach().cpu()), lp.detach().float().cpu().clone(), denominator


def _tpr_loss(model, row):
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec

    tokens, mask = row["tokens"], row["mask"].bool()
    if tokens.numel() != SEQ_LEN or mask.numel() != RESPONSE_LEN:
        raise AssertionError("Invalid real-TQ 1024-token split inputs")
    denom = int(mask.sum())
    if denom <= 0:
        raise AssertionError("Empty masked TQ response targets")
    root = SegmentSpec(0, None, tokens[:512], 0, 0)
    suffix = SegmentSpec(1, 0, tokens[512:], 512, 512)
    plan = SegmentPlan((root, suffix), root_id=0, topology_only=True)
    layers = tuple(sorted(
        module.layer_number for module in model.modules()
        if isinstance(module, TPRSelfAttention)
    ))
    if len(layers) != 28:
        raise AssertionError(f"Expected Qwen3 28 FA layers; got {len(layers)}")

    root_owned = int(mask[0].item())
    suffix_owned = int(mask[1:].sum().item())
    calls = {0: 0, 1: 0}
    captured_lp = {}

    def segment_loss(segment, logits):
        index = segment.segment_id
        if index == 0:
            # Root query 511 predicts response token 0; mask controls ownership.
            query = logits[0, 511:512].float()
            targets = tokens[512:513].to(device=logits.device)
            owned_mask = mask[:1].to(device=logits.device)
            expected = root_owned
        elif index == 1:
            # Suffix queries 0..510 predict response tokens 1..511.
            query = logits[0, :511].float()
            targets = tokens[513:1024].to(device=logits.device)
            owned_mask = mask[1:].to(device=logits.device)
            expected = suffix_owned
        else:
            raise AssertionError(f"Unexpected TPR physical segment id {index}")
        lp = F.log_softmax(query, dim=-1).gather(1, targets[:, None]).squeeze(1)
        if int(owned_mask.sum()) != expected:
            raise AssertionError(f"Segment {index} logical mask mismatch")
        calls[index] += 1
        captured_lp[index] = lp.detach().float().cpu().clone()
        return -lp[owned_mask].sum() / float(denom), {}

    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=layers,
        segment_loss_fn=segment_loss,
        segment_loss_term_counts={0: root_owned, 1: suffix_owned},
    )
    executor.push(0)
    child = executor.visit_leaf(1)
    parent = executor.pop(0)
    executor.kv_stack.assert_empty()
    if child.backward.loss_term_count + parent.loss_term_count != denom:
        raise AssertionError(f"TPR dropped or double-counted masked targets: {row['id']}")
    if calls[0] != int(root_owned > 0) or calls[1] != int(suffix_owned > 0):
        raise AssertionError(f"TPR loss callback count mismatch: {calls}")
    if suffix_owned and parent.relayed_layer_count != len(layers):
        raise AssertionError("TPR Prefix gradient was not relayed from Suffix")
    return {
        "loss": float(
            (child.backward.normalized_loss + parent.normalized_loss).detach().cpu()
        ),
        "supervised": denom,
        "prefix_grad_relays": parent.relayed_layer_count,
        # Real response masks may contain zero bits, so do not pretend there
        # are always 512 supervised targets as in single-row P0-3.
        "root_masked_target_count": root_owned,
        "suffix_masked_target_count": suffix_owned,
    }


def _optimizer_step(model, named, masters, optimizer, lr, clip_norm):
    # The current CP=1 model is an ordinary undivided replica: no rank
    # all_reduce and no artificial CP gradient scaling or FP32 GEMM.
    optimizer.zero_grad(set_to_none=True)
    for (name, param), master in zip(named, masters, strict=True):
        if param.grad is None:
            raise AssertionError(f"Missing model parameter grad: {name}")
        grad = param.grad.detach()
        if not bool(torch.isfinite(grad).all().item()):
            raise AssertionError(f"Nonfinite native BF16 gradient: {name}")
        master.grad = grad.float()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        masters, max_norm=clip_norm, foreach=False,
    )
    if not bool(torch.isfinite(grad_norm).item()):
        raise AssertionError(f"Nonfinite FP32 master grad norm: {grad_norm}")
    optimizer.step()
    with torch.no_grad():
        for (_, param), master in zip(named, masters, strict=True):
            param.copy_(master.to(dtype=param.dtype))
    model.zero_grad(set_to_none=True)
    return {
        "preclip_grad_norm": float(grad_norm.item()),
        "was_clipped": bool(grad_norm.item() > clip_norm),
        "lr": lr,
    }


def _common_full_snapshot(model, train_probe, validation, named, masters):
    # Both training branches share EXACTLY this Full-M1024 evaluation
    # backend. The only changing input is their trained parameters.
    model.eval()
    try:
        with torch.no_grad():
            probe_loss, probe_lp, probe_n = _full_loss(
                model, train_probe, backward=False,
            )
            evaluations = []
            for val in validation:
                val_loss, logprobs, supervised = _full_loss(
                    model, val, backward=False,
                )
                evaluations.append({
                    "row": val["id"], "loss": val_loss,
                    "supervised": supervised,
                    "response_mask": val["mask"].bool().cpu().clone(),
                    "response_logprobs": logprobs,
                })
    finally:
        model.train()
    return {
        "train_probe_loss": probe_loss,
        "train_probe_supervised": probe_n,
        "train_probe_response_logprobs": probe_lp,
        "train_probe_response_mask": train_probe["mask"].bool().cpu().clone(),
        "validation_loss_mean": sum(x["loss"] for x in evaluations) / len(evaluations),
        "validation_by_row": evaluations,
        "sampled_weights": _sample_weights(named, masters),
    }


def test_real_tq_full_vs_tpr_fifty_steps():
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("P0-4 is single-rank CP1; native CP2 is a separate control")
    mode = os.getenv("TPR_P0_TPR50_MODE", "").lower()
    if mode not in ("full", "tpr"):
        raise AssertionError("Set TPR_P0_TPR50_MODE=full or tpr")
    steps = int(os.getenv("TPR_P0_TPR50_STEPS", "50"))
    if not 1 <= steps <= 100:
        raise AssertionError("TPR50 steps must be within 1..100")
    lr = float(os.getenv("TPR_P0_TPR50_LR", "2e-5"))
    clip_norm = float(os.getenv("TPR_P0_TPR50_CLIP_NORM", "1.0"))
    if lr <= 0 or clip_norm <= 0:
        raise AssertionError("Nonpositive optimizer hyperparameter")
    checkpoint = Path(os.getenv(
        "TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B",
    ))
    tq_path = Path(os.environ["TPR_REAL_TQ_BATCH"])
    outdir = Path(os.getenv(
        "TPR_P0_TPR50_OUTPUT_DIR", "/tmp/tpr_p0_tpr50",
    ))
    with _runtime(1) as runtime:
        from ..correctness import test_tpr_qwen3_compatibility_npu as fixture

        fixture.QWEN_MODEL_PATH = checkpoint
        model = fixture._make_qwen_model(
            runtime.device, tpr=True, max_sequence_length=SEQ_LEN,
        )
        initial_signature = _weight_signature(model)
        train, validation, skipped, source_sha = _load_real_tq_rows(
            tq_path, model.vocab_size,
        )
        if not train or not validation:
            raise AssertionError("Training and held-out real-TQ windows are required")
        train_trajectories = sorted({r["trajectory_id"] for r in train})
        val_trajectories = sorted({r["trajectory_id"] for r in validation})
        if set(train_trajectories) & set(val_trajectories):
            raise AssertionError("Same original real-TQ trajectory leaked into validation")
        schedule = [train[i % len(train)]["id"] for i in range(steps)]
        stream_hash = hashlib.sha256(
            (source_sha + ":" + str(schedule)).encode("ascii")
        ).hexdigest()
        named, masters, optimizer = _make_optimizer(model, lr)
        checkpoints, events, grad_snapshots = {}, [], {}
        milestones = sorted({m for m in MILESTONES if m <= steps} | {steps})

        def persist(step):
            # Snapshots are keyed by STRING step IDs for stable torch-save /
            # comparator lookup; callers pass an integer training step.
            if str(step) not in checkpoints:
                raise AssertionError(f"Cannot persist missing checkpoint {step}")
            outdir.mkdir(parents=True, exist_ok=True)
            payload = {
                "tag": "P0_4_REAL_QWEN_TPR_VS_FULL_50STEP_NLL",
                "mode": mode, "steps_requested": steps,
                "steps_completed": step,
                "checkpoint": str(checkpoint), "model": "Qwen3-1.7B",
                "torch": str(torch.__version__), "dtype": "bf16",
                "data_path": str(tq_path), "data_sha": source_sha,
                "stream_sha": stream_hash, "schedule": schedule,
                "train_row_ids": [r["id"] for r in train],
                "val_row_ids": [r["id"] for r in validation],
                "train_trajectory_ids": train_trajectories,
                "val_trajectory_ids": val_trajectories,
                "skipped_rows": skipped,
                "initial_sampled_weight_sha": initial_signature,
                "objective": "response_masked_NLL_including_prefix_boundary",
                "seq_len": SEQ_LEN, "prefix_len": 512, "suffix_len": 512,
                "lr": lr, "clip_norm": clip_norm, "betas": (0.9, 0.95),
                "eps": 1e-8, "weight_decay": 0.0,
                "optimizer": "FP32_MASTER_ADAMW",
                "gradient_reduce": "NONE_CP1",
                "train_backend": (
                    "TPR_SEGMENT_EXECUTOR_PUSH_VISIT_POP"
                    if mode == "tpr" else "NO_CONTEXT_CONTROLLED_FULL"
                ),
                "eval_backend": "SAME_NO_CONTEXT_CONTROLLED_FULL_M1024",
                "events": events, "checkpoints": checkpoints,
                "preclip_grad_samples_by_step": grad_snapshots,
            }
            path = outdir / f"{mode}_train{steps}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            print(f"P0_4 mode={mode} saved={path} step={step}", flush=True)

        def snapshot(step):
            checkpoints[str(step)] = _common_full_snapshot(
                model, train[0], validation, named, masters,
            )
            persist(step)

        snapshot(0)
        print(
            f"P0_4 mode={mode} begin steps={steps} "
            f"real_windows={len(train)} "
            f"train_trajectories={train_trajectories} "
            f"val_trajectories={val_trajectories} "
            f"heldout_windows={len(validation)} "
            f"initial_weight_sha={initial_signature} "
            f"low_independent_trajectory_count={len(train_trajectories)<64}",
            flush=True,
        )

        for step in range(1, steps + 1):
            row = train[(step - 1) % len(train)]
            model.train()
            model.zero_grad(set_to_none=True)
            if mode == "full":
                loss, _lp, supervised = _full_loss(
                    model, row, backward=True,
                )
                result = {"loss": loss, "supervised": supervised,
                          "prefix_grad_relays": 0}
            else:
                result = _tpr_loss(model, row)
            if step in milestones:
                # Same real input row at corresponding step across Full/TPR.
                # At later steps this includes model/optimizer-trajectory drift.
                grad_snapshots[str(step)] = _sample_grads(model)
            result.update(_optimizer_step(
                model, named, masters, optimizer, lr, clip_norm,
            ))
            result.update(step=step, row=row["id"])
            events.append(result)
            print(
                f"P0_4 mode={mode} step={step}/{steps} "
                f"row={row['id']} NLL={result['loss']:.9f} "
                f"targets={result['supervised']} "
                f"grad_norm={result['preclip_grad_norm']:.8g} "
                f"clipped={result['was_clipped']}",
                flush=True,
            )
            if step in milestones:
                snapshot(step)

        print(
            f"P0_4 mode={mode} complete steps={steps} "
            f"train_last={events[-1]['loss']:.9f} "
            f"val0={checkpoints['0']['validation_loss_mean']:.9f} "
            f"val{steps}={checkpoints[str(steps)]['validation_loss_mean']:.9f} "
            "status=DIAGNOSTIC_ONLY_NOT_PPO_NOT_MULTI_SIBLING",
            flush=True,
        )
