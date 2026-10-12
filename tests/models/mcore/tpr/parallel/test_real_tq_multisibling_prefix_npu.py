"""P0-5: real-TQ sibling Prefix KV reuse and multi-branch backward.

Select >=2 DIFFERENT real SWE trajectories with IDENTICAL actual prompt-tail
512 tokens and nonidentical first 512 response tokens. No synthetic branches.

Full: independently forward/backward each original P512+S512 trajectory.
TPR: exactly one Push(root512), Visit(each REAL sibling leaf512), Pop(root),
     including all 28 layers of accumulated prefix dKV and fork-token logits.

Objective "nll" is real-response-masked NLL. "ppo" is STRICTLY opt-in:
recorded raw actor old_log_probs and advantages must BOTH exist and match
the original full-length real response, with no proxy values or recomputation.
The reference Qwen1.7B may differ from TQ actor provenance (e.g. Qwen8B).
A matched PPO objective replay is NOT an on-policy RL training claim.

Both modes can run 1..20 real FP32-master AdamW optimizer updates. Evaluate
both weights on the same nonsegmented Full M1024 oracle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available
from .test_native_cp1_cp2_cp4_drift_npu import _runtime, _weight_signature
from .test_native_cp_multistep_npu import _make_optimizer, _sample_weights
from .test_real_tq_tpr_full_multistep_npu import _optimizer_step
from .test_real_tq_tpr_p512s512_triplet_npu import _sample_grads

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_P0_REAL_SIBLINGS") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="Opt-in real-TQ multi-sibling Ascend test",
)


def _load_true_siblings(path, vocab_size, *, sibling_count, objective):
    from ..correctness.test_qwen3_1_7b_real_tq_ppo_npu import _rows

    if not path.is_file():
        raise FileNotFoundError(f"No real TQ file: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    # Real rollout dumps and actual pre-train actor-update captures share
    # a TensorDict payload, but only the latter can establish that PPO
    # old_log_probs/advantages came from the actor-update boundary.
    source_stage = raw.get("capture_stage", "tq_rollout")
    if source_stage not in ("tq_rollout", "actor_update_mini_batch"):
        raise AssertionError(f"Unknown real-TQ/actor capture stage: {source_stage}")
    if objective == "ppo" and source_stage != "actor_update_mini_batch":
        raise AssertionError(
            "Real PPO replay requires the ORIGINAL actor-update mini_batch "
            "capture (capture_stage=actor_update_mini_batch), NOT the earlier "
            "tq_batch.pt rollout file. Set TPR_REAL_TQ_BATCH to the actor "
            "capture path; no fabricated advantages or old_log_probs."
        )
    data = raw["tensordict"]
    keys = tuple(raw["keys"])
    required = ("prompts", "responses", "input_ids", "response_mask")
    if any(k not in data for k in required):
        raise AssertionError(f"TQ dump lacks required real trajectory fields: {required}")
    rows = {key: _rows(data, key) for key in required}
    total = len(rows["prompts"])
    if len(keys) != total or any(len(v) != total for v in rows.values()):
        raise AssertionError("TQ trajectory key/row counts differ")
    if objective == "ppo" and (
        not all(isinstance(k, str) and bool(k) for k in keys)
        or len(set(keys)) != len(keys)
    ):
        raise AssertionError(
            "Actor-update PPO replay needs distinct exact trajectory keys, "
            "not synthetic row-number identities"
        )

    old_key = next((k for k in ("old_log_probs", "old_logprobs") if k in data), None)
    adv_key = "advantages" if "advantages" in data else None
    has_ppo = bool(old_key and adv_key)
    if objective == "ppo" and not has_ppo:
        raise AssertionError(
            "REAL PPO FIELDS MISSING: original TQ must contain BOTH "
            "old_log_probs and advantages; no invented advantages, zero old "
            "logprobs, or recomputed on-policy labels are allowed"
        )
    if objective == "ppo":
        if "loss_mask" not in data or "temperature" not in data:
            raise AssertionError(
                "PPO actor update must provide original loss_mask and temperature"
            )
        loss_masks = _rows(data, "loss_mask")
        if len(loss_masks) != total:
            raise AssertionError("Actor loss_mask row count mismatch")
        for i, lm in enumerate(loss_masks):
            lm = lm.detach().cpu().bool()
            full_len = int(rows["input_ids"][i].numel())
            response_len = int(rows["responses"][i].numel())
            if lm.numel() == full_len:
                if bool(lm[:full_len - response_len].any()):
                    raise AssertionError(f"Actor PPO loss_mask supervises prompt row {i}")
                lm = lm[-response_len:]
            if not torch.equal(lm, rows["response_mask"][i].detach().cpu().bool()):
                raise AssertionError(f"Original loss_mask/response_mask mismatch row {i}")
        temperature = torch.as_tensor(data["temperature"], dtype=torch.float32).detach().cpu()
        if not bool(torch.isfinite(temperature).all()):
            raise AssertionError("Nonfinite actor PPO temperature")
        if temperature.numel() not in (1, total) or not bool((temperature > 0).all()):
            raise AssertionError("P0-5 PPO temperature must be positive scalar or per-row")
        rows["temperature"] = temperature.flatten().tolist()
        if len(rows["temperature"]) == 1:
            rows["temperature"] *= total
        rows["old"] = _rows(data, old_key)
        rows["adv"] = _rows(data, adv_key)
        if len(rows["old"]) != total or len(rows["adv"]) != total:
            raise AssertionError("Original PPO field trajectory counts disagree")

    candidates = []
    for i in range(total):
        prompt = rows["prompts"][i].detach().cpu().long()
        response = rows["responses"][i].detach().cpu().long()
        ids = rows["input_ids"][i].detach().cpu().long()
        mask = rows["response_mask"][i].detach().cpu().bool()
        if not torch.equal(ids, torch.cat((prompt, response))) or mask.numel() != response.numel():
            raise AssertionError(f"Original TQ trajectory {i} prompt/response mismatch")
        if (len(prompt) < 512 or len(response) < 512
                or not bool(mask[0]) or not bool(mask[1:512].any())):
            # The fixed P512/S512 branch witness requires a true fork
            # supervision bit and at least one true leaf supervision bit.
            continue
        cropped = torch.cat((prompt[-512:], response[:512])).contiguous()
        if int(cropped.min()) < 0 or int(cropped.max()) >= vocab_size:
            raise AssertionError(f"Real TQ row {i} token outside Qwen vocabulary")
        row = {
            "row": i, "key": str(keys[i]), "tokens": cropped,
            "prefix": prompt[-512:].contiguous(),
            "response": response[:512].contiguous(),
            "mask": mask[:512].contiguous(),
        }
        if objective == "ppo":
            old = rows["old"][i].detach().cpu().float()
            adv = rows["adv"][i].detach().cpu().float()
            if old.shape != response.shape or adv.shape != response.shape:
                raise AssertionError(f"Original PPO fields must match full response row={i}")
            if not bool(torch.isfinite(old[:512]).all() and torch.isfinite(adv[:512]).all()):
                raise AssertionError(f"Nonfinite original PPO fields row={i}")
            row["temperature"] = float(rows["temperature"][i])
            row["old"] = old[:512].clone()
            row["adv"] = adv[:512].clone()
        candidates.append(row)

    groups = {}
    for row in candidates:
        digest = hashlib.sha256(row["prefix"].numpy().tobytes()).hexdigest()
        groups.setdefault(digest, []).append(row)

    selected, best_rank = None, None
    for digest, group in groups.items():
        if len(group) < sibling_count:
            continue
        # Prefer a real branch at the exact response boundary. If all
        # candidates share response[0], a later real divergence is allowed,
        # with its first-difference position recorded explicitly.
        distinct = []
        for row in group:
            if not any(torch.equal(row["response"], old["response"]) for old in distinct):
                distinct.append(row)
        if len(distinct) < sibling_count:
            continue
        distinct.sort(key=lambda r: r["row"])
        trial = distinct[:sibling_count]
        first_difference = next(
            (j for j in range(512) if len({int(r["response"][j]) for r in trial}) > 1),
            None,
        )
        if first_difference is None:
            continue
        rank = (int(first_difference == 0), -first_difference, len(distinct))
        if best_rank is None or rank > best_rank:
            selected, best_rank = trial, rank
    if selected is None:
        raise AssertionError(
            f"No {sibling_count} DISTINCT true real response branches with "
            f"an identical last-512 prompt prefix. eligible_rows="
            f"{[r['row'] for r in candidates]}; do not fabricate a sibling pair."
        )
    for other in selected[1:]:
        if not torch.equal(other["prefix"], selected[0]["prefix"]):
            raise AssertionError("Selected sibling prefix tokens differ")

    first_divergence = next(
        j for j in range(512)
        if len({int(r["response"][j]) for r in selected}) > 1
    )
    source = hashlib.sha256()
    for row in selected:
        source.update(str(row["row"]).encode("ascii"))
        source.update(row["tokens"].numpy().tobytes())
        source.update(row["mask"].to(torch.uint8).numpy().tobytes())
        if objective == "ppo":
            source.update(str(row["temperature"]).encode("ascii"))
            source.update(row["old"].numpy().tobytes())
            source.update(row["adv"].numpy().tobytes())
    info = {
        "selected_original_rows": [r["row"] for r in selected],
        "source_stage": source_stage,
        "actor_update_capture": source_stage == "actor_update_mini_batch",
        "selected_keys": [r["key"] for r in selected],
        "first_different_response_offset": first_divergence,
        "prefix_sha256": hashlib.sha256(selected[0]["prefix"].numpy().tobytes()).hexdigest(),
        "source_sha256": source.hexdigest(),
        "raw_has_old_logprobs": bool(old_key),
        "raw_has_advantages": bool(adv_key),
        "original_response_lengths": [
            int(rows["responses"][r["row"]].numel()) for r in selected
        ],
        "supervised_per_sibling": [int(r["mask"].sum()) for r in selected],
        "p512_s512_cropped_real": True,
        "raw_dump_path": str(path),
    }
    return selected, info


def _objective(logprobs, row, normalizer, kind, clip):
    selected = row["mask"].to(device=logprobs.device, dtype=torch.bool)
    if kind == "nll":
        numerator = -logprobs[selected].sum()
        clipfrac = 0.0
    else:
        old = row["old"].to(device=logprobs.device)[selected]
        adv = row["adv"].to(device=logprobs.device)[selected]
        ratio = (logprobs[selected] - old).exp()
        surrogate1 = ratio * adv
        surrogate2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
        numerator = -torch.minimum(surrogate1, surrogate2).sum()
        clipfrac = float(((ratio - 1).abs() > clip).float().mean().detach().cpu())
    return numerator / normalizer, clipfrac


def _response_lp(logits, tokens, temperature=1.0):
    if tuple(logits.shape[:2]) != (1, 1024):
        raise AssertionError(f"Expected original full logits M1024; got {logits.shape}")
    queries = (logits[0, 511:1023] / torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)).float()
    labels = tokens[512:1024].to(device=logits.device)
    return F.log_softmax(queries, dim=-1).gather(1, labels[:, None]).squeeze(1)


def _full(model, siblings, kind, clip, *, backwards):
    total = sum(int(r["mask"].sum()) for r in siblings)
    device = next(model.parameters()).device
    losses, lps, clipfracs = [], {}, []
    ratio_outside_clip_count = 0.0
    positions = torch.arange(1024, device=device).unsqueeze(0)
    for row in siblings:
        tokens = row["tokens"].to(device=device)
        logits = model(tokens.unsqueeze(0), positions, None)
        lp = _response_lp(logits, row["tokens"], row.get("temperature", 1.0) if kind == "ppo" else 1.0)
        term, clipfrac = _objective(lp, row, total, kind, clip)
        if backwards:
            term.backward()
        losses.append(float(term.detach().cpu()))
        lps[str(row["row"])] = lp.detach().float().cpu().clone()
        clipfracs.append(clipfrac)
        ratio_outside_clip_count += clipfrac * int(row["mask"].sum())
    return {"loss": sum(losses), "logprobs": lps, "clipfracs": clipfracs,
            "ratio_outside_clip_frac": ratio_outside_clip_count / total,
            "supervised": total}


def _tpr(model, siblings, kind, clip):
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import KVPrefixState

    total = sum(int(r["mask"].sum()) for r in siblings)
    root = SegmentSpec(0, None, siblings[0]["prefix"], 0, 0)
    leaves = [SegmentSpec(i + 1, 0, row["response"], 512, 512)
              for i, row in enumerate(siblings)]
    plan = SegmentPlan((root, *leaves), root_id=0, topology_only=True)
    layers = tuple(sorted(m.layer_number for m in model.modules()
                          if isinstance(m, TPRSelfAttention)))
    if len(layers) != 28:
        raise AssertionError("P0-5 needs all 28 true TPR attention layers")
    masks = {i + 1: int(row["mask"][1:].sum()) for i, row in enumerate(siblings)}
    root_count = sum(int(row["mask"][0]) for row in siblings)
    counts = {0: root_count, **masks}
    callback_count = {i: 0 for i in counts}
    captured, root_loss, child_losses, clipfracs = {}, None, [], []
    ratio_outside_clip_count = [0.0]
    byid = {i + 1: row for i, row in enumerate(siblings)}

    def segment_loss(segment, logits):
        nonlocal root_loss
        sid = segment.segment_id
        if sid == 0:
            # A fork logit can predict DIFFERENT first response tokens of
            # distinct real trajectories: keep all sibling-owned targets.
            valid = [r for r in siblings if bool(r["mask"][0])]
            selected = logits[0, 511:512]
            targets = torch.tensor([int(r["response"][0]) for r in valid],
                                   device=logits.device, dtype=torch.long)
            per_row = []
            for row in valid:
                scale = row.get("temperature", 1.0) if kind == "ppo" else 1.0
                scaled = (selected / torch.as_tensor(scale, dtype=selected.dtype, device=selected.device)).float()
                per_row.append(F.log_softmax(scaled, dim=-1)[0, int(row["response"][0])])
            lp = torch.stack(per_row)
            for row, value in zip(valid, lp, strict=True):
                captured.setdefault(str(row["row"]), {})[0] = value.detach().float().cpu().reshape(1)
            # Every sibling gets its own actual PPO advantage and old LP.
            terms = []
            for row, value in zip(valid, lp, strict=True):
                one = dict(row)
                one["mask"] = torch.tensor([True])
                if kind == "ppo":
                    one["old"] = row["old"][:1]
                    one["adv"] = row["adv"][:1]
                term, clipfrac = _objective(value.reshape(1), one, total, kind, clip)
                terms.append(term)
                clipfracs.append(clipfrac)
                ratio_outside_clip_count[0] += clipfrac
            objective = torch.stack(terms).sum()
            root_loss = float(objective.detach().cpu())
        else:
            row = byid[sid]
            selected = logits[0, :511]
            scale = row.get("temperature", 1.0) if kind == "ppo" else 1.0
            selected = (selected / torch.as_tensor(scale, dtype=selected.dtype, device=selected.device)).float()
            targets = row["response"][1:].to(device=logits.device)
            lp = F.log_softmax(selected, dim=-1).gather(1, targets[:, None]).squeeze(1)
            captured.setdefault(str(row["row"]), {})[1] = lp.detach().float().cpu().clone()
            one = dict(row)
            one["mask"] = row["mask"][1:]
            if kind == "ppo":
                one["old"] = row["old"][1:]
                one["adv"] = row["adv"][1:]
            objective, clipfrac = _objective(lp, one, total, kind, clip)
            child_losses.append(float(objective.detach().cpu()))
            clipfracs.append(clipfrac)
            ratio_outside_clip_count[0] += clipfrac * int(one["mask"].sum())
        callback_count[sid] += 1
        return objective, {}

    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=layers,
        segment_loss_fn=segment_loss, segment_loss_term_counts=counts,
    )
    executor.push(0)

    # Instrument the EXISTING PrefixState.accumulate call without changing
    # its arithmetic. Capture BF16 contributions at 32 deterministic
    # positions per layer and verify exactly one accumulation per sibling.
    original_accumulate = KVPrefixState.accumulate_layer_gradients
    calls = {i + 1: {} for i in range(len(siblings))}
    active = [None]
    def audited_accumulate(state, layer_number, key_grad, value_grad):
        if state.segment_id == 0 and active[0] is not None:
            witness = []
            for grad in (key_grad, value_grad):
                flat = grad.detach().reshape(-1)
                ids = torch.linspace(0, flat.numel()-1, 32, device=flat.device).long()
                witness.append(flat.index_select(0, ids).detach().clone())
            if layer_number in calls[active[0]]:
                raise AssertionError(f"Duplicate Prefix KV accumulation on leaf {active[0]} layer {layer_number}")
            calls[active[0]][layer_number] = tuple(witness)
        return original_accumulate(state, layer_number, key_grad, value_grad)

    with patch.object(KVPrefixState, "accumulate_layer_gradients", audited_accumulate):
        for sid in range(1, len(siblings) + 1):
            active[0] = sid
            result = executor.visit_leaf(sid)
            if result.backward.loss_term_count != counts[sid]:
                raise AssertionError(f"Sibling {sid} lost mask-supervised targets")
            if tuple(sorted(calls[sid])) != layers:
                raise AssertionError(
                    f"Sibling {sid} did not contribute exactly once to all 28 Prefix KV layers: "
                    f"got={list(calls[sid])}"
                )
        active[0] = None

    actual = executor.kv_stack.top().gradients
    if tuple(actual) != layers:
        raise AssertionError("Shared Prefix KV gradient buffer incomplete")
    for layer in layers:
        key, value = actual[layer]
        for kind_idx, grad in enumerate((key, value)):
            flat = grad.detach().reshape(-1)
            ids = torch.linspace(0, flat.numel()-1, 32, device=flat.device).long()
            observed = flat.index_select(0, ids)
            expected = calls[1][layer][kind_idx].clone()
            for sid in range(2, len(siblings) + 1):
                expected.add_(calls[sid][layer][kind_idx])
            if not torch.equal(observed, expected):
                raise AssertionError(
                    f"Actual BF16 accumulated Prefix dKV differs from exactly-once "
                    f"per-sibling sum at layer {layer} K/V={kind_idx}"
                )
    del calls
    root_result = executor.pop(0)
    executor.kv_stack.assert_empty()
    if root_result.relayed_layer_count != 28:
        raise AssertionError("Root did not backward all 28 accumulated Prefix dKV")
    if sum(counts.values()) != total or root_result.loss_term_count != root_count:
        raise AssertionError("Real fork-token supervised ownership mismatch")
    for sid, owned in counts.items():
        if callback_count[sid] != int(owned > 0):
            raise AssertionError(
                f"Segment {sid} loss callback count {callback_count[sid]} != "
                f"expected {int(owned > 0)}"
            )
    lps = {}
    for row in siblings:
        rid = str(row["row"])
        # If response[0] is masked out, no prefix logprob was captured.
        # For comparison, obtain its logprob from a post-backward no-grad
        # root probe only when needed; the masked training objective did not
        # depend on that element. Default requires first token valid.
        if 0 not in captured.get(rid, {}):
            raise AssertionError("P0-5 requires mask[0]=True to compare fork logits")
        lps[rid] = torch.cat((captured[rid][0], captured[rid][1]))
        if lps[rid].numel() != 512:
            raise AssertionError(f"Sibling {rid} response logits not fully captured")
    return {
        "loss": float(sum(child_losses) + (root_loss or 0.0)),
        "logprobs": lps, "clipfracs": clipfracs,
        "ratio_outside_clip_frac": ratio_outside_clip_count[0] / total,
        "supervised": total,
        "root_forward_count": 1, "leaf_backward_count": len(siblings),
        "root_backward_count": 1, "prefix_relay_layers": 28,
        "audited_prefix_kv_sibling_accumulations": 28 * len(siblings),
        "audited_prefix_kv_sample_exact": True,
        "fork_loss_count": root_count,
    }


def test_true_real_tq_multisibling_branch():
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("P0-5 is a CP1 root/leaf gradient control")
    mode = os.getenv("TPR_P0_BRANCH_MODE", "")
    if mode not in ("full", "tpr"):
        raise AssertionError("P0-5 needs TPR_P0_BRANCH_MODE=full or tpr")
    objective = os.getenv("TPR_P0_BRANCH_OBJECTIVE", "nll").lower()
    if objective not in ("nll", "ppo"):
        raise AssertionError("P0-5 objective must be nll or STRICT real-field ppo")
    siblings_requested = int(os.getenv("TPR_P0_BRANCH_SIBLINGS", "2"))
    if siblings_requested not in (2, 3):
        raise AssertionError("Select 2 or 3 true real siblings only")
    steps = int(os.getenv("TPR_P0_BRANCH_STEPS", "1"))
    if not 1 <= steps <= 20:
        raise AssertionError("P0-5 supports 1..20 optimizer steps")
    clip = float(os.getenv("TPR_P0_BRANCH_PPO_CLIP", "0.2"))
    if not (0 < clip < 1):
        raise AssertionError("PPO clip ratio must be in (0,1)")
    lr = float(os.getenv("TPR_P0_BRANCH_LR", "2e-5"))
    grad_clip = float(os.getenv("TPR_P0_BRANCH_GRAD_CLIP", "1"))
    checkpoint = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
    tq = Path(os.environ["TPR_REAL_TQ_BATCH"])
    outdir = Path(os.getenv("TPR_P0_BRANCH_OUTPUT_DIR", "/tmp/tpr_p0_real_siblings"))
    with _runtime(1) as runtime:
        from ..correctness import test_tpr_qwen3_compatibility_npu as fixture
        fixture.QWEN_MODEL_PATH = checkpoint
        model = fixture._make_qwen_model(runtime.device, tpr=True, max_sequence_length=1024)
        if model.config.context_parallel_size != 1:
            raise AssertionError("P0-5 only operates at native CP1")
        signature = _weight_signature(model)
        siblings, info = _load_true_siblings(
            tq, model.vocab_size,
            sibling_count=siblings_requested, objective=objective,
        )
        # Explicitly avoid a partial-accuracy claim if a sibling's
        # fork-token contribution is not in the recorded loss mask.
        if any(not bool(row["mask"][0]) for row in siblings):
            raise AssertionError(
                "P0-5 requires the first real response token to be "
                "supervised for every selected sibling; choose another group."
            )
        named, masters, opt = _make_optimizer(model, lr)
        metadata = {
            "tag": "P0_5_REAL_TQ_SHARED_PREFIX_SIBLING_NLL_OR_RECORDED_PPO",
            "mode": mode, "objective": objective, "ppo_clip": clip,
            "checkpoint": str(checkpoint), "initial_weight_sig": signature,
            "torch": str(torch.__version__),
            "steps": steps, "num_siblings": len(siblings),
            "lr": lr, "grad_clip": grad_clip,
            "optimizer": "FP32_MASTER_ADAMW",
            "same_evaluation_backend": "NO_TPR_CONTEXT_FULL_M1024",
            "source": info, "events": [], "checkpoints": {},
        }
        captured_mask = {str(r["row"]): r["mask"].clone() for r in siblings}

        def snapshot(step):
            model.eval()
            try:
                with torch.no_grad():
                    evidence = _full(model, siblings, "nll", clip, backwards=False)
            finally:
                model.train()
            metadata["checkpoints"][str(step)] = {
                "common_eval_nll": evidence["loss"],
                "common_eval_logprobs": evidence["logprobs"],
                "common_eval_masks": captured_mask,
                "sampled_weights": _sample_weights(named, masters),
            }
            outdir.mkdir(parents=True, exist_ok=True)
            output = outdir / f"{objective}_{mode}_siblings{len(siblings)}_steps{steps}.pt"
            torch.save(metadata, output.with_suffix(".pt.tmp"))
            os.replace(output.with_suffix(".pt.tmp"), output)
            print(f"P0_5 {mode} snapshot step={step} path={output}", flush=True)

        snapshot(0)
        print(
            f"P0_5 START mode={mode} objective={objective} "
            f"true_siblings={info['selected_original_rows']} "
            f"first_response_divergence={info['first_different_response_offset']} "
            f"supervised={info['supervised_per_sibling']} "
            f"raw_has_advantages={info['raw_has_advantages']} "
            f"raw_has_old_logprobs={info['raw_has_old_logprobs']} "
            f"origin_actor_model_NOT_VERIFIED=True",
            flush=True,
        )
        for step in range(1, steps + 1):
            model.train()
            model.zero_grad(set_to_none=True)
            evidence = (_tpr(model, siblings, objective, clip) if mode == "tpr"
                        else _full(model, siblings, objective, clip, backwards=True))
            grads = _sample_grads(model)
            opt_stats = _optimizer_step(model, named, masters, opt, lr, grad_clip)
            metadata["events"].append({
                "step": step, "training_loss": evidence["loss"],
                "response_logprobs": evidence["logprobs"],
                "training_grad_samples": grads,
                "training_clipfracs": evidence["clipfracs"],
                "ratio_outside_clip_frac": evidence["ratio_outside_clip_frac"],
                "supervised": evidence["supervised"],
                "tpr_execution": {
                    k: evidence[k] for k in (
                        "root_forward_count", "leaf_backward_count",
                        "root_backward_count", "prefix_relay_layers",
                        "audited_prefix_kv_sibling_accumulations",
                        "audited_prefix_kv_sample_exact", "fork_loss_count"
                    ) if k in evidence
                },
                "optimizer": opt_stats,
            })
            if step in (1, 2, 5, 10, 20, steps):
                snapshot(step)
            print(
                f"P0_5 mode={mode} step={step} train_loss={evidence['loss']:.9g} "
                f"grad_norm={opt_stats['preclip_grad_norm']:.9g} "
                f"prefix_relay_layers={evidence.get('prefix_relay_layers',0)} "
                f"real_siblings={info['selected_original_rows']}",
                flush=True,
            )
        print(f"P0_5 DONE mode={mode} objective={objective} steps={steps} "
              "status=DIAGNOSTIC_ONLY", flush=True)
