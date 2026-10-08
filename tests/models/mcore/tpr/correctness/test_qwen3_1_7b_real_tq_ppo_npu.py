"""Phase-5 real-checkpoint numerical gate: Qwen3-1.7B + real SWE TQ tokens.

NO synthetic model. The original 8 recorded trajectories are loaded from the
TQ dump. The default fixture keeps ALL 8 real unpadded trajectories, including
multi-level branches. Shortened windows are explicit debug-only mode.

When the TQ dump has no actor advantages, deterministic *probe coefficients*
are used ONLY for a numerical gradient equivalence gate (never reported as RL
advantages). For true end-to-end PPO, capture and load the actor-update batch.
"""

from __future__ import annotations

import gc
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Qwen3-1.7B TPR PPO correctness requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_PPO") != "1",
    reason="Set TPR_RUN_QWEN17_PPO=1 to run real Qwen3-1.7B PPO correctness",
)

_DEFAULT_TQ = Path(
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt"
)


class _VanillaPPOConfig:
    policy_loss = {"loss_mode": "vanilla"}
    loss_agg_mode = "token-mean"
    clip_ratio = 0.2
    clip_ratio_low = None
    clip_ratio_high = None
    entropy_coeff = 0.0
    use_kl_loss = False
    loss_scale_factor = None

    def __init__(self):
        self.global_batch_info = {}

    def get(self, key, default=None):
        return getattr(self, key, default)


def _rows(batch, field):
    value = batch[field]
    return list(value) if isinstance(value, (list, tuple)) else list(value.unbind())


def _as_jagged(rows):
    """Keep *all* unpadded real tokens from every variable-length TQ row."""
    return torch.nested.as_nested_tensor(
        [row.detach().cpu().contiguous() for row in rows], layout=torch.jagged
    )


def _load_real_tq_probe(
    *, prompt_length: int | None = None, response_length: int | None = None
):
    """Full 8-row real TQ by default; optional opt-in cropped-real-data smoke.

    Cropped mode is enabled ONLY when both lengths are explicitly specified.
    In full mode the original token sequence, prompt boundary, response mask,
    and compressed trie are preserved exactly.
    """
    path = Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ)))
    if not path.is_file():
        pytest.fail(f"real TQ dump is required (no synthetic fallback): {path}")
    if (prompt_length is None) != (response_length is None):
        pytest.fail("Set BOTH TPR_QWEN17_PPO_PROMPT and _RESPONSE, or neither")
    if prompt_length is not None and (prompt_length <= 0 or response_length <= 0):
        pytest.fail("Explicit cropped-real-data window lengths must be positive")

    dump = torch.load(path, map_location="cpu", weights_only=False)
    original = dump["tensordict"]
    original_keys = tuple(dump["keys"])
    full_rows = _rows(original, "input_ids")
    prompt_rows = _rows(original, "prompts")
    response_rows = _rows(original, "responses")
    response_masks = _rows(original, "response_mask")
    if len(full_rows) != 8 or len(original_keys) != 8:
        pytest.fail(f"expected recorded 8 SWE trajectories; found {len(full_rows)}")

    token_rows, prompt_out, response_out, masks, advantage_rows = [], [], [], [], []
    source_advantages = _rows(original, "advantages") if "advantages" in original else None
    if source_advantages is None:
        print("PHASE5: TQ has no actor advantages. Using diagnostic signed "
              "(+1.0/-0.5) coefficients ONLY for native-vs-TPR gradient equivalence.")

    for row, (full, prompt, resp, mask) in enumerate(zip(
        full_rows, prompt_rows, response_rows, response_masks, strict=True
    )):
        if not torch.equal(full[:prompt.numel()], prompt):
            pytest.fail(f"real TQ row {row} has an inconsistent prompt boundary")
        if not torch.equal(full[prompt.numel():], resp):
            pytest.fail(f"real TQ row {row} has an inconsistent response suffix")
        if mask.numel() != resp.numel():
            pytest.fail(f"real TQ row {row} response/mask lengths disagree")

        if prompt_length is None:
            p, r = prompt.clone().long(), resp.clone().long()
            selected_mask = mask.clone().bool()
        else:
            if prompt.numel() < prompt_length or resp.numel() < response_length:
                pytest.fail(f"TQ row {row} too short for cropped REAL token window")
            p, r = prompt[-prompt_length:].clone().long(), resp[:response_length].clone().long()
            selected_mask = mask[:response_length].clone().bool()

        token_rows.append(torch.cat((p, r)))
        prompt_out.append(p)
        response_out.append(r)
        masks.append(selected_mask)

        if source_advantages is None:
            # These are NOT rollout advantages. They ensure nonzero, signed
            # gradients in a mathematical equivalence test on REAL inputs.
            arange = torch.arange(len(r), dtype=torch.long)
            # Keep both advantage signs without an exactly cancelling
            # zero native objective. A near-zero scalar hides meaningful
            # relative-loss drift and makes PPO clipping harder to interpret.
            a = torch.where((arange + row) % 2 == 0, 1.0, -0.5).float()
        else:
            a = source_advantages[row][:len(r)].float().clone()
            if a.numel() != len(r):
                pytest.fail(f"TQ row {row} advantage length does not match response")
        advantage_rows.append(a)

    count = sum(int(mask.sum()) for mask in masks)
    if count == 0:
        pytest.fail("no supervised response tokens in real TQ fixture")

    # A real actor-update dump should supply a loss_mask, whose response
    # portion must agree with this vanilla token-mean PPO test.
    if "loss_mask" in original and prompt_length is None:
        for row, (original_loss_mask, mask, full) in enumerate(zip(
            _rows(original, "loss_mask"), masks, token_rows, strict=True
        )):
            if original_loss_mask.numel() == full.numel():
                original_loss_mask = original_loss_mask[-mask.numel():]
            if not torch.equal(original_loss_mask.bool(), mask.bool()):
                pytest.fail(
                    f"real TQ row {row}: loss_mask does not match response_mask; "
                    "cannot safely reuse a PPO token-mean denominator"
                )

    batch = TensorDict({
        "input_ids": _as_jagged(token_rows),
        "prompts": _as_jagged(prompt_out),
        "responses": _as_jagged(response_out),
        "attention_mask": _as_jagged([
            torch.ones_like(row) for row in token_rows
        ]),
        "response_mask": _as_jagged(masks),
        "loss_mask": _as_jagged([mask.clone() for mask in masks]),
        "advantages": _as_jagged(advantage_rows),
        "old_log_probs": _as_jagged([
            torch.zeros_like(mask, dtype=torch.float32) for mask in masks
        ]),
        "temperature": torch.ones(8, dtype=torch.float32),
    }, batch_size=[8])
    tu.assign_non_tensor(
        batch,
        tpr_trajectory_keys=original_keys,
        batch_num_tokens=count,
        global_batch_size=8,
        dp_size=1,
    )
    mode = "FULL_REAL_TQ" if prompt_length is None else "CROPPED_REAL_TQ_DEBUG"
    print(
        f"PHASE5 DATA MODE={mode}; rows=8; logical_tokens="
        f"{sum(len(row) for row in token_rows)}; supervised_tokens={count}; "
        f"lengths={[len(row) for row in token_rows]}"
    )
    return batch


def _native_response_logprobs(model, token_row, *, prompt_length, temperature):
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    dev = token_row.device
    positions = torch.arange(token_row.numel(), device=dev).unsqueeze(0)
    logits = model(
        input_ids=token_row.unsqueeze(0),
        position_ids=positions,
        attention_mask=None,
    )
    assert logits.ndim == 3 and tuple(logits.shape[:2]) == (1, token_row.numel())
    # VERL's shift convention: query at position t-1 predicts input_ids[t].
    shifted_labels = torch.cat((token_row[1:], token_row[-1:]))
    logits = logits[0] / torch.as_tensor(temperature, dtype=logits.dtype, device=dev)
    log_probs = vocab_parallel_log_probs_from_logits(logits, shifted_labels)
    return log_probs, log_probs[prompt_length-1:-1]


def _selected_gradient_snapshot(model):
    output = {}
    for name, parameter in model.named_parameters():
        # Sample first/middle/last attention, normalization and MLP matrices.
        sample = any(f"decoder.layers.{layer}." in name for layer in (0, 13, 27))
        if not sample and not any(term in name for term in (
            "embedding.word_embeddings", "final_layernorm", "output_layer",
        )):
            continue
        grad = parameter.grad
        if grad is None:
            raise AssertionError(f"missing gradient for real model parameter {name}")
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"non-finite gradient in {name}")
        output[name] = grad.detach().float().cpu().clone()
    if not output:
        raise AssertionError("could not select any Qwen3-1.7B parameter gradients")
    return output


def _compare_grads(reference, actual):
    assert reference.keys() == actual.keys()
    square_diff = square_ref = dot = square_actual = 0.0
    for name, ref in reference.items():
        got = actual[name]
        delta = got - ref
        square_diff += torch.sum(delta.square()).item()
        square_ref += torch.sum(ref.square()).item()
        square_actual += torch.sum(got.square()).item()
        dot += torch.sum(got * ref).item()
        rel_l2 = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref).clamp_min(1e-12))
        print(f"grad {name}: relative_l2={rel_l2:.6f}")
    relative_l2 = (square_diff / max(square_ref, 1e-24)) ** .5
    cosine = dot / max((square_ref * square_actual) ** .5, 1e-24)
    print(f"Qwen3-1.7B PPO grad: global_relative_l2={relative_l2:.6f}, cosine={cosine:.8f}")
    assert relative_l2 < .08 and cosine > .997



def _logprob_diagnostics(batch, captured, reference_lp):
    """Inspect exact real-token ownership without modifying acceptance criteria."""
    from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans

    keys = tu.get_non_tensor_data(batch, "tpr_trajectory_keys", default=None)
    if keys is None:
        raise AssertionError("Real TQ diagnostic requires original trajectory keys")
    forest = build_tree_execution_plans(tuple(keys), batch)
    owner = {}
    for tree in forest.trees:
        for ref in tree.objective_refs:
            segment = tree.segment_plan.get(ref.segment_id)
            key = (ref.sample_row, ref.response_offset)
            if key in owner:
                raise AssertionError(f"Duplicate real logical token in Forest: {key}")
            owner[key] = {
                "tree": tree.tree.key,
                "segment_id": ref.segment_id,
                "segment_start": segment.position_start,
                "segment_end": segment.position_end,
                "query_abs": segment.position_start + ref.query_offset,
                "target_id": ref.target_token_id,
                "root": ref.segment_id == tree.segment_plan.root_id,
            }

    all_keys = sorted(owner)
    assert set(captured) == set(owner), "Captured TPR tokens != real trajectory objective refs"
    expected = torch.tensor([float(reference_lp[row][offset]) for row, offset in all_keys])
    actual = torch.tensor([captured[key] for key in all_keys])
    diff = (actual - expected).abs()
    threshold = .2 + .02 * expected.abs()
    bad = diff > threshold
    relative_l2 = torch.linalg.vector_norm(actual - expected) / (
        torch.linalg.vector_norm(expected).clamp_min(1e-12)
    )
    rms = torch.sqrt(torch.mean(diff.square()))
    # PPO optimizes exp(new_lp - old_lp), not the relative error in a
    # negative logprob number. A 0.37 logprob shift at -16.6 may be
    # "within_tol" here yet already change the clipped PPO branch.
    ratio = torch.exp(actual - expected)
    outside_clip = (ratio < 0.8) | (ratio > 1.2)
    print(
        "TPR PPO RATIO DIAG (using Native checkpoint logprob as old policy): "
        f"outside_clip_0.8_1.2={int(outside_clip.sum())}/{len(all_keys)} "
        f"min_ratio={ratio.min().item():.6g} "
        f"max_ratio={ratio.max().item():.6g}"
    )
    print(
        "TPR LOGPROB DIAG:"
        f" total={len(all_keys)} bad={int(bad.sum())} "
        f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g} "
        f"rms={rms.item():.6g} rel_l2={relative_l2.item():.6g}"
    )
    for row in range(len(batch["input_ids"])):
        indices = [i for i, (r, _) in enumerate(all_keys) if r == row]
        d = diff[indices]
        print(
            f"  row={row} tokens={len(indices)} "
            f"mismatch={int(bad[indices].sum())} max_abs={d.max().item():.6g}"
        )
    for root in (True, False):
        indices = [i for i, key in enumerate(all_keys) if owner[key]["root"] == root]
        if indices:
            print(
                f"  owner={'root' if root else 'nonroot'} "
                f"tokens={len(indices)} mismatches={int(bad[indices].sum())} "
                f"max_abs={diff[indices].max().item():.6g}"
            )
    for i in torch.argsort(diff, descending=True)[:min(20, len(all_keys))].tolist():
        row, offset = all_keys[i]
        info = owner[(row, offset)]
        print(
            f"  token row={row} response={offset} query_abs={info['query_abs']} "
            f"target={info['target_id']} tree={info['tree']} "
            f"segment={info['segment_id']}[{info['segment_start']}:{info['segment_end']}] "
            f"native={expected[i].item():.6g} tpr={actual[i].item():.6g} "
            f"abs_diff={diff[i].item():.6g} "
            f"ratio={ratio[i].item():.6g} "
            f"ratio_clipped={bool(outside_clip[i])} "
            f"threshold={threshold[i].item():.6g}"
            + (" FAIL" if bad[i] else " within_tol")
        )
    return all_keys, expected, actual, bad, outside_clip


def _probe_tpr_model_native_forward(model, batch, reference_lp, *, max_length=256):
    """If small enough, separate model/checkpoint drift from forest shape drift.

    Same real checkpoint, same TPR model, but its context-free native
    SelfAttention.forward, not the Push/Visit/Pop path. Diagnostic only,
    never a substitute for the strict Forest gate.
    """
    longest = max(len(row) for row in _rows(batch, "input_ids"))
    if longest > max_length:
        print(
            f"TPR NATIVE PATH PROBE skipped: longest={longest} > {max_length}; "
            "avoid full-context OOM"
        )
        return
    differences = []
    with torch.no_grad():
        for row in range(len(batch["input_ids"])):
            ids = batch["input_ids"][row].to(next(model.parameters()).device)
            _, lp = _native_response_logprobs(
                model, ids, prompt_length=len(batch["prompts"][row]), temperature=1.0
            )
            mask = batch["response_mask"][row].to(bool).cpu()
            expected = reference_lp[row][mask].float().cpu()
            current = lp.detach().float().cpu()[mask]
            differences.append((current - expected).abs())
    diffs = torch.cat(differences)
    print(
        "TPR MODEL NATIVE-FORWARD vs REFERENCE MODEL NATIVE-FORWARD: "
        f"max_abs={diffs.max().item():.6g} "
        f"mean_abs={diffs.mean().item():.6g}"
    )
    if diffs.max().item() > 0.2:
        print(
            "DIAG: model/checkpoint/forward-path mismatch may exist even "
            "WITHOUT Tree execution; inspect spec, weights and reproducibility."
        )
    else:
        print(
            "DIAG: model native paths agree within 0.2; remaining difference "
            "is concentrated in TPR segmented execution or kernel shapes."
        )



def _probe_whole_segment_tpr_forward(model, batch, reference_lp, *, row: int):
    """Same TPRSelfAttention/kernel, but one physical node; isolate split effects.

    Run only for the worst row of the cropped real-TQ smoke. Single-segment
    TPR uses identical token IDs and absolute position IDs as native.
    """
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.segment_plan import (
        SegmentLossTerm,
        SegmentPlan,
        SegmentSpec,
    )
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    tokens = batch["input_ids"][row].detach().cpu().long()
    if tokens.numel() > 256:
        print(f"ONE-SEGMENT TPR probe skipped: row={row} length={tokens.numel()}")
        return
    segment = SegmentSpec(
        segment_id=0, parent_id=None, token_ids=tokens,
        position_start=0, prefix_length=0,
        # Only to satisfy the legacy CE constructor; _forward never uses CE.
        loss_terms=(SegmentLossTerm(0, int(tokens[1])),),
    )
    plan = SegmentPlan((segment,), root_id=0)
    layers = tuple(sorted(
        layer.layer_number for layer in model.modules()
        if isinstance(layer, TPRSelfAttention)
    ))
    executor = SegmentExecutor(model, plan, expected_layer_numbers=layers)
    with torch.no_grad():
        _, logits = executor._forward(
            segment, past_key_values={}, no_grad=True
        )
        prompt_len = len(batch["prompts"][row])
        targets = tokens[prompt_len:].to(logits.device)
        query_logits = logits[0, prompt_len-1 : -1, :]
        assert len(query_logits) == len(targets)
        current = vocab_parallel_log_probs_from_logits(
            query_logits, targets
        ).detach().float().cpu()
    mask = batch["response_mask"][row].bool().cpu()
    difference = (current[mask] - reference_lp[row].float().cpu()[mask]).abs()
    print(
        f"WHOLE-SEGMENT TPR vs Native, row={row}: "
        f"max_abs={difference.max().item():.6g} "
        f"mean_abs={difference.mean().item():.6g}. "
        "Compare with Forest drift on the same row to separate shape "
        "sensitivity from segment/KV errors."
    )



def _attention_trace_hooks(model, *, monitored_positions, owner_segments=None):
    """Capture real Qwen layer attention inputs/outputs at selected absolute tokens.

    Native: full sequence at absolute index. TPR: match the exact owning
    physical segment's absolute start and length; reject sibling collisions.
    This is a diagnostic-only hook, not a forward modification.
    """
    from verl.models.mcore.tpr.context import get_tpr_attention_context

    traces = {}
    handles = []
    for layer in model.decoder.layers:
        attention = layer.self_attention
        layer_number = attention.layer_number

        def capture(module, args, output, layer_number=layer_number):
            ctx = get_tpr_attention_context()
            hidden = args[0]
            projected = output[0] if isinstance(output, tuple) else output
            if not isinstance(projected, torch.Tensor):
                return
            for absolute in monitored_positions:
                if owner_segments is None:
                    if ctx is not None:
                        continue
                    local = absolute
                else:
                    if ctx is None or absolute not in owner_segments:
                        continue
                    segment = owner_segments[absolute]
                    if (ctx.prefix_length != segment.position_start or
                            ctx.suffix_length != segment.length):
                        continue
                    local = absolute - ctx.prefix_length
                if not 0 <= local < hidden.shape[0] or local >= projected.shape[0]:
                    continue
                traces[(layer_number, absolute)] = (
                    hidden[local].detach().float().cpu().clone(),
                    projected[local].detach().float().cpu().clone(),
                )

        handles.append(attention.register_forward_hook(capture))

        # Q/K/V GEMM and attention-core output are distinct possible sources
        # of bf16 shape sensitivity. Instrument the two native Megatron
        # projection modules (no changes to their forward implementation).
        def capture_projection(module, args, output, layer_number=layer_number, stage=None):
            ctx = get_tpr_attention_context()
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0]
            y = output[0] if isinstance(output, tuple) else output
            if not isinstance(y, torch.Tensor) or x.ndim != 3 or y.ndim != 3:
                return
            for absolute in monitored_positions:
                if owner_segments is None:
                    if ctx is not None:
                        continue
                    local = absolute
                else:
                    if ctx is None or absolute not in owner_segments:
                        continue
                    owner = owner_segments[absolute]
                    if ctx.prefix_length != owner.position_start or ctx.suffix_length != owner.length:
                        continue
                    local = absolute - ctx.prefix_length
                if not 0 <= local < x.shape[0] or local >= y.shape[0]:
                    continue
                traces[(layer_number, absolute, stage)] = (
                    x[local].detach().float().cpu().clone(),
                    y[local].detach().float().cpu().clone(),
                )

        from functools import partial as _partial
        handles.append(
            attention.linear_qkv.register_forward_hook(
                _partial(capture_projection, stage="qkv")
            )
        )
        handles.append(
            attention.linear_proj.register_forward_hook(
                _partial(capture_projection, stage="proj")
            )
        )
    return traces, handles


def _print_attention_layer_drift(reference, segmented, positions):
    """Output relative L2 by layer, focusing on *first* deviation."""
    for absolute in positions:
        print(f"ATTENTION LAYER TRACE: query_abs={absolute}")
        previous = None
        first = None
        matched = 0
        for layer in sorted({k[0] for k in reference}):
            key = (layer, absolute)
            if key not in reference or key not in segmented:
                continue
            matched += 1
            (native_in, native_out) = reference[key]
            (tpr_in, tpr_out) = segmented[key]
            def metric(a, b):
                d = (a - b).float()
                return (
                    float(torch.linalg.vector_norm(d) /
                          torch.linalg.vector_norm(a).clamp_min(1e-12)),
                    float(d.abs().max()),
                )
            in_rel, in_max = metric(native_in, tpr_in)
            out_rel, out_max = metric(native_out, tpr_out)
            # Report every layer: a small first-layer BF16 discrepancy can
            # amplify greatly through a 28-layer real checkpoint.
            extra = []
            for stage, input_name, output_name in (
                ("qkv", "qkv_gemm_input", "qkv_gemm_output"),
                ("proj", "core_attention_output", "o_proj_output"),
            ):
                subkey = (layer, absolute, stage)
                if subkey not in reference or subkey not in segmented:
                    continue
                n_x, n_y = reference[subkey]
                t_x, t_y = segmented[subkey]
                sub_in_rel, sub_in_max = metric(n_x, t_x)
                sub_out_rel, sub_out_max = metric(n_y, t_y)
                extra.append(
                    f"{input_name}_rel={sub_in_rel:.6g} "
                    f"{input_name}_max={sub_in_max:.6g} "
                    f"{output_name}_rel={sub_out_rel:.6g} "
                    f"{output_name}_max={sub_out_max:.6g}"
                )
            print(
                f"  layer={layer:02d} "
                f"attention_input_rel_l2={in_rel:.6g} max_abs={in_max:.6g} "
                f"attention_output_rel_l2={out_rel:.6g} max_abs={out_max:.6g} "
                + " | ".join(extra)
            )
            if first is None and (in_max > 0 or out_max > 0):
                first = (layer, in_max, out_max)
            previous = (layer, in_max, out_max)
        print(
            f"ATTENTION TRACE SUMMARY query_abs={absolute}: "
            f"matched_layers={matched} first_deviation={first} last={previous}"
        )
        if not matched:
            print(
                "ATTENTION TRACE unavailable for this position: "
                "check original real-TQ tree ownership and hook scope."
            )


def _get_trace_owner_segments(batch, *, row=0, response_offsets=(6, 13)):
    """Exact row-path mapping, never assume branch IDs based on one run log."""
    from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
    keys = tu.get_non_tensor_data(batch, key="tpr_trajectory_keys", default=None)
    forest = build_tree_execution_plans(tuple(keys), batch)
    offsets = set(response_offsets)
    segments = {}
    for tree in forest.trees:
        for ref in tree.objective_refs:
            if ref.sample_row == row and ref.response_offset in offsets:
                segment = tree.segment_plan.get(ref.segment_id)
                absolute = segment.position_start + ref.query_offset
                segments[absolute] = segment
    if len(segments) != len(offsets):
        raise AssertionError(
            f"layer trace requires unique real query owners for {sorted(offsets)}"
        )
    return segments



def _native_truncation_controls(model, batch, original_logprobs, full_trace, owners):
    """Native-only causal control: same weights and tokens, different sequence S.

    Causal invariance: changing total S from 128 to the real radix node end
    (70 or 94) cannot mathematically affect logits at q < S. If this *alone*
    causes comparable error, the NPU native kernel's shape-dependent rounding
    is a confounder, not proof of incorrect TPR Prefix KV or objective mapping.
    """
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    if not owners:
        return
    row = 0
    tokens = batch["input_ids"][row].detach().cpu()
    device = next(model.parameters()).device
    full_length = tokens.numel()
    if full_length > 256:
        print("NATIVE TRUNCATION CONTROL skipped for full-length real TQ (OOM guard)")
        return
    grouped = {}
    for absolute, segment in sorted(owners.items()):
        if absolute + 1 >= full_length:
            continue
        grouped.setdefault(segment.position_end, []).append(absolute)
    for cutoff, positions in sorted(grouped.items()):
        if cutoff >= full_length:
            print(f"NATIVE TRUNCATION cutoff={cutoff}: already full length, no shape comparison")
            continue
        trace, handles = _attention_trace_hooks(
            model, monitored_positions=tuple(positions)
        )
        try:
            with torch.no_grad():
                ids = tokens[:cutoff].to(device)
                positions_tensor = torch.arange(cutoff, device=device)[None, :]
                logits = model(
                    input_ids=ids.unsqueeze(0),
                    position_ids=positions_tensor,
                    attention_mask=None,
                )
                for absolute in positions:
                    target = tokens[absolute + 1].to(device).reshape(1)
                    lp = vocab_parallel_log_probs_from_logits(
                        logits[0, absolute: absolute + 1], target
                    ).detach().float().cpu().item()
                    response_offset = absolute + 1 - len(batch["prompts"][row])
                    baseline = float(original_logprobs[row][response_offset])
                    delta = lp - baseline
                    print(
                        "NATIVE TRUNCATION LOGPROB "
                        f"query_abs={absolute} cutoff={cutoff} full={full_length} "
                        f"full_lp={baseline:.8f} cutoff_lp={lp:.8f} "
                        f"delta={delta:.8f} abs_diff={abs(delta):.8f}"
                    )
        finally:
            for hook in handles:
                hook.remove()
        for absolute in positions:
            matched = [
                layer for layer in range(1, len(model.decoder.layers) + 1)
                if (layer, absolute) in full_trace and (layer, absolute) in trace
            ]
            print(
                f"NATIVE TRUNCATION LAYER TRACE query_abs={absolute} "
                f"cutoff={cutoff} matched_layers={len(matched)}"
            )
            for layer in matched:
                native_input, native_output = full_trace[(layer, absolute)]
                trunc_input, trunc_output = trace[(layer, absolute)]
                delta_in = (trunc_input - native_input).abs()
                delta_out = (trunc_output - native_output).abs()
                if layer <= 6 or layer in (14, 21, 28):
                    print(
                        f"  native_cutoff layer={layer:02d} "
                        f"attn_in_max={delta_in.max().item():.6g} "
                        f"attn_out_max={delta_out.max().item():.6g}"
                    )




def _tpr_single_segment_cutoff_controls(model, batch, original_logprobs, owners):
    """TPR node starting at 0 ending at real radix cutoff; no KV splitting.

    Compare logprob to *full-length Native*. Together with the native-only
    cutoff control this cleanly separates context-length effects from
    rectangular suffix query/KV concatenation.
    """
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    row = 0
    tokens = batch["input_ids"][row].detach().cpu().long()
    if tokens.numel() > 256:
        print("TPR SINGLE-CUTOFF CONTROL skipped for full real TQ (OOM guard)")
        return
    layers = tuple(sorted(
        attention.layer_number for attention in model.modules()
        if isinstance(attention, TPRSelfAttention)
    ))
    for absolute, owner in sorted(owners.items()):
        cutoff = owner.position_end
        if cutoff >= tokens.numel() or absolute + 1 >= tokens.numel():
            continue
        shortened = tokens[:cutoff]
        segment = SegmentSpec(
            segment_id=0,
            parent_id=None,
            token_ids=shortened,
            position_start=0,
            prefix_length=0,
            loss_terms=(SegmentLossTerm(0, int(shortened[1])),),
        )
        plan = SegmentPlan((segment,), root_id=0)
        executor = SegmentExecutor(
            model, plan, expected_layer_numbers=layers
        )
        with torch.no_grad():
            _ctx, logits = executor._forward(
                segment, past_key_values={}, no_grad=True
            )
            label = tokens[absolute + 1].to(logits.device).reshape(1)
            lp = vocab_parallel_log_probs_from_logits(
                logits[0, absolute: absolute + 1], label
            ).detach().float().cpu().item()
        response_offset = absolute + 1 - len(batch["prompts"][row])
        baseline = float(original_logprobs[row][response_offset])
        print(
            "TPR SINGLE-CUTOFF LOGPROB "
            f"query_abs={absolute} cutoff={cutoff} full={tokens.numel()} "
            f"full_native_lp={baseline:.8f} single_tpr_lp={lp:.8f} "
            f"delta={lp - baseline:.8f} abs_diff={abs(lp - baseline):.8f}"
        )



def test_real_qwen3_1_7b_tq_ppo_loss_and_gradients():
    """Real-weight native row-wise PPO vs TPR Forest (CP=TP=PP=DP=1)."""
    # Sanitize before importing Qwen fixtures: some MindSpeed revisions
    # construct dataclasses during their module import/bootstrap.
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)

    from ..profiling._qwen3_profile_target import resolve_qwen3_profile_target
    from . import test_tpr_qwen3_compatibility_npu as qwen_fixture
    from verl.workers.utils.losses import ppo_loss

    # Reuse the existing REAL checkpoint selector rather than inheriting the
    # 0.6B fixture's import-time global default or an unrelated env value.
    target = resolve_qwen3_profile_target()
    if target.size != "1.7B":
        pytest.fail(f"Phase5 explicitly requires Qwen3-1.7B; got {target.label}")
    qwen_fixture.QWEN_MODEL_PATH = target.path
    print(f"PHASE5 MODEL=Qwen3-1.7B checkpoint={target.path}")

    # Same MindSpeed pytest-runtime hygiene used by the existing 1.7B Ring
    # profile: some versions accidentally include an empty key in full args;
    # turning that key into a dataclass field raises
    # TypeError: Field names must be valid identifiers: ''.
    # Delay Engine import until after the args are sanitized; importing it
    # triggers the MindSpeed compatibility/transformer patch stack.
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    _initialize_single_rank_megatron = qwen_fixture._initialize_single_rank_megatron
    _make_qwen_model = qwen_fixture._make_qwen_model

    # Default is the UNMODIFIED TQ forest. Both env vars are opt-in CROPPED
    # smoke controls only, and must be absent to validate all real tokens.
    raw_p = os.getenv("TPR_QWEN17_PPO_PROMPT")
    raw_r = os.getenv("TPR_QWEN17_PPO_RESPONSE")
    prompt_len = int(raw_p) if raw_p is not None else None
    response_len = int(raw_r) if raw_r is not None else None
    batch = _load_real_tq_probe(prompt_length=prompt_len, response_length=response_len)
    device = torch.device("npu")
    _initialize_single_rank_megatron()
    longest_row = max(len(row) for row in _rows(batch, "input_ids"))
    print(f"PHASE5 max real sequence length={longest_row}; initializing Native reference")
    ref = _make_qwen_model(device, tpr=False, max_sequence_length=longest_row)
    ref_params = target.assert_model_scale(ref)

    # The OLD policy is the real frozen Qwen checkpoint prior to this update,
    # not invented log-probabilities. Recompute once on the actual TQ tokens.
    old_probs = []
    with torch.no_grad():
        for row in range(8):
            _, response_lp = _native_response_logprobs(
                ref, batch["input_ids"][row].to(device),
                prompt_length=len(batch["prompts"][row]), temperature=1.0
            )
            old_probs.append(response_lp.detach().float().cpu())
    batch["old_log_probs"] = _as_jagged(old_probs)
    # Old-policy log_probs and the current model should be computed from the
    # same real weights. Their equality is a numerical check, not rollout truth.
    tu.assign_non_tensor(batch, tpr_capture_log_probs=True)

    loss_fn = partial(ppo_loss, config=_VanillaPPOConfig())
    native_loss = torch.zeros((), device=device, dtype=torch.float32)
    native_repeat_abs = []
    ref.zero_grad(set_to_none=True)
    for row in range(8):
        token_row = batch["input_ids"][row].to(device)
        lp, _ = _native_response_logprobs(
            ref, token_row, prompt_length=len(batch["prompts"][row]), temperature=1.0
        )
        # Compare repeated Native forward of the very same checkpoint, real
        # row and attention shape. Native nonrepeatability is a separate noise
        # floor and must not be attributed to prefix reuse.
        row_start = len(batch["prompts"][row]) - 1
        row_new = lp[row_start:-1].detach().float().cpu()
        row_mask = batch["response_mask"][row].bool().cpu()
        native_repeat_abs.append(
            (row_new[row_mask] - old_probs[row][row_mask]).abs()
        )
        # Rebuild a one-row native VERL loss view, rather than slicing the
        # full TQ metadata (whose NonTensorData keys may not be row-indexable).
        mini = TensorDict({
            key: batch[key][row].unsqueeze(0).to(device)
            for key in (
                "prompts", "responses", "attention_mask", "response_mask",
                "old_log_probs", "advantages",
            )
        }, batch_size=[1])
        tu.assign_non_tensor(
            mini, batch_num_tokens=int(batch["response_mask"].sum()),
            global_batch_size=8, dp_size=1,
        )
        # Native PPO uses the ORIGINAL global denominator for each row.
        mini_model_output = {"log_probs": lp}
        row_loss, _ = loss_fn(model_output=mini_model_output, data=mini, dp_group=None)
        row_loss.backward()
        native_loss += row_loss.detach().float()
    repeat_differences = torch.cat(native_repeat_abs)
    print(
        "NATIVE REPEATABILITY (same model, same row, no-grad vs grad-enabled): "
        f"max_abs={repeat_differences.max().item():.6g} "
        f"mean_abs={repeat_differences.mean().item():.6g}"
    )
    native_grads = _selected_gradient_snapshot(ref)

    # No synthetic model or token fixtures. Trace the exact original row-0
    # query positions for early/root and shared nonroot response tokens.
    trace_owners = None
    native_trace = {}
    if os.getenv("TPR_QWEN17_PPO_TRACE_LAYERS", "1") == "1":
        trace_owners = _get_trace_owner_segments(batch)
        trace_positions = tuple(sorted(trace_owners))
        print(f"REAL QWEN TRACE POSITIONS: {trace_positions}")
        native_trace, native_hooks = _attention_trace_hooks(
            ref, monitored_positions=trace_positions
        )
        try:
            with torch.no_grad():
                _native_response_logprobs(
                    ref, batch["input_ids"][0].to(device),
                    prompt_length=len(batch["prompts"][0]), temperature=1.0
                )
        finally:
            for handle in native_hooks:
                handle.remove()
        if os.getenv("TPR_QWEN17_PPO_NATIVE_CUTOFF", "1") == "1":
            _native_truncation_controls(
                ref, batch, old_probs, native_trace, trace_owners
            )

    del ref
    gc.collect()
    torch.npu.empty_cache()

    print("PHASE5: Native loss and gradients collected; initializing TPR reference")
    tpr = _make_qwen_model(device, tpr=True, max_sequence_length=longest_row)
    assert target.assert_model_scale(tpr) == ref_params
    tpr.config.no_sync_func = None
    tpr.config.grad_scale_func = lambda value: value
    tpr.config.finalize_model_grads_func = lambda *args, **kwargs: None
    tpr.config.calculate_per_token_loss = False
    tpr.zero_grad(set_to_none=True)
    # Test the ACTUAL patched Engine entry, not just direct adapter invocation.
    # This confirms global batch normalization happens before Tree building.
    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [tpr]
    engine.engine_config = SimpleNamespace(
        tpr_enabled=True, tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
        use_fused_kernels=False, dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = tpr.config
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None

    tpr_trace, tpr_hooks = ({}, [])
    if trace_owners is not None:
        tpr_trace, tpr_hooks = _attention_trace_hooks(
            tpr, monitored_positions=tuple(sorted(trace_owners)),
            owner_segments=trace_owners,
        )
    try:
        output = engine.forward_backward_batch(
            batch, loss_function=partial(ppo_loss, config=_VanillaPPOConfig()),
            forward_only=False,
        )
    finally:
        for handle in tpr_hooks:
            handle.remove()
    if trace_owners is not None:
        _print_attention_layer_drift(
            native_trace, tpr_trace, tuple(sorted(trace_owners))
        )
    tpr_grads = _selected_gradient_snapshot(tpr)
    captured = getattr(engine, "_tpr_captured_log_probs", None)
    assert captured is not None, "TPR logical new-logprob capture was not enabled"
    expected_keys = {
        (row, offset)
        for row in range(8)
        for offset in torch.nonzero(batch["response_mask"][row]).flatten().tolist()
    }
    assert set(captured) == expected_keys
    baseline_lp = torch.tensor(
        [float(old_probs[row][offset]) for row, offset in sorted(expected_keys)]
    )
    tpr_lp = torch.tensor([captured[key] for key in sorted(expected_keys)])
    diag_keys, diag_expected, diag_actual, bad, outside_clip = _logprob_diagnostics(
        batch, captured, old_probs
    )
    if bool(bad.any()):
        _probe_tpr_model_native_forward(tpr, batch, old_probs)
        if trace_owners is not None and os.getenv("TPR_QWEN17_PPO_NATIVE_CUTOFF", "1") == "1":
            try:
                _tpr_single_segment_cutoff_controls(
                    tpr, batch, old_probs, trace_owners
                )
            except Exception as exc:
                print(f"TPR SINGLE-CUTOFF DIAG unavailable: {type(exc).__name__}: {exc}")
        worst_i = (diag_actual - diag_expected).abs().argmax().item()
        worst_row = diag_keys[worst_i][0]
        try:
            _probe_whole_segment_tpr_forward(
                tpr, batch, old_probs, row=worst_row
            )
        except Exception as exc:
            # Diagnostic-only probe must not obscure the original numerical
            # gate failure; the NPU stack may reject probe-only forward calls.
            print(f"ONE-SEGMENT TPR DIAG unavailable: {type(exc).__name__}: {exc}")
    else:
        print(f"Qwen3-1.7B real TQ new_log_probs: {len(expected_keys)} logical tokens aligned")
    check_failures = []
    if bool(outside_clip.any()):
        check_failures.append(
            f"PPO clip regime disagrees with Native old policy at "
            f"{int(outside_clip.sum())} logical response tokens"
        )
    try:
        torch.testing.assert_close(tpr_lp, baseline_lp, rtol=2e-2, atol=2e-1)
    except AssertionError as exc:
        check_failures.append(f"logprob mismatch: {str(exc).splitlines()[0]}")
    actual_loss = float(sum(output["loss"]))
    expected_loss = float(native_loss.item())
    print(f"Qwen3-1.7B real TQ PPO native_loss={expected_loss:.8f}, tpr_loss={actual_loss:.8f}")
    try:
        torch.testing.assert_close(
            torch.tensor(actual_loss), torch.tensor(expected_loss),
            rtol=2e-2, atol=1e-3,
        )
    except AssertionError as exc:
        check_failures.append(f"PPO loss mismatch: {str(exc).splitlines()[0]}")
    try:
        _compare_grads(native_grads, tpr_grads)
    except AssertionError as exc:
        check_failures.append(f"parameter gradient mismatch: {str(exc).splitlines()[0]}")
    assert output["metrics"]["tpr/forest_trees"] == [1]
    assert output["metrics"]["tpr/logical_loss_tokens"] == [int(batch["response_mask"].sum())]
    if check_failures:
        pytest.fail(
            "Qwen3-1.7B real TQ PPO numerical gate FAILED:\n  "
            + "\n  ".join(check_failures)
        )
    print("QWEN3-1.7B REAL TQ PPO NUMERICAL GATE: PASS")
