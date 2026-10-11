"""P0-3: true real-Qwen TPR P512/S512 vs native CP1/CP2 P0-1 evidence.

Native CP2 sharding and TPR segmentation both expose Linear local M=512,
but are *not equivalent attention schedules*: CP2 Zigzag (256+256) + Ring
vs TPR contiguous Prefix512, Suffix512 + external KV rectangular attention.

Produce two same-process TPR-family results:
  TPR_CONTROL_FULL   one 1024-token GPT forward/backward with no TPR context;
  TPR_ONE_NODE       one 1024-token SegmentExecutor visit_leaf root: same
                     tree-mode rectangular Attention but no split/KV relay;
  TPR_SEGMENTED      real SegmentExecutor push(512) -> visit_leaf(512)
                     -> pop(512), with root query 511 predicting response[0].

The follow-up comparator joins the saved existing UNMODIFIED MindSpeed native
CP1/CP2 P0-1 evidence (same real TQ row0 and weights) and prints all pairs.
All four paths use the same 512-target NLL (F.log_softmax + gather), native
BF16 kernels and no FP32 GEMM / artificial padding / optimizer changes.
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
    SEQ_LEN, RESPONSE_LEN, _input_tokens, _runtime, _weight_signature,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_P0_TPR_P512S512") != "1"
    or not is_torch_npu_available(check_device=True),
    reason="Set TPR_P0_TPR_P512S512=1 for real Qwen TPR/Full diagnostic",
)


def _sample_grads(model):
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
    samples = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            raise AssertionError(f"No gradient for {name}")
        grad = param.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"Nonfinite gradient for {name}")
        index = torch.tensor(
            exact_sample_indices(grad.numel(), 512),
            device=grad.device, dtype=torch.long,
        )
        samples[name] = {
            "shape": tuple(grad.shape),
            "numel": int(grad.numel()),
            "indices": index.cpu(),
            "values": grad.flatten().index_select(0, index).float().cpu(),
        }
    if not samples:
        raise AssertionError("No trainable parameters")
    return samples


def _response_logprobs(logits, tokens):
    """Global query 511..1022 predicts target tokens 512..1023."""
    if tuple(logits.shape[:2]) != (1, SEQ_LEN):
        raise AssertionError(f"Unexpected unsplit logits shape {tuple(logits.shape)}")
    labels = tokens[SEQ_LEN - RESPONSE_LEN:].to(device=logits.device)
    queries = logits[0, SEQ_LEN - RESPONSE_LEN - 1:SEQ_LEN - 1].float()
    if queries.shape[0] != RESPONSE_LEN:
        raise AssertionError("Full response query count mismatch")
    return F.log_softmax(queries, dim=-1).gather(
        1, labels.unsqueeze(1),
    ).squeeze(1)


def _run_control_full(model, tokens):
    model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    inputs = tokens.to(device)
    positions = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
    logits = model(inputs.unsqueeze(0), positions, None)
    lps = _response_logprobs(logits, inputs)
    loss = -lps.sum() / RESPONSE_LEN
    loss.backward()
    ret = {
        "loss": float(loss.detach().cpu()),
        "response_logprobs": lps.detach().float().cpu().clone(),
        "gradients": _sample_grads(model),
    }
    model.zero_grad(set_to_none=True)
    return ret


def _run_one_node_tpr(model, tokens):
    """Full M=1024 through TPR tree Attention, with NO Prefix/Suffix split.

    Isolates native-without-context vs TPR rectangular Attention at the
    *same* physical GEMM M. There is only a root leaf and no cached KV relay.
    """
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec
    from verl.models.mcore.tpr.attention import TPRSelfAttention

    root = SegmentSpec(
        0, None, tokens, position_start=0, prefix_length=0,
    )
    plan = SegmentPlan((root,), root_id=0, topology_only=True)
    layers = tuple(sorted(
        module.layer_number for module in model.modules()
        if isinstance(module, TPRSelfAttention)
    ))
    if len(layers) != 28:
        raise AssertionError(f"Expected real 28-layer TPR Qwen; got {len(layers)}")
    snapshots = {}
    callback_count = [0]

    def loss_fn(segment, logits):
        if segment.segment_id != 0 or logits.shape[1] != SEQ_LEN:
            raise AssertionError("One-node TPR oracle is not full-sequence")
        lp = _response_logprobs(logits, tokens)
        callback_count[0] += 1
        snapshots["response_logprobs"] = lp.detach().float().cpu().clone()
        return -lp.sum() / RESPONSE_LEN, {}

    model.zero_grad(set_to_none=True)
    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=layers,
        segment_loss_fn=loss_fn,
        segment_loss_term_counts={0: RESPONSE_LEN},
    )
    leaf = executor.visit_leaf(0)
    executor.kv_stack.assert_empty()
    if callback_count != [1] or leaf.backward.loss_term_count != RESPONSE_LEN:
        raise AssertionError("TPR one-node full loss ownership mismatch")
    result = {
        "loss": float(leaf.backward.normalized_loss.detach().cpu()),
        "response_logprobs": snapshots["response_logprobs"],
        "gradients": _sample_grads(model),
        "segment_lengths": [SEQ_LEN],
        "loss_term_count": RESPONSE_LEN,
        "prefix_grad_relays": 0,
    }
    model.zero_grad(set_to_none=True)
    return result


def _run_two_node_tpr(model, tokens):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec
    from verl.models.mcore.tpr.attention import TPRSelfAttention

    # Real Prefix/Suffix boundaries. This is a *chain*, not a synthetic
    # shared-prefix forest; it tests backward relay exactly once for one child.
    p, s = RESPONSE_LEN, RESPONSE_LEN
    prefix = SegmentSpec(0, None, tokens[:p], 0, 0)
    suffix = SegmentSpec(1, 0, tokens[p:], p, p)
    plan = SegmentPlan((prefix, suffix), root_id=0, topology_only=True)
    layers = tuple(sorted(module.layer_number for module in model.modules()
                          if isinstance(module, TPRSelfAttention)))
    if len(layers) != 28:
        raise AssertionError(f"Expected 28 real TPR SelfAttention layers; got {layers}")
    model.zero_grad(set_to_none=True)
    recorded = {}
    do_calls = {0: 0, 1: 0}

    def loss_fn(segment, logits):
        if segment.segment_id == 0:
            # Prefix last query at global 511 predicts response token 0.
            selected = logits[0, p - 1:p].float()
            labels = tokens[p:p + 1].to(logits.device)
            expected = 1
        elif segment.segment_id == 1:
            # Suffix local queries 0..510 predict response tokens 1..511.
            selected = logits[0, :s - 1].float()
            labels = tokens[p + 1:p + s].to(logits.device)
            expected = s - 1
        else:
            raise AssertionError("Unexpected TPR plan node")
        lps = F.log_softmax(selected, dim=-1).gather(
            1, labels.unsqueeze(1),
        ).squeeze(1)
        if lps.numel() != expected:
            raise AssertionError(f"Wrong TPR logprob owner count {segment.segment_id}")
        do_calls[segment.segment_id] += 1
        recorded[segment.segment_id] = lps.detach().float().cpu().clone()
        return -lps.sum() / float(RESPONSE_LEN), {}

    executor = SegmentExecutor(
        model, plan, expected_layer_numbers=layers,
        segment_loss_fn=loss_fn, segment_loss_term_counts={0: 1, 1: s - 1},
    )
    executor.push(0)
    child = executor.visit_leaf(1)
    parent = executor.pop(0)
    executor.kv_stack.assert_empty()
    if do_calls != {0: 1, 1: 1}:
        raise AssertionError(f"TPR loss callbacks ran unexpected times: {do_calls}")
    if child.backward.loss_term_count != s - 1 or parent.loss_term_count != 1:
        raise AssertionError("TPR loss ownership lost or duplicated response targets")
    if parent.relayed_layer_count != 28:
        raise AssertionError(
            f"Expected 28 prefix KV gradient relays; got {parent.relayed_layer_count}"
        )
    response_lp = torch.cat((recorded[0], recorded[1]))
    if response_lp.shape != (s,):
        raise AssertionError(f"TPR response logprob count mismatch {response_lp.shape}")
    loss = float((child.backward.normalized_loss + parent.normalized_loss).cpu())
    ret = {
        "loss": loss, "response_logprobs": response_lp,
        "gradients": _sample_grads(model),
        "prefix_grad_relays": parent.relayed_layer_count,
        "segment_lengths": [p, s],
        "loss_term_count": s,
    }
    model.zero_grad(set_to_none=True)
    return ret


def test_p512s512_real_tq_tpr_vs_controlled_full():
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise AssertionError("P0-3 TPR branch is single-rank CP1")
    # MindSpeed bootstrap and official checkpoint mapping reuse P0-1's runtime.
    with _runtime(1) as runtime:
        from ..correctness import test_tpr_qwen3_compatibility_npu as fixture
        checkpoint = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
        fixture.QWEN_MODEL_PATH = checkpoint
        model = fixture._make_qwen_model(
            runtime.device, tpr=True, max_sequence_length=SEQ_LEN,
        )
        tokens, source = _input_tokens(model.vocab_size)
        if "real_tq" not in source or "row0:" not in source:
            raise AssertionError("P0-3 requires same real TQ row0 as native P0-1")
        signature = _weight_signature(model)
        controlled = _run_control_full(model, tokens)
        tpr_one_node = _run_one_node_tpr(model, tokens)
        tpr = _run_two_node_tpr(model, tokens)
        backend_gap = controlled["response_logprobs"] - tpr_one_node["response_logprobs"]
        segmentation_gap = tpr_one_node["response_logprobs"] - tpr["response_logprobs"]
        print(
            "P0_3 ONE_NODE_M1024_VS_TWO_NODE_M512 "
            f"one_node_loss={tpr_one_node['loss']:.9f} "
            f"split_loss={tpr['loss']:.9f} "
            f"backend_mean_logprob_abs={float(backend_gap.abs().mean()):.9g} "
            f"segment_mean_logprob_abs={float(segmentation_gap.abs().mean()):.9g} "
            f"segment_max_logprob_abs={float(segmentation_gap.abs().max()):.9g}",
            flush=True,
        )
        raw = controlled["response_logprobs"] - tpr["response_logprobs"]
        print(
            f"P0_3 P512S512 CONTROLLED_FULL_VS_TPR "
            f"loss_full={controlled['loss']:.9f} loss_tpr={tpr['loss']:.9f} "
            f"mean_logprob_abs={float(raw.abs().mean()):.9g} "
            f"max_logprob_abs={float(raw.abs().max()):.9g} "
            f"prefix_grad_relays={tpr['prefix_grad_relays']} "
            f"parameters={len(tpr['gradients'])}", flush=True,
        )
        outdir = Path(os.getenv("TPR_P0_3_OUTPUT_DIR", "/tmp/tpr_p0_tpr_p512s512"))
        outdir.mkdir(parents=True, exist_ok=True)
        output = outdir / "tpr_p512s512.pt"
        payload = {
            "tag": "P0_3_REAL_QWEN_1_7B_P512S512_SINGLE_PATH",
            "checkpoint": str(checkpoint), "model": "Qwen3-1.7B",
            "dtype": "bf16", "torch": str(torch.__version__),
            "source": source,
            "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            "sampled_weight_sha256_rank0": signature,
            "sequence_length": SEQ_LEN, "response_length": RESPONSE_LEN,
            "objective": "512_response_targets_including_prompt_boundary_nll",
            "denominator": RESPONSE_LEN,
            "controlled_full": controlled,
            "tpr_one_node_m1024": tpr_one_node,
            "tpr_p512s512": tpr,
            "note": "CP1 TPR engine SegmentExecutor; controlled full attention is "
                    "not official CP1 core attention; compare separately.",
        }
        torch.save(payload, output)
        print(f"P0_3 saved={output} status=DIAGNOSTIC_ONLY", flush=True)
