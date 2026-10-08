"""Real-Qwen3 long-sequence Full vs external-KV Split diagnostic.

Two independent controls:
1. Real pretrained Qwen3-1.7B first SelfAttention with actual token-embedding
   + input-RMSNorm activations: Full vs Prefix/External-KV Suffix, including
   suffix/prefix hidden grads, QKV/projection grads, and past-KV grads.
2. The SAME complete 28-layer Qwen3 GPTModel, Full vs one Prefix/Suffix
   split, without trie, SegmentExecutor, PPO or a test-only GEMM pad.
   Captures where the two paths FIRST numerically diverge.

Real recorded TQ input tokens only, never generated synthetic text. The full
reference uses the SAME controlled CANN core as TPR (not 'unmodified Megatron
DotProductAttention'), so the only change is segmented execution/KV reuse.
No model/TPR implementation, gradient thresholds or attention math modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_RUN_QWEN17_SPLIT") != "1",
    reason="Set TPR_RUN_QWEN17_SPLIT=1 for real-Qwen long-split controls",
)


def _setup_real_model():
    # Keep the bootstrap identical to the working real-TQ Phase-5 test.
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead  # noqa: F401

    from ..profiling._qwen3_profile_target import resolve_qwen3_profile_target
    from . import test_tpr_qwen3_compatibility_npu as fixture

    target = resolve_qwen3_profile_target()
    if target.size != "1.7B":
        pytest.fail(f"this test requires Qwen3-1.7B, not {target.label}")
    fixture.QWEN_MODEL_PATH = target.path
    fixture._initialize_single_rank_megatron()
    return fixture, target


def _lengths():
    p = int(os.environ.get("TPR_QWEN17_SPLIT_P", "1024"))
    s = int(os.environ.get("TPR_QWEN17_SPLIT_S", "128"))
    if p <= 0 or s < 2 or p + s > 16384:
        pytest.fail("Require 0<P, 2<=S, P+S<=16384 for single-rank diagnostic")
    return p, s


def _real_recorded_tokens(p, s):
    from .test_qwen3_1_7b_real_tq_ppo_npu import _DEFAULT_TQ

    source = Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ)))
    if not source.is_file():
        pytest.fail(f"recorded TQ tokens are required: {source}")
    record = torch.load(source, map_location="cpu", weights_only=False)["tensordict"]
    prefix = record["prompts"][0].detach().cpu().long()
    response = record["responses"][0].detach().cpu().long()
    if prefix.numel() < p or response.numel() < s:
        pytest.fail(
            f"real TQ row too short: actual prompt={prefix.numel()} response={response.numel()}, "
            f"requested P={p} S={s}"
        )
    # Match Phase-5 cropped-real mode: LAST P recorded prompt tokens and
    # FIRST S recorded response tokens; absolute positions reset to 0.
    return torch.cat((prefix[-p:], response[:s])).contiguous()


def _rope(model, start, length):
    from verl.models.mcore.tpr.rope import build_suffix_rotary_pos_emb

    return build_suffix_rotary_pos_emb(
        model.rotary_pos_emb, prefix_length=start, suffix_length=length
    )


def _stats(name, observed, expected):
    a = observed.detach().float().cpu()
    b = expected.detach().float().cpu()
    if a.shape != b.shape:
        raise AssertionError(f"{name} shape mismatch {a.shape} != {b.shape}")
    error = a - b
    rel = float(torch.linalg.vector_norm(error) /
                torch.linalg.vector_norm(b).clamp_min(1e-12))
    max_abs = float(error.abs().max())
    equal = torch.equal(a, b)
    print(f"QWEN SPLIT {name} rel_l2={rel:.8g} max_abs={max_abs:.8g} bitwise={equal}", flush=True)
    return rel, max_abs


def test_real_qwen_first_attention_full_vs_external_kv():
    """Actual Qwen layer-1 weights + real-token LN activations, P+S up to 16K."""
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context
    fixture, target = _setup_real_model()
    p, s = _lengths()
    tokens = _real_recorded_tokens(p, s).to("npu")
    model = fixture._make_qwen_model(torch.device("npu"), tpr=True, max_sequence_length=p+s)
    assert target.assert_model_scale(model) > 1_500_000_000
    attention = model.decoder.layers[0].self_attention
    assert type(attention).__name__ == "TPRSelfAttention"
    print(f"QWEN SPLIT ATTENTION P={p} S={s} layer=1 real_weights=True", flush=True)

    # This tensor is precisely the distribution seen by the first Qwen
    # Attention after word-embedding and input RMSNorm, not random hidden data.
    with torch.no_grad():
        embeddings = model.embedding.word_embeddings(tokens[None, :]).transpose(0, 1)
        attention_input = model.decoder.layers[0].input_layernorm(embeddings).detach()

    full_hidden = attention_input.clone().requires_grad_(True)
    torch.manual_seed(2026)
    upstream = torch.randn((s, 1, full_hidden.shape[-1]), device="npu", dtype=full_hidden.dtype)
    attention.zero_grad(set_to_none=True)
    full_mask = torch.triu(
        torch.ones((1, 1, p+s, p+s), device="npu", dtype=torch.bool), diagonal=1
    )
    full_out, full_bias = attention(full_hidden, full_mask, rotary_pos_emb=_rope(model, 0, p+s))
    assert full_bias is None
    full_out[-s:].backward(upstream)
    full_output = full_out[-s:].detach().float().cpu().clone()
    full_in_grad = full_hidden.grad.detach().float().cpu().clone()
    full_params = {
        n: param.grad.detach().float().cpu().clone()
        for n, param in attention.named_parameters() if param.requires_grad and param.grad is not None
    }
    del full_out, full_hidden
    attention.zero_grad(set_to_none=True)

    prefix_hidden = attention_input[:p].clone().requires_grad_(True)
    suffix_hidden = attention_input[p:].clone().requires_grad_(True)
    prefix_ctx = TPRAttentionContext(
        0, p, suffix_rotary_pos_emb=_rope(model, 0, p)
    )
    with use_tpr_attention_context(prefix_ctx):
        attention(prefix_hidden, None)
    prefix_ctx.assert_new_kv_layers((attention.layer_number,))
    past = dict(prefix_ctx.new_key_values)
    for k, v in past.values():
        k.retain_grad()
        v.retain_grad()
    suffix_ctx = TPRAttentionContext(
        p, s, past_key_values=past, suffix_rotary_pos_emb=_rope(model, p, s)
    )
    with use_tpr_attention_context(suffix_ctx):
        suffix_out, suffix_bias = attention(suffix_hidden, None)
    assert suffix_bias is None
    suffix_out.backward(upstream)

    _stats("ATTN_OUTPUT", suffix_out, full_output)
    _stats("ATTN_SUFFIX_HIDDEN_GRAD", suffix_hidden.grad, full_in_grad[p:])
    _stats("ATTN_PREFIX_HIDDEN_GRAD", prefix_hidden.grad, full_in_grad[:p])
    assert prefix_hidden.grad is not None and suffix_hidden.grad is not None
    assert all(k.grad is not None and v.grad is not None for k, v in past.values())
    assert all(
        torch.isfinite(k.grad).all() and torch.isfinite(v.grad).all()
        for k, v in past.values()
    ), "non-finite gradients in cached prefix K/V"
    assert all(
        k.grad.float().norm() > 0 and v.grad.float().norm() > 0
        for k, v in past.values()
    ), "no gradient relayed to cached prefix K/V"
    assert set(full_params) == {
        n for n, param in attention.named_parameters() if param.requires_grad and param.grad is not None
    }
    split_params = dict(attention.named_parameters())
    for name, expected in full_params.items():
        if name.startswith(("linear_qkv", "linear_proj", "q_layernorm", "k_layernorm")):
            actual = split_params[name].grad
            _stats("ATTN_PARAM_" + name, actual, expected)
            torch.testing.assert_close(
                actual.detach().float().cpu(), expected, atol=1e-2, rtol=1e-2
            )

    # Preserve the original foundational Attention gate; do not weaken it
    # simply because this uses pretrained (rather than tiny random) weights.
    torch.testing.assert_close(
        suffix_out.detach().float().cpu(), full_output, atol=6e-3, rtol=6e-3
    )
    torch.testing.assert_close(
        suffix_hidden.grad.detach().float().cpu(), full_in_grad[p:], atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        prefix_hidden.grad.detach().float().cpu(), full_in_grad[:p], atol=1e-2, rtol=1e-2
    )
    print("QWEN SPLIT REAL ATTENTION EQUIVALENCE: PASS", flush=True)


def test_real_qwen_full_gpt_vs_single_split():
    """Complete 28-layer real-Qwen Full vs two segments; no Forest/PPO/offload."""
    from verl.models.mcore.tpr.context import (
        TPRAttentionContext, get_tpr_attention_context, use_tpr_attention_context,
    )
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    fixture, target = _setup_real_model()
    p, s = _lengths()
    tokens = _real_recorded_tokens(p, s).to("npu")
    model = fixture._make_qwen_model(torch.device("npu"), tpr=True, max_sequence_length=p+s)
    assert target.assert_model_scale(model) > 1_500_000_000
    layers = tuple(layer.self_attention.layer_number for layer in model.decoder.layers)
    print(f"QWEN SPLIT FULL-GPT P={p} S={s} n_layers={len(layers)}", flush=True)

    captured = {}
    handles = []

    def add_capture(layer, module, stage):
        def hook(_module, args, output):
            ctx = get_tpr_attention_context()
            mode = "full" if ctx is None else (
                "split" if ctx.prefix_length == p and ctx.suffix_length == s else None
            )
            if mode is None:
                return
            t = args[0] if stage.endswith("_in") else (
                output[0] if isinstance(output, tuple) else output
            )
            # First two layers suffice to identify whether divergence first
            # appears in the QKV/Attention path or MLP FC2.
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"unexpected {stage} capture: {type(t)}")
            value = t[-s:] if mode == "full" else t
            captured[(mode, layer, stage)] = value.detach().float().cpu().contiguous()
        handles.append(module.register_forward_hook(hook))

    for layer in model.decoder.layers[:2]:
        i = layer.self_attention.layer_number
        add_capture(i, layer.self_attention, "attn_out")
        add_capture(i, layer.mlp.linear_fc1, "fc1_out")
        add_capture(i, layer.mlp.linear_fc2, "fc2_in")
        add_capture(i, layer.mlp.linear_fc2, "fc2_out")

    try:
        with torch.no_grad():
            positions = torch.arange(p+s, device="npu").unsqueeze(0)
            full_logits = model(tokens[None, :], positions, attention_mask=None)
            full_tail = full_logits[0, p:p+s].detach().float().cpu().clone()
            del full_logits

            prefix_ctx = TPRAttentionContext(
                0, p, suffix_rotary_pos_emb=_rope(model, 0, p)
            )
            with use_tpr_attention_context(prefix_ctx):
                prefix_logits = model(
                    tokens[None, :p],
                    positions[:, :p],
                    attention_mask=None,
                )
            del prefix_logits
            prefix_ctx.assert_new_kv_layers(layers)
            suffix_ctx = TPRAttentionContext(
                p, s, past_key_values=dict(prefix_ctx.new_key_values),
                suffix_rotary_pos_emb=_rope(model, p, s),
            )
            with use_tpr_attention_context(suffix_ctx):
                split_logits = model(
                    tokens[None, p:], positions[:, p:], attention_mask=None
                )
            split_tail = split_logits[0].detach().float().cpu().clone()
            del split_logits

        first = None
        for layer in layers[:2]:
            for stage in ("attn_out", "fc1_out", "fc2_in", "fc2_out"):
                a, b = captured[("split", layer, stage)], captured[("full", layer, stage)]
                rel, mx = _stats(f"GPT_LAYER{layer:02d}_{stage}", a, b)
                if first is None and mx != 0:
                    first = (layer, stage, mx)
        print(f"QWEN SPLIT GPT FIRST_CAPTURED_DRIFT={first}", flush=True)

        _stats("GPT_SUFFIX_LOGITS", split_tail, full_tail)
        labels = tokens[p+1:p+s]
        ref_lp = vocab_parallel_log_probs_from_logits(
            full_tail[:-1].to("npu"), labels
        ).detach().float().cpu()
        split_lp = vocab_parallel_log_probs_from_logits(
            split_tail[:-1].to("npu"), labels
        ).detach().float().cpu()
        _, max_lp = _stats("GPT_SUFFIX_LOGPROBS", split_lp, ref_lp)
        max_ratio_error = float((torch.exp(split_lp-ref_lp)-1).abs().max())
        print(f"QWEN SPLIT GPT PPO_RATIO_MAX_DEVIATION={max_ratio_error:.8g}", flush=True)
        # This is intentionally stricter than the broad legacy per-element
        # rtol=2e-2 PPO logprob check, which can conceal clipping changes.
        assert max_lp < 0.02, (
            f"Qwen pretrained Full/Split changes PPO-sensitive logprobs by {max_lp:.6g}"
        )
        print("QWEN SPLIT FULL-GPT EQUIVALENCE: PASS", flush=True)
    finally:
        for handle in handles:
            handle.remove()
