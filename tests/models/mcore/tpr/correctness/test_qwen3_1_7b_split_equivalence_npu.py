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


def _core_fp32_sparse_row_checks(full_core, rect_tail, *, prefix_length: int):
    """Three independent FP32 causal rows on the *same* real post-RoPE QKV.

    This discriminates an actual masking/indexing error from BF16 kernel
    reduction differences without materializing O(L^2) attention scores.
    """
    import math

    q = full_core["q"].detach().float().cpu()
    k = full_core["k"].detach().float().cpu()
    v = full_core["v"].detach().float().cpu()
    square = full_core["core"].detach().float().cpu()
    rect = rect_tail.detach().float().cpu()
    assert q.ndim == k.ndim == v.ndim == 4
    qheads, kvheads = q.shape[2], k.shape[2]
    assert q.shape[1] == k.shape[1] == 1 and qheads % kvheads == 0
    scale = full_core["scale"]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[3])
    group_size = qheads // kvheads
    # K and V are shared inside each GQA group, exactly as in Megatron.
    rows = tuple(sorted({0, max(0, rect.shape[0] // 2), rect.shape[0] - 1}))
    for local in rows:
        absolute = prefix_length + local
        qr = q[absolute, 0]  # [Hq, D]
        kr = k[:absolute+1, 0].permute(1, 0, 2)
        vr = v[:absolute+1, 0].permute(1, 0, 2)
        kr = kr.repeat_interleave(group_size, dim=0)
        vr = vr.repeat_interleave(group_size, dim=0)
        # The valid mask for absolute query position i is keys 0..i.
        scores = torch.einsum("hd,hkd->hk", qr, kr) * float(scale)
        probs = torch.softmax(scores, dim=-1)
        oracle = torch.einsum("hk,hkd->hd", probs, vr).reshape(-1)
        sq_out = square[absolute, 0]
        rc_out = rect[local, 0]
        print(
            f"QWEN SPLIT ATTN_FP32_ROW local={local} abs={absolute} "
            f"square_max_abs={float((sq_out-oracle).abs().max()):.8g} "
            f"rect_max_abs={float((rc_out-oracle).abs().max()):.8g} "
            f"square_rect_max_abs={float((sq_out-rc_out).abs().max()):.8g}",
            flush=True,
        )


def test_real_qwen_first_attention_full_vs_external_kv(monkeypatch):
    """Actual Qwen layer-1 weights + real-token LN activations, P+S up to 16K."""
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context
    from verl.models.mcore.tpr import rectangular_attention as cann_adapter
    from verl.models.mcore.tpr import attention as tpr_attention_module

    # Both the controlled Full reference and TPR external-KV implementation
    # ultimately call THIS same CANN adapter. Instrument both entry points
    # without changing Q/K/V or the backend/rounding. Distinguish:
    #   (a) Q/K/V before core, (b) square-vs-rectangle FA with identical QKV,
    #   (c) core-vs-linear-proj numerical differences.
    modes = {"value": "not-started"}
    captures = {}
    original_cann = cann_adapter.rectangular_causal_attention

    def traced_cann(query, key, value, **kwargs):
        mode = modes["value"]
        if mode in ("full", "prefix", "suffix"):
            if mode in captures:
                raise AssertionError(f"duplicate Attention Core call in mode={mode}")
            result = original_cann(query, key, value, **kwargs)
            captures[mode] = {
                "q": query.detach().clone(),
                "k": key.detach().clone(),
                "v": value.detach().clone(),
                "core": result.detach().clone(),
                "scale": kwargs.get("softmax_scale"),
            }
            return result
        return original_cann(query, key, value, **kwargs)

    monkeypatch.setattr(cann_adapter, "rectangular_causal_attention", traced_cann)
    monkeypatch.setattr(tpr_attention_module, "rectangular_causal_attention", traced_cann)
    fixture, target = _setup_real_model()
    p, s = _lengths()
    tokens = _real_recorded_tokens(p, s).to("npu")
    model = fixture._make_qwen_model(torch.device("npu"), tpr=True, max_sequence_length=p+s)
    assert target.assert_model_scale(model) > 1_500_000_000
    attention = model.decoder.layers[0].self_attention
    assert type(attention).__name__ == "TPRSelfAttention"
    print(f"QWEN SPLIT ATTENTION P={p} S={s} layer=1 real_weights=True", flush=True)
    # QKV's linear output can agree while Megatron's per-head Q/K RMSNorm
    # or subsequent RoPE still differs. Capture these before core Attention.
    pre_core = {}
    projection = {}
    qkv_linear = {}
    qkv_output_grads = {}

    def capture_qkv_linear(_module, args, output):
        mode = modes["value"]
        if mode in ("full", "prefix", "suffix"):
            tensor = output[0] if isinstance(output, tuple) else output
            qkv_linear[mode] = (args[0].detach().clone(), tensor.detach().clone())
            if tensor.requires_grad:
                tensor.register_hook(
                    lambda grad, name=mode: qkv_output_grads.__setitem__(
                        name, grad.detach().clone()
                    )
                )

    attention.linear_qkv.register_forward_hook(capture_qkv_linear)

    def capture_norm(name):
        def hook(_module, args, output):
            mode = modes["value"]
            if mode in ("full", "prefix", "suffix"):
                assert isinstance(output, torch.Tensor)
                pre_core[(mode, name)] = output.detach().clone()
        return hook

    def capture_projection(_module, args, output):
        mode = modes["value"]
        if mode not in ("full", "prefix", "suffix"):
            return
        out = output[0] if isinstance(output, tuple) else output
        projection[mode] = (
            args[0].detach().clone(), out.detach().clone(),
        )

    if attention.q_layernorm is not None:
        attention.q_layernorm.register_forward_hook(capture_norm("q_norm"))
    if attention.k_layernorm is not None:
        attention.k_layernorm.register_forward_hook(capture_norm("k_norm"))
    attention.linear_proj.register_forward_hook(capture_projection)

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
    modes["value"] = "full"
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

    # Optional *test-only* causal intervention, run only on this standalone
    # first Attention module: use the exact same physical M=P+S for the
    # Prefix QKV GEMM as the Full path. This holds the QKV weight and all
    # real-token inputs constant, and does not modify TPR core or any model
    # code. Compare with the default run to test whether prefix QKV M
    # sensitivity causes the downstream core/projection divergence.
    if os.environ.get("TPR_QWEN17_SPLIT_PREFIX_QKV_FIXED_M", "0") == "1":
        original_qkv_forward = attention.linear_qkv.forward

        def fixed_m_prefix_qkv_forward(hidden_states, *args, **kwargs):
            if modes["value"] != "prefix":
                return original_qkv_forward(hidden_states, *args, **kwargs)
            if hidden_states.shape[0] != p:
                raise AssertionError(
                    f"unexpected Prefix QKV physical M {hidden_states.shape[0]} vs {p}"
                )
            physically_padded = torch.cat(
                (
                    hidden_states,
                    hidden_states.new_zeros((s, *hidden_states.shape[1:])),
                ),
                dim=0,
            )
            result = original_qkv_forward(physically_padded, *args, **kwargs)
            if isinstance(result, tuple):
                return (result[0][:p], *result[1:])
            if not isinstance(result, torch.Tensor):
                raise TypeError(f"unexpected QKV result: {type(result)!r}")
            return result[:p]

        monkeypatch.setattr(
            attention.linear_qkv, "forward", fixed_m_prefix_qkv_forward
        )
        print(
            f"QWEN SPLIT DIAGNOSTIC ONLY: Prefix QKV physical M={p+s} "
            f"instead of M={p}; all other model modules unchanged",
            flush=True,
        )

    prefix_hidden = attention_input[:p].clone().requires_grad_(True)
    suffix_hidden = attention_input[p:].clone().requires_grad_(True)
    prefix_ctx = TPRAttentionContext(
        0, p, suffix_rotary_pos_emb=_rope(model, 0, p)
    )
    modes["value"] = "prefix"
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
    modes["value"] = "suffix"
    with use_tpr_attention_context(suffix_ctx):
        suffix_out, suffix_bias = attention(suffix_hidden, None)
    modes["value"] = "oracle"
    assert suffix_bias is None

    # First compare actual post-RoPE QKV. If these differ, investigating
    # rectangular Attention arithmetic before QKV/RoPE would be misleading.
    assert set(captures) == {"full", "prefix", "suffix"}, (
        f"missing core captures: {set(captures)}"
    )
    full_core = captures["full"]
    prefix_core = captures["prefix"]
    suffix_core = captures["suffix"]
    # The suffix QKV GEMM was already seen to match exactly. The prefix
    # QKV GEMM instead sees M=P rather than M=P+S; compare THAT input/output.
    assert set(qkv_linear) == {"full", "prefix", "suffix"}, (
        f"missing QKV linear capture: {set(qkv_linear)}"
    )
    _stats(
        "ATTN_PREFIX_QKV_LINEAR_INPUT",
        qkv_linear["prefix"][0], qkv_linear["full"][0][:p],
    )
    _stats(
        "ATTN_PREFIX_QKV_LINEAR_OUTPUT",
        qkv_linear["prefix"][1], qkv_linear["full"][1][:p],
    )
    _stats(
        "ATTN_SUFFIX_QKV_LINEAR_OUTPUT",
        qkv_linear["suffix"][1], qkv_linear["full"][1][-s:],
    )
    if ("suffix", "q_norm") in pre_core:
        _stats(
            "ATTN_Q_AFTER_NORM_BEFORE_ROPE",
            pre_core[("suffix", "q_norm")],
            pre_core[("full", "q_norm")][-s:],
        )
    if ("suffix", "k_norm") in pre_core:
        _stats(
            "ATTN_K_AFTER_NORM_BEFORE_ROPE",
            pre_core[("suffix", "k_norm")],
            pre_core[("full", "k_norm")][-s:],
        )
        _stats(
            "ATTN_PREFIX_K_AFTER_NORM",
            pre_core[("prefix", "k_norm")],
            pre_core[("full", "k_norm")][:p],
        )
    _stats("ATTN_Q_POST_ROPE", suffix_core["q"], full_core["q"][-s:])
    _stats("ATTN_K_PREFIX_POST_ROPE", prefix_core["k"], full_core["k"][:p])
    _stats("ATTN_V_PREFIX", prefix_core["v"], full_core["v"][:p])
    _stats("ATTN_K_ALL_POST_ROPE", suffix_core["k"], full_core["k"])
    _stats("ATTN_V_ALL", suffix_core["v"], full_core["v"])
    _stats("ATTN_CORE_FULL_VS_SPLIT", suffix_core["core"], full_core["core"][-s:])
    _stats(
        "ATTN_PROJ_INPUT",
        projection["suffix"][0], projection["full"][0][-s:],
    )
    _stats(
        "ATTN_PROJ_OUTPUT",
        projection["suffix"][1], projection["full"][1][-s:],
    )

    # The strongest shape oracle: a RECTANGULAR Attention call on the
    # exact, unchanged post-RoPE Q/K/V used by the FULL 1152x1152 kernel.
    # If this already differs from the Full tail, the CANN square/rect
    # physical execution shape is numerically noninvariant at this length.
    with torch.no_grad():
        controlled_rect_tail = original_cann(
            full_core["q"][-s:], full_core["k"], full_core["v"],
            softmax_scale=full_core["scale"], dropout_p=0.0
        )
    _stats("ATTN_SAME_QKV_SQUARE_VS_RECT", controlled_rect_tail, full_core["core"][-s:])
    _stats("ATTN_SAME_QKV_RECT_VS_SPLIT_CORE", suffix_core["core"], controlled_rect_tail)

    # Independent no-grad intervention: swap ONLY the cache (Full's genuine
    # post-RoPE K/V) while keeping the real Split suffix Q and its S-shaped
    # CANN Attention launch unchanged. Quantifies the causal contribution
    # from the Prefix QKV path, rather than blaming the rectangular mask.
    with torch.no_grad():
        core_using_full_kv = original_cann(
            suffix_core["q"],
            full_core["k"], full_core["v"],
            softmax_scale=suffix_core["scale"], dropout_p=0.0,
        )
    _stats(
        "ATTN_INJECT_FULL_KV_CORE_VS_FULL",
        core_using_full_kv, full_core["core"][-s:],
    )
    _stats(
        "ATTN_INJECT_FULL_KV_CORE_VS_SPLIT",
        core_using_full_kv, suffix_core["core"],
    )
    with torch.no_grad():
        out_using_full_kv = attention.linear_proj(core_using_full_kv)
        if isinstance(out_using_full_kv, tuple):
            out_using_full_kv = out_using_full_kv[0]
    _stats(
        "ATTN_INJECT_FULL_KV_PROJ_VS_FULL",
        out_using_full_kv, projection["full"][1][-s:],
    )

    # One more causal intervention: apply the real output projection to
    # EXACTLY the full reference core's suffix tensor, but in a standalone
    # [S,1,H] GEMM rather than the full reference's [P+S,1,H] GEMM.
    # This isolates projection M-shape effects WITHOUT any KV differences.
    with torch.no_grad():
        proj_from_full_core = attention.linear_proj(full_core["core"][-s:])
        proj_from_split_core = attention.linear_proj(suffix_core["core"])
        if isinstance(proj_from_full_core, tuple):
            proj_from_full_core = proj_from_full_core[0]
        if isinstance(proj_from_split_core, tuple):
            proj_from_split_core = proj_from_split_core[0]
    full_proj_tail = projection["full"][1][-s:]
    split_proj_tail = projection["suffix"][1]
    _stats(
        "ATTN_PROJ_SAME_CORE_DIFFERENT_M",
        proj_from_full_core, full_proj_tail,
    )
    _stats(
        "ATTN_PROJ_SHORT_M_KV_ONLY_DIFF",
        proj_from_split_core, proj_from_full_core,
    )
    _stats(
        "ATTN_PROJ_REPLAY_VS_REAL_SPLIT",
        proj_from_split_core, split_proj_tail,
    )
    _core_fp32_sparse_row_checks(full_core, controlled_rect_tail, prefix_length=p)
    print(
        "QWEN SPLIT CORE DIAG: q_same="
        f"{torch.equal(suffix_core['q'],full_core['q'][-s:])} "
        "kv_same="
        f"{torch.equal(suffix_core['k'],full_core['k']) and torch.equal(suffix_core['v'],full_core['v'])}",
        flush=True
    )
    suffix_out.backward(upstream)

    # Backward decomposition is the last distinct equivalence question:
    #   (1) Are gradients delivered TO the QKV GEMM identical?
    #   (2) If yes, do full vs split dW GEMM reductions/accumulations differ?
    # The fixed-M Prefix intervention changes the physical GEMM dW path
    # too, but does not force the two dW contributions to be accumulated
    # in the same operation as Full.
    assert set(qkv_output_grads) == {"full", "prefix", "suffix"}, (
        f"missing QKV output gradient hooks: {set(qkv_output_grads)}"
    )
    full_dy = qkv_output_grads["full"]
    prefix_dy = qkv_output_grads["prefix"]
    suffix_dy = qkv_output_grads["suffix"]
    assert full_dy.shape[0] == p+s
    assert prefix_dy.shape[0] in (p, p+s)
    assert suffix_dy.shape[0] == s
    _stats("ATTN_QKV_DY_PREFIX_FULL_VS_SPLIT", prefix_dy[:p], full_dy[:p])
    _stats("ATTN_QKV_DY_SUFFIX_FULL_VS_SPLIT", suffix_dy, full_dy[p:])
    if prefix_dy.shape[0] == p+s:
        zero_tail = torch.count_nonzero(prefix_dy[p:]).item()
        print(
            f"QWEN SPLIT ATTN_QKV_PADDED_DY_TAIL nonzero={zero_tail} "
            f"total={prefix_dy[p:].numel()}",
            flush=True,
        )
        assert zero_tail == 0, "padded Prefix QKV output got nonzero gradient"

    # Report every relevant failed coordinate before deciding whether an
    # apparent per-element gate failure is materially significant. Use a
    # handful of precise FP64 scalar dot-products, *not* an expensive
    # full [out,in] FP64 matmul, as an independent mathematical dW oracle.
    wname = "linear_qkv.weight"
    ref_wgrad = full_params[wname]
    got_wgrad = dict(attention.named_parameters())[wname].grad.detach().float().cpu()
    wdelta = (got_wgrad - ref_wgrad).abs()
    tolerance = 1e-2 + 1e-2 * ref_wgrad.abs()
    mismatch = wdelta > tolerance
    print(
        "QWEN SPLIT ATTN_QKV_DW_GATE "
        f"dtype={dict(attention.named_parameters())[wname].grad.dtype} "
        f"shape={tuple(ref_wgrad.shape)} "
        f"fail_count={int(mismatch.sum())}/{mismatch.numel()} "
        f"max_abs={float(wdelta.max()):.8g} "
        f"rel_l2={float(torch.linalg.vector_norm(got_wgrad-ref_wgrad) / torch.linalg.vector_norm(ref_wgrad).clamp_min(1e-12)):.8g}",
        flush=True,
    )
    x_full = qkv_linear["full"][0]
    x_prefix = qkv_linear["prefix"][0]
    x_suffix = qkv_linear["suffix"][0]
    assert (
        full_dy.ndim == prefix_dy.ndim == suffix_dy.ndim == 3
        and tuple(x_full.shape) == (p+s, 1, ref_wgrad.shape[1])
        and x_prefix.shape[0] == prefix_dy.shape[0]
        and tuple(x_suffix.shape) == (s, 1, ref_wgrad.shape[1])
        and full_dy.shape[-1] == ref_wgrad.shape[0]
    ), "QKV linear layout changed; sampled dW oracle must be updated"
    for flat_index in torch.topk(wdelta.flatten(), k=min(5, wdelta.numel())).indices.tolist():
        oi, ii = divmod(flat_index, ref_wgrad.shape[1])
        f64_full = (
            full_dy[:, 0, oi].double().cpu() * x_full[:, 0, ii].double().cpu()
        ).sum()
        f64_split = (
            prefix_dy[:, 0, oi].double().cpu() * x_prefix[:, 0, ii].double().cpu()
        ).sum() + (
            suffix_dy[:, 0, oi].double().cpu() * x_suffix[:, 0, ii].double().cpu()
        ).sum()
        print(
            f"QWEN SPLIT ATTN_QKV_DW_FP64 out={oi} in={ii} "
            f"native_grad={float(ref_wgrad[oi, ii]):.8g} "
            f"split_grad={float(got_wgrad[oi, ii]):.8g} "
            f"oracle_full={float(f64_full):.8g} "
            f"oracle_split={float(f64_split):.8g} "
            f"oracle_delta={float(f64_split-f64_full):.8g}",
            flush=True,
        )
    # Largest absolute dW differences are not necessarily the 34
    # elements failing atol=rtol=1e-2: a 0.5 BF16 difference at |dW|=72
    # is still within the 0.73 elementwise tolerance. Inspect ACTUAL
    # failed coordinates and their independent FP64 scalar references.
    if bool(mismatch.any()):
        ratio = torch.where(
            mismatch,
            wdelta / tolerance.clamp_min(1e-12),
            torch.zeros_like(wdelta),
        )
        for flat_index in torch.topk(
            ratio.flatten(), k=min(5, int(mismatch.sum()))
        ).indices.tolist():
            oi, ii = divmod(flat_index, ref_wgrad.shape[1])
            full_exact = (
                full_dy[:, 0, oi].double().cpu()
                * x_full[:, 0, ii].double().cpu()
            ).sum()
            split_exact = (
                prefix_dy[:, 0, oi].double().cpu()
                * x_prefix[:, 0, ii].double().cpu()
            ).sum() + (
                suffix_dy[:, 0, oi].double().cpu()
                * x_suffix[:, 0, ii].double().cpu()
            ).sum()
            print(
                f"QWEN SPLIT ATTN_QKV_DW_GATE_VIOLATION "
                f"out={oi} in={ii} "
                f"native_grad={float(ref_wgrad[oi, ii]):.8g} "
                f"split_grad={float(got_wgrad[oi, ii]):.8g} "
                f"abs_diff={float(wdelta[oi, ii]):.8g} "
                f"allowed={float(tolerance[oi, ii]):.8g} "
                f"oracle_full={float(full_exact):.8g} "
                f"oracle_split={float(split_exact):.8g} "
                f"oracle_delta={float(split_exact-full_exact):.8g}",
                flush=True,
            )
    pgrad = dict(attention.named_parameters())[wname]
    print(
        f"QWEN SPLIT ATTN_QKV_GRAD_BUFFER "
        f"param_dtype={pgrad.dtype} param_grad_dtype={pgrad.grad.dtype} "
        f"has_main_grad={getattr(pgrad, 'main_grad', None) is not None}",
        flush=True,
    )

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
    param_tolerance_failures = []
    for name, expected in full_params.items():
        if name.startswith(("linear_qkv", "linear_proj", "q_layernorm", "k_layernorm")):
            actual = split_params[name].grad
            _stats("ATTN_PARAM_" + name, actual, expected)
            try:
                torch.testing.assert_close(
                    actual.detach().float().cpu(), expected, atol=1e-2, rtol=1e-2
                )
            except AssertionError as exc:
                # Preserve the strict original gate, but do not stop after
                # the FIRST QKV parameter: see Q/K normalization and every
                # projection before reporting an aggregate failure.
                param_tolerance_failures.append((name, str(exc).splitlines()[0]))
    print(
        f"QWEN SPLIT ATTN_PARAM_GATE bad_count={len(param_tolerance_failures)} "
        f"bad={param_tolerance_failures}",
        flush=True,
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
    if param_tolerance_failures:
        raise AssertionError(
            f"real Qwen Attention parameter-gradient tolerance failed: "
            f"{param_tolerance_failures}"
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
        add_capture(i, layer.self_attention, "attn_in")
        add_capture(i, layer.self_attention.linear_qkv, "qkv_out")
        add_capture(i, layer.self_attention, "attn_out")
        add_capture(i, layer.mlp.linear_fc1, "fc1_out")
        add_capture(i, layer.mlp.linear_fc2, "fc2_in")
        add_capture(i, layer.mlp.linear_fc2, "fc2_out")
        add_capture(i, layer.mlp, "mlp_out")

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
            for stage in (
                "attn_in", "qkv_out", "attn_out",
                "fc1_out", "fc2_in", "fc2_out", "mlp_out",
            ):
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
