"""Isolate full-vs-cutoff shape drift on real SWE TQ and Qwen3-1.7B.

The previously named "Native" Phase-5 reference deliberately replaces core
attention with our controlled CANN wrapper. This test compares that controlled
reference against an explicitly UNMODIFIED Megatron core-attention spec, and
uses an independent CPU FP32 causal-attention oracle on the Q/K/V captured from
each forward.

It does not create synthetic models or token sequences, does not change TPR
production code, and does not lower PPO correctness thresholds.

Run:
    TPR_RUN_QWEN17_ROOTCAUSE=1 TPR_QWEN_1_7B_PATH=... \\
      python -m pytest -s -q --tb=short \\
      tests/models/mcore/tpr/correctness/test_qwen3_1_7b_shape_rootcause_npu.py
"""

from __future__ import annotations

import gc
import os

import pytest
import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires a real Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_ROOTCAUSE") != "1",
    reason="Set TPR_RUN_QWEN17_ROOTCAUSE=1 for real checkpoint shape diagnosis",
)

# Always compare the same physical inputs, not two independently sampled rows.
_STAGES = (
    "attn_in",
    "qkv_out",
    "core_q",
    "core_k",
    "core_v",
    "core_out",
    "proj_in",
    "proj_out",
    "attn_out",
    "mlp_in",
    "fc1_in",
    "fc1_out",
    "fc2_in",
    "fc2_out",
    "mlp_out",
)


def _capture_model_stages(model):
    """Collect complete per-layer sequence tensors, including ALL prefix K/V."""
    captures = {}
    handles = []
    core_scales = {}

    def hook_for(layer_number, stage, *, which="out", index=0):
        def callback(module, args, output):
            value = args[index] if which == "arg" else output
            if isinstance(value, tuple):
                value = value[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"unexpected tensor at layer={layer_number} stage={stage}: {type(value)}"
                )
            captures[(layer_number, stage)] = value.detach().float().cpu().contiguous()
            if stage == "core_out":
                core_scales[layer_number] = getattr(module, "softmax_scale", None)
        return callback

    for layer in model.decoder.layers:
        layer_number = layer.self_attention.layer_number
        attn = layer.self_attention
        handles.extend((
            attn.register_forward_hook(hook_for(layer_number, "attn_in", which="arg")),
            attn.linear_qkv.register_forward_hook(hook_for(layer_number, "qkv_out")),
            attn.core_attention.register_forward_hook(
                hook_for(layer_number, "core_q", which="arg", index=0)
            ),
            attn.core_attention.register_forward_hook(
                hook_for(layer_number, "core_k", which="arg", index=1)
            ),
            attn.core_attention.register_forward_hook(
                hook_for(layer_number, "core_v", which="arg", index=2)
            ),
            attn.core_attention.register_forward_hook(hook_for(layer_number, "core_out")),
            attn.linear_proj.register_forward_hook(
                hook_for(layer_number, "proj_in", which="arg")
            ),
            attn.linear_proj.register_forward_hook(hook_for(layer_number, "proj_out")),
            attn.register_forward_hook(hook_for(layer_number, "attn_out")),
            layer.mlp.register_forward_hook(hook_for(layer_number, "mlp_in", which="arg")),
            layer.mlp.linear_fc1.register_forward_hook(
                hook_for(layer_number, "fc1_in", which="arg")
            ),
            layer.mlp.linear_fc1.register_forward_hook(
                hook_for(layer_number, "fc1_out")
            ),
            layer.mlp.linear_fc2.register_forward_hook(
                hook_for(layer_number, "fc2_in", which="arg")
            ),
            layer.mlp.linear_fc2.register_forward_hook(
                hook_for(layer_number, "fc2_out")
            ),
            layer.mlp.register_forward_hook(hook_for(layer_number, "mlp_out")),
        ))
    return captures, core_scales, handles


def _run_recorded_prefix(model, tokens, *, cutoff):
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    ids = tokens[:cutoff].to(next(model.parameters()).device, dtype=torch.long)
    traces, scales, handles = _capture_model_stages(model)
    try:
        with torch.no_grad():
            logits = model(
                input_ids=ids.unsqueeze(0),
                position_ids=torch.arange(cutoff, device=ids.device)[None, :],
                attention_mask=None,
            )
            if tuple(logits.shape[:2]) != (1, cutoff):
                raise AssertionError(f"unexpected GPT logits: {tuple(logits.shape)}")
            values = {}
            for query_abs in (69, 76, 99, 114, 126):
                if query_abs < cutoff and query_abs + 1 < len(tokens):
                    label = tokens[query_abs + 1].to(ids.device).reshape(1)
                    value = vocab_parallel_log_probs_from_logits(
                        logits[0, query_abs:query_abs + 1], label
                    )
                    values[query_abs] = float(value.detach().float().cpu()[0])
        return traces, scales, values
    finally:
        for handle in handles:
            handle.remove()


def _first_tensor_divergence(full_trace, cropped_trace, cutoff, backend):
    first = None
    print(f"FIRST-DIVERGENCE SCAN backend={backend} full=128 cutoff={cutoff}")
    for layer in range(1, 29):
        for stage in _STAGES:
            key = (layer, stage)
            if key not in full_trace or key not in cropped_trace:
                raise AssertionError(f"missing Qwen trace {key} for {backend}")
            a = full_trace[key][:cutoff]
            b = cropped_trace[key]
            if a.shape != b.shape:
                raise AssertionError(
                    f"trace shape mismatch at {key}: {tuple(a.shape)} vs {tuple(b.shape)}"
                )
            errors = (a - b).abs()
            max_abs = errors.max().item()
            if max_abs == 0:
                continue
            differs = errors.reshape(cutoff, -1).ne(0).any(dim=1)
            first_token = int(torch.nonzero(differs)[0, 0])
            entry = (layer, stage, first_token, max_abs)
            if first is None:
                first = entry
                print(
                    f"FIRST DIVERGENCE backend={backend} cutoff={cutoff} "
                    f"layer={layer:02d} stage={stage} "
                    f"first_token={first_token} max_abs={max_abs:.8g}"
                )
            if layer <= 3:
                print(
                    f"  DRIFT layer={layer:02d} stage={stage} "
                    f"first_token={first_token} max_abs={max_abs:.8g} "
                    f"changed_tokens={int(differs.sum())}/{cutoff}"
                )
    if first is None:
        print(f"FIRST DIVERGENCE backend={backend} cutoff={cutoff}: NONE (all stages equal)")
    return first


def _probe_first_layer_mlp_shape(model, full_trace, cropped_trace, cutoff, *, backend):
    """Identify FC1/Gate/FC2 shape-dependence on exactly equal real inputs.

    The first MLP is the first observed difference in the controlled CANN
    reference. We enforce equality at its entrance before attributing any
    difference to a particular GEMM or gate operation. No artificial tokens,
    no model replacement and no threshold changes.
    """
    layer = 1
    for stage in ("mlp_in", "fc1_in", "fc1_out", "fc2_in", "fc2_out", "mlp_out"):
        a = full_trace[(layer, stage)][:cutoff]
        b = cropped_trace[(layer, stage)]
        errors = (a - b).abs()
        max_error = float(errors.max())
        changed = int(errors.reshape(cutoff, -1).ne(0).any(dim=1).sum())
        print(
            f"MLP ROOTCAUSE backend={backend} cutoff={cutoff} layer=01 "
            f"stage={stage} max_abs={max_error:.9g} "
            f"changed_tokens={changed}/{cutoff}"
        )

    same_input = torch.equal(
        full_trace[(layer, "fc1_in")][:cutoff],
        cropped_trace[(layer, "fc1_in")],
    )
    fc1_equal = torch.equal(
        full_trace[(layer, "fc1_out")][:cutoff],
        cropped_trace[(layer, "fc1_out")],
    )
    gate_equal = torch.equal(
        full_trace[(layer, "fc2_in")][:cutoff],
        cropped_trace[(layer, "fc2_in")],
    )
    fc2_equal = torch.equal(
        full_trace[(layer, "fc2_out")][:cutoff],
        cropped_trace[(layer, "fc2_out")],
    )
    if not same_input:
        print(
            f"MLP ROOTCAUSE RESULT backend={backend} cutoff={cutoff}: "
            "FC1 input already differs; investigate preceding layer computation."
        )
        return

    if not fc1_equal:
        source = "FC1 GEMM (identical FC1 inputs, sequence-length-dependent output)"
    elif not gate_equal:
        source = "gate/activation (identical FC1 output)"
    elif not fc2_equal:
        source = "FC2 GEMM (identical FC2 inputs, sequence-length-dependent output)"
    else:
        source = "none inside MLP (check residual/add/fusion after MLP)"

    print(
        f"MLP ROOTCAUSE RESULT backend={backend} cutoff={cutoff}: "
        f"first_internal_source={source}"
    )

    # Independent CPU FP32 GEMM on *real* layer-1 FC1 inputs/weights,
    # restricted to 3 representative real tokens to avoid huge CPU work.
    if not fc1_equal:
        from torch.nn import functional as F

        sample_positions = [p for p in (2, 6, 69) if p < cutoff]
        fc1 = model.decoder.layers[0].mlp.linear_fc1
        w = fc1.weight.detach().float().cpu()
        bias = getattr(fc1, "bias", None)
        if bias is not None:
            bias = bias.detach().float().cpu()
        for pos in sample_positions:
            real_input = full_trace[(layer, "fc1_in")][pos].reshape(1, -1)
            oracle = F.linear(real_input, w, bias).reshape(-1)
            a = full_trace[(layer, "fc1_out")][pos].reshape(-1)
            b = cropped_trace[(layer, "fc1_out")][pos].reshape(-1)
            print(
                f"MLP FP32 FC1 ORACLE backend={backend} cutoff={cutoff} "
                f"token={pos} input_equal=True "
                f"full_vs_fp32_max={float((a - oracle).abs().max()):.9g} "
                f"cutoff_vs_fp32_max={float((b - oracle).abs().max()):.9g} "
                f"full_vs_cutoff_max={float((a - b).abs().max()):.9g}"
            )



def _replay_fc2_at_fixed_physical_shape(
    model, full_trace, cropped_trace, cutoff, *, backend
):
    """Test a *causal intervention*: same real FC2 inputs, fixed physical M.

    MLP's FC2 maps each token independently; padding extra rows to the
    full-context physical GEMM size cannot change the true unpadded result.
    A restored bitwise match under padding points to GEMM tiling/accumulation
    rather than FC2 weights, gate, attention, or incorrect logical positions.

    No training or production path is changed by this diagnostic.
    """
    layer = 1
    x_full = full_trace[(layer, "fc2_in")]
    x_cut = cropped_trace[(layer, "fc2_in")]
    if not torch.equal(x_full[:cutoff], x_cut):
        raise AssertionError("FC2 replay expects byte-identical real inputs")

    fc2 = model.decoder.layers[0].mlp.linear_fc2
    parameter = next(fc2.parameters())
    device, dtype = parameter.device, parameter.dtype

    def apply(x):
        y = fc2(x.to(device=device, dtype=dtype))
        if isinstance(y, tuple):
            y = y[0]
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"FC2 returned non-tensor {type(y)!r}")
        return y.detach().float().cpu()

    # Replaying both shapes rules out stale hook snapshots or hidden module
    # state as the cause of the observed mismatch.
    with torch.no_grad():
        y_full_replayed = apply(x_full)
        y_cut_replayed = apply(x_cut)
        pad = torch.zeros_like(x_full[cutoff:])
        y_padded = apply(torch.cat((x_cut, pad), dim=0))[:cutoff]

    y_full = full_trace[(layer, "fc2_out")]
    y_cut = cropped_trace[(layer, "fc2_out")]
    if y_full_replayed.shape != y_full.shape:
        raise AssertionError("FC2 replay changed physical output shape")

    def summarize(name, a, b):
        difference = (a - b).abs()
        max_abs = difference.max().item()
        different = int(
            difference.reshape(cutoff, -1).ne(0).any(dim=1).sum()
        )
        print(
            f"FC2 SHAPE REPLAY backend={backend} cutoff={cutoff} "
            f"comparison={name} max_abs={max_abs:.9g} "
            f"changed_tokens={different}/{cutoff}"
        )
        return max_abs

    summarize("full_replay_vs_original", y_full_replayed[:cutoff], y_full[:cutoff])
    summarize("short_replay_vs_original", y_cut_replayed, y_cut)
    original_drift = summarize("short_vs_full", y_cut, y_full[:cutoff])
    padding_drift = summarize("pad_to_128_vs_full", y_padded, y_full[:cutoff])
    if padding_drift == 0:
        print(
            f"FC2 PADDING RESULT backend={backend} cutoff={cutoff}: "
            "BITWISE_MATCH; padding the FC2 physical GEMM to 128 eliminated "
            "the measured shape-dependent FC2 forward difference."
        )
    else:
        print(
            f"FC2 PADDING RESULT backend={backend} cutoff={cutoff}: "
            f"RESIDUAL max_abs={padding_drift:.9g} "
            f"vs_unpadded={original_drift:.9g}; padding alone did not "
            "fully restore full-context FC2 arithmetic."
        )

    # Ground the independent high-precision reference in the actual layer
    # weights and real FC2 input, not a synthetic tensor or new model.
    from torch.nn import functional as F

    weight_fp32 = fc2.weight.detach().float().cpu()
    bias = getattr(fc2, "bias", None)
    bias_fp32 = None if bias is None else bias.detach().float().cpu()
    for pos in (2, 6, 69):
        if pos >= cutoff:
            continue
        x = x_full[pos].reshape(1, -1)
        oracle_fp32 = F.linear(x, weight_fp32, bias_fp32).reshape(-1)
        full = y_full[pos].reshape(-1)
        short = y_cut[pos].reshape(-1)
        padded = y_padded[pos].reshape(-1)
        print(
            f"FC2 FP32 ORACLE backend={backend} cutoff={cutoff} token={pos} "
            f"full_vs_fp32_max={float((full-oracle_fp32).abs().max()):.9g} "
            f"short_vs_fp32_max={float((short-oracle_fp32).abs().max()):.9g} "
            f"padded_vs_fp32_max={float((padded-oracle_fp32).abs().max()):.9g}"
        )



def _fp32_attention_from_same_real_qkv(trace, scales, layer):
    """Independent, non-CANN CPU FP32 attention with causal/GQA semantics."""
    q = trace[(layer, "core_q")]
    k = trace[(layer, "core_k")]
    v = trace[(layer, "core_v")]
    # Native Megatron layout [sequence, batch, heads, head_dim].
    assert q.ndim == k.ndim == v.ndim == 4
    q_len, batch, n_heads, head_dim = q.shape
    kv_len, _, n_kv_heads, _ = k.shape
    assert batch == 1 and n_heads % n_kv_heads == 0
    q = q.permute(1, 2, 0, 3)
    k = k.permute(1, 2, 0, 3).repeat_interleave(n_heads // n_kv_heads, dim=1)
    v = v.permute(1, 2, 0, 3).repeat_interleave(n_heads // n_kv_heads, dim=1)
    scale = scales.get(layer)
    if scale is None:
        scale = head_dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) * float(scale)
    query_positions = torch.arange(kv_len - q_len, kv_len)
    key_positions = torch.arange(kv_len)
    masked = key_positions[None, :] > query_positions[:, None]
    attn_scores = attn_scores.masked_fill(masked[None, None, :, :], float("-inf"))
    probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(probs, v)
    return output.permute(2, 0, 1, 3).reshape(q_len, batch, n_heads * head_dim)


def _print_fp32_core_oracle(trace, scales, backend, cutoff, *, layers=(1, 2, 3, 4)):
    for layer in layers:
        oracle = _fp32_attention_from_same_real_qkv(trace, scales, layer)
        physical = trace[(layer, "core_out")]
        if oracle.shape != physical.shape:
            raise AssertionError(
                f"FP32 oracle/core shape mismatch: {tuple(oracle.shape)} vs {tuple(physical.shape)}"
            )
        delta = (physical - oracle).abs()
        rel = float(torch.linalg.vector_norm(physical - oracle)
                    / torch.linalg.vector_norm(oracle).clamp_min(1e-12))
        print(
            f"FP32 CORE ORACLE backend={backend} S={cutoff} layer={layer:02d} "
            f"max_abs={delta.max().item():.8g} rel_l2={rel:.8g}"
        )


def test_real_qwen3_1_7b_first_shape_divergence():
    # This is a focused, real-TQ test, not a replacement for the Phase-5 PPO
    # correctness gate. Its default 64+64 crop is deliberately explicit.
    # Match the real NPU/MindSpeed bootstrap used by existing Qwen Ring CP
    # tests: unmodified DotProductAttention uses Megatron RNG and softmax
    # paths that may require MindSpeed's NPU compatibility repatch.
    import sys
    saved_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = saved_argv
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)
    repatch({
        "context_parallel_size": 1,
        "context_parallel_algo": "megatron_cp_algo",
    })

    from ..profiling._qwen3_profile_target import resolve_qwen3_profile_target
    from . import test_tpr_qwen3_compatibility_npu as qwen_fixture
    from .test_qwen3_1_7b_real_tq_ppo_npu import _load_real_tq_probe

    target = resolve_qwen3_profile_target()
    if target.size != "1.7B":
        pytest.fail(f"expected Qwen3-1.7B, got {target.label}")
    qwen_fixture.QWEN_MODEL_PATH = target.path

    p = int(os.environ.get("TPR_QWEN17_PPO_PROMPT", "64"))
    r = int(os.environ.get("TPR_QWEN17_PPO_RESPONSE", "64"))
    if (p, r) != (64, 64):
        pytest.fail("Focused real-TQ diagnostic currently fixes P=64, R=64")
    batch = _load_real_tq_probe(prompt_length=p, response_length=r)
    tokens = batch["input_ids"][0].detach().cpu().long()
    assert tokens.numel() == 128
    qwen_fixture._initialize_single_rank_megatron()
    print(f"ROOTCAUSE REAL CHECKPOINT: {target.path}")
    print("ROOTCAUSE TOKENS: recorded TQ row=0; cropped_real_window=64+64")

    for backend in ("controlled_cann", "unmodified_megatron"):
        import torch_npu  # noqa: F401

        if backend == "unmodified_megatron":
            mode = "native"
        else:
            mode = None

        model = qwen_fixture._make_qwen_model(
            torch.device("npu"), tpr=False,
            max_sequence_length=128, core_attention_module=mode,
        )
        print(
            f"ROOTCAUSE BACKEND={backend} "
            f"core_attention={type(model.decoder.layers[0].self_attention.core_attention).__module__}."
            f"{type(model.decoder.layers[0].self_attention.core_attention).__name__}"
        )
        if backend == "unmodified_megatron":
            assert not isinstance(
                model.decoder.layers[0].self_attention.core_attention,
                qwen_fixture._ProfileFusedCausalAttention
            ), "Unmodified reference accidentally still uses TPR CANN adapter"
        try:
            print(
                f"ROOTCAUSE RUN backend={backend} cutoff=128 START",
                flush=True,
            )
            full_trace, full_scales, full_lp = _run_recorded_prefix(
                model, tokens, cutoff=128
            )
            print(
                f"ROOTCAUSE RUN backend={backend} cutoff=128 DONE",
                flush=True,
            )
            _print_fp32_core_oracle(
                full_trace, full_scales, backend, 128
            )
            for cutoff in (70, 94, 102):
                print(
                    f"ROOTCAUSE RUN backend={backend} cutoff={cutoff} START",
                    flush=True,
                )
                crop_trace, crop_scales, crop_lp = _run_recorded_prefix(
                    model, tokens, cutoff=cutoff
                )
                print(
                    f"ROOTCAUSE RUN backend={backend} cutoff={cutoff} DONE",
                    flush=True,
                )
                for query_abs in sorted(set(full_lp).intersection(crop_lp)):
                    print(
                        f"SHAPE LOGPROB backend={backend} cutoff={cutoff} "
                        f"query_abs={query_abs} full={full_lp[query_abs]:.8f} "
                        f"cutoff_value={crop_lp[query_abs]:.8f} "
                        f"delta={crop_lp[query_abs] - full_lp[query_abs]:.8f}"
                    )
                _first_tensor_divergence(
                    full_trace, crop_trace, cutoff, backend
                )
                _probe_first_layer_mlp_shape(
                    model, full_trace, crop_trace, cutoff, backend=backend
                )
                if os.environ.get("TPR_QWEN17_ROOTCAUSE_FC2_REPLAY", "1") == "1":
                    _replay_fc2_at_fixed_physical_shape(
                        model, full_trace, crop_trace, cutoff, backend=backend
                    )
                _print_fp32_core_oracle(
                    crop_trace, crop_scales, backend, cutoff, layers=(1, 2)
                )
                del crop_trace
        finally:
            del model
            gc.collect()
            torch.npu.empty_cache()
    print("ROOTCAUSE DIAGNOSTIC COMPLETE (not a PPO correctness PASS)")
