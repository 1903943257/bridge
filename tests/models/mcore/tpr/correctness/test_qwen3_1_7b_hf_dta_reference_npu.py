"""Opt-in independent HF Qwen3-1.7B vs DTA-style DynamicCache BF16 control.

Same Ascend device, HF model object, HF checkpoint, and real TQ token rows:
  HF Full (single cache-enabled forward)
  HF DTA-style sorted LCP/DFS cached suffix forward
  HF fixed 5-chunk row5 control with the same TPR physical boundaries

This is NOT production TPR, and not DTA backward training. The comparison
never attributes HF-vs-Megatron numerical differences to prefix reuse.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from ._qwen17_dta_style_reference import (
    hf_full_logprobs,
    hf_cached_chunk_logprobs,
    hf_dta_lcp_forward,
    error_summary,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_DTA_REF") != "1",
    reason="Set TPR_RUN_QWEN17_DTA_REF=1 for real HF DTA-style NPU control",
)


def _real_cropped_rows(path: Path, *, p: int, s: int):
    if not path.is_file():
        pytest.fail(f"missing real TQ dump: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    td = data["tensordict"]
    prompts, responses, originals = (
        list(td[key]) for key in ("prompts", "responses", "input_ids")
    )
    if not (len(prompts) == len(responses) == len(originals) == 8):
        pytest.fail("DTA-style control expects the recorded 8 real SWE trajectories")
    rows = []
    for i, (prompt, response, original) in enumerate(
        zip(prompts, responses, originals, strict=True)
    ):
        if len(prompt) < p or len(response) < s:
            pytest.fail(f"real TQ row{i} shorter than requested crop")
        if not torch.equal(
            torch.cat((prompt, response)), original
        ):
            pytest.fail(f"row{i} TQ prompt/response boundary inconsistent")
        rows.append(torch.cat((prompt[-p:], response[:s])).long())
    return rows


def _single_v_capture(model, tokens, cache_factory, *, length):
    """Capture v_proj pre-input and output at the first HF model layer."""
    if not hasattr(model.model.layers[0].self_attn, "v_proj"):
        raise AssertionError("Expected Qwen3 first-layer self_attn.v_proj")
    result = {}
    def hook(_module, args, _output):
        if result:
            raise AssertionError("HF first-layer V projection called twice")
        if not args or not isinstance(args[0], torch.Tensor):
            raise AssertionError("Expected positional HF v_proj input")
        result["input"] = args[0].detach().cpu().float().clone()
        result["v"] = _output.detach().cpu().float().clone()
    h = model.model.layers[0].self_attn.v_proj.register_forward_hook(hook)
    try:
        hf_full_logprobs(model, tokens[:length], cache_factory)
    finally:
        h.remove()
    if set(result) != {"input", "v"}:
        raise AssertionError("HF first-layer V capture missing")
    return result


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="Requires physical Ascend NPU and torch_npu",
)
def test_real_qwen3_hf_full_vs_dta_style_dynamic_cache():
    from transformers import AutoModelForCausalLM, DynamicCache
    import transformers

    tq_path = Path(os.environ["TPR_REAL_TQ_BATCH"])
    checkpoint = Path(os.environ["TPR_QWEN_1_7B_PATH"])
    if not checkpoint.is_dir():
        pytest.fail(f"Missing Qwen3-1.7B checkpoint: {checkpoint}")
    p = int(os.environ.get("TPR_QWEN17_DTA_PROMPT", "128"))
    s = int(os.environ.get("TPR_QWEN17_DTA_RESPONSE", "64"))
    if p != 128 or s != 64:
        pytest.fail(
            "first DTA diagnostic must use same 128+64 crop as Megatron baseline"
        )
    attn = os.environ.get("TPR_QWEN17_DTA_HF_ATTN", "sdpa")
    if attn not in ("sdpa", "eager"):
        pytest.fail("TPR_QWEN17_DTA_HF_ATTN must be sdpa or eager")
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    tokens = [
        x.to(device).contiguous()
        for x in _real_cropped_rows(tq_path, p=p, s=s)
    ]
    print(
        "P0 DTA_HF CONFIG "
        f"checkpoint={checkpoint} model=HF_QWEN3_1_7B "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device={device} dtype=BF16 attention={attn} "
        f"lengths={[x.numel() for x in tokens]} "
        "input=CROPPED_REAL_TQ backward=False "
        "baseline=HF_FULL_WITH_EMPTY_DYNAMIC_CACHE",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    ).to(device).eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        raise AssertionError("HF Qwen3 was not loaded in BF16")

    # Three modes operate on SAME HF model/weights, not Megatron replicas.
    full = tuple(hf_full_logprobs(model, x, DynamicCache) for x in tokens)
    lcp_dfs = hf_dta_lcp_forward(model, tokens, DynamicCache)
    row_metrics, summary = error_summary(full, lcp_dfs.logprobs)
    # Match the 512 shifted response positions of the cropped weak PPO
    # comparison; prompt-token drift must not dilute actor-response errors.
    response_full = tuple(x[p - 1:p + s - 1] for x in full)
    response_dta = tuple(x[p - 1:p + s - 1] for x in lcp_dfs.logprobs)
    _, response_summary = error_summary(response_full, response_dta)

    # Higher-fidelity AReaL-DTA forward: TokenTrie leafization, optimized
    # CompressedTrie.forward_permute, and persistent KV buffers/views.
    # Preserve lexical DFS to isolate the effect of *execution schedule*.
    from ._qwen17_areal_dta_reference import hf_areal_forward_only
    areal = hf_areal_forward_only(model, tokens, DynamicCache)
    # Same fixed AReaL K/V buffers but without forward_permute:
    # separate storage/layout effects from changed GEMM M scheduling.
    lexical_buffer = hf_areal_forward_only(
        model, tokens, DynamicCache, forward_permute=False,
    )
    _, lexical_buffer_full = error_summary(full, lexical_buffer.logprobs)
    _, buffer_only = error_summary(lcp_dfs.logprobs, lexical_buffer.logprobs)
    _, permute_only = error_summary(lexical_buffer.logprobs, areal.logprobs)
    areal_rows, areal_summary = error_summary(full, areal.logprobs)
    areal_response = tuple(x[p - 1:p + s - 1] for x in areal.logprobs)
    _, areal_response_summary = error_summary(response_full, areal_response)
    _, areal_vs_lexical = error_summary(lcp_dfs.logprobs, areal.logprobs)
    assert areal_summary["num_tokens"] == 8 * 191
    assert areal_response_summary["num_tokens"] == 8 * 64
    for item in areal_rows:
        i, q = item["row"], item["worst_query"]
        owner = areal.token_owner[i][q]
        print(
            "P0 DTA_HF AREAL_ROW "
            f"row={i} max_abs={item['max_abs']:.9g} "
            f"mean_abs={item['mean_abs']:.9g} worst_query_abs={q} "
            f"owner_visit={owner} owner_start={areal.physical_starts[owner]} "
            f"owner_M={areal.physical_m[owner]}",
            flush=True,
        )
    print(
        "P0 DTA_HF AREAL_SUMMARY "
        f"num_tokens={areal_summary['num_tokens']} "
        f"max_abs={areal_summary['max_abs']:.9g} "
        f"mean_abs={areal_summary['mean_abs']:.9g} "
        f"p95_abs={areal_summary['p95_abs']:.9g} "
        f"num_gt_0p2={areal_summary['num_gt_0p2']} "
        f"leaves={areal.n_leaves} physical_m={list(areal.physical_m)} "
        f"physical_starts={list(areal.physical_starts)} "
        f"physical_attached_rows={list(areal.physical_rows)} "
        f"saved_forward_tokens={areal.dense_tokens - areal.total_processed_tokens} "
        "scheduler=AREAL_FORWARD_PERMUTE buffer=PERSISTENT_KV forward_only=True",
        flush=True,
    )
    print(
        "P0 DTA_HF AREAL_RESPONSE_SUMMARY "
        f"num_tokens={areal_response_summary['num_tokens']} "
        f"max_abs={areal_response_summary['max_abs']:.9g} "
        f"mean_abs={areal_response_summary['mean_abs']:.9g} "
        f"p95_abs={areal_response_summary['p95_abs']:.9g} "
        f"num_gt_0p2={areal_response_summary['num_gt_0p2']} "
        "same_HF_checkpoint=True numerical_parity=DIAGNOSTIC_ONLY",
        flush=True,
    )
    print(
        "P0 DTA_HF AREAL_LEXICAL_BUFFER "
        f"max_abs={lexical_buffer_full['max_abs']:.9g} "
        f"mean_abs={lexical_buffer_full['mean_abs']:.9g} "
        f"p95_abs={lexical_buffer_full['p95_abs']:.9g} "
        f"physical_m={list(lexical_buffer.physical_m)} "
        f"physical_starts={list(lexical_buffer.physical_starts)}",
        flush=True,
    )
    print(
        "P0 DTA_HF AREAL_BUFFER_ABLATION "
        f"max_abs={buffer_only['max_abs']:.9g} "
        f"mean_abs={buffer_only['mean_abs']:.9g} "
        "comparison=LEXICAL_LAST_CACHE_VS_LEXICAL_FIXED_KV",
        flush=True,
    )
    print(
        "P0 DTA_HF AREAL_PERMUTE_ABLATION "
        f"max_abs={permute_only['max_abs']:.9g} "
        f"mean_abs={permute_only['mean_abs']:.9g} "
        "comparison=LEXICAL_FIXED_KV_VS_OPTIMIZED_FIXED_KV",
        flush=True,
    )
    print(
        "P0 DTA_HF AREAL_VS_LEXICAL "
        f"max_abs={areal_vs_lexical['max_abs']:.9g} "
        f"mean_abs={areal_vs_lexical['mean_abs']:.9g} "
        f"p95_abs={areal_vs_lexical['p95_abs']:.9g} "
        "scope=EXECUTION_ORDER_PLUS_KV_BUFFER",
        flush=True,
    )
    # Identify *response* outliers with actual optimized-DFS owning M.
    outliers = sorted(
        (
            (float(error), i, p - 1 + offset)
            for i, (dense, segmented) in enumerate(
                zip(response_full, areal_response, strict=True)
            )
            for offset, error in enumerate(
                (dense.float() - segmented.float()).abs().cpu().tolist()
            )
        ), reverse=True
    )[:12]
    for diff, i, query in outliers:
        owner = areal.token_owner[i][query]
        print(
            "P0 DTA_HF AREAL_OUTLIER "
            f"row={i} response_offset={query - (p - 1)} "
            f"query_abs={query} abs_delta={diff:.9g} "
            f"owner_visit={owner} owner_start={areal.physical_starts[owner]} "
            f"owner_M={areal.physical_m[owner]}",
            flush=True,
        )
    for item in row_metrics:
        print(
            "P0 DTA_HF DFS_ROW "
            f"row={item['row']} tokens={item['count']} "
            f"max_abs={item['max_abs']:.9g} "
            f"mean_abs={item['mean_abs']:.9g} "
            f"worst_query_abs={item['worst_query']}",
            flush=True,
        )
    print(
        "P0 DTA_HF DFS_SUMMARY "
        f"num_tokens={summary['num_tokens']} "
        f"max_abs={summary['max_abs']:.9g} "
        f"mean_abs={summary['mean_abs']:.9g} "
        f"p95_abs={summary['p95_abs']:.9g} "
        f"num_gt_0p2={summary['num_gt_0p2']} "
        f"physical_m={list(lcp_dfs.physical_m)} "
        f"physical_starts={list(lcp_dfs.physical_starts)} "
        f"saved_forward_tokens={lcp_dfs.dense_tokens-lcp_dfs.total_processed_tokens} "
        "parity=DIAGNOSTIC_ONLY",
        flush=True,
    )
    print(
        "P0 DTA_HF RESPONSE_SUMMARY "
        f"num_tokens={response_summary['num_tokens']} "
        f"max_abs={response_summary['max_abs']:.9g} "
        f"mean_abs={response_summary['mean_abs']:.9g} "
        f"p95_abs={response_summary['p95_abs']:.9g} "
        f"num_gt_0p2={response_summary['num_gt_0p2']} "
        "shifted_response_positions=True same_checkpoint=True",
        flush=True,
    )
    row = int(os.environ.get("TPR_QWEN17_DTA_TRACE_ROW", "5"))
    if not 0 <= row < len(tokens):
        pytest.fail("DTA HF target row out of range")
    boundaries = (0, 134, 158, 166, 189, 192)
    if tokens[row].numel() != boundaries[-1]:
        raise AssertionError("Fixed path is valid only for 192-token rows")
    # Capture the *actual* first root chunk inside the segmented HF run,
    # rather than assuming a separate cutoff forward is representative.
    dta_root = {}
    def first_root_v(_module, args, result):
        if dta_root:
            return  # later physical chunks are intentionally not captured
        if not args or not isinstance(args[0], torch.Tensor):
            raise AssertionError("HF segmented root V lacked input tensor")
        dta_root["input"] = args[0].detach().cpu().float().clone()
        dta_root["v"] = result.detach().cpu().float().clone()

    root_handle = model.model.layers[0].self_attn.v_proj.register_forward_hook(
        first_root_v
    )
    try:
        fixed = hf_cached_chunk_logprobs(
            model, tokens[row], boundaries, DynamicCache
        )
    finally:
        root_handle.remove()
    if set(dta_root) != {"input", "v"}:
        raise AssertionError("HF DTA root v_proj not captured")
    fixed_rows, fixed_summary = error_summary((full[row],), (fixed,))
    v_full = _single_v_capture(
        model, tokens[row], DynamicCache, length=192
    )
    v_root = _single_v_capture(
        model, tokens[row], DynamicCache, length=134
    )
    if not torch.equal(v_full["input"][:, :134], v_root["input"]):
        raise AssertionError(
            "HF full/root first-layer V GEMM inputs differ; invalid M-only control"
        )
    if not torch.equal(v_root["input"], dta_root["input"]):
        raise AssertionError("HF Native cutoff and DTA root V GEMM inputs differ")
    root_to_dta_v = (
        v_root["v"].double() - dta_root["v"].double()
    ).abs()
    print(
        "P0 DTA_HF ROOT_CUTOFF_TO_DTA "
        f"root_M=134 input_max_abs=0 "
        f"v_max_abs={float(root_to_dta_v.max()):.9g} "
        "same_HF_model=True",
        flush=True,
    )
    v_delta = (
        v_full["v"][:, :134].double() - v_root["v"].double()
    ).abs()
    diff_tokens = v_delta.reshape(134, -1).amax(-1) > 0 if v_delta.ndim == 2 else (
        v_delta.squeeze(0).reshape(134, -1).amax(-1) > 0
    )
    print(
        "P0 DTA_HF ROOT_V_SHAPE "
        "full_M=192 root_M=134 input_max_abs=0 "
        f"v_max_abs={float(v_delta.max()):.9g} "
        f"v_rel_l2={float(torch.linalg.vector_norm(v_delta) / torch.linalg.vector_norm(v_full['v'][:, :134].double()).clamp_min(1e-24)):.9g} "
        f"v_changed_tokens={int(diff_tokens.sum())}/134",
        flush=True,
    )
    print(
        "P0 DTA_HF FIXED_PATH "
        f"row={row} boundaries={list(boundaries)} "
        f"max_abs={fixed_summary['max_abs']:.9g} "
        f"mean_abs={fixed_summary['mean_abs']:.9g} "
        f"p95_abs={fixed_summary['p95_abs']:.9g} "
        f"num_gt_0p2={fixed_summary['num_gt_0p2']} "
        f"worst_query_abs={fixed_rows[0]['worst_query']} "
        f"query189_abs={float((full[row][189]-fixed[189]).abs()):.9g}",
        flush=True,
    )
    # The existing Megatron TPR row5/query189 had ~0.749 forward drift:
    # do not silently substitute some other worst HF token.
    print(
        "P0 DTA_HF AREAL_TARGET "
        f"row={row} query_abs=189 "
        f"hf_full={float(full[row][189]):.9g} "
        f"hf_lexical={float(lcp_dfs.logprobs[row][189]):.9g} "
        f"hf_lexical_buffer={float(lexical_buffer.logprobs[row][189]):.9g} "
        f"hf_areal={float(areal.logprobs[row][189]):.9g} "
        f"hf_fixed={float(fixed[189]):.9g} "
        f"areal_abs={float((full[row][189]-areal.logprobs[row][189]).abs()):.9g} "
        f"fixed_abs={float((full[row][189]-fixed[189]).abs()):.9g}",
        flush=True,
    )
    # A pass means measurements were completed, NOT numeric equivalence.
    assert summary["num_tokens"] == 8 * 191
    assert response_summary["num_tokens"] == 8 * 64
    print(
        "P0 DTA_HF RESULT status=PASS execution=FORWARD_CONTROL "
        "numerical_parity=UNVERIFIED dta_backward=NOT_IMPLEMENTED "
        "cross_framework_mixed_compare=FORBIDDEN",
        flush=True,
    )
