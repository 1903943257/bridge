"""Ascend Qwen3-1.7B: AReaL-DTA *training* Pop Forward numerical control.

Ported schedule is backward_permute -> push(cache_len) -> build_cache ->
pop_byblock -> pop Forward -> merged per-attachment loss logprobs.
This experiment does not call actual torch.autograd.backward.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch

from .test_qwen3_1_7b_hf_dta_reference_npu import _real_cropped_rows
from ._qwen17_dta_style_reference import hf_full_logprobs, error_summary
from ._qwen17_areal_dta_reference import hf_areal_forward_only
from ._qwen17_areal_training_forward_reference import hf_areal_training_loss_forward

pytestmark=pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_DTA_TRAIN_FWD")!="1",
    reason="opt-in numerical reproduction of AReaL backward schedule Forward",
)


@pytest.mark.skipif(
    not hasattr(torch,"npu") or not torch.npu.is_available(),
    reason="requires Ascend NPU",
)
def test_areal_training_pop_forward_vs_full():
    from transformers import AutoModelForCausalLM, DynamicCache
    import transformers

    data_path=Path(os.environ["TPR_REAL_TQ_BATCH"])
    checkpoint=Path(os.environ["TPR_QWEN_1_7B_PATH"])
    if not checkpoint.is_dir():
        pytest.fail(f"checkpoint missing: {checkpoint}")
    p=int(os.getenv("TPR_QWEN17_DTA_PROMPT","128"))
    s=int(os.getenv("TPR_QWEN17_DTA_RESPONSE","64"))
    if (p,s)!=(128,64):
        pytest.fail("hold input fixed at 128 prompt + 64 response")
    attn=os.getenv("TPR_QWEN17_DTA_HF_ATTN","sdpa")
    if attn not in ("sdpa","eager"):
        pytest.fail("attention backend must be sdpa or eager")
    sizes=tuple(int(x.strip()) for x in
                os.getenv("TPR_QWEN17_DTA_TRAIN_BLOCK_SIZES","64,-1").split(","))
    if not sizes or any(v==0 or v < -1 for v in sizes):
        pytest.fail("invalid block size(s)")
    cut=os.getenv("TPR_QWEN17_DTA_TRAIN_CUT_F1_TAIL","1")=="1"
    max_len=int(os.getenv("TPR_QWEN17_DTA_TRAIN_MAX_SEQ_LEN","192"))
    if max_len < p+s:
        pytest.fail("max_seq_len must be >=192")
    device=torch.device("npu:0")
    torch.npu.set_device(device)
    rows=tuple(x.to(device).contiguous() for x in
               _real_cropped_rows(data_path,p=p,s=s))
    model=AutoModelForCausalLM.from_pretrained(
        str(checkpoint),torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    ).to(device).train()  # actual training mode, not model.eval()
    if next(model.parameters()).dtype!=torch.bfloat16:
        pytest.fail("HF model not BF16")
    print(
        "P0 DTA_TRAIN CONFIG "
        f"checkpoint={checkpoint} model=HF_QWEN3_1_7B "
        f"torch={torch.__version__} transformers={transformers.__version__} "
        f"device={device} dtype=BF16 attention={attn} "
        "input=REAL_TQ_128P_64S model_mode=train "
        "push_autograd=False pop_autograd=True "
        f"block_sizes={list(sizes)} cut_f1_tail={cut} max_seq_len={max_len} "
        "goal=REAL_DTA_TRAIN_LOSS_LOGPROBS actual_backward=False",
        flush=True,
    )
    full=tuple(hf_full_logprobs(model,x,DynamicCache) for x in rows)
    fwd=hf_areal_forward_only(model,rows,DynamicCache)
    full_response=tuple(x[p-1:p+s-1] for x in full)
    fwd_response=tuple(x[p-1:p+s-1] for x in fwd.logprobs)
    _,fwd_stats=error_summary(full_response,fwd_response)
    print(
        "P0 DTA_TRAIN FWD_ONLY_RESPONSE "
        f"max_abs={fwd_stats['max_abs']:.9g} "
        f"mean_abs={fwd_stats['mean_abs']:.9g} "
        f"p95_abs={fwd_stats['p95_abs']:.9g} "
        f"num_gt_0p2={fwd_stats['num_gt_0p2']}",
        flush=True,
    )
    for block_size in sizes:
        result=hf_areal_training_loss_forward(
            model,rows,DynamicCache,block_size=block_size,
            cut_f1_tail=cut,max_seq_len=max_len,
        )
        row_stats,summary=error_summary(full,result.logprobs)
        response=tuple(x[p-1:p+s-1] for x in result.logprobs)
        _,rs=error_summary(full_response,response)
        _,train_vs_fwd=error_summary(fwd.logprobs,result.logprobs)
        print(
            "P0 DTA_TRAIN SUMMARY "
            f"block_size={block_size} cut_f1_tail={cut} "
            f"num_tokens={summary['num_tokens']} "
            f"max_abs={summary['max_abs']:.9g} "
            f"mean_abs={summary['mean_abs']:.9g} "
            f"p95_abs={summary['p95_abs']:.9g} "
            f"num_gt_0p2={summary['num_gt_0p2']} "
            f"leaves={result.n_leaves} "
            f"n_cache={result.n_cache_forwards} "
            f"n_pop={result.n_pop_forwards} "
            f"cache_m={list(result.physical_m_cache)} "
            f"pop_m={list(result.physical_m_pop)} "
            f"pop_starts={list(result.pop_starts)} "
            "numerical_parity=DIAGNOSTIC_ONLY",
            flush=True,
        )
        print(
            "P0 DTA_TRAIN RESPONSE_SUMMARY "
            f"block_size={block_size} num_tokens={rs['num_tokens']} "
            f"max_abs={rs['max_abs']:.9g} "
            f"mean_abs={rs['mean_abs']:.9g} "
            f"p95_abs={rs['p95_abs']:.9g} "
            f"num_gt_0p2={rs['num_gt_0p2']} "
            "comparison=HF_FULL_VS_DTA_TRAIN_POP_LOSS",
            flush=True,
        )
        print(
            "P0 DTA_TRAIN VS_FWD_ONLY "
            f"block_size={block_size} "
            f"max_abs={train_vs_fwd['max_abs']:.9g} "
            f"mean_abs={train_vs_fwd['mean_abs']:.9g} "
            "comparison=TRAIN_BACKWARD_PERMUTE_POP_VS_FORWARD_PERMUTE_PUSH",
            flush=True,
        )
        for eid,(kind,start,end,leaf) in enumerate(result.events):
            print(
                "P0 DTA_TRAIN EVENT "
                f"block_size={block_size} event_id={eid} kind={kind} "
                f"start={start} end={end} M={end-start} leaf={leaf}",
                flush=True,
            )
        for item in row_stats:
            i=item["row"];q=item["worst_query"]
            owner=result.token_owner[i][q]
            kind,start,end,leaf=result.events[owner]
            print(
                "P0 DTA_TRAIN ROW "
                f"block_size={block_size} row={i} "
                f"max_abs={item['max_abs']:.9g} "
                f"mean_abs={item['mean_abs']:.9g} "
                f"worst_query_abs={q} "
                f"origin_kind={kind} origin_start={start} "
                f"origin_M={end-start} origin_event={owner}",
                flush=True,
            )
        outliers=sorted(
            (
                (float(error),i,p-1+offset)
                for i,(a,b) in enumerate(zip(full_response,response,strict=True))
                for offset,error in enumerate(
                    (a.float()-b.float()).abs().cpu().tolist())
            ),
            reverse=True,
        )[:12]
        for delta,i,q in outliers:
            owner=result.token_owner[i][q]
            kind,start,end,leaf=result.events[owner]
            print(
                "P0 DTA_TRAIN OUTLIER "
                f"block_size={block_size} row={i} "
                f"response_offset={q-(p-1)} query_abs={q} "
                f"abs_delta={delta:.9g} "
                f"origin_kind={kind} origin_start={start} "
                f"origin_M={end-start} origin_event={owner}",
                flush=True,
            )
        i=5;q=189
        owner=result.token_owner[i][q]
        kind,start,end,leaf=result.events[owner]
        print(
            "P0 DTA_TRAIN TARGET "
            f"block_size={block_size} row={i} query_abs={q} "
            f"full={float(full[i][q]):.9g} "
            f"forward_only={float(fwd.logprobs[i][q]):.9g} "
            f"train_pop={float(result.logprobs[i][q]):.9g} "
            f"pop_abs={float((full[i][q]-result.logprobs[i][q]).abs()):.9g} "
            f"origin_kind={kind} origin_start={start} origin_M={end-start}",
            flush=True,
        )
    if any(param.grad is not None for param in model.parameters()):
        raise AssertionError("numerical Forward oracle accumulated parameter grads")
    print(
        "P0 DTA_TRAIN RESULT status=PASS "
        "execution=TRAIN_LOSS_FORWARD_CONTROL numerical_parity=UNVERIFIED "
        "full_dta_backward=NOT_IMPLEMENTED optimizer=NOT_RUN",
        flush=True,
    )
