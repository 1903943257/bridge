"""NPU e2e DTA backward: gradient, PPO surrogate step, and GEMM precision ablations.

Test-only integration, 910B2C / Qwen3-1.7B BF16. Main comparison:
baseline independent full-row per-trajectory PPO; DTA training engine full
backward with KV / logprob / entropy / fork-logit gradient relay.
Interventions: bf16 M tile slicing and fp32 internal linear GEMMs. 
This is a PPO *proxy* unless real advantages + old_log_probs are in TQ dump.
HF-shifted B-1 labels adaptation is explicit; see engine adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
from math import sqrt
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import types

import pytest
import torch
import torch.nn.functional as F

from ._qwen17_areal_dta_reference import hf_areal_forward_only
from ._qwen17_areal_training_forward_reference import areal_backward_plan
from ._qwen17_areal_native_dta_backward import DTAEngine
from ._qwen17_dta_style_reference import (
    _logp_from_previous_logits, error_summary,
)
from .test_qwen3_1_7b_hf_dta_reference_npu import _real_cropped_rows


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_DTA_BACKWARD") != "1",
    reason="opt-in AReaL-DTA full backward & PPO update on physical NPU",
)


def _tensor_rows(value, n_rows: int, s: int):
    """Accept recorded [N,S] data only; reject silent reshaping/cropping."""
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n_rows, s):
        return None
    return value.detach().float().cpu().contiguous()


def _recorded_ppo_data(path: Path, n_rows: int, s: int):
    data = torch.load(path, map_location="cpu", weights_only=False)
    td = data.get("tensordict")
    if td is None:
        return None, None
    keys = set(td.keys())
    adv = None
    old = None
    for k in ("advantages", "advantage"):
        if k in keys:
            adv = _tensor_rows(td[k], n_rows, s)
            break
    for k in ("old_log_probs", "old_logprobs", "old_log_probs_actor"):
        if k in keys:
            old = _tensor_rows(td[k], n_rows, s)
            break
    return adv, old


def _proxy_advantages(n_rows: int, s: int):
    """Signed, nonconstant deterministic proxy. NOT rollout GAE."""
    pos = torch.arange(s, dtype=torch.float32)[None, :]
    row = torch.arange(n_rows, dtype=torch.float32)[:, None]
    return (torch.sin(pos * 0.51 + row * 0.39)
            + 0.4 * torch.cos(pos * 0.13 - row * 0.27)).contiguous()


def _ppo_row_loss(logp, entropy, old_lp, advantage, *, p, s, clip_eps, entropy_coef, n_rows, objective="ppo"):
    lp = logp[p - 1:p + s - 1].float()
    # Entropy belongs to the query position that predicts each response
    # token; like shifted logprob, this starts at p-1, not p.
    ent = entropy[p - 1:p + s - 1].float()
    if len(lp) != s or len(ent) != s:
        raise AssertionError("PPO response alignment error")
    if objective in ("ppo", "ppo_unclipped"):
        ratio = (lp - old_lp).exp()
        if objective == "ppo":
            clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
        else:
            # PPO importance ratio without clipping; derivative retains
            # exp(logp-old_lp), unlike the fixed-logprob diagnostic.
            surrogate = ratio * advantage
    elif objective == "fixed_logprob":
        # Identical d(loss)/d(logprob) at every token, regardless of
        # Full/DTA logprob drift. Diagnostic only, not PPO.
        surrogate = lp * advantage
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return (-surrogate.sum() - entropy_coef * ent.sum()) / (n_rows * s)


@contextmanager
def _linear_ablation(model, mode: str, tile_m: int):
    """Temporarily patch Qwen Linear forward; preserve autograd to BF16 weights.

    SPLIT chunks physical row dimension M *without* changing logical sequence.
    FP32 computes Linear inner GEMM in FP32 but casts output back to BF16,
    keeping LayerNorm/Attention/KV and model parameter storage BF16.
    """
    if mode not in ("native", "m_split", "fp32_linear", "m_split_fp32"):
        raise ValueError(f"unknown GEMM mode {mode}")
    if tile_m <= 0:
        raise ValueError("tile_m must be positive")
    patched = []
    try:
        if mode != "native":
            for name, layer in model.named_modules():
                if not isinstance(layer, torch.nn.Linear):
                    continue
                # named_modules() emits relative paths: unit-test modules
                # may start with "self_attn.", while HF Qwen3 prefixes them
                # with "model.layers.N.". Match component boundaries in both.
                path = f".{name}."
                if not (
                    ".self_attn." in path or ".mlp." in path
                    or name == "lm_head"
                ):
                    continue
                original = layer.forward
                def repl(self, x, *, _mode=mode, _tile=tile_m):
                    base_shape = x.shape
                    xx = x.reshape(-1, base_shape[-1])
                    chunks = xx.split(_tile, dim=0) if "m_split" in _mode else (xx,)
                    ys = []
                    for chunk in chunks:
                        if "fp32" in _mode:
                            y = F.linear(chunk.float(), self.weight.float(),
                                         None if self.bias is None else self.bias.float())
                            y = y.to(chunk.dtype)
                        else:
                            y = F.linear(chunk, self.weight, self.bias)
                        ys.append(y)
                    return torch.cat(ys, dim=0).reshape(*base_shape[:-1], -1)
                layer.forward = types.MethodType(repl, layer)
                patched.append((layer, original))
        yield len(patched)
    finally:
        for layer, original in reversed(patched):
            layer.forward = original


def _all_parameter_grads(model):
    return {
        name: None if p.grad is None else p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
    }


def _selected_post_update(model, *, slice_n=2048):
    """Bound memory: compare actual AdamW-updated parameter samples."""
    named = dict(model.named_parameters())
    preferred = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.13.self_attn.q_proj.weight",
        "model.layers.27.self_attn.o_proj.weight",
        "lm_head.weight",
    )
    return {
        name: named[name].detach().reshape(-1)[:slice_n].cpu().float().clone()
        for name in preferred if name in named
    }


def _metrics(ref, cand, *, top_k=5):
    dot=0.; norm_a=0.; norm_b=0.; diff2=0.
    ranked=[]; missing=[]
    for name, a in ref.items():
        b=cand.get(name)
        if a is None and b is None:
            continue
        if a is None or b is None:
            missing.append(name)
            continue
        if a.shape != b.shape:
            raise AssertionError(f"gradient shape mismatch {name}")
        aa=a.float().reshape(-1); bb=b.float().reshape(-1)
        d=bb-aa
        av=float(torch.linalg.vector_norm(aa.double()))
        dv=float(torch.linalg.vector_norm(d.double()))
        ranked.append((name, dv / max(av, 1e-30), av, dv))
        dot+=float((aa.double()*bb.double()).sum())
        norm_a+=av*av
        norm_b+=float((bb.double().square()).sum())
        diff2+=dv*dv
        del aa,bb,d
    return {
        "relative_l2":sqrt(diff2)/max(sqrt(norm_a),1e-30),
        "cosine":dot/max(sqrt(norm_a*norm_b),1e-30),
        "reference_norm":sqrt(norm_a),
        "candidate_norm":sqrt(norm_b),
        "worst":sorted(ranked,key=lambda x:x[1],reverse=True)[:top_k],
        "missing":missing,
        "matched_params":len(ranked),
    }


def _gradient_groups(ref, cand):
    """Group gradient drift by functional module, not just worst relative layer."""
    groups={}
    for name,a in ref.items():
        b=cand.get(name)
        if a is None or b is None:
            continue
        if ".self_attn." in name:
            family="attn."+name.split(".self_attn.",1)[1].split(".",1)[0]
        elif ".mlp." in name:
            family="mlp."+name.split(".mlp.",1)[1].split(".",1)[0]
        elif "norm" in name:
            family="other_norm"
        elif "embed_tokens" in name:
            family="embedding"
        elif "lm_head" in name:
            family="lm_head"
        else:
            family="other"
        ref2,diff2,matched=groups.get(family,(0.,0.,0))
        af=a.float().double()
        df=b.float().double()-af
        groups[family]=(ref2+float(af.square().sum()),
                        diff2+float(df.square().sum()),matched+1)
    return tuple(sorted(
        ((family,sqrt(d2)/max(sqrt(r2),1e-30),sqrt(r2),sqrt(d2),matched)
         for family,(r2,d2,matched) in groups.items()),
        key=lambda x:x[3],reverse=True,
    ))


def _full_logp_entropy(logits, tokens):
    logp = F.log_softmax(logits[0].float(), dim=-1)
    prob=logp.exp()
    full_entropy=-(prob*logp).sum(-1)
    shifted=logp[:-1].gather(-1,tokens[1:].long().unsqueeze(-1)).squeeze(-1)
    return shifted,full_entropy


def _load_model(path, attention, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(path),torch_dtype=torch.bfloat16,attn_implementation=attention,
    ).to(device).train()


def _one_step(model, mode, rows, cache_factory, old, adv, *, p,s,clip_eps,
              entropy_coef,lr,block_size,tile_m, max_seq_len,objective="ppo"):
    n_rows=len(rows)
    with _linear_ablation(model,mode,tile_m) as count:
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.0,
                                    foreach=False)
        optimizer.zero_grad(set_to_none=True)
        total_loss_value=0.0
        captured=[None]*n_rows
        captured_entropy=[None]*n_rows
        if block_size is None:
            # Independent dense reference: sum exactly the same per-row PPO loss.
            for i, tokens in enumerate(rows):
                from transformers import DynamicCache
                out=model(input_ids=tokens.unsqueeze(0),past_key_values=DynamicCache(),
                          use_cache=True)
                lp,ent=_full_logp_entropy(out.logits,tokens)
                captured[i]=lp.detach().clone()
                captured_entropy[i]=ent.detach().clone()
                loss=_ppo_row_loss(lp,ent,old[i],adv[i],p=p,s=s,
                                   clip_eps=clip_eps,entropy_coef=entropy_coef,n_rows=n_rows,
                                   objective=objective)
                total_loss_value+=float(loss.detach())
                loss.backward()
            engine_loss=None
        else:
            plan=areal_backward_plan(rows)
            attachments=[
                [({"_sequence_batch_id":row_id},length)
                 for row_id,length in attached]
                for attached in plan.attachments
            ]
            trie=SimpleNamespace(inputs=plan.sequences,attach_lists=attachments,
                                lcp_lens=plan.lcp_lens,n_sequences=n_rows)
            engine=DTAEngine(model.config,device=rows[0].device,
                             dtype=next(model.parameters()).dtype,
                             max_seq_len=max_seq_len,is_critic=False,
                             forward_only=False)
            def loss_fn(lp, entropy, attachment):
                i=attachment["_sequence_batch_id"]
                captured[i]=lp.detach().clone()
                captured_entropy[i]=entropy.detach().clone()
                return _ppo_row_loss(lp,entropy,old[i],adv[i],p=p,s=s,
                                     clip_eps=clip_eps,entropy_coef=entropy_coef,
                                     n_rows=n_rows,objective=objective)
            engine_loss=engine.backward(
                model,trie,loss_fn,block_size=block_size,cut_f1_tail=True)
            total_loss_value=float(engine_loss)
            del engine, trie
        if any(x is None for x in captured) or any(x is None for x in captured_entropy):
            raise AssertionError("PPO loss did not receive all trajectory outputs")
        grads=_all_parameter_grads(model)
        if not all(torch.isfinite(g).all() for g in grads.values() if g is not None):
            raise AssertionError("nonfinite parameter gradients")
        before=_selected_post_update(model)
        optimizer.step()
        after=_selected_post_update(model)
        updates={k:after[k]-v for k,v in before.items()}
        del optimizer
    with torch.no_grad():
        ratios=torch.cat([
            (lp[p-1:p+s-1].float().cpu()-old[i].cpu()).exp()
            for i,lp in enumerate(captured)
        ])
        ratios_finite=torch.isfinite(ratios).all().item()
        if not ratios_finite:
            raise AssertionError("nonfinite PPO ratios")
        clip_frac=float(((ratios>1+clip_eps)|(ratios<1-clip_eps)).float().mean())
        # PPO's min surrogate clips the derivative only for
        # positive-advantage upper-bound or negative-advantage lower-bound.
        signed_adv=torch.cat([x.float().cpu() for x in adv])
        effective_clip=(((signed_adv>0)&(ratios>1+clip_eps))|
                        ((signed_adv<0)&(ratios<1-clip_eps)))
        effective_clip_frac=float(effective_clip.float().mean())
    return {
        "grads":grads,
        "updated":after,
        "updates":updates,
        "logprobs":tuple(captured),
        "entropies":tuple(captured_entropy),
        "clip_frac":clip_frac,
        "effective_clip_frac":effective_clip_frac,
        # Counterfactual: only the clipped PPO objective actually suppresses
        # gradients beyond the advantage-dependent clipping threshold.
        "active_clip_frac":effective_clip_frac if objective == "ppo" else 0.0,
        "engine_loss":engine_loss,
        "loss_value":total_loss_value,
        "patched_linear_count":count,
    }


@pytest.mark.skipif(
    not hasattr(torch,"npu") or not torch.npu.is_available(),
    reason="requires physical Ascend NPU",
)
def test_real_areal_dta_full_backward_ppo_gemm():
    from transformers import DynamicCache
    checkpoint=Path(os.environ["TPR_QWEN_1_7B_PATH"])
    tq=Path(os.environ["TPR_REAL_TQ_BATCH"])
    attn=os.getenv("TPR_QWEN17_DTA_HF_ATTN","sdpa")
    p,s=128,64
    mode_list=tuple(x.strip() for x in
                    os.getenv("TPR_DTA_BWD_MODES","native,m_split,fp32_linear").split(",")
                    if x.strip())
    device=torch.device("npu:0")
    torch.npu.set_device(device)
    rows=tuple(x.to(device).contiguous() for x in _real_cropped_rows(tq,p=p,s=s))
    n_rows=int(os.getenv("TPR_DTA_BWD_ROWS","8"))
    if not 1<=n_rows<=len(rows):
        pytest.fail("invalid TPR_DTA_BWD_ROWS")
    rows=rows[:n_rows]
    block=int(os.getenv("TPR_DTA_BWD_BLOCK","64"))
    objective=os.getenv("TPR_DTA_BWD_OBJECTIVE","ppo")
    if objective not in ("ppo","ppo_unclipped","fixed_logprob"):
        pytest.fail("TPR_DTA_BWD_OBJECTIVE must be ppo, ppo_unclipped, or fixed_logprob")
    paired_full=os.getenv("TPR_DTA_BWD_PAIRED_FULL","0")=="1"
    tile=int(os.getenv("TPR_DTA_BWD_GEMM_M_TILE","32"))
    lr=float(os.getenv("TPR_DTA_BWD_ADAM_LR","1e-4"))
    eps=float(os.getenv("TPR_DTA_BWD_PPO_CLIP","0.2"))
    entropy_coef=float(os.getenv("TPR_DTA_BWD_ENTROPY_COEF","0.0"))
    max_seq_len=int(os.getenv("TPR_DTA_BWD_MAX_SEQ_LEN","192"))
    if max_seq_len<192: pytest.fail("DTA max_seq_len must >=192")
    adv_cpu,old_cpu=_recorded_ppo_data(tq,8,s)
    recorded_adv=adv_cpu is not None
    recorded_old=old_cpu is not None
    if adv_cpu is not None: adv_cpu=adv_cpu[:n_rows]
    if old_cpu is not None: old_cpu=old_cpu[:n_rows]
    if os.getenv("TPR_PPO_REQUIRE_RECORDED")=="1" and (adv_cpu is None or old_cpu is None):
        pytest.fail("actual rollout advantages and old_log_probs missing: refusing to claim real PPO")
    if adv_cpu is None: adv_cpu=_proxy_advantages(n_rows,s)
    adv=tuple(x.to(device) for x in adv_cpu)
    # Missing recorded behavior logprobs: use same-checkpoint HF Full, explicitly label proxy.
    model=_load_model(checkpoint,attn,device)
    if old_cpu is None:
        with torch.no_grad():
            old_cpu=torch.stack([
                _full_logp_entropy(model(x.unsqueeze(0),past_key_values=DynamicCache(),
                                          use_cache=True).logits,x)[0][p-1:p+s-1].cpu()
                for x in rows
            ])
    old=tuple(x.to(device) for x in old_cpu)
    source="RECORDED" if recorded_adv and recorded_old else "PPO_PROXY"
    print(
        "P1 DTA_BACKWARD CONFIG "
        f"checkpoint={checkpoint} rows={n_rows} p={p} s={s} block={block} "
        f"modes={list(mode_list)} tile_m={tile} attention={attn} "
        f"objective={objective} paired_full={paired_full} "
        f"old_source={'RECORDED' if recorded_old else 'HF_FULL_CURRENT_MODEL'} "
        f"adv_source={'RECORDED' if recorded_adv else 'DETERMINISTIC_PROXY'} "
        f"ppo_source={source} optimizer=AdamW lr={lr} clip={eps} entropy_coef={entropy_coef} "
        "grad_relay=KV_FORK_LOGPROBS_ENTROPY actual_backward=True "
        "adapter=HF_SHIFTED_LOGITS_B_MINUS_1",
        flush=True,
    )
    reference=_one_step(model,"native",rows,DynamicCache,old,adv,p=p,s=s,
                        clip_eps=eps,entropy_coef=entropy_coef,lr=lr,
                        block_size=None,tile_m=tile,max_seq_len=max_seq_len,
                        objective=objective)
    print(
        "P1 DTA_BACKWARD FULL "
        f"clip_frac={reference['clip_frac']:.9g} "
        f"matched_grad_count={sum(g is not None for g in reference['grads'].values())} "
        f"ppo_loss={reference['loss_value']:.9g} "
        f"effective_clip_frac={reference['effective_clip_frac']:.9g} "
        f"active_clip_frac={reference['active_clip_frac']:.9g} "
        "mode=BF16_FULL optimizer_step=EXECUTED",flush=True,
    )
    del model
    gc.collect()
    torch.npu.empty_cache()
    for mode in mode_list:
        model=_load_model(checkpoint,attn,device)
        current=_one_step(model,mode,rows,DynamicCache,old,adv,p=p,s=s,
                          clip_eps=eps,entropy_coef=entropy_coef,lr=lr,
                          block_size=block,tile_m=tile,max_seq_len=max_seq_len,
                          objective=objective)
        grad_stats=_metrics(reference["grads"],current["grads"])
        paired=None
        if paired_full and mode!="native":
            paired_model=_load_model(checkpoint,attn,device)
            paired=_one_step(
                paired_model,mode,rows,DynamicCache,old,adv,p=p,s=s,
                clip_eps=eps,entropy_coef=entropy_coef,lr=lr,
                block_size=None,tile_m=tile,max_seq_len=max_seq_len,
                objective=objective)
            paired_grad=_metrics(paired["grads"],current["grads"])
            _,paired_forward=error_summary(
                tuple(x[p-1:p+s-1] for x in paired["logprobs"]),
                tuple(x[p-1:p+s-1] for x in current["logprobs"]))
            print(
                "P1 DTA_BACKWARD PAIRED_FULL "
                f"mode={mode} objective={objective} "
                f"grad_rel_l2={paired_grad['relative_l2']:.9g} "
                f"grad_cosine={paired_grad['cosine']:.9g} "
                f"response_logp_mean={paired_forward['mean_abs']:.9g} "
                f"response_logp_max={paired_forward['max_abs']:.9g} "
                f"full_patched_linear_count={paired['patched_linear_count']} "
                "comparison=SAME_ABLATION_FULL_VS_DTA",
                flush=True)
            del paired,paired_model
            gc.collect()
            torch.npu.empty_cache()
        _,loss_stats=error_summary(reference["logprobs"],current["logprobs"])
        rp=tuple(x[p-1:p+s-1] for x in reference["logprobs"])
        cp=tuple(x[p-1:p+s-1] for x in current["logprobs"])
        _,response=error_summary(rp,cp)
        step_ref=reference["updated"]; step_cur=current["updated"]
        d2=0.; b2=0.
        for name in step_ref:
            d2+=float(((step_ref[name]-step_cur[name]).double().square()).sum())
            b2+=float((reference["updates"][name].double().square()).sum())
        step_rel_l2=sqrt(d2)/max(sqrt(b2),1e-30)
        print(
            "P1 DTA_BACKWARD SUMMARY "
            f"mode={mode} patched_linear_count={current['patched_linear_count']} "
            f"ppo_source={source} block={block} objective={objective} "
            f"all_logp_max={loss_stats['max_abs']:.9g} "
            f"all_logp_mean={loss_stats['mean_abs']:.9g} "
            f"response_logp_max={response['max_abs']:.9g} "
            f"response_logp_mean={response['mean_abs']:.9g} "
            f"response_logp_gt0p2={response['num_gt_0p2']} "
            f"clip_frac={current['clip_frac']:.9g} "
            f"effective_clip_frac={current['effective_clip_frac']:.9g} "
            f"active_clip_frac={current['active_clip_frac']:.9g} "
            f"ppo_loss={current['loss_value']:.9g} "
            f"ppo_loss_delta={current['loss_value']-reference['loss_value']:.9g} "
            f"grad_rel_l2={grad_stats['relative_l2']:.9g} "
            f"grad_cosine={grad_stats['cosine']:.9g} "
            f"full_grad_norm={grad_stats['reference_norm']:.9g} "
            f"dta_grad_norm={grad_stats['candidate_norm']:.9g} "
            f"param_step_sample_rel_l2={step_rel_l2:.9g} "
            f"matched_grad_params={grad_stats['matched_params']} "
            f"missing_grads={len(grad_stats['missing'])} "
            "optimizer_step=EXECUTED numerical_parity=DIAGNOSTIC_ONLY",
            flush=True,
        )
        for family,rel,norm,diff,count in _gradient_groups(reference["grads"],current["grads"]):
            print(
                "P1 DTA_BACKWARD GRAD_GROUP "
                f"mode={mode} family={family} rel_l2={rel:.9g} "
                f"full_norm={norm:.9g} diff_norm={diff:.9g} "
                f"param_count={count}",
                flush=True,
            )
        for name, rel, norm, diff in grad_stats["worst"]:
            print(
                "P1 DTA_BACKWARD WORST_GRAD "
                f"mode={mode} name={name} rel_l2={rel:.9g} "
                f"ref_norm={norm:.9g} diff_norm={diff:.9g}",flush=True,
            )
        for i,(a,b) in enumerate(zip(rp,cp,strict=True)):
            delta=(a.float()-b.float()).abs()
            idx=int(delta.argmax())
            if i==5 or i==0 or i==2:
                print(
                    "P1 DTA_BACKWARD ROW "
                    f"mode={mode} row={i} worst_query_abs={p-1+idx} "
                    f"response_max_abs={float(delta.max()):.9g} "
                    f"query189_abs={float((reference['logprobs'][i][189]-current['logprobs'][i][189]).abs()):.9g}",
                    flush=True,
                )
        del model,current
        gc.collect()
        torch.npu.empty_cache()
    print("P1 DTA_BACKWARD RESULT status=PASS backward=EXECUTED optimizer=AdamW_step numerical_parity=UNVERIFIED",flush=True)
