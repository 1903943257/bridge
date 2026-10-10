"""Matched Qwen3-1.7B Megatron Native vs real TPR Forest training probe.

Complements the independent HF Full vs AReaL-DTA test. All paths use
8 cropped real SWE TQ rows (P=128,S=64), identical deterministic advantages,
same within-backend old policy, and the same objective equation.

Measurable pairs:
   HF Full / HF DTA     : full HF gradients in the other test
   Megatron Native/TPR : this file; all 311 parameter names with *sampled*
                         gradients/AdamW steps (not falsely full gradients)
Cross-backend HF Full vs Megatron Native response-logprob control is explicit.
DOES NOT label HF-vs-Megatron gradient differences as TPR numerical drift.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

pytestmark=pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_TRIPLET_TPR")!="1",
    reason="opt-in three-way comparison: real Megatron TPR actor training",
)


def _loss_terms(new,old,adv,objective,clip):
    if objective=="fixed_logprob":
        return -new*adv
    ratio=torch.exp(new-old)
    if objective=="ppo_unclipped":
        return -ratio*adv
    if objective=="ppo":
        return -torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv)
    raise ValueError(f"invalid objective {objective}")


def _loss_adapter(*, model_output, data, dp_group=None):
    """One objective for BOTH packed native and TPR compact logical rows.

    Packed log_probs per row:
      native: [query0,..,query191] with prompt length 128
      TPR:    [each local supervised query,...,one unused last query]
    The actual per-row packed length is attention_mask.sum, never the
    maximum padded response length.
    """
    from .test_qwen3_1_7b_areal_backward_ppo_npu import _proxy_advantages
    del _proxy_advantages  # objective data must be carried in data, not rebuilt
    objective=os.environ.get("TPR_DTA_BWD_OBJECTIVE","ppo")
    clip=float(os.environ.get("TPR_DTA_BWD_PPO_CLIP","0.2"))
    logits=model_output["log_probs"]
    if logits.ndim!=1:
        raise AssertionError("expected [packed_query] Megatron logprobs")
    offset=0
    numerator=logits.float().sum()*0.0
    for i in range(len(data["responses"])):
        prompt_len=int(data["prompts"][i].numel())
        valid_len=int(data["attention_mask"][i].sum().item())
        response_mask=data["response_mask"][i].bool()
        positions=torch.nonzero(response_mask,as_tuple=False).flatten()
        if positions.numel()==0:
            offset+=valid_len
            continue
        if int(positions[-1])>=valid_len-prompt_len:
            raise AssertionError("response mask is outside current packed row")
        indices=positions.to(logits.device)+offset+prompt_len-1
        new=logits.index_select(0,indices).float()
        old=data["old_log_probs"][i].to(new.device).index_select(
            0,positions.to(data["old_log_probs"][i].device)).float()
        adv=data["advantages"][i].to(new.device).index_select(
            0,positions.to(data["advantages"][i].device)).float()
        numerator=numerator+_loss_terms(new,old,adv,objective,clip).sum()
        offset+=valid_len
    if offset!=logits.numel():
        raise AssertionError(
            f"packed logprob count={logits.numel()} != attention lengths={offset}")
    # Both paths use precisely 512 supervised response tokens.
    denominator=int(os.environ.get("TPR_TRIPLET_TOTAL_VALID_TOKENS","512"))
    return numerator/denominator, {}


def _gradient_samples(model,n=4096):
    samples={}
    total_params=0
    for name,p in model.named_parameters():
        if p.grad is None:
            raise AssertionError(f"missing TPR/Native gradient for {name}")
        if not bool(torch.isfinite(p.grad).all()):
            raise AssertionError(f"nonfinite gradient {name}")
        samples[name]=p.grad.detach().reshape(-1)[:n].float().cpu().clone()
        total_params+=1
    if total_params==0:
        raise AssertionError("no gradients")
    return samples


def _weight_samples(model,n=4096):
    return {
        name:p.detach().reshape(-1)[:n].float().cpu().clone()
        for name,p in model.named_parameters()
    }


def _compare_snapshots(native,tpr):
    if set(native)!=set(tpr):
        raise AssertionError(
            f"gradient parameter namespaces differ "
            f"only_native={sorted(set(native)-set(tpr))[:8]} "
            f"only_tpr={sorted(set(tpr)-set(native))[:8]}")
    ref2=act2=diff2=dot=0.0
    for name,a in native.items():
        b=tpr[name]
        if a.shape!=b.shape:
            raise AssertionError(f"gradient sample shape mismatch {name}")
        aa=a.double();bb=b.double();d=aa-bb
        ref2+=float(aa.square().sum())
        act2+=float(bb.square().sum())
        diff2+=float(d.square().sum())
        dot+=float((aa*bb).sum())
    return {
        "rel_l2":(diff2/max(ref2,1e-60))**0.5,
        "cosine":dot/max((ref2*act2)**0.5,1e-60),
        "count":len(native),
        "sample_values":sum(x.numel() for x in native.values()),
    }


def _clip_metrics(new,old,adv,clip):
    ratio=torch.exp(new.float()-old.float())
    outside=((ratio<1-clip)|(ratio>1+clip))
    effective=(((adv>0)&(ratio>1+clip))|
               ((adv<0)&(ratio<1-clip)))
    return float(outside.float().mean()),float(effective.float().mean())


def _engine(model):
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    engine=MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module=[model]
    engine.engine_config=SimpleNamespace(
        tpr_enabled=True,tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,context_parallel_size=1,
        expert_model_parallel_size=1,virtual_pipeline_model_parallel_size=None,
        use_fused_kernels=False,dynamic_context_parallel=False,
    )
    engine.model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config=model.config
    engine.enable_routing_replay=False
    engine.get_data_parallel_size=lambda:1
    engine.get_data_parallel_group=lambda:None
    return engine


@pytest.mark.skipif(
    not hasattr(torch,"npu") or not torch.npu.is_available(),
    reason="requires physical Ascend NPU",
)
def test_real_megatron_tpr_matches_full_and_hf_reference():
    # Keep the MindSpeed initialization sequence identical to the already
    # validated Native-vs-TPR real TQ gate. Do not initialize HF and Megatron
    # in the same execution process.
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("",None)
    from ..profiling._qwen3_profile_target import resolve_qwen3_profile_target
    from . import test_tpr_qwen3_compatibility_npu as fixture
    from .test_qwen3_1_7b_real_tq_ppo_npu import (
        _load_real_tq_probe,_rows,_as_jagged,_native_response_logprobs,
    )
    from .test_qwen3_1_7b_areal_backward_ppo_npu import _proxy_advantages

    p,s=128,64
    objective=os.getenv("TPR_DTA_BWD_OBJECTIVE","ppo")
    if objective not in ("ppo","ppo_unclipped","fixed_logprob"):
        pytest.fail(f"invalid objective: {objective}")
    target=resolve_qwen3_profile_target()
    if target.size!="1.7B":
        pytest.fail(f"requires Qwen3-1.7B, got {target.label}")
    fixture.QWEN_MODEL_PATH=target.path
    batch=_load_real_tq_probe(prompt_length=p,response_length=s)
    rows=_rows(batch,"input_ids")
    if len(rows)!=8 or any(x.numel()!=192 for x in rows):
        pytest.fail("three-way experiment needs the exact 8x192 cropped TQ")
    response_masks=_rows(batch,"response_mask")
    valid_tokens=sum(int(x.sum()) for x in response_masks)
    if valid_tokens!=512 or any(not bool(x.bool().all()) for x in response_masks):
        pytest.fail("same HF experiment uses all 8x64 response tokens; "
                    "TPR response_mask differs, cannot make three-way claim")
    from .test_qwen3_1_7b_areal_backward_ppo_npu import _recorded_ppo_data
    adv_cpu,recorded_old=_recorded_ppo_data(Path(os.environ["TPR_REAL_TQ_BATCH"]),8,s)
    if adv_cpu is None:
        adv_cpu=_proxy_advantages(8,s)
    batch["advantages"]=_as_jagged(list(adv_cpu))
    os.environ["TPR_TRIPLET_TOTAL_VALID_TOKENS"]=str(valid_tokens)

    fixture._initialize_single_rank_megatron()
    device=torch.device("npu")
    native=fixture._make_qwen_model(device,tpr=False,max_sequence_length=192)
    num_params=target.assert_model_scale(native)
    old=[]
    with torch.no_grad():
        for ids in rows:
            _,response_lp=_native_response_logprobs(
                native,ids.to(device),prompt_length=p,temperature=1.0)
            old.append(response_lp.detach().float().cpu())
    batch["old_log_probs"]=_as_jagged(old)
    native.zero_grad(set_to_none=True)
    captured_native=[]
    total_loss=0.
    for i,ids in enumerate(rows):
        packed,logp=_native_response_logprobs(
            native,ids.to(device),prompt_length=p,temperature=1.0)
        captured_native.append(logp.detach().float().cpu())
        mini=TensorDict({
            key:batch[key][i].unsqueeze(0).to(device)
            for key in ("prompts","responses","attention_mask","response_mask",
                        "old_log_probs","advantages")
        },batch_size=[1])
        value,_=_loss_adapter(model_output={"log_probs":packed},
                              data=mini,dp_group=None)
        value.backward()
        total_loss+=float(value.detach())
    native_grads=_gradient_samples(native)
    native_before=_weight_samples(native)
    lr=float(os.getenv("TPR_DTA_BWD_ADAM_LR","1e-4"))
    native_opt=torch.optim.AdamW(native.parameters(),lr=lr,
                                 weight_decay=0.0,foreach=False)
    native_opt.step()
    native_after=_weight_samples(native)
    native_step={k:native_after[k]-v for k,v in native_before.items()}
    native_lp=torch.stack(captured_native)
    print("P1 TPR_TRIPLET NATIVE "
          f"objective={objective} loss={total_loss:.9g} "
          f"num_grad_parameters={len(native_grads)} "
          f"params={num_params} optimizer_step=EXECUTED",
          flush=True)
    del native,native_opt
    gc.collect()
    torch.npu.empty_cache()

    tpr=fixture._make_qwen_model(device,tpr=True,max_sequence_length=192)
    if target.assert_model_scale(tpr)!=num_params:
        raise AssertionError("Native and TPR parameter count mismatch")
    tpr.config.no_sync_func=None
    tpr.config.grad_scale_func=lambda value:value
    tpr.config.finalize_model_grads_func=lambda *args,**kwargs:None
    tpr.config.calculate_per_token_loss=False
    tpr.zero_grad(set_to_none=True)
    tu.assign_non_tensor(batch,tpr_capture_log_probs=True)
    engine=_engine(tpr)
    output=engine.forward_backward_batch(
        batch,loss_function=_loss_adapter,forward_only=False)
    logical=engine._tpr_captured_log_probs
    keys=[(row,i) for row in range(8) for i in range(s)]
    if set(logical)!=set(keys):
        raise AssertionError(
            f"Forest capture mismatch expected={len(keys)} got={len(logical)}")
    tpr_lp=torch.tensor(
        [[logical[row,i] for i in range(s)] for row in range(8)],
        dtype=torch.float32)
    tpr_loss=float(sum(output["loss"]))
    tpr_grads=_gradient_samples(tpr)
    tpr_before=_weight_samples(tpr)
    tpr_opt=torch.optim.AdamW(tpr.parameters(),lr=lr,
                              weight_decay=0.0,foreach=False)
    tpr_opt.step()
    tpr_after=_weight_samples(tpr)
    tpr_step={k:tpr_after[k]-v for k,v in tpr_before.items()}
    diff=(tpr_lp-native_lp).abs()
    stats=_compare_snapshots(native_grads,tpr_grads)
    weight_drift=_compare_snapshots(native_before,tpr_before)
    step=_compare_snapshots(native_step,tpr_step)
    advantage=adv_cpu.float()
    clip=float(os.getenv("TPR_DTA_BWD_PPO_CLIP","0.2"))
    outside,effective=_clip_metrics(tpr_lp,torch.stack(old),advantage,clip)
    print(
        "P1 TPR_TRIPLET SUMMARY backend=MEGATRON "
        f"objective={objective} rows=8 p=128 s=64 "
        f"response_logp_max={float(diff.max()):.9g} "
        f"response_logp_mean={float(diff.mean()):.9g} "
        f"response_logp_gt0p2={int((diff>0.2).sum())} "
        f"ratio_outside_frac={outside:.9g} "
        f"effective_clip_frac={effective:.9g} "
        f"grad_sample_rel_l2={stats['rel_l2']:.9g} "
        f"grad_sample_cosine={stats['cosine']:.9g} "
        f"sampled_parameters={stats['count']} "
        f"sampled_gradient_values={stats['sample_values']} "
        f"initial_weight_sample_max_diff={max(float((native_before[k]-tpr_before[k]).abs().max()) for k in native_before):.9g} "
        f"ppo_loss_native={total_loss:.9g} "
        f"ppo_loss_tpr={tpr_loss:.9g} "
        f"ppo_loss_delta={tpr_loss-total_loss:.9g} "
        f"step_sample_rel_l2={step['rel_l2']:.9g} "
        "gradient_scope=SAMPLED_PER_PARAMETER optimizer=AdamW_step "
        "tpr_execution=REAL_FOREST numerical_parity=DIAGNOSTIC_ONLY",
        flush=True)
    if weight_drift["rel_l2"]!=0:
        raise AssertionError(
            "Megatron Native / TPR initial checkpoint weights mismatch")
    output_dir=os.getenv("TPR_DTA_TRIPLET_DIR")
    if output_dir:
        dest=Path(output_dir)
        dest.mkdir(parents=True,exist_ok=True)
        artifact={
            "backend":"MEGATRON_QWEN3_1_7B",
            "objective":objective,
            "n_rows":8,
            "prompt_length":128,
            "response_length":64,
            "checkpoint":str(target.path),
            "full_response_logprobs":native_lp,
            "tpr_response_logprobs":tpr_lp,
            "tpr_grad_sample_rel_l2":stats["rel_l2"],
            "tpr_grad_sample_cosine":stats["cosine"],
            "tpr_loss_delta":tpr_loss-total_loss,
            "tpr_step_sample_rel_l2":step["rel_l2"],
            "tpr_gradient_scope":"SAMPLED_PER_PARAMETER",
        }
        path=dest/"megatron_full_tpr.pt"
        torch.save(artifact,path)
        print(f"P1 TPR_TRIPLET MEGATRON_ARTIFACT path={path}",
              flush=True)
    print("P1 TPR_TRIPLET RESULT status=PASS "
          "native_vs_tpr=MEGATRON_PAIRED "
          "cross_framework_gradient_parity=NOT_CLAIMED "
          "numerical_parity=DIAGNOSTIC_ONLY",flush=True)
