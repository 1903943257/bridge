"""CPU-only comparison of fixed-label 10K TP/CP/DP parity matrix outputs.

Reference is a same-engine unshared CP1 replay, NOT native Megatron Full.
Do not use these metrics to claim whole-model gradient equivalence:
canonical 64-samples-per-parameter are sparse.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

CASES=(
    "unshared_tp2cp1dp1",
    "tpr_tp2cp1dp1",
    "tpr_tp2cp2dp1",
    "tpr_tp2cp1dp2",
    "tpr_tp2cp2dp2",
)
PAIRS=(
    ("unshared_vs_tpr_no_cp","unshared_tp2cp1dp1","tpr_tp2cp1dp1"),
    ("cp_only","tpr_tp2cp1dp1","tpr_tp2cp2dp1"),
    ("dp_only_cp1","tpr_tp2cp1dp1","tpr_tp2cp1dp2"),
    ("dp_only_cp2","tpr_tp2cp2dp1","tpr_tp2cp2dp2"),
    ("cp2dp2_vs_no_cp","tpr_tp2cp1dp1","tpr_tp2cp2dp2"),
)


def _vector_metrics(a, b):
    if len(a)!=len(b) or not a:
        raise AssertionError("Empty or mismatched vectors")
    if not all(math.isfinite(x) for x in a+b):
        raise ValueError("Nonfinite parity witness")
    fn2=math.fsum(x*x for x in a)
    gn2=math.fsum(x*x for x in b)
    dot=math.fsum(x*y for x,y in zip(a,b,strict=True))
    en2=math.fsum((x-y)**2 for x,y in zip(a,b,strict=True))
    return {
        "elements":len(a),
        "relative_l2": math.sqrt(en2/fn2) if fn2 else (0.0 if en2==0 else math.inf),
        "cosine": dot/math.sqrt(fn2*gn2) if fn2*gn2 else None,
        "max_abs": max(abs(x-y) for x,y in zip(a,b,strict=True)),
        "full_l2": math.sqrt(fn2),
        "candidate_l2": math.sqrt(gn2),
    }


def _labels(capture: Path):
    obj=torch.load(capture,map_location="cpu",weights_only=False)
    td=obj["tensordict"]
    tokens={}
    for row in range(8):
        mask=td["response_mask"][row].bool()
        old=td["old_log_probs"][row]
        adv=td["advantages"][row]
        if len(mask)!=len(old) or len(mask)!=len(adv):
            raise AssertionError("PPO labels are not aligned")
        for k in torch.nonzero(mask,as_tuple=False).flatten().tolist():
            tokens[(row,k)]=(float(old[k]),float(adv[k]))
    return tokens


def _effective_ppo(logp,old,adv):
    # Vanilla clip_ratio=0.2 and token-mean. This is a *diagnostic*
    # FP64 effective d(loss)/d(new_log_prob) with fixed labels;
    # the BF16 autograd graph's arithmetic may differ.
    z=logp-old
    ratio=math.exp(z) if z < 500 else math.inf
    if adv==0:
        return False, 0.0, ratio
    clipped=(adv>0 and ratio>1.2) or (adv<0 and ratio<0.8)
    return clipped,0.0 if clipped else -adv*ratio,ratio


def compare_pair(ref:dict,cur:dict,labels:dict) -> dict:
    lw,rw=ref["gradient_witness"],cur["gradient_witness"]
    for key in ("names","canonical_shapes","samples_per_name","samples_per_parameter"):
        if lw[key]!=rw[key]:
            raise AssertionError(f"Canonical TP-independent gradient coordinates differ: {key}")
    if lw["replica_max_abs"]!=0 or rw["replica_max_abs"]!=0:
        raise AssertionError("CP/DP gradient replicas must agree exactly")
    grads=_vector_metrics(lw["values"],rw["values"])
    an,bn=lw["complete_norm_gradients"],rw["complete_norm_gradients"]
    if an["names"]!=bn["names"] or an["shapes"]!=bn["shapes"]:
        raise AssertionError("Full Norm gradient coordinate layouts differ")
    complete_norm=_vector_metrics(an["values"],bn["values"])
    a={(int(x),int(y)):float(v) for x,y,v in ref["token_logprobs"]}
    b={(int(x),int(y)):float(v) for x,y,v in cur["token_logprobs"]}
    if set(a)!=set(b) or set(a)!=set(labels):
        raise AssertionError(
            f"Logical PPO token mismatch: ref={len(a)} candidate={len(b)} "
            f"expected={len(labels)}"
        )
    keys=sorted(labels)
    lpa=[a[k] for k in keys]
    lpb=[b[k] for k in keys]
    clip_changed=0
    eff_a=[];eff_b=[]
    clipped_a=0;clipped_b=0
    for k in keys:
        old,adv=labels[k]
        ca,ea,_=_effective_ppo(a[k],old,adv)
        cb,eb,_=_effective_ppo(b[k],old,adv)
        clipped_a+=int(ca)
        clipped_b+=int(cb)
        clip_changed+=int(ca!=cb)
        eff_a.append(ea/len(labels))
        eff_b.append(eb/len(labels))
    if not all(math.isfinite(x) for x in eff_a+eff_b):
        raise ValueError("Nonfinite PPO effective gradient diagnostic")
    return {
        "loss_abs_gap":abs(ref["global_loss"]-cur["global_loss"]),
        "loss_ref":ref["global_loss"],
        "loss_candidate":cur["global_loss"],
        "canonical_sampled_parameter_gradient":grads,
        "complete_norm_gradient":complete_norm,
        "token_logprob":_vector_metrics(lpa,lpb),
        "clip_state_changes":clip_changed,
        "ref_clipped_tokens":clipped_a,
        "candidate_clipped_tokens":clipped_b,
        "effective_dloss_dlogprob":_vector_metrics(eff_a,eff_b),
        "clip_caveat":"FP64 host reconstruction of vanilla PPO clip branch, not BF16 kernel bitwise oracle",
    }


def compare(root: Path):
    # TP is fixed at 2 in all five arms. No TP1 data or OOM log is required.
    cases={}
    for name in CASES:
        file=root/name/"result.json"
        if not file.is_file():
            raise FileNotFoundError(file)
        cases[name]=json.loads(file.read_text())
    inputs={(z["source_capture"],z["input_sha256"],z["checkpoint"],
             z["global_ppo_tokens"],z["gradient_sync_reduce_dtype"],
             z["hccl_deterministic"])
            for z in cases.values()}
    if len(inputs)!=1:
        raise AssertionError(f"Input/dtype/settings mismatch across matrix: {inputs}")
    if any(z["updated_weights"] or z["hccl_deterministic"]!="true" for z in cases.values()):
        raise AssertionError("Only fixed-weight deterministic HCCL runs are comparable")
    expected_topologies={
        "unshared_tp2cp1dp1":(2,1,1),
        "tpr_tp2cp1dp1":(2,1,1),
        "tpr_tp2cp2dp1":(2,2,1),
        "tpr_tp2cp1dp2":(2,1,2),
        "tpr_tp2cp2dp2":(2,2,2),
    }
    for name,xyz in expected_topologies.items():
        z=cases[name]
        if (z["topology"]["tp"],z["topology"]["cp"],z["topology"]["dp"])!=xyz:
            raise AssertionError(f"Wrong topology: {name}")
        if z["mode"]!=name.split("_")[0]:
            raise AssertionError(f"Wrong shared vs unshared mode: {name}")
    src=next(iter(cases.values()))["source_capture"]
    labels=_labels(Path(src))
    if len(labels)!=next(iter(cases.values()))["global_ppo_tokens"]:
        raise AssertionError("PPO token-count mismatch with original labels")
    computed_pairs={
        name:compare_pair(cases[a],cases[b],labels)
        for name,a,b in PAIRS
    }
    result={
        "source_capture":src,
        "input_sha256":next(iter(cases.values()))["input_sha256"],
        "global_ppo_tokens":len(labels),
        "cases":{
            key:{"topology":v["topology"],
                 "mode":v["mode"],
                 "global_loss":v["global_loss"],
                 "grad_replica_max_abs":v["gradient_witness"]["replica_max_abs"]}
            for key,v in cases.items()
        },
        "pairwise":computed_pairs,
        "complete_matrix":True,
        "tp_correctness_status":"NOT_EVALUATED_TP_FIXED_AT_2",
        "witness_warning":"Canonical sparse gradient samples only, no optimizer or multi-step validation.",
        "reference_warning":"unshared is TPR codepath with prefix sharing disabled, not Megatron Full",
    }
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("output_root",type=Path)
    args=ap.parse_args()
    result=compare(args.output_root)
    path=args.output_root/"comparison_10k.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2))
    for name,values in result["pairwise"].items():
        grad=values["canonical_sampled_parameter_gradient"]
        lp=values["token_logprob"]
        eff=values["effective_dloss_dlogprob"]
        norm=values["complete_norm_gradient"]
        print(
            f"TPR_10K_PARITY_COMPARE axis={name} loss_gap={values['loss_abs_gap']:.9g} "
            f"grad_rel_l2={grad['relative_l2']:.9g} grad_cos={grad['cosine']} "
            f"norm_full_rel_l2={norm['relative_l2']:.9g} "
            f"norm_full_cos={norm['cosine']} "
            f"logprob_max_abs={lp['max_abs']:.9g} "
            f"clip_changes={values['clip_state_changes']} "
            f"effective_grad_rel_l2={eff['relative_l2']:.9g}",
            flush=True,
        )
    print("TPR_10K_PARITY_COMPARISON="+str(path))


if __name__=="__main__":
    main()
