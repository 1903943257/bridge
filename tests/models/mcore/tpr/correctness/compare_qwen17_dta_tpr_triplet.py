"""Compare saved same-data HF Full/DTA and Megatron Full/TPR artifacts.

Outputs a grounded paired three-way comparison without implicitly mapping HF
gradients into Megatron parameter namespace.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def _stats(a,b):
    if a.shape!=b.shape or a.ndim!=2:
        raise AssertionError("unmatched response token grids")
    delta=(a.float()-b.float()).abs()
    return float(delta.mean()),float(delta.max()),int((delta>0.2).sum())


def compare(folder):
    path=Path(folder)
    hf=torch.load(path/"hf_full_dta.pt",map_location="cpu",weights_only=True)
    mg=torch.load(path/"megatron_full_tpr.pt",map_location="cpu",weights_only=True)
    for key in ("objective","n_rows","prompt_length","response_length"):
        if hf[key]!=mg[key]:
            raise AssertionError(
                f"HF and Megatron experiments do not have identical {key}: "
                f"{hf[key]} vs {mg[key]}")
    grids={
        "hf_full":hf["full_response_logprobs"],
        "hf_dta":hf["dta_response_logprobs"],
        "mg_full":mg["full_response_logprobs"],
        "mg_tpr":mg["tpr_response_logprobs"],
    }
    if {tuple(t.shape) for t in grids.values()}!={(8,64)}:
        raise AssertionError("three-way comparison requires all 8x64 valid response tokens")
    print(
        "P1 TRIPLET CONFIG "
        f"objective={hf['objective']} "
        "checkpoint_family=QWEN3_1_7B data=REAL_TQ_8x192 "
        "backend_pairing=HF_FULL_HF_DTA__MG_FULL_MG_TPR "
        "gradient_cross_framework=NOT_COMPARABLE_WITHOUT_MAPPING",
        flush=True,
    )
    for label,a,b in (
        ("HF_DTA_VS_HF_FULL","hf_full","hf_dta"),
        ("TPR_VS_MEGATRON_FULL","mg_full","mg_tpr"),
        ("MEGATRON_FULL_VS_HF_FULL","hf_full","mg_full"),
        ("TPR_VS_HF_FULL_RAW","hf_full","mg_tpr"),
        ("TPR_VS_HF_DTA_RAW","hf_dta","mg_tpr"),
    ):
        mean,mx,n_gt=_stats(grids[a],grids[b])
        print(
            f"P1 TRIPLET LOGPROB comparison={label} "
            f"mean_abs={mean:.9g} max_abs={mx:.9g} "
            f"num_gt_0p2={n_gt} tokens=512 "
            f"causal_attribution={'WITHIN_BACKEND' if ('RAW' not in label and label!='MEGATRON_FULL_VS_HF_FULL') else 'NOT_VALID_CROSS_FRAMEWORK'}",
            flush=True,
        )
    print(
        "P1 TRIPLET GRAD "
        f"backend=HF comparison=FULL_VS_DTA "
        f"rel_l2={hf['dta_grad_rel_l2']:.9g} "
        f"cosine={hf['dta_grad_cosine']:.9g} "
        "scope=ALL_PARAMETERS",
        flush=True,
    )
    print(
        "P1 TRIPLET GRAD "
        f"backend=MEGATRON comparison=FULL_VS_TPR "
        f"rel_l2={mg['tpr_grad_sample_rel_l2']:.9g} "
        f"cosine={mg['tpr_grad_sample_cosine']:.9g} "
        "scope=SAMPLED_PER_PARAMETER",
        flush=True,
    )
    print(
        "P1 TRIPLET STEP "
        f"hf_dta_sample_rel_l2={hf['dta_step_sample_rel_l2']:.9g} "
        f"megatron_tpr_sample_rel_l2={mg['tpr_step_sample_rel_l2']:.9g} "
        "sampling=BACKEND_SPECIFIC_NOT_COMPARABLE",
        flush=True,
    )
    floor_mean,floor_max,_=_stats(grids["hf_full"],grids["mg_full"])
    print(
        "P1 TRIPLET RESULT status=PASS "
        f"hf_megatron_full_floor_mean_abs={floor_mean:.9g} "
        f"hf_megatron_full_floor_max_abs={floor_max:.9g} "
        "numerical_parity=DIAGNOSTIC_ONLY "
        "raw_hf_vs_tpr_gap_includes_framework_conversion=True",
        flush=True,
    )


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory")
    args=parser.parse_args()
    compare(args.directory)
