"""Compare identical 1024 real tokens under native CP1/CP2 and TPR P512/S512.

Inputs:
  P0-1 native_cp1.pt, native_cp2.pt (unmodified official MindSpeed)
  P0-3 tpr_p512s512.pt (controlled TPR Full and SegmentExecutor split)
No guessed "normal" error thresholds. Fail if tokens, base weight bits, loss
definition or parameter sample ownership do not match.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.torch_version import TorchVersion


def _load(path):
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _stat(a, b):
    aa = a.detach().float().reshape(-1).double()
    bb = b.detach().float().reshape(-1).double()
    if aa.shape != bb.shape or not bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()):
        raise AssertionError(f"Different shapes or nonfinite values: {aa.shape}, {bb.shape}")
    delta = bb - aa
    norm = aa.norm().clamp_min(1e-24)
    denom = (aa.norm() * bb.norm()).clamp_min(1e-24)
    return {
        "count": aa.numel(),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "p95_abs": float(torch.quantile(delta.abs(), .95)),
        "rel_l2": float(delta.norm() / norm),
        "cosine": float(torch.dot(aa, bb) / denom),
        "changed": int(torch.count_nonzero(delta)),
    }


def _gradient_stat(left, right):
    if set(left) != set(right):
        raise AssertionError(
            f"Parameter name difference: {sorted(set(left) ^ set(right))[:15]}"
        )
    anorm = bnorm = diff = dot = 0.0
    worst = []
    for name in sorted(left):
        a, b = left[name], right[name]
        if a["shape"] != b["shape"] or not torch.equal(a["indices"], b["indices"]):
            raise AssertionError(f"Parameter shape/indices mismatch for {name}")
        av, bv = a["values"].double(), b["values"].double()
        stat = _stat(av, bv)
        worst.append((stat["rel_l2"], name, stat))
        anorm += float(av.square().sum())
        bnorm += float(bv.square().sum())
        diff += float((bv-av).square().sum())
        dot += float((av*bv).sum())
    return {
        "rel_l2": math.sqrt(diff / max(anorm, 1e-48)),
        "cosine": dot / max(math.sqrt(anorm*bnorm), 1e-48),
        "parameter_count": len(left),
        "mode": "STRIDED_SAMPLED_ONLY",
        "worst": sorted(worst, reverse=True)[:6],
    }


def _validate(native_cp1, native_cp2, tpr):
    if native_cp1["cp"] != 1 or native_cp2["cp"] != 2:
        raise AssertionError("Need native CP1 and CP2 oracle artifacts")
    if native_cp2["native_ring_calls_rank0"] < 28:
        raise AssertionError("CP2 artifact has no verified official Ring dispatch")
    for name in ("checkpoint", "model", "dtype", "token_sha256", "source",
                 "sequence_length", "response_length", "objective", "denominator",
                 "sampled_weight_sha256_rank0", "torch"):
        expected = native_cp1[name]
        for label, item in (("native_cp2", native_cp2), ("tpr", tpr)):
            if item[name] != expected:
                raise AssertionError(
                    f"Invalid {label} vs native_cp1 comparison: field {name}: "
                    f"{item[name]} != {expected}"
                )
    if native_cp1["source"] != "real_tq:tq_batch.pt:row0:prompt_tail512+response_head512":
        raise AssertionError("Expected true real-TQ row0/512+512 P0-1 control, not synthetic data")
    if tpr["tag"] != "P0_3_REAL_QWEN_1_7B_P512S512_SINGLE_PATH":
        raise AssertionError("Wrong TPR evidence tag")
    if tpr["tpr_p512s512"]["segment_lengths"] != [512, 512]:
        raise AssertionError("TPR did not physically separate P512 and S512")
    if "tpr_one_node_m1024" in tpr:
        one = tpr["tpr_one_node_m1024"]
        if one.get("segment_lengths") != [1024]:
            raise AssertionError("TPR one-node oracle must run physical M=1024")
        if one.get("loss_term_count") != 512 or one.get("prefix_grad_relays") != 0:
            raise AssertionError("TPR one-node oracle unexpectedly lost/relayed tokens")
    if tpr["tpr_p512s512"]["prefix_grad_relays"] != 28:
        raise AssertionError("TPR ancestor KV-gradient relay was not verified")
    if not (native_cp1["local_m"] == 1024 and native_cp2["local_m"] == 512):
        raise AssertionError("Native CP1/2 Linear M witness is inconsistent")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-dir", type=Path, default=Path("/tmp/tpr_p0_native_cp"),
    )
    parser.add_argument(
        "--tpr-dir", type=Path, default=Path("/tmp/tpr_p0_tpr_p512s512"),
    )
    args = parser.parse_args()
    paths = [
        args.native_dir / "native_cp1.pt",
        args.native_dir / "native_cp2.pt",
        args.tpr_dir / "tpr_p512s512.pt",
    ]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        parser.error(f"Missing required saved witness(s): {missing}")
    cp1, cp2, tpr = map(_load, paths)
    _validate(cp1, cp2, tpr)
    response_start = cp1["sequence_length"] - cp1["response_length"] - 1
    evidence = {
        "NATIVE_CP1": {
            "loss": cp1["loss"],
            "response_logprobs": cp1["logprobs"][response_start:],
            "gradients": cp1["gradients"],
        },
        "NATIVE_CP2": {
            "loss": cp2["loss"],
            "response_logprobs": cp2["logprobs"][response_start:],
            "gradients": cp2["gradients"],
        },
        "TPR_CONTROLLED_FULL": tpr["controlled_full"],
        "TPR_P512_S512": tpr["tpr_p512s512"],
    }
    if "tpr_one_node_m1024" in tpr:
        evidence["TPR_ONE_NODE_M1024"] = tpr["tpr_one_node_m1024"]
    else:
        print("P0_3 ONE_NODE_M1024=MISSING_OLD_ARTIFACT "
              "rerun P0-3 after test sync for the Attention-path isolation gate")
    pairs = [
        ("TPR_CONTROLLED_FULL", "TPR_P512_S512"),
        ("NATIVE_CP1", "TPR_CONTROLLED_FULL"),
        ("NATIVE_CP1", "NATIVE_CP2"),
        ("NATIVE_CP1", "TPR_P512_S512"),
        ("NATIVE_CP2", "TPR_P512_S512"),
    ]
    if "TPR_ONE_NODE_M1024" in evidence:
        pairs += [
            ("TPR_CONTROLLED_FULL", "TPR_ONE_NODE_M1024"),
            ("TPR_ONE_NODE_M1024", "TPR_P512_S512"),
            ("NATIVE_CP2", "TPR_ONE_NODE_M1024"),
        ]
    for ref, candidate in pairs:
        a, b = evidence[ref], evidence[candidate]
        metrics = _stat(a["response_logprobs"], b["response_logprobs"])
        grad = _gradient_stat(a["gradients"], b["gradients"])
        print(f"P0_3 {ref}_VS_{candidate} LOSS "
              f"ref={a['loss']:.9g} actual={b['loss']:.9g} "
              f"abs={abs(b['loss']-a['loss']):.9g}")
        print(f"P0_3 {ref}_VS_{candidate} RESPONSE_LOGPROB {metrics}")
        print(f"P0_3 {ref}_VS_{candidate} SAMPLED_GRAD "
              f"rel_l2={grad['rel_l2']:.9g} cosine={grad['cosine']:.9g} "
              f"parameter_count={grad['parameter_count']} "
              f"mode={grad['mode']}")
        for _, name, stat in grad["worst"]:
            print(f"P0_3 {ref}_VS_{candidate} GRAD_WORST name={name} {stat}")
    print("P0_3 status=DIAGNOSTIC_ONLY P512_S512_nativeCP2_M512_"
          "NOT_IDENTICAL_ATTENTION_SCHEDULE no_tpr_optimizer_training")


if __name__ == "__main__":
    main()
