"""Standalone subprocess-isolated Conv/GDR A/B. No TPR scheduler or HCCL.

Run with python (not torchrun). Each combination gets a fresh worker process.
"""

import argparse
import itertools
import os
from pathlib import Path
import subprocess
import sys
import traceback


def child(args):
    import torch
    import torch_npu  # noqa: F401
    from unittest.mock import patch
    from contextlib import ExitStack
    # Load the sibling package without importing verl or its Engine.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from parallel._stateful_ops_probe import install_ops_probe

    torch.npu.set_device(args.device)
    torch.manual_seed(451001)
    op, t, state_mode, final_mode = args.case.split(":")
    t = int(t)
    label = f"op={op} T={t} initial={state_mode} final={final_mode}"
    print(f"OPS-CASE {label} device={torch.npu.get_device_name(args.device)}", flush=True)
    device = torch.device("npu", args.device)
    gen = torch.Generator().manual_seed(451001)

    def leaf(shape, dtype=torch.bfloat16, scale=0.1, normalize=False):
        value = torch.randn(shape, generator=gen) * scale
        if normalize:
            value = torch.nn.functional.normalize(value, dim=-1)
        return value.to(device=device, dtype=dtype).requires_grad_(True)

    if op == "conv":
        from mindspeed_ops.api.triton.convolution import causal_conv1d
        inputs = dict(x=leaf((1, t, args.channels)), weight=leaf((4, args.channels)), bias=leaf((args.channels,)))
        initial = leaf((1, args.channels, 4)).detach()
        forward = lambda: causal_conv1d(**inputs, activation="silu", initial_state=initial, output_final_state=final_mode != "off")
    else:
        from mindspeed_ops.api.triton.chunk_gated_delta_rule import chunk_gated_delta_rule
        inputs = {key: leaf((1, t, 8, 128), normalize=key in ("q", "k")) for key in ("q", "k", "v")}
        inputs["g"] = (-torch.rand((1, t, 8), generator=gen)).to(device).requires_grad_(True)
        inputs["beta"] = torch.rand((1, t, 8), generator=gen).to(device, dtype=torch.bfloat16).requires_grad_(True)
        initial = leaf((1, 8, 128, 128), dtype=torch.float32).detach()
        forward = lambda: chunk_gated_delta_rule(**inputs, initial_state=initial, output_final_state=final_mode != "off", head_first=False, chunk_size=64)
    initial = None if state_mode == "none" else initial.requires_grad_(state_mode == "train")
    with ExitStack() as stack:
        class Patches:
            def setattr(self, owner, name, value):
                stack.enter_context(patch.object(owner, name, value))
        probe = install_ops_probe(Patches(), label, operators=(op,))
        with probe.span("forward"):
            output, final = forward()
        dy = leaf(output.shape, dtype=output.dtype, scale=1e-3).detach()
        roots, grads = [output], [dy]
        if final_mode == "nonzero":
            roots.append(final)
            grads.append(leaf(final.shape, dtype=final.dtype, scale=1e-3).detach())
        with probe.span("backward"):
            torch.autograd.backward(roots, grads)
        for name, tensor in {**inputs, **({"initial": initial} if state_mode == "train" else {})}.items():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all().item(), name
            print(f"OPS-GRAD {label} {name} norm={tensor.grad.float().norm().item():.9e}", flush=True)
        print(f"OPS-CASE COMPLETE {label}; finite smoke only, not equivalence PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ops", default="conv,gdr")
    parser.add_argument("--lengths", default="64,128,256,512,1024")
    parser.add_argument("--states", default="none,frozen,train")
    parser.add_argument("--finals", default="unused", help="off,unused,nonzero; unused matches leaf forward returning final state")
    parser.add_argument("--channels", type=int, choices=(3072, 6144), default=3072)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        try:
            child(args)
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # The parent records failure; do not run device/HCCL destructors
            # in a poisoned context. This worker never owns a process group.
            os._exit(1)
        return 0
    ops, lengths, states, finals = args.ops.split(","), args.lengths.split(","), args.states.split(","), args.finals.split(",")
    assert set(ops) <= {"conv", "gdr"} and set(states) <= {"none", "frozen", "train"}
    assert set(finals) <= {"off", "unused", "nonzero"} and all(int(t) > 0 for t in lengths)
    results = []
    for combo in itertools.product(ops, lengths, states, finals):
        case = ":".join(combo)
        print(f"OPS-MATRIX START {case}", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--case", case,
               "--device", str(args.device), "--channels", str(args.channels)]
        try:
            code = subprocess.run(cmd, timeout=args.timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            code = 124
        results.append((case, code))
        print(f"OPS-MATRIX RESULT {case} exit={code}", flush=True)
    print("OPS-MATRIX SUMMARY " + repr(results), flush=True)
    return int(any(code != 0 for _, code in results))


if __name__ == "__main__":
    sys.exit(main())
