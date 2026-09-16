"""Single-op reproducer; no kernel patches, model imports or distributed setup.

Default: run A/B/C sequentially in fresh processes (a device fault may poison
the context). PASS means execution completed, not gradient numerical accuracy.
The current checkout may already contain the store fix: record its provenance.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import platform
import subprocess
import sys
import traceback


CASES = ("none", "state_no_grad", "state_grad")
B, T, D, W = 1, 64, 3072, 4


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed / metadata unavailable"


def git_info(source):
    root = next((p for p in source.parents if (p / ".git").exists()), None)
    if root is None:
        print("MindSpeed-Ops git commit: unavailable (installed package without .git)")
        return
    for label, args in (("commit", ["rev-parse", "HEAD"]),
                        ("working tree", ["status", "--short"])):
        try:
            result = subprocess.run(["git", "-C", str(root), *args],
                                    capture_output=True, text=True, check=True)
            print(f"MindSpeed-Ops {label}: {result.stdout.strip() or 'clean'}")
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"MindSpeed-Ops {label}: unavailable ({exc})")


def run_case(args):
    print(f"\n{'=' * 60}\nCase {args.case}\n{'=' * 60}", flush=True)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    for name in ("torch", "torch_npu", "triton", "triton-ascend"):
        print(f"{name}: {version(name)}")
    stage = "Setup"
    try:
        import torch
        import torch_npu  # registers torch.npu
        import triton
        from mindspeed_ops.api.triton import convolution as api
        from mindspeed_ops.utils import is_arch35

        torch.npu.set_device(args.device)
        print(f"device index: {args.device}")
        print(f"device name: {torch.npu.get_device_name(args.device)}")
        print(f"is_arch35: {is_arch35()}")
        print(f"API module: {api.__file__}")
        print(f"API signature: {inspect.signature(api.causal_conv1d)}")
        git_info(Path(api.__file__).resolve())
        if is_arch35():
            raise RuntimeError("This reproducer targets arch32; API rejects arch35")
        from mindspeed_ops.arch32.triton import convolution as impl

        print(f"dispatch module: {impl.__file__}")
        source = Path(impl.__file__).read_bytes()
        print(f"kernel source SHA256: {hashlib.sha256(source).hexdigest()}")
        print("Source excerpts (inspect guards; tile count alone is not proof of OOB):")
        for number, line in enumerate(source.decode().splitlines(), 1):
            if ("if USE_INITIAL_STATE" in line or "dh0 +" in line
                    or "dh0 = initial_state" in line):
                print(f"  {number}: {line.strip()}")
        dtype = torch.bfloat16
        print(f"dtype: {dtype}; B={B} T={T} D={D} W={W}")
        print("activation=None; bias=None; residual=None; cu_seqlens=None; "
              "output_final_state=False (isolate initial-state backward)")
        cores = impl.get_vector_num()
        bt = min(8 if args.case != "none" else 32,
                 triton.next_power_of_2(triton.cdiv(max(16, B * T), cores)))
        nt = triton.cdiv(T, bt)
        slots = min(nt, triton.cdiv(W, bt))
        print(f"Pre-call diagnostic from inspected checkout formula: cores={cores}, BT={bt}, "
              f"eff_NT={nt}, dh0_allocated_tiles_if_stateful={slots}")
        print(f"unguarded_store_tiles={nt - slots}; "
              f"potential_oob_if_store_unguarded={nt > slots}; "
              "actual backward locals checked below")
        torch.manual_seed(1234)
        device = f"npu:{args.device}"
        x = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(W, D, device=device, dtype=dtype, requires_grad=True)
        state = None if args.case == "none" else torch.randn(
            B, D, W, device=device, dtype=dtype,
            requires_grad=args.case == "state_grad")
        for name, tensor in (("x", x), ("weight", weight), ("initial_state", state)):
            print(f"{name}: None" if tensor is None else
                  f"{name}: shape={tuple(tensor.shape)}, stride={tensor.stride()}, "
                  f"dtype={tensor.dtype}, requires_grad={tensor.requires_grad}")

        # Observe Python launch locals without wrapping/replacing any kernel or
        # changing arguments. This also proves backward runs for state_no_grad.
        observed = set()
        def trace(frame, event, arg):
            if (event == "line" and frame.f_code.co_name == "causal_conv1d_bwd_impl"
                    and Path(frame.f_code.co_filename).resolve() == Path(impl.__file__).resolve()):
                local = frame.f_locals
                if "entry" not in observed:
                    observed.add("entry")
                    print(f"Actual backward entered: {frame.f_code.co_filename}", flush=True)
                if "dh0" in local and "allocation" not in observed:
                    observed.add("allocation")
                    h = local["dh0"]
                    print(f"Actual launch locals: BT={local.get('BT')}, NT={local.get('NT')}, "
                          f"eff_NT={local.get('eff_NT')}, "
                          f"dh0.shape={None if h is None else tuple(h.shape)}", flush=True)
            return trace

        torch.npu.synchronize()
        stage = "Forward"
        y, final_state = api.causal_conv1d(x, weight, initial_state=state,
                                         output_final_state=False)
        torch.npu.synchronize()
        print(f"output: shape={tuple(y.shape)}, dtype={y.dtype}; final_state={final_state}")
        print("Forward: PASS", flush=True)
        stage = "Backward"
        previous_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            y.float().sum().backward()
            torch.npu.synchronize()
        finally:
            sys.settrace(previous_trace)
        # Autograd may invoke Python backward on a worker thread; sys.settrace
        # only observes the installing thread. Missing telemetry is not a
        # backward failure and does not prove which implementation executed.
        if "entry" not in observed:
            print("Diagnostic: Python trace did not observe arch32 backward; "
                  "actual runtime path/BT/allocation unverified. "
                  "Pre-call values remain diagnostic estimates.", flush=True)
        elif "allocation" not in observed:
            print("Diagnostic: backward entered, but launch locals were not captured.",
                  flush=True)
        assert x.grad is not None and weight.grad is not None
        if args.case == "state_grad":
            assert state.grad is not None
        print("Backward: PASS", flush=True)
        return 0
    except Exception:
        print(f"{stage}: FAIL", flush=True)
        traceback.print_exc()
        # Do not issue further device operations after a runtime fault.
        return 2 if stage == "Setup" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.case != "all":
        return run_case(args)
    results = []
    for case in CASES:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "--case", case, "--device", str(args.device)])
        results.append((case, result.returncode))
    print("\nResult summary (fresh process per case):", flush=True)
    for case, code in results:
        status = "PASS" if code == 0 else "SETUP FAILED / NOT RUN" if code == 2 else "FAIL"
        print(f"{case:16s}: {status} (exit={code})")
    return int(any(code for _, code in results))


if __name__ == "__main__":
    sys.exit(main())
