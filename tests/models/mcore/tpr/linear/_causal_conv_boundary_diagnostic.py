"""Read-only diagnostics for the actual Stage 3.2 causal-conv backward launch."""

from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
from pathlib import Path
from unittest.mock import patch

import torch


@contextmanager
def capture_conv_backward_launches():
    """Observe runtime tile choices and tensor operands; delegate launch unchanged."""
    import mindspeed_ops.arch32.triton.convolution as convolution

    original = convolution.causal_conv1d_bwd_kernel
    original_impl = convolution.causal_conv1d_bwd_impl
    records = []

    class LaunchProbe:
        def __getitem__(self, grid):
            launch = original[grid]

            def traced(*args, **kwargs):
                result = launch(*args, **kwargs)
                # Keep operands alive until backward completes. No synchronization
                # or tensor modification is inserted into the kernel launch itself.
                records.append({"grid": grid, **kwargs})
                return result

            return traced

    @wraps(original_impl)
    def implementation(*args, **kwargs):
        before = len(records)
        result = original_impl(*args, **kwargs)
        if len(records) != before + 1:
            raise AssertionError("expected one causal-conv kernel launch per backward implementation call")
        records[-1]["returned_dh0"] = result[4]
        records[-1]["returned_dx"] = result[0]
        return result

    with patch.object(convolution, "causal_conv1d_bwd_kernel", LaunchProbe()), patch.object(
        convolution, "causal_conv1d_bwd_impl", implementation
    ):
        yield records


def _cpu(tensor):
    return None if tensor is None else tensor.detach().cpu().float()


def print_conv_backward_source():
    import mindspeed_ops.arch32.triton.convolution as convolution

    path = Path(convolution.__file__).resolve()
    print(
        f"STAGE-3.2 CONV-SOURCE path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}",
        flush=True,
    )
    # Print the server's actual dh0 loop, not a guess based on the local checkout.
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if "b_dh0_s =" in line]
    if not matches:
        print("STAGE-3.2 CONV-SOURCE dh0 loop marker absent; inspect server source", flush=True)
    for index in matches:
        end = next((j + 1 for j in range(index, len(lines)) if "fp_downcast_rounding" in lines[j]), index + 35)
        for j in range(max(0, index - 3), min(end + 3, len(lines))):
            print(f"STAGE-3.2 CONV-SOURCE {j + 1}: {lines[j]}", flush=True)
    implementation = inspect.unwrap(convolution.causal_conv1d_bwd_impl)
    source, start = inspect.getsourcelines(implementation)
    for offset, line in enumerate(source):
        if "BT =" in line or "BD =" in line or "dh0 =" in line:
            print(f"STAGE-3.2 CONV-SOURCE {start + offset}: {line.rstrip()}", flush=True)


def snapshot_conv_launches(records, *, mode):
    snapshots = []
    for record in records:
        if record.get("db") is not None:
            raise AssertionError("this diagnostic expects the bias-free Stage 3.2 convolution")
        initial = record.get("initial_state")
        state_dtype = None if initial is None else initial.dtype
        tensors = {
            name: _cpu(record.get(name))
            for name in ("x", "weight", "initial_state", "y", "dy", "dht", "dh0", "dx")
        }
        # Capture the actual API result as well as raw tile buffers; do not assume
        # that the server's dirty implementation reduces them in a particular way.
        if tensors["dh0"] is not None:
            tensors["dh0_tiles"] = tensors["dh0"]
        tensors["dh0"] = _cpu(record["returned_dh0"])
        tensors["dx"] = _cpu(record["returned_dx"])
        metadata = {name: record[name] for name in ("B", "T", "D", "W", "BT", "BD", "ACTIVATION")}
        print(
            f"STAGE-3.2 CONV-LAUNCH {mode}: {metadata} grid={record['grid']} "
            f"has_h0={initial is not None} has_dht={record.get('dht') is not None} "
            f"x_dtype={record['x'].dtype} h0_dtype={state_dtype}", flush=True,
        )
        snapshots.append({**metadata, **tensors, "state_dtype": state_dtype})
    return snapshots


def _reference_dh0(record, *, use_kernel_preactivation):
    """CPU FP32 formula over real GDN operands, independent of tiled kernel indexing.

    State layout is [B,D,W], with slot zero unused by the next convolution.
    dh0[j] = sum(t=0..j-1) dz[t] * weight[j-1-t], j=1..W-1.
    """
    x, weight, h0 = record["x"], record["weight"], record["initial_state"]
    width = record["W"]
    length = min(width - 1, x.shape[1])
    if use_kernel_preactivation:
        z = None if record["y"] is None else record["y"][:, :length]
    else:
        z = torch.zeros_like(x[:, :length])
        for t in range(length):
            for w in range(width):
                position = t + w - width + 1
                sample = x[:, position] if position >= 0 else h0[:, :, width + position]
                z[:, t] += sample * weight[w]
    dz = record["dy"][:, :length].clone()
    if record["ACTIVATION"] in ("silu", "swish"):
        sigmoid = z.sigmoid()
        dz *= sigmoid * (1 + z * (1 - sigmoid))
    elif record["ACTIVATION"] is not None:
        raise AssertionError(f"unsupported diagnostic activation: {record['ACTIVATION']}")
    dh0 = torch.zeros_like(h0)
    for j in range(1, width):
        for t in range(min(j, length)):
            dh0[:, :, j] += dz[:, t] * weight[j - 1 - t]
    return dh0


def report_conv_boundary(full, split, *, prefix_length, compare):
    """Compare dh0 and the resulting prefix-tail error, with no new thresholds."""
    if len(full) != 1 or len(split) != 2:
        raise AssertionError(f"expected full=1/split=2 conv launches, got {len(full)}/{len(split)}")
    suffix = [record for record in split if record["initial_state"] is not None]
    prefix = [record for record in split if record["initial_state"] is None]
    if len(suffix) != 1 or len(prefix) != 1:
        raise AssertionError("cannot identify prefix/suffix conv backward launches")
    full, prefix, suffix = full[0], prefix[0], suffix[0]
    width = suffix["W"]
    if prefix_length < width or suffix["T"] < width or suffix["B"] != 1:
        raise AssertionError("conv boundary diagnostic requires B=1 and segments at least W tokens")
    for name in ("dh0", "y"):
        if suffix[name] is None:
            raise AssertionError(f"suffix backward is missing {name}")
    if prefix["dht"] is None:
        raise AssertionError("prefix conv backward did not receive dht")
    print(
        f"STAGE-3.2 CONV-BOUNDARY suffix BT={suffix['BT']} W={width} "
        f"BT<W={suffix['BT'] < width}", flush=True,
    )
    mathematical = _reference_dh0(suffix, use_kernel_preactivation=False)
    matched = _reference_dh0(suffix, use_kernel_preactivation=True)
    actual = suffix["dh0"]
    for label, reference in (("fp32-formula", mathematical), ("kernel-preactivation-formula", matched)):
        compare(f"conv-dh0/{label}-vs-actual", {"dh0": reference}, {"dh0": actual})
    compare("conv-relay/dh0-vs-prefix-dht", {"state": actual}, {"state": prefix["dht"]})
    for index, tile in enumerate(suffix["dh0_tiles"]):
        print(f"STAGE-3.2 CONV-DH0-TILE index={index} norm={tile.norm().item():.6e}", flush=True)
    for j in range(width):
        compare(f"conv-dh0/slot{j}", {"dh0": matched[:, :, j]}, {"dh0": actual[:, :, j]})

    ref_dx = full["dx"][:, :prefix_length]
    split_dx = prefix["dx"]
    for label, start, end in (("interior", 0, prefix_length - width), ("tail", prefix_length - width, prefix_length)):
        if end > start:
            compare(f"conv-prefix-dx/{label}", {"dx": ref_dx[:, start:end]}, {"dx": split_dx[:, start:end]})
    tail_delta = (split_dx - ref_dx)[:, -width:]
    # Full vs split also includes BF16 rounding of dx; report the residual rather
    # than assuming an exact identity between tail_delta and the dh0 error.
    predicted = (actual - matched).transpose(1, 2)
    compare("conv-prefix-tail/dh0-error-vs-observed-dx-error", {"error": predicted}, {"error": tail_delta})
    print(
        f"STAGE-3.2 CONV-TAIL observed_error={tail_delta.norm().item():.6e} "
        f"predicted_error={predicted.norm().item():.6e} "
        f"unexplained_residual={(tail_delta - predicted).norm().item():.6e}", flush=True,
    )
