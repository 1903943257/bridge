"""Test-only OPS boundary/launch audit; never changes gates or tiling."""

import hashlib
import inspect
from pathlib import Path
import sys

from ._stage45_sync_trace import SyncTrace


def conv_dh0_allocation_audit(t, bt, b, d, w, allocated_elements):
    """Bounds implied by the current UNGUARDED per-time-tile store formula.

    This is not proof that a server kernel has this store; inspect its source.
    Even zero-valued stores need valid addresses.
    """
    nt = (t + bt - 1) // bt
    required = nt * b * d * w
    return dict(time_tiles=nt, allocated_elements=allocated_elements,
                unguarded_store_extent=required,
                unguarded_store_out_of_bounds=required > allocated_elements)


def install_ops_probe(patch, label, operators=("conv", "gdr")):
    import importlib
    import torch

    emit = lambda message: print(message, flush=True)
    trace = SyncTrace(torch.npu.synchronize, emit, label)
    import triton
    import torch_npu
    from mindspeed_ops.utils import is_arch35
    emit(f"OPS-ENV {label} torch={torch.__version__} torch_npu={getattr(torch_npu, '__version__', 'unknown')} "
         f"triton={triton.__version__} is_arch35={is_arch35()}")
    native_gdn = sys.modules.get("mindspeed.core.ssm.gated_delta_net")
    if native_gdn is not None:
        emit(f"OPS-NATIVE-GATE HAVE_FLA={getattr(native_gdn, 'HAVE_FLA', 'absent')} "
             f"file={getattr(native_gdn, '__file__', None)}")
        constructor = getattr(getattr(native_gdn, "GatedDeltaNet", None), "__init__", None)
        if constructor is not None:
            try:
                lines, start = inspect.getsourcelines(constructor)
            except (OSError, TypeError):
                emit("OPS-NATIVE-GATE constructor source unavailable")
            else:
                for number, line in enumerate(lines, start):
                    if any(word in line for word in ("HAVE_FLA", "raise", "FLA is", "support")):
                        emit(f"OPS-NATIVE-GATE source:{number} {line.strip()}")
    modules = []
    if "conv" in operators:
        modules.append(importlib.import_module("mindspeed_ops.api.triton.convolution"))
        # API may contain the implementation itself on older server versions.
        try:
            modules.append(importlib.import_module("mindspeed_ops.arch32.triton.convolution"))
        except ModuleNotFoundError as exc:
            if not "mindspeed_ops.arch32" in str(exc):
                raise
    if "gdr" in operators:
        modules.append(importlib.import_module("mindspeed_ops.api.triton.chunk_gated_delta_rule"))
    # Discover actual binding owners instead of assuming api/arch32 layout.
    for mod in list(modules):
        for name, value in list(vars(mod).items()):
            if inspect.isfunction(value) and "mindspeed_ops" in getattr(value, "__module__", ""):
                owner = inspect.getmodule(value)
                if owner is not None and owner not in modules:
                    modules.append(owner)
    seen_files = set()
    for mod in modules:
        filename = getattr(mod, "__file__", None)
        if filename and filename not in seen_files:
            seen_files.add(filename)
            data = Path(filename).read_bytes()
            emit(f"OPS-SOURCE module={mod.__name__} file={filename} sha256={hashlib.sha256(data).hexdigest()}")
            # Source hints, not dynamically executed capabilities.
            for number, line in enumerate(data.decode('utf-8', errors='replace').splitlines(), 1):
                if any(word in line for word in ("NotImplementedError", "is_arch35()", "assert K", "D % BD", "set_materialize_grads", "needs_input_grad")):
                    emit(f"OPS-GATE-SOURCE {filename}:{number} {line.strip()}")
                if "dh0" in line and any(word in line for word in ("new_zeros", "empty_like", "i_t *", "USE_INITIAL_STATE")):
                    emit(f"OPS-DH0-SOURCE {filename}:{number} {line.strip()}")

    def meta(value):
        if isinstance(value, torch.Tensor):
            return f"shape={tuple(value.shape)},stride={value.stride()},dtype={value.dtype},requires_grad={value.requires_grad}"
        return repr(value)

    wrapped_count = 0
    # Wrap module bindings, including the aliases actually called by API.
    for mod in modules:
        for name, original in list(vars(mod).items()):
            if inspect.isfunction(original) and "mindspeed_ops" in getattr(original, "__module__", "") and (
                name.startswith("causal_conv1d_") or name in (
                    "chunk_gated_delta_rule_bwd", "chunk_gated_delta_rule_bwd_dhu",
                    "chunk_bwd_dqkwg", "chunk_bwd_dv_local", "recompute_w_u_fwd",
                    "chunk_gated_delta_rule_fwd_h", "prepare_wy_repr_bwd")):
                signature = inspect.signature(original)

                def wrapped(*args, original=original, name=name, signature=signature, **kwargs):
                    with trace.span(name):
                        bound = signature.bind(*args, **kwargs)
                        bound.apply_defaults()
                        for key in ("x", "q", "k", "v", "dy", "do", "initial_state", "h0", "dht", "dh0", "chunk_size"):
                            if key in bound.arguments:
                                value = bound.arguments[key]
                                emit(f"OPS-ARG {label} fn={name} {key}: {meta(value)}")
                                if key == "dht" and isinstance(value, torch.Tensor):
                                    emit(f"OPS-DHT {label} fn={name} nonzero={torch.count_nonzero(value).item()}")
                        return original(*args, **kwargs)

                patch.setattr(mod, name, wrapped)
                wrapped_count += 1
            elif "kernel" in name and ("causal_conv1d" in name or "dhu" in name) and hasattr(original, "__getitem__"):
                class LaunchProbe:
                    def __init__(self, kernel, name):
                        self.kernel, self.name = kernel, name

                    def __getattr__(self, name):
                        return getattr(self.kernel, name)

                    def __getitem__(self, grid):
                        launch = self.kernel[grid]

                        def call(*args, **kwargs):
                            with trace.span(self.name + "/launch"):
                                info = {k: kwargs[k] for k in ("B", "T", "D", "W", "H", "K", "V", "BT", "BD", "BV", "NUM_CHKS", "NUM_BLKS_D") if k in kwargs}
                                emit(f"OPS-LAUNCH {label} kernel={self.name} grid={grid} tiling={info}")
                                for key in ("initial_state", "dh0", "dht"):
                                    emit(f"OPS-LAUNCH {label} {key}: {meta(kwargs.get(key))}")
                                if "causal_conv1d_bwd" in self.name and kwargs.get("dh0") is not None:
                                    audit = conv_dh0_allocation_audit(kwargs["T"], kwargs["BT"], kwargs["B"], kwargs["D"], kwargs["W"], kwargs["dh0"].numel())
                                    emit(f"OPS-DH0-BOUNDS {label} conditional_on_unguarded_store=True {audit}")
                                return launch(*args, **kwargs)
                        return call

                patch.setattr(mod, name, LaunchProbe(original, name))
    emit(f"OPS-PROBE {label} wrapped_bindings={wrapped_count}; no gate/shape overrides")
    return trace
