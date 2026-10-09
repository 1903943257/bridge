"""Opt-in per-physical-Segment trace: Native cutoff vs real Forest leaf.

No GEMM, Attention, or training operator is replaced. Module forward hooks
capture a single logical query's inputs and outputs in FP32 on CPU. The Forest
hook MUST be gated by the exact Segment ID, not just the [start:end] shape:
sibling nodes can share the same physical length and absolute positions.
This is a heavy synchronizing numerical diagnostic, never a performance test.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


STAGES = (
    "input_norm", "qkv", "q_norm", "k_norm", "attn_proj",
    "attention", "pre_mlp_norm", "mlp_fc1", "mlp_fc2", "mlp", "layer",
)


def _as_tensor(value):
    """Unwrap a Megatron module's first tensor output or positional input."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _first_input_tensor(args, kwargs):
    """Handle TransformerLayer.forward(hidden_states=...) and positional calls.

    Megatron TransformerBlock calls TransformerLayer with keyword-only
    hidden_states in some versions. For those calls, forward-hook args is
    empty even though the input tensor is present in kwargs. The former
    positional-only trace raised during an otherwise healthy Native run.
    """
    from_args = _as_tensor(args)
    if from_args is not None:
        return from_args
    for name in ("hidden_states", "input", "x"):
        value = _as_tensor(kwargs.get(name))
        if value is not None:
            return value
    # Only use a generic fallback when the input is unambiguous. Silently
    # choosing a mask or position tensor would invalidate the comparison.
    tensors = [value for value in kwargs.values()
               if isinstance(value, torch.Tensor)]
    return tensors[0] if len(tensors) == 1 else None


@contextmanager
def capture_query_stages(model, *, query_position: int, get_active_span):
    """Capture stage tensors at one *absolute* query position, by segment.

    get_active_span() returns (start, end) for only the intended active
    Segment (or None to suppress a hook). For Native, return (0, full_S).
    Multiple passes at the same layer/stage are rejected, since silently
    overwriting a sibling's snapshot invalidates the numerical attribution.
    """
    output = {}
    handles = []
    try:
        for i, layer in enumerate(model.decoder.layers):
            modules = {
                "input_norm": layer.input_layernorm,
                "qkv": layer.self_attention.linear_qkv,
                "q_norm": layer.self_attention.q_layernorm,
                "k_norm": layer.self_attention.k_layernorm,
                "attn_proj": layer.self_attention.linear_proj,
                "attention": layer.self_attention,
                "pre_mlp_norm": layer.pre_mlp_layernorm,
                "mlp_fc1": layer.mlp.linear_fc1,
                "mlp_fc2": layer.mlp.linear_fc2,
                "mlp": layer.mlp,
                "layer": layer,
            }
            for stage, module in modules.items():
                def hook(_module, args, kwargs, result, *, layer_index=i, stage_name=stage):
                    span = get_active_span()
                    if span is None:
                        return
                    start, end = span
                    if not start <= query_position < end:
                        return
                    pos = query_position - start
                    x = _first_input_tensor(args, kwargs)
                    y = _as_tensor(result)
                    if x is None or y is None:
                        raise AssertionError(
                            f"trace layer={layer_index} stage={stage_name} has no tensor I/O "
                            f"(args={len(args)}, kwargs={tuple(kwargs)})"
                        )
                    if not (x.ndim >= 1 and y.ndim >= 1 and
                            pos < x.shape[0] and pos < y.shape[0]):
                        raise AssertionError(
                            f"trace invalid position pos={pos}, span={span}, "
                            f"layer={layer_index}, stage={stage_name}, "
                            f"input={tuple(x.shape)} output={tuple(y.shape)}"
                        )
                    key = (layer_index, stage_name)
                    if key in output:
                        raise AssertionError(
                            f"trace duplicate capture at {key}; verify Segment-ID gating"
                        )
                    output[key] = (
                        x[pos].detach().float().cpu().clone(),
                        y[pos].detach().float().cpu().clone(),
                    )
                handles.append(module.register_forward_hook(hook, with_kwargs=True))
        yield output
    finally:
        for handle in handles:
            handle.remove()


def _metric(ref: torch.Tensor, value: torch.Tensor):
    if ref.shape != value.shape:
        raise AssertionError(
            f"mismatched stage tensors: native={tuple(ref.shape)}, "
            f"tpr={tuple(value.shape)}"
        )
    if not bool(torch.isfinite(ref).all() and torch.isfinite(value).all()):
        raise AssertionError("nonfinite trace tensor")
    delta = (ref.double() - value.double()).abs()
    return float(delta.max()), float(
        torch.linalg.vector_norm(delta) /
        torch.linalg.vector_norm(ref.double()).clamp_min(1e-24)
    )


def describe_trace(native, tpr, *, layers: int) -> list[str]:
    expected = {(i, stage) for i in range(layers) for stage in STAGES}
    if set(native) != expected or set(tpr) != expected:
        raise AssertionError(
            "incomplete Native/TPR layer trace: "
            f"missing_native={sorted(expected-set(native))[:15]} "
            f"missing_tpr={sorted(expected-set(tpr))[:15]} "
            f"extra_native={sorted(set(native)-expected)[:15]} "
            f"extra_tpr={sorted(set(tpr)-expected)[:15]}"
        )
    reports = []
    first = None
    first_material = None
    # Elide tiny float32 GEMM layout rounding (~1e-7) from the
    # "first material deviation" diagnostic; raw per-stage diffs
    # remain fully printed, including the first nonzero ULP.
    absolute_floor = 1e-6
    relative_floor = 1e-5
    for layer in range(layers):
        for stage in STAGES:
            nx, ny = native[layer, stage]
            tx, ty = tpr[layer, stage]
            in_max, in_rel = _metric(nx, tx)
            out_max, out_rel = _metric(ny, ty)
            reports.append(
                f"P0 WEAK_TQ TRACE layer={layer:02d} stage={stage} "
                f"input_max={in_max:.9g} input_rel={in_rel:.9g} "
                f"output_max={out_max:.9g} output_rel={out_rel:.9g}"
            )
            if first is None and (in_max > 0 or out_max > 0):
                first = (layer, stage, in_max, out_max)
            if first_material is None and (
                (in_max > absolute_floor and in_rel > relative_floor)
                or (out_max > absolute_floor and out_rel > relative_floor)
            ):
                first_material = (layer, stage, in_max, out_max)
    reports.append(
        "P0 WEAK_TQ TRACE_SUMMARY "
        f"compared_stages={len(expected)} first_nonzero={first} "
        f"first_material={first_material} "
        "trace_numeric_parity=DIAGNOSTIC_ONLY "
        "note=first_nonzero_may_be_BF16_rounding_not_first_causal_error"
    )
    return reports


def describe_kv_trace(native_kv, tpr_kv, path, *, layers=(1, 2, 3, 4, 14, 28)):
    """Compare actual post-RoPE K/raw-V seen by Native vs TPR leaf.

    Native tensors come from the untouched core_attention forward-hook.
    TPR tensors are concatenations of the *actual* past KV state and
    current segment's context.new_key_values for the same layer.
    path lists (segment_id, start, end) along the target leaf's ancestor path.
    Report per-path region to identify shape-sensitive Prefix production
    separately from positional or assembly faults. Pure diagnostic.
    """
    if not path or path[0][1] != 0:
        raise AssertionError("KV diagnostic path must start at zero")
    last = 0
    for _, start, end in path:
        if start != last or end <= start:
            raise AssertionError(f"KV path is not contiguous: {path}")
        last = end
    if set(native_kv) != set(layers) or set(tpr_kv) != set(layers):
        raise AssertionError(
            f"KV trace missing native={sorted(set(layers)-set(native_kv))} "
            f"tpr={sorted(set(layers)-set(tpr_kv))}"
        )
    report = []
    max_error = (-1.0, None)
    for number in layers:
        for field, i in (("K_POST_ROPE", 0), ("V_RAW", 1)):
            full_native = native_kv[number][i].detach().float().cpu()
            full_tpr = tpr_kv[number][i].detach().float().cpu()
            if full_native.shape != full_tpr.shape or full_native.shape[0] != last:
                raise AssertionError(
                    f"KV mismatch layer={number} {field} "
                    f"native={tuple(full_native.shape)} "
                    f"tpr={tuple(full_tpr.shape)} total_length={last}"
                )
            if not bool(torch.isfinite(full_native).all() and torch.isfinite(full_tpr).all()):
                raise AssertionError(f"KV nonfinite in layer={number} {field}")
            for segment_id, start, end in path:
                a, b = full_native[start:end].double(), full_tpr[start:end].double()
                delta = (a - b).abs()
                delta_max = float(delta.max())
                rel = float(
                    torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(a).clamp_min(1e-24)
                )
                positions = (delta.reshape(end-start, -1).amax(dim=1) > 0)
                first_local = int(torch.nonzero(positions).flatten()[0]) if bool(positions.any()) else -1
                first_abs = start + first_local if first_local >= 0 else -1
                report.append(
                    "P0 WEAK_TQ KV_TRACE "
                    f"layer={number:02d} field={field} "
                    f"segment={segment_id}[{start}:{end}] "
                    f"max_abs={delta_max:.9g} rel_l2={rel:.9g} "
                    f"differing_tokens={int(positions.sum())}/{end-start} "
                    f"first_differing_abs={first_abs}"
                )
                if delta_max > max_error[0]:
                    max_error = (delta_max, (number, field, segment_id, start, end))
    report.append(
        "P0 WEAK_TQ KV_SUMMARY "
        f"compared_layers={len(layers)} "
        f"segments_per_layer={len(path)} "
        f"rows={len(report)} "
        f"worst_max_abs={max_error[0]:.9g} "
        f"worst_location={max_error[1]} "
        "same_model_checkpoint=True probe_only=True"
    )
    return report
