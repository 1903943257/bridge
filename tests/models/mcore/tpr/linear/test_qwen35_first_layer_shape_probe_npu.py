"""Diagnostic only: isolate the first 128-vs-64 forward divergence in layer 1.

Uses the real Qwen3.5 model/embedding/first layer, but does not run 24 layers.
Replays the first divergent callable with canonical identical prefix inputs
and identical output VJP seed (zero suffix seed in the 128-token execution).
No numeric acceptance threshold, optimizer, or kernel changes.
"""

from contextlib import contextmanager

import torch

from .test_qwen35_hybrid_push_branch_pop_npu import (
    runtime,  # noqa: F401 -- shared NPU runtime fixture, patches before model imports
    _plan,
    _no_cp_probe,
)
from baseline._qwen35_baseline_utils import make_qwen35_model, gradient_map_diagnostics


def _tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _snapshot(value):
    return value.detach().clone() if isinstance(value, torch.Tensor) else value


def _metric(name, full, short, axis=0):
    left = full.narrow(axis, 0, 64).detach().cpu().float()
    right = short.detach().cpu().float()
    assert left.shape == right.shape, (name, left.shape, right.shape)
    assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
    diag = gradient_map_diagnostics({name: left}, {name: right}).aggregate
    exact = torch.equal(left, right)
    print(f"FIRST-LAYER {name}: exact={exact}, {diag}, "
          f"max_abs={(right-left).abs().max().item():.9e}", flush=True)
    return exact


class _Trace:
    def __init__(self):
        self.records = {}

    def save(self, name, value, axis):
        assert name not in self.records, name
        self.records[name] = {"value": _snapshot(value), "axis": axis}

    def call(self, name, fn, args, kwargs, axes, out_axis):
        # Snapshot BEFORE invocation; some kernels may use mutable buffers.
        copied_args = tuple(_snapshot(x) for x in args)
        copied_kwargs = {k: _snapshot(v) for k, v in kwargs.items()}
        for key, axis in axes.items():
            value = args[key] if isinstance(key, int) else kwargs[key]
            self.save(f"{name}.input.{key}", value, axis)
        output = fn(*args, **kwargs)
        self.save(f"{name}.output", _tensor(output), out_axis)
        self.records[f"{name}.output"]["replay"] = (fn, copied_args, copied_kwargs, axes)
        return output


@contextmanager
def _trace_layer(layer, monkeypatch):
    import verl.models.mcore.tpr.gated_delta_net as gdn_module

    trace = _Trace()
    handles = []
    attention = layer.self_attention
    # Wrappers call the original module.forward; modules/parameters stay intact.
    modules = (
        ("input_norm", layer.input_layernorm),
        ("in_proj", attention.in_proj),
        ("out_proj", attention.out_proj),
        ("mlp_norm", layer.pre_mlp_layernorm),
        ("mlp_fc1", layer.mlp.linear_fc1),
        ("mlp_fc2", layer.mlp.linear_fc2),
    )
    with monkeypatch.context() as patch:
        for name, module in modules:
            original = module.forward

            def wrapped(*args, _name=name, _fn=original, **kwargs):
                return trace.call(_name, _fn, args, kwargs, {0: 0}, 0)

            patch.setattr(module, "forward", wrapped)
        original_conv = gdn_module._stage1_causal_conv1d
        original_gdr = gdn_module._stage1_gated_delta_rule
        original_norm = attention._apply_gated_norm

        def conv(*args, **kwargs):
            return trace.call("conv", original_conv, args, kwargs, {0: 1}, 1)

        def gdr(*args, **kwargs):
            return trace.call("gdr", original_gdr, args, kwargs,
                              {0: 1, 1: 1, 2: 1, "g": 1, "beta": 1}, 1)

        def norm(*args, **kwargs):
            return trace.call("gated_norm", original_norm, args, kwargs, {0: 1, 1: 1}, 1)

        patch.setattr(gdn_module, "_stage1_causal_conv1d", conv)
        patch.setattr(gdn_module, "_stage1_gated_delta_rule", gdr)
        patch.setattr(attention, "_apply_gated_norm", norm)
        # Explicit residual seams; do not reconstruct residual/BDA math.
        handles.append(layer.pre_mlp_layernorm.register_forward_pre_hook(
            lambda module, args: trace.save("attention_residual", args[0], 0)))
        try:
            yield trace
        finally:
            for handle in handles:
                handle.remove()


def _forward(layer, hidden):
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context

    # Layer 1 is GDN: no RoPE/FA, external state, or packed metadata is needed.
    context = TPRAttentionContext(prefix_length=0, suffix_length=hidden.shape[0])
    with use_tpr_attention_context(context):
        return _tensor(layer(hidden_states=hidden, attention_mask=None))


def _replay(layer, record):
    fn, original_args, original_kwargs, axes = record["replay"]
    out_axis = record["axis"]
    results = []
    seed = None
    for length in (128, 64):
        layer.zero_grad(set_to_none=True)
        leaves = {}

        def leaf(key, value):
            if not isinstance(value, torch.Tensor):
                return value
            if key in axes:
                value = value.narrow(axes[key], 0, length)
            value = value.detach().clone()
            if value.is_floating_point():
                value.requires_grad_(True)
                leaves[key] = value
            return value

        args = tuple(leaf(i, v) for i, v in enumerate(original_args))
        kwargs = {k: leaf(k, v) for k, v in original_kwargs.items()}
        output = _tensor(fn(*args, **kwargs))
        assert output.shape[out_axis] == length
        prefix = output.narrow(out_axis, 0, 64)
        if seed is None:
            seed = torch.sin(torch.arange(prefix.numel(), device=output.device).float())
            seed = (seed.reshape(prefix.shape) / prefix.numel()).to(output.dtype)
        upstream = torch.zeros_like(output)
        upstream.narrow(out_axis, 0, 64).copy_(seed)
        output.backward(upstream)
        grads = {}
        for key, value in leaves.items():
            if key in axes:
                assert value.grad is not None, f"missing isolated sequence-input gradient: {key}"
            if value.grad is not None:
                grad = value.grad
                if key in axes:
                    grad = grad.narrow(axes[key], 0, 64)
                grads[f"input.{key}"] = grad.detach().cpu().float().clone()
        for name, parameter in layer.named_parameters():
            if parameter.grad is not None:
                grads[f"parameter.{name}"] = parameter.grad.detach().cpu().float().clone()
        assert grads, "isolated replay produced no gradients"
        results.append((output.detach().cpu(), grads))
    _metric("isolated/canonical-input/output", results[0][0], results[1][0], out_axis)
    left, right = results[0][1], results[1][1]
    assert left.keys() == right.keys()
    for name in left:
        assert torch.isfinite(left[name]).all() and torch.isfinite(right[name]).all(), name
        print(f"FIRST-LAYER isolated/canonical-input/{name}: "
              f"{gradient_map_diagnostics({name: left[name]}, {name: right[name]}).aggregate}", flush=True)


def test_first_layer_full_vs_prefix(runtime, monkeypatch):
    torch.manual_seed(353301)
    model = make_qwen35_model(runtime, cp_size=1, tpr=True)
    layer = model.decoder.layers[0]
    assert layer.layer_number == 1
    plan = _plan()
    tokens = torch.cat((plan.get(0).token_ids, plan.get(1).token_ids)).to(runtime.device).unsqueeze(0)
    positions = torch.arange(128, device=runtime.device).unsqueeze(0)
    with _no_cp_probe(monkeypatch):
        with torch.no_grad():
            full_input = model.embedding(input_ids=tokens, position_ids=positions)
            short_input = model.embedding(input_ids=tokens[:, :64], position_ids=positions[:, :64])
            _metric("embedding", full_input, short_input)
        with torch.enable_grad():
            traces = []
            # Deliberately eliminate embedding differences from downstream tests.
            for length in (128, 64):
                hidden = full_input[:length].detach().clone().requires_grad_(True)
                with _trace_layer(layer, monkeypatch) as trace:
                    trace.save("layer_input", hidden, 0)
                    output = _forward(layer, hidden)
                    trace.save("layer_output_after_mlp_residual", output, 0)
                traces.append(trace.records)
                del output
        full, short = traces
        assert full.keys() == short.keys()
        first = None
        for name, record in full.items():
            exact = _metric(name, record["value"], short[name]["value"], record["axis"])
            if not exact and first is None:
                first = name
        print(f"FIRST-LAYER first-nonzero={first} (not a significance threshold)", flush=True)
        if first is not None and "replay" in full[first]:
            # Full run's inputs are canonical for BOTH calls, so inherited
            # upstream differences cannot contaminate this operator replay.
            _replay(layer, full[first])
        elif first is not None:
            print("FIRST-LAYER first divergence is an input/residual seam; "
                  "no standalone callable at this seam, inspect its preceding operation", flush=True)
        print("FIRST-LAYER DIAGNOSTIC COMPLETE (not Stage 3.3 correctness PASS)", flush=True)
