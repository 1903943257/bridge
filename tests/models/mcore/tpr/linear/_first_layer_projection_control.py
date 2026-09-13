"""Opt-in causal intervention for tests only; never installed in production."""

from contextlib import contextmanager
import os

import torch


def _controlled_forward(original, counts):
    # Factory binds each projection and its own counters independently.
    def controlled(hidden, *args, **kwargs):
        assert hidden.ndim == 3 and hidden.shape[1] == 1, hidden.shape
        if hidden.shape[0] == 64:
            counts["prefix64_unchanged"] += 1
            return original(hidden, *args, **kwargs)
        assert hidden.shape[0] == 128, f"control only supports 64/128 tokens, got {hidden.shape}"
        counts["full128_to_2x64"] += 1
        pieces = [original(part.contiguous(), *args, **kwargs) for part in hidden.split(64, dim=0)]
        if isinstance(pieces[0], tuple):
            assert all(isinstance(part, tuple) and len(part) == 2 for part in pieces)
            first_bias, second_bias = pieces[0][1], pieces[1][1]
            # This Qwen fixture has add_bias_linear=False. Do not silently
            # invent semantics for an unfamiliar projection return value.
            assert first_bias is None and second_bias is None, "control requires bias-free projection"
            return torch.cat([part[0] for part in pieces], dim=0), None
        assert all(isinstance(part, torch.Tensor) for part in pieces)
        return torch.cat(pieces, dim=0)

    return controlled


@contextmanager
def first_layer_projection_control(model, monkeypatch):
    modes = {}
    for name, variable in (("out_proj", "STAGE33_OUT_PROJ_CHUNK64"),
                           ("mlp_fc2", "STAGE33_MLP_FC2_CHUNK64")):
        mode = os.getenv(variable, "0")
        if mode not in ("0", "1"):
            raise ValueError(f"{variable} must be 0 or 1")
        modes[name] = mode == "1"
        print(f"STAGE-3.3 CONTROL first-layer-{name}-chunk64={mode}", flush=True)
    if not any(modes.values()):
        yield False
        return

    layer = model.decoder.layers[0]
    assert layer.layer_number == 1
    assert getattr(layer.self_attention, "tpr_state_kind", None) == "gdn"
    assert model.config.context_parallel_size == 1
    projections = {}
    if modes["out_proj"]:
        projections["out_proj"] = layer.self_attention.out_proj
    if modes["mlp_fc2"]:
        projections["mlp_fc2"] = layer.mlp.linear_fc2
    counters = {name: {"full128_to_2x64": 0, "prefix64_unchanged": 0} for name in projections}
    with monkeypatch.context() as patch:
        for name, projection in projections.items():
            patch.setattr(projection, "forward", _controlled_forward(projection.forward, counters[name]))
        try:
            yield True
        finally:
            for name, counts in counters.items():
                print(f"STAGE-3.3 CONTROL {name} calls={counts}; restored on exit", flush=True)
    for name, counts in counters.items():
        assert counts["full128_to_2x64"] > 0, f"{name} intervention did not execute"
        assert counts["prefix64_unchanged"] > 0, f"{name} unchanged short path did not execute"
