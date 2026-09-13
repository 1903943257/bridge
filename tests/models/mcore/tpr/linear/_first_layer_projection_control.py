"""Opt-in causal intervention for tests only; never installed in production."""

from contextlib import contextmanager
import os

import torch


@contextmanager
def first_layer_projection_control(model, monkeypatch):
    mode = os.getenv("STAGE33_OUT_PROJ_CHUNK64", "0")
    if mode not in ("0", "1"):
        raise ValueError("STAGE33_OUT_PROJ_CHUNK64 must be 0 or 1")
    print(f"STAGE-3.3 CONTROL first-layer-out-proj-chunk64={mode}", flush=True)
    if mode == "0":
        yield False
        return

    layer = model.decoder.layers[0]
    assert layer.layer_number == 1
    assert getattr(layer.self_attention, "tpr_state_kind", None) == "gdn"
    assert model.config.context_parallel_size == 1
    projection = layer.self_attention.out_proj
    original = projection.forward
    counts = {"full128_to_2x64": 0, "prefix64_unchanged": 0}

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

    with monkeypatch.context() as patch:
        patch.setattr(projection, "forward", controlled)
        try:
            yield True
        finally:
            print(f"STAGE-3.3 CONTROL calls={counts}; first-layer projection restored on exit", flush=True)
    assert counts["full128_to_2x64"] > 0, "intervention did not execute"
    assert counts["prefix64_unchanged"] > 0, "unchanged short path did not execute"
