"""Test-only CP1 first-layer projection shape intervention; default off."""

from contextlib import contextmanager
import os

import torch


def zigzag_control_forward(original, counts):
    def forward(hidden, *args, **kwargs):
        assert hidden.ndim == 3 and hidden.shape[:2] == (128, 1), hidden.shape
        # CP2 rank0 owns chunks 0,3; rank1 owns 1,2. This is NOT split(64).
        groups = ((0, 3), (1, 2))
        indices = [torch.cat([torch.arange(c * 32, (c + 1) * 32, device=hidden.device)
                              for c in group]) for group in groups]
        parts = [original(hidden.index_select(0, index).contiguous(), *args, **kwargs)
                 for index in indices]
        is_tuple = isinstance(parts[0], tuple)
        if is_tuple:
            assert all(isinstance(p, tuple) and len(p) == 2 and p[1] is None for p in parts), \
                "Qwen control requires bias-free (output, None)"
            tensors = [p[0] for p in parts]
        else:
            assert all(isinstance(p, torch.Tensor) for p in parts)
            tensors = parts
        # Differentiable inverse permutation routes input/weight VJPs normally.
        inverse = torch.argsort(torch.cat(indices))
        output = torch.cat(tensors, dim=0).index_select(0, inverse)
        counts["full128_to_zigzag_2x64"] += 1
        return (output, None) if is_tuple else output
    return forward


@contextmanager
def first_gdn_zigzag_control(model, monkeypatch):
    mode = os.getenv("STAGE43_OUT_PROJ_ZIGZAG64", "0")
    if mode not in ("0", "1"):
        raise ValueError("STAGE43_OUT_PROJ_ZIGZAG64 must be 0 or 1")
    print(f"STAGE-4.3 CONTROL layer1-out-proj-zigzag64={mode}", flush=True)
    counts = {"full128_to_zigzag_2x64": 0}
    if mode == "0":
        yield False, counts
        return
    layer = model.decoder.layers[0]
    assert layer.layer_number == 1 and layer.self_attention.tpr_state_kind == "gdn"
    assert model.config.context_parallel_size == 1
    projection = layer.self_attention.out_proj
    with monkeypatch.context() as patch:
        patch.setattr(projection, "forward", zigzag_control_forward(projection.forward, counts))
        try:
            yield True, counts
        finally:
            print(f"STAGE-4.3 CONTROL calls={counts}; CP1 projection restored on exit", flush=True)
