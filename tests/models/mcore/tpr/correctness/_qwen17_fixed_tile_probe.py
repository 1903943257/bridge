"""Test-only shape-invariant BF16 GEMM for cropped real-TQ PPO triage.

This is intentionally slow. Every Megatron Linear is called using physical
M=tile_size, padding ONLY the last GEMM tile with zero input rows, and slicing
away dummy output rows. No attention mask, token position, or KV is padded.

Install on BOTH Native and Forest models. This is a diagnostic numerical
oracle and does not promise backward wgrad numerical equivalence or speed.
"""
from __future__ import annotations

import torch


def install_test_only_fixed_tile_gemm(
    model, monkeypatch, *, tile_size: int = 128,
    groups=("qkv", "proj", "fc1", "fc2"),
):
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    groups = set(groups)
    candidates = {"qkv", "proj", "fc1", "fc2"}
    if not groups or not groups.issubset(candidates):
        raise ValueError(f"Invalid fixed-tile groups={groups}")
    if model.config.tensor_model_parallel_size != 1:
        raise ValueError("test-only fixed GEMM requires TP=1")
    count = 0
    for layer in model.decoder.layers:
        linear_modules = {
            "qkv": layer.self_attention.linear_qkv,
            "proj": layer.self_attention.linear_proj,
            "fc1": layer.mlp.linear_fc1,
            "fc2": layer.mlp.linear_fc2,
        }
        for group in sorted(groups):
            module = linear_modules[group]
            original = module.forward
            label = f"L{layer.self_attention.layer_number:02d}_{group}"

            def fixed_forward(
                x, *args, _orig=original, _label=label, **kwargs,
            ):
                if (
                    not isinstance(x, torch.Tensor)
                    or x.ndim != 3 or x.shape[1] != 1
                    or x.dtype != torch.bfloat16
                ):
                    raise AssertionError(
                        f"{_label}: require BF16 [T,1,H], "
                        f"got shape={getattr(x,'shape',None)} "
                        f"dtype={getattr(x,'dtype',None)}"
                    )
                if x.shape[0] == 0:
                    raise AssertionError(f"{_label}: cannot run empty GEMM")
                pieces = []
                tuple_output = None
                for begin in range(0, x.shape[0], tile_size):
                    part = x[begin:begin+tile_size].contiguous()
                    valid = part.shape[0]
                    if valid < tile_size:
                        # Pad along M only. The linear output does not mix
                        # tokens; fake padded rows never reach the caller.
                        filler = part.new_zeros(
                            (tile_size-valid, *part.shape[1:])
                        )
                        part = torch.cat([part, filler], dim=0)
                    ret = _orig(part, *args, **kwargs)
                    if isinstance(ret, tuple):
                        if len(ret) != 2 or ret[1] is not None:
                            raise AssertionError(
                                f"{_label}: unsupported nonzero bias"
                            )
                        y, _ = ret
                        is_tuple = True
                    else:
                        y = ret
                        is_tuple = False
                    if tuple_output is None:
                        tuple_output = is_tuple
                    elif tuple_output != is_tuple:
                        raise AssertionError(
                            f"{_label}: inconsistent Linear return type"
                        )
                    if y.shape[0] != tile_size:
                        raise AssertionError(
                            f"{_label}: wrong GEMM output shape={y.shape}"
                        )
                    pieces.append(y[:valid])
                output = torch.cat(pieces, dim=0)
                return (output, None) if tuple_output else output

            monkeypatch.setattr(module, "forward", fixed_forward)
            count += 1
    print(
        "QWEN17 PPO TEST-ONLY SYMMETRIC_BF16_FIXED_TILE "
        f"tile={tile_size} groups={sorted(groups)} "
        f"patched_linear_modules={count} "
        "partial_tail_padded=True attention_unchanged=True "
        "forward_and_backward_diagnostic=True",
        flush=True,
    )
