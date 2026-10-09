"""Qwen3-1.7B pretrained GEMM physical-M benchmark on Ascend NPU.

Opt-in: TPR_RUN_QWEN17_GEMM_BENCH=1.
This is a FORWARD-ONLY microbenchmark, with real Qwen3 weights but
synthetic BF16 hidden states. It does not establish training/backward support
for npu_grouped_matmul, or end-to-end TPR speedup. No production patch.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_RUN_QWEN17_GEMM_BENCH") != "1",
    reason="Set TPR_RUN_QWEN17_GEMM_BENCH=1",
)


def _out(result):
    if isinstance(result, tuple):
        if len(result) != 2 or result[1] is not None:
            raise AssertionError("Expected bias-free Qwen3 Megatron Linear")
        return result[0]
    if not isinstance(result, torch.Tensor):
        raise AssertionError(f"Unexpected Linear result {type(result)}")
    return result


def _timed(name, call, *, warmup, repeats):
    with torch.no_grad():
        for _ in range(warmup):
            output = call()
            del output
        torch.npu.synchronize()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = call()
            torch.npu.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            del output
    median = statistics.median(timings)
    print(
        f"QWEN17 GEMM BENCH {name} median_ms={median:.4f} "
        f"min_ms={min(timings):.4f} max_ms={max(timings):.4f} "
        f"warmup={warmup} repeats={repeats}",
        flush=True,
    )
    return median


def _error(name, candidate, reference):
    a = candidate.detach().float()
    b = reference.detach().float()
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shapes differ {a.shape} vs {b.shape}")
    delta = a-b
    norm = torch.linalg.vector_norm(b).clamp_min(1e-12)
    rel = float(torch.linalg.vector_norm(delta)/norm)
    max_abs = float(delta.abs().max())
    eq = torch.equal(candidate, reference)
    print(
        f"QWEN17 GEMM BENCH NUMERICS {name} "
        f"rel_l2={rel:.8g} max_abs={max_abs:.8g} bitwise={eq}",
        flush=True,
    )
    return eq


def test_real_qwen17_physical_m_gemm_benchmark():
    from ..correctness.test_qwen3_1_7b_split_equivalence_npu import _setup_real_model

    fixture, target = _setup_real_model()
    device = torch.device("npu")
    model = fixture._make_qwen_model(
        device, tpr=True,
        max_sequence_length=int(os.environ.get("TPR_QWEN17_GEMM_MAX_SEQ", "1152")),
    )
    assert target.assert_model_scale(model) > 1_500_000_000

    tile = int(os.environ.get("TPR_QWEN17_GEMM_TILE", "128"))
    warmup = int(os.environ.get("TPR_QWEN17_GEMM_WARMUP", "2"))
    repeats = int(os.environ.get("TPR_QWEN17_GEMM_REPEATS", "8"))
    if tile <= 0 or warmup < 0 or repeats < 1:
        raise AssertionError("invalid GEMM tile/warmup/repeat settings")
    rows = tuple(
        int(x.strip()) for x in
        os.environ.get("TPR_QWEN17_GEMM_M", "128,1024,1152").split(",")
    )
    if not rows or any(m <= 0 or m % tile for m in rows):
        raise AssertionError(f"All M must be positive multiples of tile={tile}")
    groups = {
        g.strip().lower() for g in os.environ.get(
            "TPR_QWEN17_GEMM_GROUPS", "qkv,proj,fc1,fc2"
        ).split(",")
    }
    available = {
        "qkv": model.decoder.layers[0].self_attention.linear_qkv,
        "proj": model.decoder.layers[0].self_attention.linear_proj,
        "fc1": model.decoder.layers[0].mlp.linear_fc1,
        "fc2": model.decoder.layers[0].mlp.linear_fc2,
    }
    if not groups or not groups.issubset(available):
        raise AssertionError(f"Invalid GEMM groups: {groups}")

    try_grouped = os.environ.get("TPR_QWEN17_GEMM_TRY_GROUPED", "0") == "1"
    try:
        import torch_npu
    except ImportError:
        torch_npu = None
    grouped_api = getattr(torch_npu, "npu_grouped_matmul", None) if torch_npu else None
    print(
        "QWEN17 GEMM BENCH CONFIG "
        f"tile={tile} M={rows} groups={sorted(groups)} "
        f"grouped_api_present={grouped_api is not None} "
        f"try_grouped={try_grouped} "
        f"torch={torch.__version__} "
        f"torch_npu={getattr(torch_npu,'__version__','unavailable')}",
        flush=True,
    )

    for name in sorted(groups):
        module = available[name]
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16:
            raise AssertionError(f"Expected loaded BF16 weight for {name}")
        nin = int(weight.shape[1])
        # Real pretrained Qwen weights, controlled BF16 input; separate from
        # recorded-token correctness tests (which use actual activations).
        for m in rows:
            x = torch.randn((m,1,nin), device=device, dtype=torch.bfloat16)

            def baseline():
                return _out(module(x))

            def tiled():
                return torch.cat(
                    [_out(module(chunk.contiguous()))
                     for chunk in x.split(tile, dim=0)],
                    dim=0,
                )

            with torch.no_grad():
                full_y = baseline()
                tile_y = tiled()
                _error(f"{name} M={m} tiled_vs_full", tile_y, full_y)
                del full_y, tile_y

            baseline_ms = _timed(
                f"{name} M={m} baseline", baseline, warmup=warmup, repeats=repeats
            )
            tile_ms = _timed(
                f"{name} M={m} tiled", tiled, warmup=warmup, repeats=repeats
            )
            print(
                f"QWEN17 GEMM BENCH SLOWDOWN {name} M={m} "
                f"tiled_over_baseline={tile_ms/max(baseline_ms,1e-9):.4f}",
                flush=True,
            )

            if try_grouped:
                if grouped_api is None:
                    print(
                        f"QWEN17 GEMM BENCH GROUPED {name} M={m} "
                        "status=UNAVAILABLE missing torch_npu API",
                        flush=True,
                    )
                else:
                    # torch_npu 2.9 requires an explicit group_type.
                    # Compare two PUBLIC supported operand layouts:
                    #   -1: multiple x, multiple weight, multiple y
                    #    0: single x, multiple weight, single y, cumsum
                    # Both lists point at ONE weight tensor. No full weight
                    # duplication in Python; the kernel may still reload it.
                    # Weight transposition/packing below is OUTSIDE timings.
                    x_parts = [
                        part.contiguous()
                        for part in x[:, 0, :].split(tile, dim=0)
                    ]
                    x_full = x[:, 0, :].contiguous()
                    w = weight.detach().transpose(0, 1).contiguous()
                    weight_list = [w] * len(x_parts)
                    group_cumsum = torch.arange(
                        tile, m + 1, tile,
                        device=device, dtype=torch.int64,
                    )

                    def multi_x_multi_w():
                        ys = grouped_api(
                            x_parts, weight_list,
                            group_type=-1, split_item=0,
                        )
                        if not isinstance(ys, (tuple, list)) or len(ys) != len(x_parts):
                            raise AssertionError(
                                f"GROUPED_MMM returned {type(ys)} "
                                f"length={len(ys) if hasattr(ys, '__len__') else '?'}"
                            )
                        return torch.cat(ys, dim=0).unsqueeze(1)

                    def single_x_multi_w():
                        ys = grouped_api(
                            [x_full], weight_list,
                            group_list=group_cumsum, group_type=0,
                            group_list_type=0, split_item=2,
                        )
                        if not isinstance(ys, (tuple, list)) or len(ys) != 1:
                            raise AssertionError(
                                f"GROUPED_SMM returned {type(ys)} "
                                f"length={len(ys) if hasattr(ys, '__len__') else '?'}"
                            )
                        return ys[0].unsqueeze(1)

                    for mode, grouped in (
                        ("multi_x_multi_w", multi_x_multi_w),
                        ("single_x_multi_w", single_x_multi_w),
                    ):
                        try:
                            with torch.no_grad():
                                out_grouped = grouped()
                                out_tiled = tiled()
                                exact = _error(
                                    f"{name} M={m} {mode} grouped_vs_tiled",
                                    out_grouped, out_tiled,
                                )
                                del out_grouped, out_tiled
                            grouped_ms = _timed(
                                f"{name} M={m} {mode}",
                                grouped, warmup=warmup, repeats=repeats,
                            )
                            print(
                                f"QWEN17 GEMM BENCH GROUPED {name} M={m} "
                                f"mode={mode} status=FORWARD_OK bitwise_tile={exact} "
                                f"grouped_over_tiled={grouped_ms/max(tile_ms,1e-9):.4f} "
                                f"grouped_over_baseline={grouped_ms/max(baseline_ms,1e-9):.4f} "
                                "backward_supported=UNVERIFIED",
                                flush=True,
                            )
                        except (RuntimeError, TypeError, ValueError, AssertionError) as exc:
                            print(
                                f"QWEN17 GEMM BENCH GROUPED {name} M={m} "
                                f"mode={mode} status=UNSUPPORTED "
                                f"error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    del x_parts, x_full, weight_list, w, group_cumsum
            del x
    print("QWEN17 GEMM BENCH COMPLETE: forward microbenchmark only", flush=True)
