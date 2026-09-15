"""Ordinary CP2: CP1 / AG / old Ring / native Ring; no numerical gate waiver.

Fixed full-Q upstream tests both zigzag halves, unlike suffix-only probes.
FA shape assertions count kernel launches, not the public Ring wrapper.
"""

import importlib

import torch

from .test_full_qwen35_tree_cp_npu import runtime, gather_native_zigzag
from .test_fa_transport_matrix_npu import metric
from ._whole_fa_allgather import install_whole_fa_allgather
from ._ordinary_ring_control import install_ordinary_ring
from ._native_ring_tnd_layout import install_native_tnd_layout


def test_ordinary_ring_schedule(runtime, monkeypatch):
    import torch_npu
    from verl.models.mcore.tpr.parallel import ring_attention as ring
    from verl.models.mcore.tpr.rectangular_attention import rectangular_causal_attention

    native = importlib.import_module(
        "mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel"
    )
    gen = torch.Generator().manual_seed(445501)
    base = [torch.randn(128, 1, h, 256, generator=gen).bfloat16()
            for h in (8, 2, 2)]
    upstream = torch.randn(128, 1, 2048, generator=gen).bfloat16() * 1e-3
    shard = ring.make_ring_sequence_shard(128, cp_rank=runtime.cp_group.rank(), cp_size=2)
    idx = shard.global_indices(device="cpu")
    results = {}
    for backend in ("cp1", "cp1_sbh", "allgather", "old_ring", "native_ring", "native_tnd"):
        cp = backend not in ("cp1", "cp1_sbh")
        inputs = [(x[idx] if cp else x).to(runtime.device).clone().requires_grad_() for x in base]
        calls, backward_calls, stats = [], [], []
        with monkeypatch.context() as patch:
            if backend == "allgather":
                install_whole_fa_allgather(patch)
            elif backend in ("native_ring", "native_tnd"):
                install_ordinary_ring(patch)
            original_fwd = torch_npu.npu_fusion_attention
            original_bwd = torch_npu.npu_fusion_attention_grad
            original_update = native.causal_out_update
            original_finalize = ring._finalize_attention_result

            def finalize(*args, **kwargs):
                result = original_finalize(*args, **kwargs)
                stats.append((result[1].detach().clone(), result[2].detach().clone()))
                return result

            def forward(*args, **kwargs):
                result = original_fwd(*args, **kwargs)
                calls.append((tuple(args[0].shape), tuple(args[1].shape), args[4], kwargs["sparse_mode"]))
                if backend in ("cp1", "cp1_sbh"):
                    stats.append((result[1].detach().clone(), result[2].detach().clone()))
                return result

            def backward(*args, **kwargs):
                backward_calls.append((tuple(args[0].shape), tuple(args[1].shape), args[5], kwargs["sparse_mode"]))
                return original_bwd(*args, **kwargs)

            def update(*args, **kwargs):
                result = original_update(*args, **kwargs)
                stats[:] = [(result[1].detach().clone(), result[2].detach().clone())]
                return result

            patch.setattr(torch_npu, "npu_fusion_attention", forward)
            patch.setattr(torch_npu, "npu_fusion_attention_grad", backward)
            patch.setattr(native, "causal_out_update", update)
            if backend == "old_ring":
                patch.setattr(ring, "_finalize_attention_result", finalize)
            if backend == "native_tnd":
                # Install outside the trace so trace sees real TND kernel args,
                # not the native SBH arguments entering the conversion shim.
                install_native_tnd_layout(patch, torch_npu)
            if cp:
                output = ring.ring_cp_attention(*inputs, prefix_blocks=(), current_shard=shard,
                    cp_group=runtime.cp_group, softmax_scale=0.0625)
            elif backend == "cp1_sbh":
                # Same scale/window/mask/precision as CP1 TND; layout only.
                output = torch_npu.npu_fusion_attention(
                    *(x.flatten(2).contiguous() for x in inputs), 8, "SBH",
                    pse=None, padding_mask=None,
                    atten_mask=ring._compressed_causal_mask(runtime.device),
                    scale=0.0625, pre_tockens=ring._MAX_TOKENS, next_tockens=0,
                    keep_prob=1.0, inner_precise=0, sparse_mode=3,
                )[0]
            else:
                output = rectangular_causal_attention(*inputs, softmax_scale=0.0625)
            grads = torch.autograd.grad(output, inputs, (upstream[idx] if cp else upstream).to(runtime.device))

        if backend == "old_ring":
            assert len(calls) == len(backward_calls) == 5, (calls, backward_calls)
        if backend == "cp1_sbh":
            assert calls == [((128, 1, 2048), (128, 1, 512), "SBH", 3)], calls
        if backend in ("native_ring", "native_tnd"):
            remote = (32, 64) if runtime.cp_group.rank() == 0 else (64, 32)
            expected = [(64, 64), remote]
            assert [(q[0], k[0]) for q, k, _, _ in calls] == expected, calls
            layout = "TND" if backend == "native_tnd" else "SBH"
            assert [item[2] for item in calls] == [layout, layout], calls
            assert [mode for _, _, _, mode in calls] == [3, 0], calls
            shapes = ((64, 8, 256), (64, 2, 256)) if layout == "TND" else ((64, 1, 2048), (64, 1, 512))
            assert calls[0][:2] == shapes, calls
            assert backward_calls == list(reversed(calls)), backward_calls

        def full(x):
            x = x.detach()
            return (gather_native_zigzag(x, runtime.cp_group) if cp else x).cpu().float()

        result = dict(zip(("output", "dQ", "dK", "dV"), map(full, (output, *grads))))
        results[backend] = result
        for name in result:
            metric(f"whole/CP1-vs-{backend}/{name}", {name: results["cp1"][name]},
                   {name: result[name]}, runtime.rank)
        if stats:
            # SBH native stats may retain [B,H,2,half,8]; canonicalize to [T,H].
            maximum = torch.cat([m.reshape(1, 8, -1, 8)[0, :, :, 0].T for m, _ in stats])
            total = torch.cat([s.reshape(1, 8, -1, 8)[0, :, :, 0].T for _, s in stats])
            lse = full(maximum.float() + total.float().log())
            results[backend]["lse"] = lse
            metric(f"whole/CP1-vs-{backend}/LSE", {"lse": results["cp1"]["lse"]}, {"lse": lse}, runtime.rank)
        print(f"ORDINARY-SCHEDULE r={runtime.rank} {backend} fwd={calls} bwd={backward_calls}", flush=True)
    for left, right, label in (
        ("cp1", "cp1_sbh", "layout-whole-TND-vs-SBH"),
        ("native_tnd", "native_ring", "layout-native-2call-TND-vs-SBH"),
        ("old_ring", "native_tnd", "old-5call-vs-native-2call-TND"),
        ("cp1_sbh", "native_ring", "SBH-whole-vs-native-2call"),
    ):
        for name in ("output", "dQ", "dK", "dV", "lse"):
            metric(f"whole/{label}/{name}", {name: results[left][name]},
                   {name: results[right][name]}, runtime.rank)
    print("NOTE: old-5call-vs-native-2call also differs in merge/casts/backward reduction; "
          "the two layout comparisons keep their respective schedules fixed.", flush=True)
    print("ORDINARY-SCHEDULE COMPLETE: diagnostic metrics, not a training correctness PASS", flush=True)
