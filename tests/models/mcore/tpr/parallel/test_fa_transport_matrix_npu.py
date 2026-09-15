"""Whole vs rectangular attention transport matrix; no kernel/threshold edits."""

import torch

from .test_full_qwen35_tree_cp_npu import runtime, gradient_map_diagnostics, gather_native_zigzag
from ._whole_fa_allgather import install_whole_fa_allgather


def metric(label, left, right, rank):
    assert left.keys() == right.keys()
    for n in left:
        assert left[n].shape == right[n].shape, n
        assert torch.isfinite(left[n]).all() and torch.isfinite(right[n]).all(), n
    p = gradient_map_diagnostics(left, right)
    print(f"FA-MATRIX r={rank} {label}: rel={p.aggregate.relative_l2:.6e} "
          f"cos={p.aggregate.cosine:.9f} ratio={p.aggregate.norm_ratio:.6f} "
          f"worst={p.worst_name}:{p.worst.relative_l2:.6e}", flush=True)


def test_attention_whole_vs_prefix_transport(runtime, monkeypatch):
    from verl.models.mcore.tpr.rectangular_attention import rectangular_causal_attention
    from verl.models.mcore.tpr.parallel import ring_attention as ring
    from .test_small_hybrid_tree_cp_npu import _communication_probe
    gen = torch.Generator().manual_seed(445501)
    base = {n: torch.randn(128, 1, heads, 256, generator=gen).bfloat16()
            for n, heads in (("q", 8), ("k", 2), ("v", 2))}
    do = torch.randn(64, 1, 2048, generator=gen).bfloat16() * 1e-3
    results = {}
    for rectangular in (False, True):
        for backend in ("cp1", "allgather", "ring"):
            cp = 1 if backend == "cp1" else 2
            shard = ring.make_ring_sequence_shard(64 if rectangular else 128,
                                                cp_rank=runtime.cp_group.rank(), cp_size=2)
            def leaf(tensor):
                if cp == 2:
                    tensor = tensor.index_select(0, shard.global_indices(device="cpu"))
                return tensor.to(runtime.device).clone().requires_grad_()
            inputs = {n: leaf(t[64:] if rectangular else t) for n, t in base.items()}
            if rectangular:
                inputs.update(pk=leaf(base["k"][:64]), pv=leaf(base["v"][:64]))
            with monkeypatch.context() as patch:
                counts = install_whole_fa_allgather(patch) if backend == "allgather" else None
                with _communication_probe(patch) as (comm, a2a):
                    if cp == 1:
                        k = torch.cat((inputs["pk"], inputs["k"])) if rectangular else inputs["k"]
                        v = torch.cat((inputs["pv"], inputs["v"])) if rectangular else inputs["v"]
                        out = rectangular_causal_attention(inputs["q"], k, v, softmax_scale=0.0625)
                    else:
                        blocks = (ring.RingLocalKVBlock(0, shard, inputs["pk"], inputs["pv"]),) if rectangular else ()
                        out = ring.ring_cp_attention(inputs["q"], inputs["k"], inputs["v"], prefix_blocks=blocks,
                                                    current_shard=shard, cp_group=runtime.cp_group, softmax_scale=0.0625)
                    upstream = do if rectangular else torch.cat((torch.zeros_like(do), do))
                    if cp == 2:
                        upstream = upstream.index_select(0, shard.global_indices(device="cpu"))
                    grads = torch.autograd.grad(out, tuple(inputs.values()), upstream.to(runtime.device))
                assert not a2a.calls
                if backend == "allgather":
                    assert counts["all_gather"] == counts["reduce_scatter"] == (5 if rectangular else 3)
                    assert comm["ring_p2p"] == 0
                elif backend == "ring":
                    assert comm["fa_ring"] == 1 and comm["ring_p2p"] == (4 if rectangular else 2)
                else:
                    assert not comm
            def full(t):
                return (gather_native_zigzag(t.detach(), runtime.cp_group) if cp == 2 else t.detach()).cpu().float()
            values = dict(zip(inputs, (full(g) for g in grads)))
            output = full(out)
            if not rectangular:
                values = dict(q=values["q"][64:], k=values["k"][64:], v=values["v"][64:],
                              pk=values["k"][:64], pv=values["v"][:64])
                output = output[64:]
            result = dict(output=output, **values)
            label = f"{backend}/{'rectangular' if rectangular else 'whole'}"
            results[label] = result
            for name in result:
                metric(f"CP1-whole-vs-{label}/{name}", {name: results["cp1/whole"][name]}, {name: result[name]}, runtime.rank)
    for backend in ("cp1", "allgather", "ring"):
        metric(f"{backend}/whole-vs-rectangular", results[f"{backend}/whole"], results[f"{backend}/rectangular"], runtime.rank)
    print("FA-MATRIX COMPLETE: core-only fixed QKV, S-only dO; diagnostic, no correctness gate waived", flush=True)
