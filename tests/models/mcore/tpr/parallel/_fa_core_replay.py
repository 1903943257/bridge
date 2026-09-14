"""Opt-in L4/L24, S2 attention-core replay; no model weights or gates changed."""

from contextlib import contextmanager
import torch
import torch.distributed as dist


class FACoreReplay:
    def __init__(self, cp):
        self.cp = cp
        self.records = {}

    @contextmanager
    def capture(self, model, sid, monkeypatch):
        if sid != 2:
            yield
            return
        from verl.models.mcore.tpr import attention
        from verl.models.mcore.tpr.parallel import ring_attention as ring
        active = []
        handles = []
        rect, ring_fn = attention.rectangular_causal_attention, ring.ring_cp_attention

        def record(q, k, v, pk, pv, scale, output):
            layer = active[-1]
            assert layer not in self.records
            tensors = dict(q=q, k=k, v=v, pk=pk, pv=pv)
            assert all(t.shape[0] == 128 // self.cp for t in tensors.values())
            entry = {name: t.detach().cpu().clone() for name, t in tensors.items()}
            entry.update(scale=scale, output=output.detach().cpu().clone(), upstream=None)
            self.records[layer] = entry
            def gradient(value):
                assert entry["upstream"] is None, "core output VJP invoked twice"
                entry["upstream"] = value.detach().cpu().clone()
            output.register_hook(gradient)

        def local(q, k, v, **kwargs):
            output = rect(q, k, v, **kwargs)
            if active and active[-1] in (4, 24):
                assert k.shape[0] == v.shape[0] == 256 and q.shape[0] == 128
                assert kwargs.get("dropout_p", 0) == 0
                assert kwargs.get("attention_mask") is None
                record(q, k[128:], v[128:], k[:128], v[:128], kwargs.get("softmax_scale"), output)
            return output

        def distributed(q, k, v, **kwargs):
            output = ring_fn(q, k, v, **kwargs)
            if active and active[-1] in (4, 24):
                blocks = kwargs["prefix_blocks"]
                assert len(blocks) == 1 and blocks[0].segment_id == 0
                assert blocks[0].shard.global_length == kwargs["current_shard"].global_length == 128
                record(q, k, v, blocks[0].key, blocks[0].value, kwargs.get("softmax_scale"), output)
            return output

        for layer in model.decoder.layers:
            if layer.layer_number not in (4, 24):
                continue
            def before(module, args):
                active.append(module.layer_number)
            def after(module, args, output):
                assert active.pop() == module.layer_number
            handles.append(layer.self_attention.register_forward_pre_hook(before))
            handles.append(layer.self_attention.register_forward_hook(after))
        try:
            with monkeypatch.context() as patch:
                patch.setattr(attention, "rectangular_causal_attention", local)
                patch.setattr(ring, "ring_cp_attention", distributed)
                yield
        finally:
            for handle in handles:
                handle.remove()

    def compare(self, actual, runtime, metrics):
        from verl.models.mcore.tpr.rectangular_attention import rectangular_causal_attention
        from verl.models.mcore.tpr.parallel.ring_attention import (
            ring_cp_attention, make_ring_sequence_shard, RingLocalKVBlock,
        )
        assert set(self.records) == set(actual.records) == {4, 24}
        rank = runtime.cp_group.rank()
        shard = make_ring_sequence_shard(128, cp_rank=rank, cp_size=2)
        indices = shard.global_indices(device="cpu")
        names = ("q", "k", "v", "pk", "pv")

        def sliced(value):
            return value.index_select(0, indices).contiguous()

        def run(entry, upstream, cp):
            inputs = {n: entry[n].to(runtime.device).detach().clone().requires_grad_(True) for n in names}
            with torch.enable_grad():
                if cp == 1:
                    output = rectangular_causal_attention(inputs["q"], torch.cat((inputs["pk"], inputs["k"])),
                        torch.cat((inputs["pv"], inputs["v"])), softmax_scale=entry["scale"], dropout_p=0.)
                else:
                    block = RingLocalKVBlock(0, shard, inputs["pk"], inputs["pv"])
                    output = ring_cp_attention(inputs["q"], inputs["k"], inputs["v"],
                        prefix_blocks=(block,), current_shard=shard, cp_group=runtime.cp_group,
                        softmax_scale=entry["scale"])
                # The Ring backward already returns owner-local KV gradients.
                # Do NOT SUM or divide the replay gradients by CP again.
                grads = torch.autograd.grad(output, tuple(inputs.values()), grad_outputs=upstream.to(runtime.device))
            return {"output": output.detach().cpu(), **{n: g.detach().cpu() for n, g in zip(names, grads)}}

        for layer in (4, 24):
            reference, own = self.records[layer], actual.records[layer]
            assert reference["scale"] == own["scale"]
            assert reference["upstream"] is not None and own["upstream"] is not None
            # CP1 runs independently on each rank; its backward can differ.
            # Broadcast full canonical inputs AND dO before sharding so every
            # Ring rank uses one coherent global objective (never mixed dO).
            global_canonical = {}
            for n in (*names, "upstream"):
                tensor = reference[n].to(runtime.device).clone()
                dist.broadcast(tensor, src=0, group=runtime.cp_group)
                global_canonical[n] = tensor.cpu()
            global_canonical["scale"] = reference["scale"]
            canonical = {n: sliced(global_canonical[n]) for n in names}
            canonical["scale"] = reference["scale"]
            for n in names:
                assert canonical[n].dtype == own[n].dtype and canonical[n].shape == own[n].shape
            full_upstream = global_canonical["upstream"]
            local_upstream = sliced(full_upstream)

            def report(label, left, right):
                assert left.shape == right.shape, (label, left.shape, right.shape)
                left, right = left.float(), right.float()
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                pair = metrics({label: left}, {label: right}).aggregate
                print(f"STAGE-4.4 FA-REPLAY r={rank} L{layer:02d} S2 {label}: "
                      f"exact={torch.equal(left,right)} norm={pair.reference_norm:.4e}/{pair.actual_norm:.4e} "
                      f"rel={pair.relative_l2:.5e} cos={pair.cosine:.9f} "
                      f"max={(right-left).abs().max().item():.5e}", flush=True)

            print(f"STAGE-4.4 FA-REPLAY r={rank} L{layer:02d}: QKV post-RoPE, "
                  f"prefix=128 current=128 local=64 scale={reference['scale']} "
                  f"dtype={reference['q'].dtype}; no projections/output-gate; canonical inputs/dO=rank0 CP1 captured", flush=True)
            for n in names:
                report("captured-input/"+n, canonical[n], own[n])
            report("captured-output", sliced(reference["output"]), own["output"])
            report("captured-upstream/not-used-for-replay", local_upstream, own["upstream"])
            # Identical explicit dO for all replays: own-input variants vary
            # only Q/K/V, not upstream gradient. Full CP1 covers all queries.
            full_own = run(reference, full_upstream, 1)
            full = run(global_canonical, full_upstream, 1)
            full_repeat = run(global_canonical, full_upstream, 1)
            cp_own = run(own, local_upstream, 2)
            cp_canonical = run(canonical, local_upstream, 2)
            cp_repeat = run(canonical, local_upstream, 2)
            report("own-input-reproduce/CP1", reference["output"], full_own["output"])
            report("own-input-reproduce/CP2", own["output"], cp_own["output"])
            for n in ("output", *names):
                report("canonical/"+n, sliced(full[n]), cp_canonical[n])
                report("CP1-repeat/"+n, full[n], full_repeat[n])
                report("CP2-repeat/"+n, cp_canonical[n], cp_repeat[n])
                report("CP2-canonical-vs-own/"+n, cp_canonical[n], cp_own[n])
