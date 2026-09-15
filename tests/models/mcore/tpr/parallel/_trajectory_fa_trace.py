"""L4-only kernel/dispatch audit. Records metadata, never retains tensors."""

from contextlib import contextmanager
import importlib


def check_whole_trace(records, backend, rank):
    if backend == "native_ring":
        remote = ((32, 1, 2048), (64, 1, 512)) if rank == 0 else (
            (64, 1, 2048), (32, 1, 512))
        expected = [((64, 1, 2048), (64, 1, 512), "SBH", 3), (*remote, "SBH", 0)]
        assert records["native_entry"] == 1, records
        assert records["fwd"] == expected, (
            "L4 native dispatch/schedule differs from single-layer native_ring", records, expected)
        assert records["bwd"] == list(reversed(expected)), records
    elif backend == "ring":
        assert records["native_entry"] == 0, records
        assert len(records["fwd"]) == len(records["bwd"]) == 5, records
        assert all(item[2] == "TND" for item in records["fwd"] + records["bwd"]), records


@contextmanager
def trace_l4(model, patch, *, backend, rank, enabled):
    records = {"fwd": [], "bwd": [], "native_entry": 0}
    if not enabled:
        yield records
        return
    import torch_npu
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    native = importlib.import_module(
        "mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel")
    layer = next(layer for layer in model.decoder.layers if layer.layer_number == 4)
    active = [False]
    original_layer = layer.self_attention.forward

    def layer_forward(*args, **kwargs):
        previous = active[0]
        active[0] = True
        try:
            return original_layer(*args, **kwargs)
        finally:
            active[0] = previous

    # Native/custom autograd backward executes outside module.forward. Associate
    # its ctx with L4 when forward runs, then enable trace only for that ctx.
    for cls in (native.AttentionWithCp, ring._RingTPRAttention):
        contexts = set()
        forward, backward = cls.forward, cls.backward

        def tagged_forward(ctx, *args, original=forward, contexts=contexts, **kwargs):
            if active[0]:
                contexts.add(id(ctx))
            return original(ctx, *args, **kwargs)

        def tagged_backward(ctx, *args, original=backward, contexts=contexts, **kwargs):
            previous = active[0]
            active[0] = id(ctx) in contexts
            try:
                return original(ctx, *args, **kwargs)
            finally:
                active[0] = previous

        patch.setattr(cls, "forward", staticmethod(tagged_forward))
        patch.setattr(cls, "backward", staticmethod(tagged_backward))

    for name, key, layout_index in (("npu_fusion_attention", "fwd", 4),
                                    ("npu_fusion_attention_grad", "bwd", 5)):
        original = getattr(torch_npu, name)

        def kernel(*args, original=original, key=key, layout_index=layout_index, **kwargs):
            if active[0]:
                records[key].append((tuple(args[0].shape), tuple(args[1].shape),
                                     args[layout_index], kwargs.get("sparse_mode", 0)))
            return original(*args, **kwargs)

        patch.setattr(torch_npu, name, kernel)

    original_entry = ring.ordinary_ring_cp_attention

    def entry(*args, **kwargs):
        if active[0]:
            records["native_entry"] += 1
        return original_entry(*args, **kwargs)

    patch.setattr(ring, "ordinary_ring_cp_attention", entry)
    patch.setattr(layer.self_attention, "forward", layer_forward)
    try:
        yield records
    finally:
        print(f"TRAJECTORY-L4 r={rank} backend={backend} native_entry={records['native_entry']} "
              f"fwd={records['fwd']} bwd={records['bwd']}", flush=True)
    check_whole_trace(records, backend, rank)
