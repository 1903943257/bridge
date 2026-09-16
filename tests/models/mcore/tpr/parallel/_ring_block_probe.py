"""Untimed execution trace for the actual Ring FA dispatch loops."""

import os
from collections import Counter
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import verl.models.mcore.tpr.parallel.ring_attention as ring
from verl.models.mcore.tpr import SegmentExecutor


@contextmanager
def ring_block_probe(model, runtime):
    counts = Counter()
    scope = {"operation": None, "segment": None, "layer": None}
    original_forward = ring._RingTPRAttention.forward
    original_backward = ring._RingTPRAttention.backward
    first_leaf = []
    layout_reported = False
    detailed = os.getenv("TPR_RING_BLOCK_TRACE", "0") == "1"

    def trace(**event):
        if event["fa_called"]:
            counts[(scope["operation"], event["phase"], event["segment_type"])] += 1
        if (detailed and runtime.rank == 0 and scope["operation"] == "visit_leaf"
                and scope["layer"] == model.decoder.layers[0].self_attention.layer_number
                and event["segment_type"] == "prefix" and not first_leaf):
            first_leaf.append(scope["segment"])
        if (detailed and runtime.rank == 0 and first_leaf
                and scope["operation"] == "visit_leaf" and scope["segment"] == first_leaf[0]
                and scope["layer"] == model.decoder.layers[0].self_attention.layer_number):
            print(f"Ring block trace: {scope} {event}")

    def forward(ctx, *args):
        nonlocal layout_reported
        ctx.profile_block_scope = dict(scope)
        if (detailed and runtime.rank == 0 and not layout_reported
                and scope["operation"] == "visit_leaf"
                and scope["layer"] == model.decoder.layers[0].self_attention.layer_number
                and len(args[-1].segment_lengths) > 1):
            query, config = args[0], args[-1]
            q = query.squeeze(1)
            layout_reported = True
            print(f"Q layout: base_ptr={q.untyped_storage().data_ptr():#x}, "
                  f"shape={tuple(q.shape)}, stride={q.stride()}, offset={q.storage_offset()}, "
                  f"can_zero_copy_merge={q.is_contiguous()}, "
                  f"fast_path_eligible={ring._can_coalesce_prefix_query(query, config)}")
            for item in ring._iter_range_slices(q, config.current_shard):
                print(f"  Q chunk: logical={item.logical_range}, rows={item.local_slice}, "
                      f"base_ptr={item.tensor.untyped_storage().data_ptr():#x}, "
                      f"offset={item.tensor.storage_offset()}, stride={item.tensor.stride()}")
        return original_forward(ctx, *args)

    def backward(ctx, *args):
        previous = dict(scope)
        scope.update(ctx.profile_block_scope)
        try:
            return original_backward(ctx, *args)
        finally:
            scope.update(previous)

    def wrap_operation(name, original):
        def call(executor, segment_id):
            previous = dict(scope)
            scope.update(operation=name, segment=segment_id)
            try:
                return original(executor, segment_id)
            finally:
                scope.update(previous)
        return call

    handles = [layer.self_attention.register_forward_pre_hook(
        lambda module, _args: scope.update(layer=module.layer_number)
    ) for layer in model.decoder.layers]
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(ring, "_trace_ring_block", trace))
            stack.enter_context(patch.object(ring._RingTPRAttention, "forward", staticmethod(forward)))
            stack.enter_context(patch.object(ring._RingTPRAttention, "backward", staticmethod(backward)))
            for name in ("push", "visit_leaf", "pop"):
                stack.enter_context(patch.object(
                    SegmentExecutor, name, wrap_operation(name, getattr(SegmentExecutor, name)),
                ))
            yield counts
    finally:
        for handle in handles:
            handle.remove()
    if runtime.rank == 0:
        print("Untimed Ring FA probe (rank 0; step means receive origin, not execution order):")
        for key, count in sorted(counts.items()):
            print(f"  {key}: {count}")
        for phase in ("forward", "backward"):
            total = sum(value for key, value in counts.items() if key[1] == phase)
            print(f"  total {phase}: {total}")
