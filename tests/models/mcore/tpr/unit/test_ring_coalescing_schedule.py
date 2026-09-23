"""Dependency-free dispatch tests; real Ring loops with shape-only FA stubs.

These verify scheduling and buffer aliasing, NOT numerical kernel correctness.
Run directly with Python when PyTorch/NPU is unavailable.
"""
import ast
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace


class Tensor:
    dtype = "bf16"
    device = "shape-only"

    def __init__(self, shape, storage=None, contiguous=True):
        self.shape = tuple(shape)
        self.storage = storage if storage is not None else object()
        self.contiguous_flag = contiguous

    def is_contiguous(self):
        return self.contiguous_flag

    def view(self, *shape):
        return Tensor(shape, self.storage)

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self.storage))

    def storage_offset(self):
        return 0

    def stride(self):
        stride = []
        value = 1
        for dim in reversed(self.shape):
            stride.append(value)
            value *= dim
        return tuple(reversed(stride))

    def new_empty(self, shape):
        return Tensor(shape)

    def copy_(self, other):
        assert self.shape == other.shape
        return self

    def squeeze(self, dim):
        return Tensor(self.shape[:dim] + self.shape[dim + 1:], self.storage, self.contiguous_flag)

    def unsqueeze(self, dim):
        return Tensor(self.shape[:dim] + (1,) + self.shape[dim:], self.storage)

    def contiguous(self):
        return self if self.is_contiguous() else Tensor(self.shape)

    def reshape(self, *shape):
        return Tensor(shape, self.storage)

    def new_zeros(self, shape):
        return Tensor(shape)

    def __getitem__(self, item):
        indices = item if isinstance(item, tuple) else (item,)
        result_shape = list(self.shape)
        for dim, index in enumerate(indices):
            assert isinstance(index, slice)
            start, stop, step = index.indices(self.shape[dim])
            result_shape[dim] = len(range(start, stop, step))
        return Tensor(result_shape, self.storage)

    def __setitem__(self, item, value):
        assert self[item].shape == value.shape

    def add_(self, other):
        assert self.shape == other.shape
        return self


def shard(length, *, cp_rank, cp_size, padded_length=None):
    padded_length = padded_length or length
    chunk = padded_length // (2 * cp_size)
    ranges = ((cp_rank * chunk, (cp_rank + 1) * chunk),
              ((2 * cp_size - cp_rank - 1) * chunk, (2 * cp_size - cp_rank) * chunk))
    return SimpleNamespace(global_length=length, padded_length=padded_length,
                           local_length=2 * chunk, physical_global_ranges=ranges,
                           global_ranges=ranges)


def load_dispatch():
    source = Path(__file__).resolve().parents[5] / "verl/models/mcore/tpr/parallel/ring_attention.py"
    wanted = {"RingBlockKind", "_RingRangeSlice", "_iter_range_slices", "_prefix_full_slices",
              "classify_ring_block", "_physical_block_attention_mask", "_RingTPRAttention",
              "_can_coalesce_prefix_query", "_slice_tnd_result", "_trace_merged_query",
              "_source_schedule"}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    preamble = ast.parse("from __future__ import annotations").body
    ns = dict(dataclass=dataclass, Enum=Enum,
              torch=SimpleNamespace(autograd=SimpleNamespace(Function=object),
                                    zeros_like=lambda t: Tensor(t.shape)),
              _RingAttentionConfig=SimpleNamespace, make_ring_sequence_shard=shard)
    exec(compile(ast.Module(body=preamble + nodes, type_ignores=[]), str(source), "exec"), ns)
    return ns


class DispatchTests(unittest.TestCase):
    def run_attention(self, cp, rank, prefix, enabled, query_enabled=False):
        ns = load_dispatch()
        events, calls = [], {"forward": 0, "backward": 0}
        query_storage = []
        communication = dict(forward=0, backward=0)
        def circulate(k, v, cfg, consume):
            communication["forward"] += 1
            for step in range(cp):
                consume((rank - step) % cp, k, v)
        def reduce(k, v, cfg, consume):
            communication["backward"] += 1
            dk, dv = Tensor(k.shape), Tensor(v.shape)
            for step in range(cp):
                consume((rank + step + 1) % cp, k, v, dk, dv)
            return dk, dv
        def forward(q, k, v, **kwargs):
            calls["forward"] += 1
            if query_enabled and enabled and prefix:
                assert q.storage is query_storage[0]
            return Tensor(q.shape), Tensor((q.shape[0], q.shape[1], 8)), Tensor((q.shape[0], q.shape[1], 8))
        def backward(q, k, v, grad, **kwargs):
            calls["backward"] += 1
            return Tensor(q.shape), Tensor(k.shape), Tensor(v.shape)
        ns.update(_trace_ring_block=lambda **event: events.append(event),
                  _observe_ring_storage=lambda *args: None,
                  _circulate_kv=circulate,
                  _block_attention_forward=forward, _block_attention_backward=backward,
                  _merge_attention=lambda previous, current, **kw: current,
                  _finalize_attention_result=lambda result, **kw: result,
                  _reduce_ring_gradients_to_owner=reduce)
        current = shard(16384, cp_rank=rank, cp_size=cp)
        config = SimpleNamespace(segment_lengths=(16384,) * (2 if prefix else 1),
                                 segment_padded_lengths=(16384,) * (2 if prefix else 1),
                                 cp_size=cp, cp_rank=rank, current_shard=current,
                                 query_heads=16, head_dim=128, softmax_scale=1,
                                 coalesce_prefix_full=enabled, coalesce_prefix_query=query_enabled)
        q = Tensor((current.local_length, 1, 16, 128))
        query_storage.append(q.storage)
        kv = [Tensor((current.local_length, 1, 8, 128)) for _ in range(2 * len(config.segment_lengths))]
        ctx = SimpleNamespace()
        ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
        output = ns["_RingTPRAttention"].forward(ctx, q, *kv, config)
        self.assertEqual(ctx.block_tensor_count, len(kv))
        self.assertEqual(ctx.saved_tensors[1:1 + len(kv)], tuple(kv))
        if ctx.query_coalesced:
            self.assertEqual(len(ctx.saved_tensors) - 1 - ctx.block_tensor_count, 3)
            self.assertIs(ctx.saved_tensors[-3].storage, output.storage)
            self.assertEqual(ctx.saved_tensors[-2].shape, (current.local_length, 16, 8))
        gradients = ns["_RingTPRAttention"].backward(ctx, Tensor(output.shape))
        assert len(gradients) == 2 + len(kv)
        assert gradients[0].shape == q.shape
        for gradient, value in zip(gradients[1:-1], kv):
            assert gradient.shape == value.shape
        assert communication == dict(forward=len(config.segment_lengths), backward=len(config.segment_lengths))
        return calls, events

    def test_dispatch_counts_all_ranks(self):
        for cp in (2, 4):
            for rank in range(cp):
                for enabled in (False, True):
                    calls, events = self.run_attention(cp, rank, True, enabled)
                    expected = 2 * cp + 1 + (2 if enabled else 4) * cp
                    self.assertEqual(calls, dict(forward=expected, backward=expected))
                    full = [e for e in events if e["phase"] == "forward" and e["segment_type"] == "prefix"]
                    self.assertEqual(len(full), (2 if enabled else 4) * cp)
                    plain, _ = self.run_attention(cp, rank, False, enabled)
                    self.assertEqual(plain, dict(forward=2 * cp + 1, backward=2 * cp + 1))
        self.assertEqual((9 + 8 * 25 + 9) * 28, 6104)
        self.assertEqual((8 * 25 + 9) * 28, 5852)
        self.assertEqual((9 + 8 * 17 + 9) * 28, 4312)
        self.assertEqual((8 * 17 + 9) * 28, 4060)

    def test_query_merge_counts_and_transport_all_ranks(self):
        for cp in (2, 4):
            for rank in range(cp):
                calls, events = self.run_attention(cp, rank, True, True, True)
                expected = 3 * cp + 1
                self.assertEqual(calls, dict(forward=expected, backward=expected))
                for phase in ("forward", "backward"):
                    prefix = [e for e in events if e["phase"] == phase and e["segment_type"] == "prefix"]
                    self.assertEqual(len(prefix), cp)
                    self.assertTrue(all(e["query_coalesced"] for e in prefix))
                plain, _ = self.run_attention(cp, rank, False, True, True)
                self.assertEqual(plain, dict(forward=2 * cp + 1, backward=2 * cp + 1))
        self.assertEqual((9 + 8 * 13 + 9) * 28, 3416)
        self.assertEqual((8 * 13 + 9) * 28, 3164)
        self.assertEqual((1 + 8 * 2 + 1) * 28, 504)
        self.assertEqual((8 * 2 + 1) * 28, 476)

    def test_query_eligibility_fallback(self):
        ns = load_dispatch()
        config = SimpleNamespace(coalesce_prefix_full=True, coalesce_prefix_query=True,
                                 segment_lengths=(128, 64), segment_padded_lengths=(128, 64))
        q = Tensor((16, 1, 4, 8))
        self.assertTrue(ns["_can_coalesce_prefix_query"](q, config))
        q.contiguous_flag = False
        self.assertFalse(ns["_can_coalesce_prefix_query"](q, config))
        q.contiguous_flag = True
        config.segment_padded_lengths = (128, 72)
        self.assertFalse(ns["_can_coalesce_prefix_query"](q, config))

    def test_buffer_alias_and_padding_fallback(self):
        ns = load_dispatch()
        tensor = Tensor((4096, 8, 128))
        full = shard(16384, cp_rank=0, cp_size=4)
        items = ns["_prefix_full_slices"](tensor, full, enabled=True)
        self.assertIs(items[0].tensor, tensor)
        self.assertEqual(items[0].local_slice, slice(0, 4096))
        padded = shard(16383, cp_rank=0, cp_size=4, padded_length=16384)
        items = ns["_prefix_full_slices"](tensor, padded, enabled=True)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[-1].valid_length, 2047)


if __name__ == "__main__":
    unittest.main()
