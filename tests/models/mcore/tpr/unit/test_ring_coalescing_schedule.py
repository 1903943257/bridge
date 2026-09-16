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

    def __init__(self, shape):
        self.shape = tuple(shape)

    def squeeze(self, dim):
        return Tensor(self.shape[:dim] + self.shape[dim + 1:])

    def unsqueeze(self, dim):
        return Tensor(self.shape[:dim] + (1,) + self.shape[dim:])

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return Tensor(shape)

    def new_zeros(self, shape):
        return Tensor(shape)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self.shape[0])
            return Tensor((len(range(start, stop, step)), *self.shape[1:]))
        raise AssertionError(item)

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
              "classify_ring_block", "_physical_block_attention_mask", "_RingTPRAttention"}
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
    def run_attention(self, cp, rank, prefix, enabled):
        ns = load_dispatch()
        events, calls = [], {"forward": 0, "backward": 0}
        def forward(q, k, v, **kwargs):
            calls["forward"] += 1
            return Tensor(q.shape), Tensor((1,)), Tensor((1,))
        def backward(q, k, v, grad, **kwargs):
            calls["backward"] += 1
            return Tensor(q.shape), Tensor(k.shape), Tensor(v.shape)
        ns.update(_trace_ring_block=lambda **event: events.append(event),
                  _circulate_kv=lambda k, v, cfg: tuple((k, v) for _ in range(cp)),
                  _block_attention_forward=forward, _block_attention_backward=backward,
                  _merge_attention=lambda previous, current, **kw: current,
                  _finalize_attention_result=lambda result, **kw: result,
                  _reduce_ring_gradients_to_owner=lambda contributions, cfg: contributions[rank])
        current = shard(16384, cp_rank=rank, cp_size=cp)
        config = SimpleNamespace(segment_lengths=(16384,) * (2 if prefix else 1),
                                 segment_padded_lengths=(16384,) * (2 if prefix else 1),
                                 cp_size=cp, cp_rank=rank, current_shard=current,
                                 query_heads=16, head_dim=128, softmax_scale=1,
                                 coalesce_prefix_full=enabled)
        q = Tensor((current.local_length, 1, 16, 128))
        kv = [Tensor((current.local_length, 1, 8, 128)) for _ in range(2 * len(config.segment_lengths))]
        ctx = SimpleNamespace()
        ctx.save_for_backward = lambda *args: setattr(ctx, "saved_tensors", args)
        output = ns["_RingTPRAttention"].forward(ctx, q, *kv, config)
        gradients = ns["_RingTPRAttention"].backward(ctx, Tensor(output.shape))
        assert len(gradients) == 2 + len(kv)
        assert gradients[0].shape == q.shape
        for gradient, value in zip(gradients[1:-1], kv):
            assert gradient.shape == value.shape
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
