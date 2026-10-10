"""CPU-only parity for the AReaL-style forward scheduler and KV buffers.

The tiny cache-aware model is deterministic; these tests validate token
alignment and traversal semantics, not actual BF16 NPU numerical parity.
"""
from types import SimpleNamespace

import torch

from ..correctness._qwen17_areal_dta_reference import (
    _areal_forward_order, areal_forward_plan, hf_areal_forward_only,
)
from ..correctness._qwen17_dta_style_reference import hf_full_logprobs


class FakeDynamicCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, layer_idx):
        if layer_idx == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        else:
            layer = self.layers[layer_idx]
            layer.keys = torch.cat((layer.keys, keys), dim=-2)
            layer.values = torch.cat((layer.values, values), dim=-2)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            num_hidden_layers=2, num_key_value_heads=1,
            num_attention_heads=1, hidden_size=1,
        )
        self.calls = []

    def forward(self, *, input_ids, past_key_values, use_cache):
        assert use_cache and input_ids.shape[0] == 1
        layers = past_key_values.layers
        prefix = (
            layers[0].keys[0, 0, :, 0].long()
            if layers else input_ids.new_empty(0)
        )
        current = input_ids.flatten()
        self.calls.append((len(prefix), current.numel()))
        all_ids = torch.cat((prefix, current))
        sums = all_ids.float().cumsum(0)[len(prefix):]
        logits = torch.stack(
            (sums/7, -sums/9, sums/13, sums*0+0.1, -sums/16,
             sums/17, sums/23, sums/29, -sums/31, sums*0-0.1),
            dim=-1,
        ).unsqueeze(0)
        for i in range(2):
            k = current.float().reshape(1, 1, -1, 1)
            past_key_values.update(k, k+1, i)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def row(tokens):
    return torch.tensor(tokens, dtype=torch.long)


def assert_full_parity(sequences):
    model = FakeModel().eval()
    full = tuple(
        hf_full_logprobs(model, seq, FakeDynamicCache) for seq in sequences
    )
    model.calls.clear()
    result = hf_areal_forward_only(model, sequences, FakeDynamicCache)
    for a, b in zip(result.logprobs, full, strict=True):
        assert torch.equal(a, b)
    assert tuple(m for _, m in model.calls) == result.physical_m
    assert result.total_processed_tokens == sum(result.physical_m)
    assert all(
        len(owner) == len(seq) - 1
        for owner, seq in zip(result.token_owner, sequences)
    )
    assert all(
        0 <= visit < len(result.physical_m)
        for row_owners in result.token_owner for visit in row_owners
    )
    return result


def test_compressed_forward_order():
    assert _areal_forward_order([5], []) == [0]
    # Explicit upstream CompressedTrie forward-priority order for a depth-skewed tree.
    assert _areal_forward_order([5, 4, 3, 2], [3, 2, 1]) == [3, 2, 1, 0]
    assert_full_parity((row([1, 2, 3, 4, 5]),))


def test_leafization_nested_prefix_and_duplicates():
    sequences = (
        row([1, 2, 3, 7, 8]), row([1, 2, 3, 4, 5]),
        row([1, 2, 9, 0]), row([1, 2, 3, 4, 5]),
        row([1, 2, 3]),
    )
    plan = areal_forward_plan(sequences)
    assert len(plan.sequences) == 3
    assert sorted(i for group in plan.attachments for i, _ in group) == list(range(5))
    result = assert_full_parity(sequences)
    assert result.n_leaves == 3
    assert result.total_processed_tokens < result.dense_tokens


def test_disjoint_rows_and_skewed_branches():
    assert_full_parity((row([8, 2, 1]), row([1, 2, 3]), row([4, 2, 3])))
    sequences = (
        row([1, 2, 3, 1]), row([1, 2, 3, 2, 4, 5, 6]),
        row([1, 2, 8, 4, 5, 6]), row([9, 3, 1]),
    )
    result = assert_full_parity(sequences)
    assert result.n_leaves == len(sequences)


def test_fork_logits_and_kv_buffer_reuse():
    sequences = (
        row([1, 2, 3, 4, 5, 6]), row([1, 2, 3, 7, 8]),
        row([1, 2, 4, 5, 6, 7]), row([1, 2, 3, 4, 5, 6]),
    )
    result = assert_full_parity(sequences)
    assert result.total_processed_tokens < result.dense_tokens
    assert result.logprobs[0].shape == (5,)


def test_lexical_fixed_kv_ablation_has_identical_toy_outputs():
    sequences = (
        row([1, 2, 3, 4, 5]), row([1, 2, 3, 6, 7]),
        row([1, 2, 8, 7, 6]), row([9, 2, 1, 0]),
    )
    model = FakeModel().eval()
    dense = tuple(hf_full_logprobs(model, seq, FakeDynamicCache) for seq in sequences)
    lexical = hf_areal_forward_only(
        model, sequences, FakeDynamicCache, forward_permute=False
    )
    optimized = hf_areal_forward_only(
        model, sequences, FakeDynamicCache, forward_permute=True
    )
    for baseline, a, b in zip(dense, lexical.logprobs, optimized.logprobs, strict=True):
        assert torch.equal(baseline, a)
        assert torch.equal(baseline, b)
    assert lexical.total_processed_tokens <= lexical.dense_tokens
