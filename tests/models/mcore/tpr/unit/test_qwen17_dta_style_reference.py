"""CPU-only shape/shift/LCP controls for independent HF-DTA forward oracle.

A tiny fake HF DynamicCache model makes this test independent of an NPU,
transformers and a 1.7B checkpoint. This does NOT test actual HF NPU kernels.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ..correctness._qwen17_dta_style_reference import (
    error_summary,
    hf_cached_chunk_logprobs,
    hf_dta_lcp_forward,
    hf_full_logprobs,
    longest_common_prefix,
)


class FakeDynamicCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, idx):
        if idx == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        elif idx < len(self.layers):
            layer = self.layers[idx]
            layer.keys = torch.cat((layer.keys, keys), dim=-2)
            layer.values = torch.cat((layer.values, values), dim=-2)
        else:
            raise AssertionError("KV layer index skipped")


class TinyCausalCacheModel:
    """A deterministic position-sensitive toy model with the HF KV API."""

    def __call__(self, *, input_ids, past_key_values, use_cache):
        assert use_cache is True and input_ids.shape[0] == 1
        if past_key_values.layers:
            prev = past_key_values.layers[0].keys[0, 0, :, 0].long()
        else:
            prev = input_ids.new_empty((0,))
        all_ids = torch.cat((prev, input_ids.flatten()))
        sums = all_ids.float().cumsum(0)[len(prev):]
        # Logits at position t predict token t+1. Values depend on
        # all preceding tokens, so incorrect cache use is observable.
        logits = torch.stack(
            (sums / 7, -sums / 9, sums / 13, sums * 0 + 0.1,
             -sums / 16, sums / 17, sums / 23, sums / 29,
             -sums / 31, sums * 0 - 0.1),
            dim=-1,
        ).unsqueeze(0)
        keys = input_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(keys, keys + 1, 0)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _row(tokens):
    return torch.tensor(tokens, dtype=torch.long)


def test_lcp_semantics_and_invalid_shapes():
    assert longest_common_prefix(_row([1, 2, 3]), _row([1, 2, 4])) == 2
    assert longest_common_prefix(_row([1, 2]), _row([1, 2, 3])) == 2
    assert longest_common_prefix(_row([5]), _row([2])) == 0
    with pytest.raises(ValueError):
        longest_common_prefix(torch.zeros(2, 1), torch.zeros(2, 1))


def test_repeated_cache_chunks_match_full_with_boundary_labels():
    model = TinyCausalCacheModel()
    tokens = _row([1, 2, 3, 4, 5, 6, 7, 8, 9])
    dense = hf_full_logprobs(model, tokens, FakeDynamicCache)
    for boundaries in (
        (0, 9),
        (0, 4, 9),
        (0, 3, 4, 6, 7, 9),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ):
        cached = hf_cached_chunk_logprobs(
            model, tokens, boundaries, FakeDynamicCache
        )
        assert torch.equal(cached, dense), boundaries


def test_lcp_dfs_with_duplicates_and_nested_prefixes():
    model = TinyCausalCacheModel()
    sequences = (
        _row([1, 2, 3, 7, 8]),
        _row([1, 2, 3, 4, 5]),
        _row([1, 2, 9, 0]),
        _row([1, 2, 3, 4, 5]),
        _row([1, 2, 3]),
    )
    baseline = tuple(
        hf_full_logprobs(model, row, FakeDynamicCache)
        for row in sequences
    )
    result = hf_dta_lcp_forward(model, sequences, FakeDynamicCache)
    assert len(result.logprobs) == len(baseline)
    for actual, expected in zip(result.logprobs, baseline, strict=True):
        assert torch.equal(actual, expected)
    assert len(result.physical_m) < len(sequences)  # duplicate leaf reuses KV
    assert result.total_processed_tokens < result.dense_tokens
    rows, summary = error_summary(baseline, result.logprobs)
    assert len(rows) == len(sequences)
    assert summary["max_abs"] == 0
    assert summary["num_gt_0p2"] == 0


def test_lcp_dfs_also_works_without_any_shared_prefix():
    model = TinyCausalCacheModel()
    rows = (_row([8, 2, 1]), _row([1, 2, 3]), _row([4, 2, 3]))
    baseline = tuple(hf_full_logprobs(model, x, FakeDynamicCache) for x in rows)
    got = hf_dta_lcp_forward(model, rows, FakeDynamicCache)
    assert got.physical_starts == (0, 0, 0)
    assert got.total_processed_tokens == got.dense_tokens
    assert all(torch.equal(a, b) for a, b in zip(baseline, got.logprobs))


def test_rejects_bad_boundaries_and_output_shapes():
    model = TinyCausalCacheModel()
    tokens = _row([1, 2, 3])
    with pytest.raises(ValueError, match="cover"):
        hf_cached_chunk_logprobs(model, tokens, (0, 2), FakeDynamicCache)
    with pytest.raises(ValueError, match="increase"):
        hf_cached_chunk_logprobs(model, tokens, (0, 2, 2, 3), FakeDynamicCache)
    with pytest.raises(AssertionError, match="shape mismatch"):
        error_summary((torch.zeros(2),), (torch.zeros(3),))
