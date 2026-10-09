"""CPU-only regression for an Ascend 507035 weak-E2E sampling bug."""
from __future__ import annotations

import pytest
import torch

from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices


@pytest.mark.parametrize(
    "numel",
    [
        1, 2, 3, 127, 511, 512, 513, 1024,
        (1 << 24), (1 << 24) + 1,
        151936 * 1024,
        151936 * 2048,  # Qwen vocab x hidden: FP32 rounds numel-1 up to numel
        (1 << 32) + 4096,
    ],
)
def test_exact_indices_valid_across_fp32_precision_boundary(numel):
    idx = exact_sample_indices(numel)
    assert len(idx) == min(numel, 512)
    assert idx[0] == 0
    assert idx[-1] == numel - 1 if len(idx) > 1 else idx[-1] == 0
    assert len(set(idx)) == len(idx)
    assert all(0 <= i < numel for i in idx)
    assert list(idx) == sorted(idx)


def test_large_embedding_float32_endpoint_would_be_out_of_bounds():
    numel = 151936 * 2048
    # An FP32 linspace silently rounds the endpoint (numel-1) to numel.
    rounded_endpoint = int(torch.tensor(float(numel - 1), dtype=torch.float32))
    assert rounded_endpoint == numel
    assert exact_sample_indices(numel)[-1] == numel - 1


@pytest.mark.parametrize("max_samples", [1, 2, 3, 16, 512, 1024])
def test_sample_counts_and_integer_endpoints(max_samples):
    idx = exact_sample_indices(100000003, max_samples)
    assert len(idx) == max_samples
    assert idx[0] == 0
    assert idx[-1] == 0 if max_samples == 1 else idx[-1] == 100000002


@pytest.mark.parametrize("numel", [0, -1, None, 0.5, True])
def test_reject_invalid_sizes(numel):
    with pytest.raises(ValueError):
        exact_sample_indices(numel)


@pytest.mark.parametrize("count", [0, -10, None, False, 1.5])
def test_reject_invalid_counts(count):
    with pytest.raises(ValueError):
        exact_sample_indices(1024, count)
