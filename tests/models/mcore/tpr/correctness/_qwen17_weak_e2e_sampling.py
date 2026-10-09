"""NPU-safe integer-only strided sampling for weak real-TQ optimizer checks.

IMPORTANT: torch.linspace(..., device="npu").long() defaults to float32
and can round the last valid element index *up* to numel for large
embedding matrices (>2**24 entries). Passing that invalid index to
NPU index_select can asynchronously crash an Ascend Vector Core, later
surfacing as ACL stream synchronize error 507035 at .cpu().
"""
from __future__ import annotations


def exact_sample_indices(numel: int, max_samples: int = 512) -> tuple[int, ...]:
    """Return valid integer indices spanning [0, numel - 1], no floats.

    Generates at most max_samples deterministic, monotonically increasing
    indices. Allocation is O(max_samples), independent of tensor size.
    """
    if not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0:
        raise ValueError("numel must be a positive integer")
    if (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer")
    count = min(numel, max_samples)
    if count == 1:
        return (0,)
    last = numel - 1
    values = tuple(i * last // (count - 1) for i in range(count))
    assert values[0] == 0 and values[-1] == last
    assert all(0 <= v < numel for v in values)
    assert all(a < b for a, b in zip(values, values[1:]))
    return values
