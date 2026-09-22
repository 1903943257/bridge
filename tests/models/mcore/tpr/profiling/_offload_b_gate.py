"""Test-only per-parameter OFF-repeat noise budgets (no runtime policy changes)."""

import math


NOISE_FACTOR = 1.5
REL_L2_FLOOR = 2e-3
SMALL_TENSOR_ELEMENTS = 4096
SMALL_TENSOR_REL_L2_FLOOR = 2e-2


def parameter_gate(metrics, baseline):
    """Baseline-aware parameter-gradient gate.

    Hard acceptance is based on finite values and relative-L2 only. max_abs and
    mismatch_fraction stay diagnostic: a single BF16 outlier can dominate those
    metrics on very large or very small tensors without indicating model-wide
    gradient corruption. Small norm/bias tensors use a wider relative-L2 floor
    because one quantization step is a large fraction of their norm.
    """
    valid = lambda row: (
        row["finite"]
        and math.isfinite(row["relative_l2"])
        and row["relative_l2"] >= 0
    )
    if not valid(metrics) or (baseline is not None and not valid(baseline)):
        return False, {}, float("inf")
    if baseline is None:
        # OFF repeat calibrates native Ring/NPU backward noise.
        return True, {}, 0.0

    floor = (
        SMALL_TENSOR_REL_L2_FLOOR
        if metrics["elements"] < SMALL_TENSOR_ELEMENTS
        else REL_L2_FLOOR
    )
    limit = max(
        NOISE_FACTOR * baseline["relative_l2"],
        baseline["relative_l2"] + floor,
    )
    severity = metrics["relative_l2"] / max(limit, 1e-30)
    return severity <= 1.0, {"relative_l2": limit}, severity
