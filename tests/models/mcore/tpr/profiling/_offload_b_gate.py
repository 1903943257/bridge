"""Test-only OFF-repeat noise budgets for gradient acceptance."""

import math


NOISE_FACTOR = 1.5
REL_L2_FLOOR = 2e-3
SMALL_TENSOR_ELEMENTS = 4096
SMALL_TENSOR_REL_L2_FLOOR = 2e-2


def gradient_gate(metrics, baseline):
    """Return pass/limit/severity against calibrated OFF-repeat noise.

    Hard acceptance uses finite values and relative-L2 only. max_abs and
    mismatch_fraction remain diagnostics because isolated BF16 quantization
    outliers are not a robust correctness signal.
    """
    valid = lambda row: (
        row["finite"]
        and math.isfinite(row["relative_l2"])
        and row["relative_l2"] >= 0
    )
    if not valid(metrics) or baseline is None or not valid(baseline):
        return False, {}, float("inf")

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


def merge_baseline(previous, metrics):
    """Keep the noisier OFF-repeat sample as the per-tensor baseline."""
    if previous is None or metrics["relative_l2"] > previous["relative_l2"]:
        return metrics.copy()
    return previous
