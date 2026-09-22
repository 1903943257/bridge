"""Test-only per-parameter OFF-repeat noise budgets (no runtime policy changes)."""

import math


NOISE_FACTOR = 1.5
NOISE_FLOORS = {"relative_l2": 2e-3, "max_abs": 2e-4}


def parameter_gate(metrics, baseline):
    """Baseline-aware parameter gate.

    mismatch_fraction is diagnostic only: on tiny tensors, one BF16 outlier can
    make the fraction look enormous (1/128 == 0.0078125) even when the absolute
    and aggregate errors remain small. Hard acceptance therefore uses finite,
    relative-L2 and max-absolute error only.
    """
    valid = lambda row: row["finite"] and all(
        math.isfinite(row[key]) and row[key] >= 0 for key in NOISE_FLOORS)
    if not valid(metrics) or (baseline is not None and not valid(baseline)):
        return False, {}, float("inf")
    if baseline is None:
        # OFF repeat calibrates native Ring/NPU backward noise.
        return True, {}, 0.0
    limits = {
        key: max(NOISE_FACTOR * baseline[key], baseline[key] + floor)
        for key, floor in NOISE_FLOORS.items()
    }
    severity = max(metrics[key] / limits[key] for key in limits)
    return severity <= 1.0, limits, severity
