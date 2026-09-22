"""Test-only per-parameter OFF-repeat noise budgets (no runtime policy changes)."""

import math


NOISE_FACTOR = 1.5
NOISE_FLOORS = {"relative_l2": 2e-3, "max_abs": 2e-4, "mismatch_fraction": 1e-3}


def parameter_gate(metrics, baseline):
    """Return pass, limits, severity; invalid baseline is never calibrated away."""
    valid = lambda row: row["finite"] and all(
        math.isfinite(row[key]) and row[key] >= 0 for key in NOISE_FLOORS)
    if not valid(metrics) or (baseline is not None and not valid(baseline)):
        return False, {}, float("inf")
    if baseline is None:
        # OFF repeat measures noise, not a pointwise parameter-gradient gate.
        return True, {}, 0.0
    limits = {key: max(NOISE_FACTOR * baseline[key], baseline[key] + floor)
              for key, floor in NOISE_FLOORS.items()}
    severity = max(metrics[key] / limits[key] for key in limits)
    return severity <= 1.0, limits, severity
