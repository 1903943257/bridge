"""Dependency-free reporting/configuration for the AllGather reuse benchmark."""

from statistics import median


def benchmark_cases(branches="2,4,8,16", lengths="8192:1024,8192:8192"):
    ns = tuple(int(n) for n in branches.split(","))
    pairs = tuple(tuple(int(v) for v in pair.split(":")) for pair in lengths.split(","))
    if not ns or any(n < 2 for n in ns) or len(set(ns)) != len(ns):
        raise ValueError("branch counts must be unique integers >= 2")
    if not pairs or any(len(pair) != 2 or any(v <= 0 or v % 64 for v in pair) for pair in pairs):
        raise ValueError("P:S lengths must be positive multiples of 64")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate P:S cases")
    return tuple((n, p, s) for p, s in pairs for n in ns)


def summarize_case(n, p, s, ref, tpr):
    """Inputs are rank-max samples; medians of components need not add up."""
    result = {"N": n, "P": p, "S": s}
    for label, rows in (("ref", ref), ("tpr", tpr)):
        if len(rows) < 3 or any(row.keys() != rows[0].keys() for row in rows):
            raise ValueError("need >= 3 samples with identical fields")
        result[label] = {key: median(row[key] for row in rows) for key in rows[0]}
    result["ideal_token_speedup"] = n * (p + s) / (p + n * s)
    # Illustration only: F:B=1:2, Push(P)+recompute(P)+backward(P).
    result["recompute_adjusted_token_speedup_F1_B2"] = 3 * n * (p + s) / (4 * p + 3 * n * s)
    result["measured_speedup"] = result["ref"]["total_ms"] / result["tpr"]["total_ms"]
    result["ideal_time_saving_fraction"] = 1 - 1 / result["ideal_token_speedup"]
    result["actual_time_saving_fraction"] = 1 - 1 / result["measured_speedup"]
    result["saving_realization_fraction"] = (
        result["actual_time_saving_fraction"] / result["ideal_time_saving_fraction"]
    )
    result["loss_relative_diff"] = abs(result["tpr"]["loss"] - result["ref"]["loss"]) / max(
        abs(result["ref"]["loss"]), 1e-12
    )
    return result
