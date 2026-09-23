#!/usr/bin/env python3
"""Summarize machine-readable Ref-vs-TPR profile results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RESULT_MARKER = "TPR_REF_TPR_RESULT "
STAGE_MARKER = "TPR_REF_TPR_STAGE "
LOG_RE = re.compile(
    r"cp(?P<cp>\d+)_p(?P<p>\d+)_s(?P<s>\d+)_n(?P<n>\d+)\.log$"
)


def _json_records(text: str, marker: str):
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        pos = text.find(marker, pos)
        if pos < 0:
            return
        pos += len(marker)
        try:
            record, consumed = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        yield record
        pos += consumed


def _gib(value):
    return None if value is None else float(value) / (1024 ** 3)


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def parse_log(path: Path):
    text = path.read_text(errors="replace")
    results = [
        row for row in _json_records(text, RESULT_MARKER)
        if isinstance(row, dict)
    ]
    if results:
        rows = []
        for record in results:
            ref = record["reference"]
            tpr = record["tpr"]
            rows.append({
                "cp": int(record["cp_size"]),
                "prefix": int(record["prefix"]),
                "suffix": int(record["suffix"]),
                "siblings": int(record["siblings"]),
                "status": "OK",
                "failed_stage": None,
                "offload": bool(record.get("offload")),
                "loss_chunk_size": record.get("loss_chunk_size"),
                "ref_median_ms": float(ref["median_ms"]),
                "tpr_median_ms": float(tpr["median_ms"]),
                "speedup": float(record["speedup"]),
                "ref_peak_gib": _gib(ref["peak_allocated"]),
                "tpr_peak_gib": _gib(tpr["peak_allocated"]),
                "peak_saved_gib": _gib(ref["peak_allocated"] - tpr["peak_allocated"]),
                "peak_saved_pct": (
                    100.0 * (ref["peak_allocated"] - tpr["peak_allocated"])
                    / max(float(ref["peak_allocated"]), 1.0)
                ),
                "ref_incremental_gib": _gib(ref["incremental_peak"]),
                "tpr_incremental_gib": _gib(tpr["incremental_peak"]),
                "incremental_reduction_pct": float(
                    record["incremental_peak_reduction_pct"]
                ),
                "source": str(path),
            })
        return rows

    match = LOG_RE.match(path.name)
    if not match:
        return []
    meta = {key: int(value) for key, value in match.groupdict().items()}
    stages = [
        row for row in _json_records(text, STAGE_MARKER)
        if isinstance(row, dict)
    ]
    failed_stage = stages[-1].get("path") if stages else None
    lowered = text.lower()
    status = "OOM" if "out of memory" in lowered or "memory_allocation_failure" in lowered else "FAILED"
    return [{
        "cp": meta["cp"],
        "prefix": meta["p"],
        "suffix": meta["s"],
        "siblings": meta["n"],
        "status": status,
        "failed_stage": failed_stage,
        "offload": True,
        "loss_chunk_size": None,
        "ref_median_ms": None,
        "tpr_median_ms": None,
        "speedup": None,
        "ref_peak_gib": None,
        "tpr_peak_gib": None,
        "peak_saved_gib": None,
        "peak_saved_pct": None,
        "ref_incremental_gib": None,
        "tpr_incremental_gib": None,
        "incremental_reduction_pct": None,
        "source": str(path),
    }]


def print_markdown(rows):
    headers = [
        "CP", "P", "S", "N", "Status", "Failed path",
        "Ref ms", "TPR ms", "Speedup",
        "Ref peak GiB", "TPR peak GiB", "Saved GiB", "Saved %",
        "Ref incr. GiB", "TPR incr. GiB", "Incr. reduction %",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(map(str, [
            row["cp"], row["prefix"], row["suffix"], row["siblings"],
            row["status"], row["failed_stage"] or "-",
            _fmt(row["ref_median_ms"]), _fmt(row["tpr_median_ms"]),
            _fmt(row["speedup"]),
            _fmt(row["ref_peak_gib"]), _fmt(row["tpr_peak_gib"]),
            _fmt(row["peak_saved_gib"]), _fmt(row["peak_saved_pct"], 2),
            _fmt(row["ref_incremental_gib"]), _fmt(row["tpr_incremental_gib"]),
            _fmt(row["incremental_reduction_pct"], 2),
        ])) + " |")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir", nargs="?",
        default="tests/models/mcore/tpr/logs/ref_vs_tpr_offload",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    root = Path(args.log_dir)
    paths = [root] if root.is_file() else sorted(root.glob("cp*_p*_s*_n*.log"))
    rows = []
    for path in paths:
        rows.extend(parse_log(path))
    rows.sort(key=lambda row: (
        row["cp"], row["prefix"] + row["suffix"],
        row["prefix"], row["suffix"], row["siblings"],
    ))
    if not rows:
        raise SystemExit(f"No Ref-vs-TPR logs found under {root}")

    print_markdown(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
