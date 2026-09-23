#!/usr/bin/env python3
"""Collect existing Phase-B activation-offload results without OFF-vs-ON pairing.

The collector is intentionally marker-driven rather than filename-driven. It
recursively scans log/text files, extracts TPR_OFFLOAD_B_MAX records, drops
warmups, and prints one aggregate row per (file, CP, P, S, offload) run.

By default only offload=ON rows are shown because Phase B acceptance is already
complete. Pass --include-off to retain historical OFF rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MARKER = "TPR_OFFLOAD_B_MAX "
FAILURE_MARKER = "TPR_OFFLOAD_B_FAILURE "
_SUFFIXES = (".log", ".txt", ".out")


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


def _median(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else statistics.median(values)


def _maximum(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else max(values)


def _fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def collect_file(path: Path, warmup: int, include_off: bool):
    text = path.read_text(errors="replace")
    max_rows = [
        item.get("max_rank", {})
        for item in _json_records(text, MARKER)
        if isinstance(item, dict) and isinstance(item.get("max_rank"), dict)
    ]
    failures = [
        item for item in _json_records(text, FAILURE_MARKER)
        if isinstance(item, dict)
    ]
    grouped = {}
    for row in max_rows:
        required = ("cp_size", "prefix", "suffix", "offload", "iteration")
        if any(key not in row for key in required):
            continue
        offload = bool(row["offload"])
        if not include_off and not offload:
            continue
        key = (
            int(row["cp_size"]),
            int(row["prefix"]),
            int(row["suffix"]),
            offload,
        )
        grouped.setdefault(key, []).append(row)

    result = []
    for (cp, prefix, suffix, offload), rows in sorted(grouped.items()):
        measured = [
            row for row in rows
            if int(row.get("iteration", -1)) >= warmup
        ]
        relevant_failures = [
            row for row in failures
            if int(row.get("cp_size", cp)) == cp
            and int(row.get("prefix", prefix)) == prefix
            and int(row.get("suffix", suffix)) == suffix
            and bool(row.get("offload", offload)) == offload
        ]
        oom = any(
            bool(row.get("oom"))
            or "out of memory" in str(row.get("error", "")).lower()
            for row in relevant_failures
        )
        status = "OOM" if oom else ("OK" if measured else "NO_SAMPLES")
        result.append({
            "source": str(path),
            "cp": cp,
            "prefix": prefix,
            "suffix": suffix,
            "total_tokens": prefix + suffix,
            "offload": int(offload),
            "status": status,
            "samples": len(measured),
            "latency_s_median": _median(measured, "latency_s"),
            "peak_allocated_gib": _gib(_maximum(measured, "peak_allocated")),
            "incremental_peak_gib": _gib(_maximum(measured, "incremental_peak")),
            "peak_reserved_gib": _gib(_maximum(measured, "peak_reserved")),
            "d2h_gib": _gib(_median(measured, "d2h_bytes")),
            "released_gib": _gib(_median(measured, "released_bytes")),
            "h2d_gib": _gib(_median(measured, "h2d_bytes")),
            "restore_peak_gib": _gib(_maximum(measured, "restore_sample_peak_allocated")),
        })
    return result


def iter_inputs(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUFFIXES:
            yield path


def print_markdown(rows):
    headers = [
        "CP", "P", "S", "P+S", "Status", "Samples",
        "Median s", "Peak GiB", "Incr. GiB", "Reserved GiB",
        "D2H GiB", "Released GiB", "H2D GiB", "Restore peak GiB", "Source",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(map(str, [
            row["cp"], row["prefix"], row["suffix"], row["total_tokens"],
            row["status"], row["samples"],
            _fmt(row["latency_s_median"]),
            _fmt(row["peak_allocated_gib"]),
            _fmt(row["incremental_peak_gib"]),
            _fmt(row["peak_reserved_gib"]),
            _fmt(row["d2h_gib"]),
            _fmt(row["released_gib"]),
            _fmt(row["h2d_gib"]),
            _fmt(row["restore_peak_gib"]),
            row["source"],
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
        "root", nargs="?",
        default="tests/models/mcore/tpr/logs",
        help="Log file or directory to scan recursively.",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--include-off", action="store_true",
        help="Also include historical offload=0 rows.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for path in iter_inputs(Path(args.root)):
        rows.extend(collect_file(path, args.warmup, args.include_off))
    rows.sort(key=lambda row: (
        row["cp"], row["total_tokens"], row["prefix"], row["suffix"],
        row["source"],
    ))
    if not rows:
        raise SystemExit(f"No Phase-B result markers found under {args.root}")

    print_markdown(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
