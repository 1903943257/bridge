#!/usr/bin/env python3
"""Summarize Phase-B activation-offload capacity/performance logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LOG_RE = re.compile(r"cp(?P<cp>\d+)_p(?P<p>\d+)_s(?P<s>\d+)_off(?P<off>[01])\.log$")


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
    return None if value is None else value / (1024 ** 3)


def _median(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else statistics.median(values)


def _maximum(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else max(values)


def parse_log(path: Path, warmup: int, repeats: int):
    match = LOG_RE.match(path.name)
    if not match:
        return None
    meta = {key: int(value) for key, value in match.groupdict().items()}
    text = path.read_text(errors="replace")

    max_rows = [
        item.get("max_rank", {})
        for item in _json_records(text, "TPR_OFFLOAD_B_MAX ")
        if isinstance(item, dict)
    ]
    measured = [
        row for row in max_rows
        if int(row.get("iteration", -1)) >= warmup
    ]
    failures = list(_json_records(text, "TPR_OFFLOAD_B_FAILURE "))
    oom = any(bool(item.get("oom")) for item in failures if isinstance(item, dict))

    if oom:
        status = "OOM"
    elif len(measured) >= repeats:
        status = "OK"
    elif measured:
        status = f"INCOMPLETE({len(measured)}/{repeats})"
    else:
        status = "FAILED"

    return {
        **meta,
        "path": str(path),
        "status": status,
        "samples": len(measured),
        "latency_s": _median(measured, "latency_s"),
        # Memory is a capacity metric: report the worst measured iteration.
        "peak_allocated": _maximum(measured, "peak_allocated"),
        "incremental_peak": _maximum(measured, "incremental_peak"),
        # Transfer counts should be stable; median avoids a stray instrumentation sample.
        "d2h_bytes": _median(measured, "d2h_bytes"),
        "released_bytes": _median(measured, "released_bytes"),
        "h2d_bytes": _median(measured, "h2d_bytes"),
    }


def _pct(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return 100.0 * numerator / denominator


def compare(off, on):
    row = {
        "cp": (off or on)["cp"],
        "prefix": (off or on)["p"],
        "suffix": (off or on)["s"],
        "off_status": off["status"] if off else "MISSING",
        "on_status": on["status"] if on else "MISSING",
        "off_latency_ms": None if not off or off["latency_s"] is None else off["latency_s"] * 1000,
        "on_latency_ms": None if not on or on["latency_s"] is None else on["latency_s"] * 1000,
        "off_peak_gib": None if not off else _gib(off["peak_allocated"]),
        "on_peak_gib": None if not on else _gib(on["peak_allocated"]),
        "off_incremental_gib": None if not off else _gib(off["incremental_peak"]),
        "on_incremental_gib": None if not on else _gib(on["incremental_peak"]),
        "on_released_gib": None if not on else _gib(on["released_bytes"]),
    }
    if off and on and off["status"] == "OK" and on["status"] == "OK":
        row["latency_overhead_pct"] = _pct(
            on["latency_s"] - off["latency_s"], off["latency_s"]
        )
        row["memory_saved_gib"] = _gib(off["peak_allocated"] - on["peak_allocated"])
        row["memory_saved_pct"] = _pct(
            off["peak_allocated"] - on["peak_allocated"], off["peak_allocated"]
        )
    else:
        row["latency_overhead_pct"] = None
        row["memory_saved_gib"] = None
        row["memory_saved_pct"] = None
    return row


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def print_markdown(rows):
    headers = [
        "CP", "P", "S", "OFF", "ON",
        "OFF latency ms", "ON latency ms", "Δ latency %",
        "OFF peak GiB", "ON peak GiB", "Saved GiB", "Saved %",
        "ON released GiB",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = [
            row["cp"], row["prefix"], row["suffix"],
            row["off_status"], row["on_status"],
            fmt(row["off_latency_ms"]), fmt(row["on_latency_ms"]),
            fmt(row["latency_overhead_pct"]),
            fmt(row["off_peak_gib"]), fmt(row["on_peak_gib"]),
            fmt(row["memory_saved_gib"]), fmt(row["memory_saved_pct"]),
            fmt(row["on_released_gib"]),
        ]
        print("| " + " | ".join(map(str, values)) + " |")


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir", nargs="?",
        default="tests/models/mcore/tpr/logs/phase_b_perf",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for path in sorted(Path(args.log_dir).glob("cp*_p*_s*_off*.log")):
        record = parse_log(path, args.warmup, args.repeats)
        if record is not None:
            records.append(record)

    grouped = {}
    for record in records:
        key = (record["cp"], record["p"], record["s"])
        grouped.setdefault(key, {})[record["off"]] = record

    rows = [
        compare(pair.get(0), pair.get(1))
        for _, pair in sorted(grouped.items())
    ]
    if not rows:
        raise SystemExit(f"No Phase-B logs found under {args.log_dir}")

    print_markdown(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
