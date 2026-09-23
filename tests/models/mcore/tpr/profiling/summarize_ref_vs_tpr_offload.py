#!/usr/bin/env python3
"""Summarize independent Reference and TPR profile torchruns."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


PATH_RESULT_MARKER = "TPR_REF_TPR_PATH_RESULT "
COMBINED_RESULT_MARKER = "TPR_REF_TPR_RESULT "
NEW_LOG_RE = re.compile(
    r"cp(?P<cp>\d+)_p(?P<p>\d+)_s(?P<s>\d+)_n(?P<n>\d+)_(?P<path>reference|tpr)\.log$"
)
OLD_LOG_RE = re.compile(
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


def _failure_status(text: str) -> str:
    lowered = text.lower()
    if (
        "out of memory" in lowered
        or "memory_allocation_failure" in lowered
        or "failed to allocate memory" in lowered
    ):
        return "OOM"
    if "timed out" in lowered or "timeout" in lowered:
        return "TIMEOUT"
    return "FAILED"


def _path_record(path: Path):
    match = NEW_LOG_RE.match(path.name)
    if not match:
        return None
    meta = match.groupdict()
    text = path.read_text(errors="replace")
    expected_path = meta["path"]
    results = [
        row for row in _json_records(text, PATH_RESULT_MARKER)
        if isinstance(row, dict) and row.get("path") == expected_path
    ]
    if results:
        record = results[-1]
        stats = record["stats"]
        return {
            "cp": int(record["cp_size"]),
            "prefix": int(record["prefix"]),
            "suffix": int(record["suffix"]),
            "siblings": int(record["siblings"]),
            "path": expected_path,
            "status": "OK",
            "median_ms": float(stats["median_ms"]),
            "mean_ms": float(stats["mean_ms"]),
            "std_ms": float(stats["std_ms"]),
            "peak_gib": _gib(stats["peak_allocated"]),
            "incremental_gib": _gib(stats["incremental_peak"]),
            "reserved_gib": _gib(stats["peak_reserved"]),
            "source": str(path),
        }
    return {
        "cp": int(meta["cp"]),
        "prefix": int(meta["p"]),
        "suffix": int(meta["s"]),
        "siblings": int(meta["n"]),
        "path": expected_path,
        "status": _failure_status(text),
        "median_ms": None,
        "mean_ms": None,
        "std_ms": None,
        "peak_gib": None,
        "incremental_gib": None,
        "reserved_gib": None,
        "source": str(path),
    }


def _combined_records(path: Path):
    if not OLD_LOG_RE.match(path.name):
        return []
    text = path.read_text(errors="replace")
    rows = []
    for record in _json_records(text, COMBINED_RESULT_MARKER):
        if not isinstance(record, dict):
            continue
        for path_name in ("reference", "tpr"):
            stats = record[path_name]
            rows.append({
                "cp": int(record["cp_size"]),
                "prefix": int(record["prefix"]),
                "suffix": int(record["suffix"]),
                "siblings": int(record["siblings"]),
                "path": path_name,
                "status": "OK",
                "median_ms": float(stats["median_ms"]),
                "mean_ms": float(stats["mean_ms"]),
                "std_ms": float(stats["std_ms"]),
                "peak_gib": _gib(stats["peak_allocated"]),
                "incremental_gib": _gib(stats["incremental_peak"]),
                "reserved_gib": _gib(stats["peak_reserved"]),
                "source": str(path),
            })
    return rows


def _combine(pair):
    ref = pair.get("reference")
    tpr = pair.get("tpr")
    any_row = ref or tpr
    row = {
        "cp": any_row["cp"],
        "prefix": any_row["prefix"],
        "suffix": any_row["suffix"],
        "siblings": any_row["siblings"],
        "ref_status": ref["status"] if ref else "MISSING",
        "tpr_status": tpr["status"] if tpr else "MISSING",
        "ref_median_ms": ref["median_ms"] if ref else None,
        "tpr_median_ms": tpr["median_ms"] if tpr else None,
        "ref_peak_gib": ref["peak_gib"] if ref else None,
        "tpr_peak_gib": tpr["peak_gib"] if tpr else None,
        "ref_incremental_gib": ref["incremental_gib"] if ref else None,
        "tpr_incremental_gib": tpr["incremental_gib"] if tpr else None,
        "ref_reserved_gib": ref["reserved_gib"] if ref else None,
        "tpr_reserved_gib": tpr["reserved_gib"] if tpr else None,
        "ref_source": ref["source"] if ref else None,
        "tpr_source": tpr["source"] if tpr else None,
    }
    if ref and tpr and ref["status"] == "OK" and tpr["status"] == "OK":
        row["speedup"] = ref["median_ms"] / tpr["median_ms"]
        row["peak_saved_gib"] = ref["peak_gib"] - tpr["peak_gib"]
        row["peak_saved_pct"] = 100.0 * row["peak_saved_gib"] / max(ref["peak_gib"], 1e-12)
        row["incremental_reduction_pct"] = (
            100.0 * (ref["incremental_gib"] - tpr["incremental_gib"])
            / max(ref["incremental_gib"], 1e-12)
        )
    else:
        row["speedup"] = None
        row["peak_saved_gib"] = None
        row["peak_saved_pct"] = None
        row["incremental_reduction_pct"] = None
    return row


def print_markdown(rows):
    headers = [
        "CP", "P", "S", "N",
        "Ref", "TPR",
        "Ref ms", "TPR ms", "Speedup",
        "Ref peak GiB", "TPR peak GiB", "Saved GiB", "Saved %",
        "Ref incr. GiB", "TPR incr. GiB", "Incr. reduction %",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(map(str, [
            row["cp"], row["prefix"], row["suffix"], row["siblings"],
            row["ref_status"], row["tpr_status"],
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
    paths = [root] if root.is_file() else sorted(root.glob("*.log"))
    records = []
    for path in paths:
        record = _path_record(path)
        if record is not None:
            records.append(record)
        else:
            records.extend(_combined_records(path))

    # Prefer new independent-path logs if both old combined and new logs exist.
    grouped = {}
    for record in records:
        key = (
            record["cp"], record["prefix"], record["suffix"], record["siblings"]
        )
        pair = grouped.setdefault(key, {})
        previous = pair.get(record["path"])
        if previous is None or NEW_LOG_RE.match(Path(record["source"]).name):
            pair[record["path"]] = record

    rows = [_combine(pair) for _, pair in sorted(grouped.items())]
    if not rows:
        raise SystemExit(f"No Ref-vs-TPR logs found under {root}")

    print_markdown(rows)
    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
