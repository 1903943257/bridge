#!/usr/bin/env python3
"""Safely add the Phase-4 PPO dispatch to an already CE-patched VERL checkout.

Unlike a unified diff, this script uses semantic anchors inside
MegatronEngine.forward_backward_batch and does not depend on upstream line
numbers or the existence of the optional routed_num_tokens branch.

Usage:
  python /path/to/bridge/patches/apply_tpr_phase4.py --check \\
    /workspace/uni-agent/verl/verl/workers/engine/megatron/transformer_impl.py
  python /path/to/bridge/patches/apply_tpr_phase4.py \\
    /workspace/uni-agent/verl/verl/workers/engine/megatron/transformer_impl.py

No git operations, no changes to config, and no changes to the old CE
tpr_forward_backward_request path. Writes are atomic and preserve mode bits.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import os
from pathlib import Path
import re
import tempfile


DISPATCH = (
    "        # TPR PPO: keep all logical trajectories together for Forest execution.\n"
    "        # Native batch_num_tokens / dp_size have already been attached.\n"
    "        # Must run before native prepare_micro_batches or DCP scheduling.\n"
    "        if self.engine_config.tpr_enabled and not forward_only:\n"
    "            from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch\n"
    "\n"
    "            return run_tpr_forward_backward_batch(\n"
    "                self, data, loss_function, forward_only=forward_only\n"
    "            )\n"
    "\n"
)


class PatchError(RuntimeError):
    pass


def _method_range(source: str) -> tuple[int, int]:
    """Identify only the top-level MegatronEngine.forward_backward_batch body."""
    tree = ast.parse(source)
    candidates = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine":
            candidates.extend(
                f for f in node.body
                if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                and f.name == "forward_backward_batch"
            )
    if len(candidates) != 1:
        raise PatchError(
            "Expected exactly one MegatronEngine.forward_backward_batch; "
            f"found {len(candidates)}. Leave local changes untouched."
        )
    method = candidates[0]
    if not method.body:
        raise PatchError("forward_backward_batch has an empty body")
    lines = source.splitlines(keepends=True)
    # The function body starts after the potentially multi-line function header.
    return method.body[0].lineno - 1, method.end_lineno


def transform(source: str) -> tuple[str, str]:
    """Return (patched_source, status), refusing ambiguous or unsafe sites."""
    start, end = _method_range(source)
    lines = source.splitlines(keepends=True)
    body = "".join(lines[start:end])

    if "run_tpr_forward_backward_batch" in body:
        return source, "already-patched"

    if "prepare_micro_batches(" not in body:
        raise PatchError("Could not find native prepare_micro_batches in method")
    # The native normalization metadata is required by the PPO objective.
    dp_matches = [
        i for i in range(start, end)
        if re.search(
            r"tu\.assign_non_tensor\s*\(\s*data\s*,\s*dp_size\s*=",
            lines[i],
        )
    ]
    token_matches = [
        i for i in range(start, end)
        if re.search(
            r"tu\.assign_non_tensor\s*\(\s*data\s*,\s*batch_num_tokens\s*=",
            lines[i],
        )
    ]
    if len(dp_matches) != 1 or len(token_matches) != 1:
        raise PatchError(
            "Expected one native batch_num_tokens and one dp_size assignment "
            "in this version of forward_backward_batch. No changes made."
        )
    metadata_end = max(dp_matches[0], token_matches[0])

    # Prefer the start of native micro-batch preparation, after optional routed
    # token metadata. A few upstream versions omit the BSHD prelude; use
    # narrower fallbacks only when their indentation and relative order match.
    anchors = (
        r"^        # BSHD path only:",
        r"^        pad_bshd_to_minibatch_max\s*=",
        r"^        vpp_size\s*=",
        r"^        dcp_group\s*=",
    )
    insertion = None
    for pattern in anchors:
        matches = [
            i for i in range(metadata_end + 1, end)
            if re.search(pattern, lines[i])
        ]
        if len(matches) == 1:
            insertion = matches[0]
            break
    if insertion is None:
        raise PatchError(
            "No unique native micro-batch preparation anchor found. "
            "Do not guess or force the patch; inspect forward_backward_batch."
        )
    if not metadata_end < insertion:
        raise PatchError("Dispatch would run before global batch metadata")
    if "prepare_micro_batches(" not in "".join(lines[insertion:end]):
        raise PatchError("Candidate insertion is after native micro-batch splitting")

    # The patch is intentionally restricted to the current method.
    updated = "".join(lines[:insertion]) + DISPATCH + "".join(lines[insertion:])
    ast.parse(updated)
    return updated, "patched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transformer_impl",
        type=Path,
        help="The existing local VERL transformer_impl.py (never a repo directory)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Validate and show a unified diff without modifying any file",
    )
    args = parser.parse_args()
    path = args.transformer_impl.expanduser().resolve()
    if not path.is_file():
        parser.error(f"Not a file: {path}")
    source = path.read_text(encoding="utf-8")
    try:
        patched, status = transform(source)
    except (PatchError, SyntaxError) as exc:
        parser.error(str(exc))
    print(f"Phase-4 dispatch: {status}: {path}")
    if status == "already-patched":
        return 0

    diff = "".join(difflib.unified_diff(
        source.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=str(path) + " (before)",
        tofile=str(path) + " (after)",
    ))
    print(diff, end="")
    if args.check:
        print("CHECK ONLY: no file modified")
        return 0

    mode = path.stat().st_mode
    fd, temp_name = tempfile.mkstemp(
        prefix=".tpr-phase4-", suffix=".py", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as writer:
            writer.write(patched)
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print("APPLIED: previous CE entry and other code left unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
