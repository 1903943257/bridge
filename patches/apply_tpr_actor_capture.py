"""Idempotent, opt-in P0 real actor-update capture integration for VERL.

Patches ONLY the call inside TrainingWorker.train_mini_batch:
    actor_output = self.train_batch(mini_batch_td)

A capture is made immediately before the actual training step, using
TPR_CAPTURE_ACTOR_MINIBATCH_DIR. This does NOT change optimizer behavior
or create fake PPO advantages/old_log_probs.

Examples (no git pull):
    python patches/apply_tpr_actor_capture.py --check \\
        /workspace/uni-agent/verl/verl/workers/engine_workers.py
    python patches/apply_tpr_actor_capture.py \\
        /workspace/uni-agent/verl/verl/workers/engine_workers.py
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


_MARKER = "# TPR P0 real actor-update capture (opt-in; before train_batch)"
_MATCH = re.compile(r"(?m)^(?P<indent>[ \t]+)actor_output = self\.train_batch\(mini_batch_td\)[ \t]*$")


def transform(source: str) -> tuple[str, str]:
    if _MARKER in source:
        return source, "already_applied"
    matches = list(_MATCH.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one actor_output = self.train_batch(mini_batch_td) "
            f"in current VERL TrainingWorker.train_mini_batch; got {len(matches)}. "
            "Inspect the exact worker source instead of forcing a patch."
        )
    matched = matches[0]
    indent = matched.group("indent")
    # Body is entirely behind the env variable. The helper refuses to
    # overwrite existing captures and preserves real data identity.
    extra = "\n".join([
        f"{indent}{_MARKER}",
        f"{indent}import os as _tpr_actor_capture_os",
        f'{indent}_tpr_actor_dir = _tpr_actor_capture_os.environ.get("TPR_CAPTURE_ACTOR_MINIBATCH_DIR")',
        f"{indent}if _tpr_actor_dir:",
        f"{indent}    from verl.models.mcore.tpr.actor_capture import capture_real_actor_update_minibatch",
        f"{indent}    capture_real_actor_update_minibatch(",
        f"{indent}        mini_batch_td, _tpr_actor_dir, batch_idx=batch_idx",
        f"{indent}    )",
    ]) + "\n"
    result = source[: matched.start()] + extra + source[matched.start() :]
    ast.parse(result)
    return result, "patched"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", type=Path, help="Actual VERL engine_workers.py")
    ap.add_argument("--check", action="store_true", help="Check without writing")
    args = ap.parse_args()
    target: Path = args.target
    original = target.read_text(encoding="utf-8")
    changed, status = transform(original)
    print(f"P0 actor capture patch status={status} file={target}")
    if args.check or status == "already_applied":
        return 0
    # Atomic replacement; do not overwrite an unrelated process's updates.
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix=".tpr-actor-", suffix=".py", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(changed)
        if target.read_text(encoding="utf-8") != original:
            raise RuntimeError("Target changed during patch; refusing to overwrite")
        os.chmod(tmp, target.stat().st_mode)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
