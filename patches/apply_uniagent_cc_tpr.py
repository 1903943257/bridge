#!/usr/bin/env python3
"""Idempotently install Bridge TPR hooks into a locally modified Uni-Agent-CC VERL.

Run from any directory:
  python /workspace/bridge/patches/apply_uniagent_cc_tpr.py --check /workspace/uni-agent-cc/verl
  python /workspace/bridge/patches/apply_uniagent_cc_tpr.py /workspace/uni-agent-cc/verl

Unlike the version-pinned unified diff, this preserves unrelated worker edits.
Wiring only; TPR training support/accuracy must be validated separately.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
from pathlib import Path
import re
import tempfile

import apply_tpr_actor_capture
import apply_tpr_phase4


class PatchError(RuntimeError):
    """Fail closed when an anchor is absent, ambiguous, or partially patched."""


def _class_method(source: str, class_name: str, method_name: str):
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(matches) != 1:
        raise PatchError(f"Expected one {class_name} class; got {len(matches)}")
    methods = [node for node in matches[0].body if isinstance(node, ast.FunctionDef) and node.name == method_name]
    if len(methods) != 1:
        raise PatchError(f"Expected one {class_name}.{method_name}; got {len(methods)}")
    return matches[0], methods[0]


def patch_config(source: str) -> tuple[str, str]:
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "McoreEngineConfig"]
    if len(classes) != 1:
        raise PatchError(f"Expected one McoreEngineConfig class; got {len(classes)}")
    fields = {}
    for node in classes[0].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            fields[node.target.id] = node
    present = {name for name in ("tpr_enabled", "tpr_cp_backend") if name in fields}
    if len(present) == 2:
        if not isinstance(fields["tpr_enabled"].value, ast.Constant) or fields["tpr_enabled"].value.value is not False:
            raise PatchError("Existing tpr_enabled field has unexpected default; inspect manually")
        return source, "already-patched"
    if present:
        raise PatchError("Partial TPR config fields detected; inspect manually")
    anchor = fields.get("use_megatron_fsdp")
    if anchor is None or not isinstance(anchor.value, ast.Constant) or anchor.value.value is not False:
        raise PatchError("Cannot locate expected use_megatron_fsdp=False config field")
    if "from typing import Optional" not in source and not re.search(r"from typing import [^\n]*\bOptional\b", source):
        raise PatchError("Expected Optional to be imported in config.py")
    lines = source.splitlines(keepends=True)
    line = lines[anchor.end_lineno - 1]
    if not line.startswith("    use_megatron_fsdp:") or not line.endswith("\n"):
        raise PatchError("Unexpected config anchor indentation or newline")
    insertion = "    tpr_enabled: bool = False\n    tpr_cp_backend: Optional[str] = None\n"
    updated = "".join(lines[:anchor.end_lineno]) + insertion + "".join(lines[anchor.end_lineno:])
    ast.parse(updated)
    return updated, "patched"


def patch_model_spec(source: str) -> tuple[str, str]:
    _, method = _class_method(source, "MegatronEngine", "_build_megatron_module")
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1 : method.end_lineno])
    if "install_tpr_module_spec" in body:
        if not any(call in body for call in (
            "install_tpr_module_spec(self.provider, bridge=self.bridge)",
            "install_tpr_module_spec(self.provider)",
        )):
            raise PatchError("Partial or nonstandard TPR Module Spec hook detected")
        return source, "already-patched"
    calls = [node for node in method.body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "make_megatron_module"]
    if len(calls) != 1:
        raise PatchError(f"Expected exactly one direct make_megatron_module assignment; got {len(calls)}")
    anchor = calls[0].lineno - 1
    if lines[anchor].strip() != "module, updated_tf_config = make_megatron_module(":
        raise PatchError("Model constructor is not at expected standalone assignment")
    preceding = "".join(lines[method.lineno - 1 : anchor])
    if "override_ddp_config = self._resolve_override_ddp_config()" not in preceding:
        raise PatchError("Expected DDP configuration to be resolved before TPR module injection")
    insertion = (
        "        if self.engine_config.tpr_enabled:\n"
        "            from verl.models.mcore.tpr.megatron_adapter import install_tpr_module_spec\n"
        "\n"
        "            install_tpr_module_spec(self.provider, bridge=self.bridge)\n"
        "\n"
    )
    updated = "".join(lines[:anchor]) + insertion + "".join(lines[anchor:])
    ast.parse(updated)
    return updated, "patched"


def plan(root: Path):
    mapping = {
        "config": root / "verl/workers/config/engine.py",
        "engine": root / "verl/workers/engine/megatron/transformer_impl.py",
        "worker": root / "verl/workers/engine_workers.py",
    }
    for name, path in mapping.items():
        if not path.is_file():
            raise PatchError(f"Missing {name}: {path}")
    tpr_adapter = root / "verl/models/mcore/tpr/megatron_adapter.py"
    if not tpr_adapter.is_file():
        raise PatchError(f"TPR files were not copied into the target VERL: {tpr_adapter}")
    originals = {name: p.read_text(encoding="utf-8") for name, p in mapping.items()}
    updated = {}
    statuses = {}
    updated["config"], statuses["config"] = patch_config(originals["config"])
    spec, status_spec = patch_model_spec(originals["engine"])
    updated["engine"], status_dispatch = apply_tpr_phase4.transform(spec)
    statuses["engine"] = f"spec={status_spec}, dispatch={status_dispatch}"
    updated["worker"], statuses["worker"] = apply_tpr_actor_capture.transform(originals["worker"])
    for name, source in updated.items():
        ast.parse(source, filename=str(mapping[name]))
    return mapping, originals, updated, statuses


def _replace_verified(path: Path, old: str, new: str):
    if path.read_text(encoding="utf-8") != old:
        raise PatchError(f"Target changed during patch: {path}")
    fd, tmp = tempfile.mkstemp(prefix=".tpr-uniagent-cc-", suffix=".py", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(new)
        os.chmod(tmp, path.stat().st_mode)
        if path.read_text(encoding="utf-8") != old:
            raise PatchError(f"Target changed while writing temporary file: {path}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verl_root", type=Path, help="Target Uni-Agent-CC VERL repository root")
    parser.add_argument("--check", action="store_true", help="Preflight and show diff without writing")
    args = parser.parse_args()
    root = args.verl_root.expanduser().resolve()
    try:
        mapping, originals, updated, statuses = plan(root)
        for name in ("config", "engine", "worker"):
            print(f"{name}: {statuses[name]}: {mapping[name]}")
            if originals[name] != updated[name]:
                diff = difflib.unified_diff(
                    originals[name].splitlines(keepends=True),
                    updated[name].splitlines(keepends=True),
                    fromfile=str(mapping[name]) + " (before)",
                    tofile=str(mapping[name]) + " (after)",
                )
                print("".join(diff), end="")
        if args.check:
            print("CHECK ONLY: no files modified")
            return 0
        written = []
        try:
            for name in ("config", "engine", "worker"):
                if originals[name] == updated[name]:
                    continue
                _replace_verified(mapping[name], originals[name], updated[name])
                written.append(name)
        except Exception:
            for name in reversed(written):
                if mapping[name].read_text(encoding="utf-8") == updated[name]:
                    _replace_verified(mapping[name], updated[name], originals[name])
            raise
        print("APPLIED: TPR hooks installed; unrelated local edits retained")
        return 0
    except (PatchError, ValueError, SyntaxError, apply_tpr_phase4.PatchError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
