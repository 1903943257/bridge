"""CPU-only tests for the minimal, idempotent real actor capture patcher."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


def _patcher():
    root = Path(__file__).resolve().parents[5]
    script = root / "patches" / "apply_tpr_actor_capture.py"
    spec = importlib.util.spec_from_file_location("_actor_capture_patch", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = """def train_mini_batch(self, data):
    for batch_idx, mini_batch_td in enumerate(data):
        actor_output = self.train_batch(mini_batch_td)
    return actor_output
"""


def test_actor_capture_patch_preserves_actual_train_batch_and_is_opt_in():
    patcher = _patcher()
    updated, status = patcher.transform(_SOURCE)
    assert status == "patched"
    assert "TPR_CAPTURE_ACTOR_MINIBATCH_DIR" in updated
    assert "capture_real_actor_update_minibatch(" in updated
    assert updated.count("actor_output = self.train_batch(mini_batch_td)") == 1
    assert updated.index("capture_real_actor_update_minibatch(") < updated.index(
        "actor_output = self.train_batch(mini_batch_td)"
    )
    ast.parse(updated)
    again, status_again = patcher.transform(updated)
    assert status_again == "already_applied"
    assert again == updated


def test_actor_capture_patch_rejects_unknown_worker_source():
    with pytest.raises(ValueError, match="exactly one"):
        _patcher().transform("def train_batch(self, data): pass\n")


def test_actor_capture_patch_rejects_duplicate_train_calls():
    with pytest.raises(ValueError, match="got 2"):
        _patcher().transform(_SOURCE + "\n" + _SOURCE)
