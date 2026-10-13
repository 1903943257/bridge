"""CPU-only contract: a real actor-update capture has an unambiguous step and ordinal."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from verl.models.mcore.tpr import actor_capture
from verl.utils import tensordict_utils as tu


def _batch(step: int = 1, rows: int = 2) -> TensorDict:
    return TensorDict(
        {
            "global_steps": torch.full((rows,), step, dtype=torch.int64),
            "input_ids": torch.zeros((rows, 4), dtype=torch.int64),
        },
        batch_size=[rows],
    )


def test_capture_step_accepts_repeated_real_step_and_scalar_metadata():
    batch = _batch(step=5, rows=2)
    assert actor_capture._capture_global_step(batch) == 5

    batch = TensorDict(
        {"input_ids": torch.zeros((2, 4), dtype=torch.int64)},
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, global_steps=6)
    assert actor_capture._capture_global_step(batch) == 6


def test_capture_step_rejects_missing_inconsistent_and_noninteger_values():
    batch = _batch()
    del batch["global_steps"]
    with pytest.raises(ValueError, match="requires the original global_steps"):
        actor_capture._capture_global_step(batch)

    batch = _batch()
    batch["global_steps"] = torch.tensor([1, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="differs"):
        actor_capture._capture_global_step(batch)

    batch["global_steps"] = torch.tensor([1.0, 1.0])
    with pytest.raises(TypeError, match="must be integer"):
        actor_capture._capture_global_step(batch)


def test_capture_unique_across_steps_and_repeated_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(actor_capture, "_real_keys", lambda _: ("row-0", "row-1"))
    first = actor_capture.capture_real_actor_update_minibatch(
        _batch(step=1), tmp_path, batch_idx=0, capture_index=0
    )
    second_same_step = actor_capture.capture_real_actor_update_minibatch(
        _batch(step=1), tmp_path, batch_idx=0, capture_index=1
    )
    third = actor_capture.capture_real_actor_update_minibatch(
        _batch(step=2), tmp_path, batch_idx=0, capture_index=2
    )
    assert len({first.name, second_same_step.name, third.name}) == 3
    assert "step000001" in first.name
    assert "capture000001" in second_same_step.name
    assert "step000002" in third.name

    payload = torch.load(first, map_location="cpu", weights_only=False)
    assert payload["capture_stage"] == "actor_update_mini_batch"
    assert payload["capture_global_step"] == 1
    assert payload["capture_index"] == 0
    assert payload["keys"] == ("row-0", "row-1")
    assert payload["tensordict"].batch_size[0] == 2

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        actor_capture.capture_real_actor_update_minibatch(
            _batch(step=1), tmp_path, batch_idx=0, capture_index=0
        )
