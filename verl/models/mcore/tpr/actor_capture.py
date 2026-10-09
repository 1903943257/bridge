"""Opt-in capture of the *actual* VERL actor-update mini_batch_td.

Called immediately before TrainingWorker.train_mini_batch dispatches the
actor step; this module does NOT patch the optimizer or change the mini-batch.
Captures may contain sensitive trajectory data: store on a private server.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from verl.utils import tensordict_utils as tu


def _real_keys(batch) -> tuple[str, ...]:
    keys = tu.get_non_tensor_data(
        batch, key="tpr_trajectory_keys", default=None
    )
    if keys is not None:
        return tuple(keys)
    # Real per-trajectory uid is a supported identity only when genuinely
    # supplied by the data loader. Never fabricate a row-number identity.
    if "uid" in batch:
        value = batch["uid"]
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        if isinstance(value, torch.Tensor):
            if value.ndim == 1:
                return tuple(str(v.item()) for v in value)
    return ()


def capture_real_actor_update_minibatch(
    mini_batch_td, directory: str | os.PathLike[str], *, batch_idx: int
) -> Path:
    """Persist one immutable actor-update batch without modifying it."""
    if not isinstance(batch_idx, int) or batch_idx < 0:
        raise ValueError("batch_idx must be a non-negative integer")
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available()
        and torch.distributed.is_initialized()
        else 0
    )
    target = base / f"actor_update_rank{rank}_batch{batch_idx}.pt"
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite a previous real actor-update capture: {target}"
        )
    keys = _real_keys(mini_batch_td)
    # Deep-ish TensorDict clone keeps the runtime mini-batch unmodified.
    # Conversion to CPU also ensures the saved payload is NPU-independent.
    payload = {
        "capture_stage": "actor_update_mini_batch",
        "tensordict": mini_batch_td.clone().cpu(),
        "keys": keys,
        "capture_dp_rank": rank,
        "capture_mini_batch_idx": batch_idx,
    }
    temp = target.with_suffix(".pt.tmp")
    if temp.exists():
        raise FileExistsError(f"Capture temp path already exists: {temp}")
    try:
        torch.save(payload, temp)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
    print(
        f"P0 ACTOR_UPDATE_CAPTURE path={target} "
        f"rows={mini_batch_td.batch_size[0]} keys={len(keys)} "
        "stage=before_optimizer_step",
        flush=True,
    )
    return target
