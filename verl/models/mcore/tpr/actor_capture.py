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
    """Capture EXACTLY the keys the production PPO Engine will use.

    A uid is a sample/group identifier, not a unique rollout identifier.
    The Engine resolves repeated uid values to uid_tpr_<mini-batch-row>,
    preserving group membership while distinguishing sibling trajectories.
    Do not independently duplicate its fallback implementation.
    """
    from .megatron_adapter import _trajectory_keys_from_minibatch

    return _trajectory_keys_from_minibatch(batch)


def _capture_global_step(batch) -> int:
    """Resolve one real global step, shared by all rows in an actor mini-batch.

    Fail closed rather than silently producing misordered Replay traces.
    Both TensorDict tensor and NonTensorData/NonTensorStack representations
    are supported; all rows in one mini-batch must share the same step.
    """
    raw = tu.get(batch, "global_steps", default=None)
    if raw is None:
        raise ValueError(
            "Actor capture requires the original global_steps field; "
            "cannot label a multi-step PPO Replay without it"
        )
    try:
        values = torch.as_tensor(raw)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"unsupported global_steps value: {type(raw)!r}") from exc
    if values.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise TypeError(f"global_steps must be integer, got {values.dtype}")
    flat = values.detach().cpu().reshape(-1)
    rows = batch.batch_size[0]
    if flat.numel() not in (1, rows) or flat.numel() == 0:
        raise ValueError(
            f"global_steps expects scalar or {rows} rows, got {flat.numel()}"
        )
    if not bool(torch.all(flat == flat[0])):
        raise ValueError("global_steps differs across mini-batch rows")
    step = int(flat[0].item())
    if step < 0:
        raise ValueError(f"global_steps must be non-negative, got {step}")
    return step


def capture_real_actor_update_minibatch(
    mini_batch_td, directory: str | os.PathLike[str], *,
    batch_idx: int, capture_index: int = 0,
) -> Path:
    """Persist one immutable actor-update batch without modifying it."""
    if not isinstance(batch_idx, int) or batch_idx < 0:
        raise ValueError("batch_idx must be a non-negative integer")
    if not isinstance(capture_index, int) or capture_index < 0:
        raise ValueError("capture_index must be a non-negative integer")
    global_step = _capture_global_step(mini_batch_td)
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available()
        and torch.distributed.is_initialized()
        else 0
    )
    target = base / (
        f"actor_update_step{global_step:06d}_rank{rank}_"
        f"capture{capture_index:06d}_batch{batch_idx}.pt"
    )
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
        "capture_dp_rank": rank,  # legacy key: historically the global rank
        "capture_global_rank": rank,
        "capture_global_step": global_step,
        "capture_index": capture_index,
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
        f"step={global_step} rank={rank} capture_index={capture_index} "
        f"batch_idx={batch_idx} rows={mini_batch_td.batch_size[0]} "
        f"keys={len(keys)} stage=before_optimizer_step",
        flush=True,
    )
    return target
