# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve the formal Megatron context-parallel runtime used by TPR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch import Tensor

from megatron.core import parallel_state

from .backend import ALLGATHER_CP_BACKEND, TPRCPBackend, resolve_tpr_cp_backend
from .execution_context import resolve_cp_group


@dataclass(frozen=True, slots=True)
class TPREngineCPRuntime:
    """One validated CP group and backend selected for an Engine request."""

    group: Any | None
    backend: TPRCPBackend | None
    size: int
    rank: int

    @property
    def enabled(self) -> bool:
        return self.size > 1

    @property
    def backend_name(self) -> str:
        return "none" if self.backend is None else self.backend.backend_name


def _setting(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _configured_cp_algorithm(engine: Any, model_config: Any) -> str:
    explicit_backend = _setting(engine.engine_config, "tpr_cp_backend")
    if explicit_backend not in (None, "auto"):
        return explicit_backend

    sources = (
        model_config,
        getattr(engine, "tf_config", None),
        _setting(engine.engine_config, "override_transformer_config"),
        _setting(engine.engine_config, "mcore_kwargs"),
    )
    for source in sources:
        algorithm = _setting(source, "context_parallel_algo")
        if algorithm is not None:
            return algorithm

    # MindSpeed keeps the selected CP algorithm in Megatron's global args in
    # addition to copying it onto some TransformerConfig construction paths.
    try:
        from megatron.training import get_args

        global_args = get_args()
    except (AssertionError, ImportError, RuntimeError):
        global_args = None
    algorithm = _setting(global_args, "context_parallel_algo")
    return ALLGATHER_CP_BACKEND if algorithm is None else algorithm


def resolve_engine_cp_runtime(engine: Any, model_config: Any) -> TPREngineCPRuntime:
    """Resolve TPR CP from the initialized Megatron topology and real config."""

    configured_size = _setting(engine.engine_config, "context_parallel_size")
    if not isinstance(configured_size, int) or isinstance(configured_size, bool) or configured_size <= 0:
        raise ValueError(
            "engine_config.context_parallel_size must be a positive integer, "
            f"got {configured_size!r}"
        )
    explicit_backend = _setting(engine.engine_config, "tpr_cp_backend")
    if configured_size == 1:
        if explicit_backend not in (None, "auto"):
            raise ValueError(
                f"tpr_cp_backend={explicit_backend!r} requires context_parallel_size > 1"
            )
        return TPREngineCPRuntime(group=None, backend=None, size=1, rank=0)

    if _setting(engine.engine_config, "dynamic_context_parallel"):
        raise NotImplementedError("TPR formal entry does not support dynamic context parallelism")
    if not dist.is_initialized() or not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("TPR CP requires initialized torch.distributed and Megatron model parallelism")

    try:
        cp_group = parallel_state.get_context_parallel_group()
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError("TPR CP could not resolve Megatron's context-parallel group") from exc
    actual_size, actual_rank = resolve_cp_group(cp_group)
    if actual_size != configured_size:
        raise RuntimeError(
            "TPR CP topology mismatch: "
            f"engine_config requests CP={configured_size}, initialized group has CP={actual_size}"
        )

    algorithm = _configured_cp_algorithm(engine, model_config)
    backend = resolve_tpr_cp_backend(
        algorithm,
        cp_group=cp_group,
        parallel_size=actual_size,
        parallel_rank=actual_rank,
    )
    return TPREngineCPRuntime(
        group=cp_group,
        backend=backend,
        size=actual_size,
        rank=actual_rank,
    )


def aggregate_cp_loss(
    loss_sum: Tensor,
    normalized_loss: Tensor,
    runtime: TPREngineCPRuntime,
) -> tuple[Tensor, Tensor]:
    """Return the same global TPR loss scalars on every CP rank."""

    global_loss_sum = loss_sum.detach().clone()
    global_normalized_loss = normalized_loss.detach().clone()
    if runtime.enabled:
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=runtime.group)
        dist.all_reduce(global_normalized_loss, op=dist.ReduceOp.SUM, group=runtime.group)
    return global_loss_sum, global_normalized_loss
