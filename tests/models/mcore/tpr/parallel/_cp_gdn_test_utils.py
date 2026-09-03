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

"""Distributed helpers for the native MindSpeed CP/GDN capability tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn


@dataclass(frozen=True)
class TensorComparison:
    """FP32 comparison metrics for two tensors."""

    relative_l2: float
    cosine: float
    max_abs_diff: float


@dataclass(frozen=True)
class AllToAllCall:
    """One observed low-level CP All-to-All invocation."""

    direction: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class AllToAllProbe:
    """Trace the actual low-level A2A functions used by MindSpeed GDN."""

    def __init__(self, gated_delta_net_module: Any):
        self._module = gated_delta_net_module
        self._original_cp2hp = gated_delta_net_module._all_to_all_cp2hp
        self._original_hp2cp = gated_delta_net_module._all_to_all_hp2cp
        self.calls: list[AllToAllCall] = []
        self._installed = False

    def install(self) -> AllToAllProbe:
        if self._installed:
            raise RuntimeError("AllToAllProbe is already installed")

        def cp2hp(input_: Tensor, cp_group: dist.ProcessGroup) -> Tensor:
            output = self._original_cp2hp(input_, cp_group)
            self.calls.append(
                AllToAllCall(
                    direction="cp2hp",
                    input_shape=tuple(input_.shape),
                    output_shape=tuple(output.shape),
                    dtype=input_.dtype,
                    device=input_.device,
                )
            )
            return output

        def hp2cp(input_: Tensor, cp_group: dist.ProcessGroup) -> Tensor:
            output = self._original_hp2cp(input_, cp_group)
            self.calls.append(
                AllToAllCall(
                    direction="hp2cp",
                    input_shape=tuple(input_.shape),
                    output_shape=tuple(output.shape),
                    dtype=input_.dtype,
                    device=input_.device,
                )
            )
            return output

        self._module._all_to_all_cp2hp = cp2hp
        self._module._all_to_all_hp2cp = hp2cp
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._module._all_to_all_cp2hp = self._original_cp2hp
        self._module._all_to_all_hp2cp = self._original_hp2cp
        self._installed = False

    def clear(self) -> None:
        self.calls.clear()

    def count(self, direction: str) -> int:
        return sum(call.direction == direction for call in self.calls)

    def __enter__(self) -> AllToAllProbe:
        return self.install()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.uninstall()


def native_zigzag_shard(tensor: Tensor, cp_group: dist.ProcessGroup, *, seq_dim: int = 0) -> Tensor:
    """Return the two native Megatron CP chunks owned by this rank."""

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    sequence_length = tensor.shape[seq_dim]
    num_chunks = 2 * cp_size
    if sequence_length % num_chunks != 0:
        raise ValueError(
            f"sequence length {sequence_length} must be divisible by 2 * CP size ({num_chunks})"
        )

    chunk_length = sequence_length // num_chunks
    sequence_first = tensor.movedim(seq_dim, 0)
    chunks = sequence_first.reshape(num_chunks, chunk_length, *sequence_first.shape[1:])
    local = torch.cat((chunks[cp_rank], chunks[num_chunks - cp_rank - 1]), dim=0)
    return local.movedim(0, seq_dim).contiguous()


def gather_native_zigzag(
    local_tensor: Tensor,
    cp_group: dist.ProcessGroup,
    *,
    seq_dim: int = 0,
) -> Tensor:
    """Gather native Megatron CP shards and restore the global sequence order."""

    cp_size = cp_group.size()
    local_sequence_length = local_tensor.shape[seq_dim]
    if local_sequence_length % 2 != 0:
        raise ValueError(
            f"local CP sequence length must contain two equal chunks, got {local_sequence_length}"
        )

    local_first = local_tensor.movedim(seq_dim, 0).contiguous()
    gathered = [torch.empty_like(local_first) for _ in range(cp_size)]
    dist.all_gather(gathered, local_first, group=cp_group)

    chunk_length = local_sequence_length // 2
    global_chunks: list[Tensor | None] = [None] * (2 * cp_size)
    for rank, rank_tensor in enumerate(gathered):
        first, second = rank_tensor.split(chunk_length, dim=0)
        global_chunks[rank] = first
        global_chunks[2 * cp_size - rank - 1] = second

    if any(chunk is None for chunk in global_chunks):
        raise RuntimeError("failed to reconstruct every native CP sequence chunk")
    global_first = torch.cat([chunk for chunk in global_chunks if chunk is not None], dim=0)
    return global_first.movedim(0, seq_dim).contiguous()


def broadcast_module_state(module: nn.Module, *, src: int = 0) -> None:
    """Make every rank start from bitwise-identical parameters and buffers."""

    with torch.no_grad():
        for tensor in module.state_dict().values():
            dist.broadcast(tensor, src=src)


def reduce_parameter_gradients(module: nn.Module, cp_group: dist.ProcessGroup) -> None:
    """Sum CP-local parameter-gradient contributions into global gradients."""

    for parameter in module.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=cp_group)


def clone_parameter_gradients(module: nn.Module) -> dict[str, Tensor]:
    """Clone every materialized named parameter gradient in FP32."""

    return {
        name: parameter.grad.detach().float().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def assert_named_tensors_finite(tensors: Mapping[str, Tensor], *, label: str) -> None:
    """Fail with the precise tensor name when NaN or Inf is observed."""

    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{label} tensor {name!r} contains NaN or Inf")


def tensor_comparison(reference: Tensor, actual: Tensor) -> TensorComparison:
    """Compute scale-aware metrics after converting inputs to flat FP32 tensors."""

    reference_flat = reference.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if reference_flat.shape != actual_flat.shape:
        raise ValueError(
            f"comparison shape mismatch: reference={tuple(reference.shape)}, actual={tuple(actual.shape)}"
        )

    difference = actual_flat - reference_flat
    reference_norm = torch.linalg.vector_norm(reference_flat)
    difference_norm = torch.linalg.vector_norm(difference)
    relative_l2 = difference_norm / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)

    actual_norm = torch.linalg.vector_norm(actual_flat)
    denominator = reference_norm * actual_norm
    if denominator == 0:
        cosine = torch.tensor(
            1.0 if reference_norm == 0 and actual_norm == 0 else 0.0,
            device=reference_flat.device,
        )
    else:
        cosine = torch.dot(reference_flat, actual_flat) / denominator

    max_abs_diff = difference.abs().max() if difference.numel() else difference.new_zeros(())
    return TensorComparison(
        relative_l2=float(relative_l2.item()),
        cosine=float(cosine.item()),
        max_abs_diff=float(max_abs_diff.item()),
    )


def named_tensor_comparison(
    reference: Mapping[str, Tensor], actual: Mapping[str, Tensor]
) -> tuple[TensorComparison, tuple[str, TensorComparison]]:
    """Compare a named tensor collection globally and return its worst tensor."""

    if reference.keys() != actual.keys():
        missing = sorted(reference.keys() - actual.keys())
        unexpected = sorted(actual.keys() - reference.keys())
        raise AssertionError(
            f"named tensor sets differ: missing={missing}, unexpected={unexpected}"
        )
    if not reference:
        raise AssertionError("named tensor comparison requires at least one tensor")

    names = sorted(reference)
    reference_flat = torch.cat([reference[name].reshape(-1) for name in names])
    actual_flat = torch.cat([actual[name].reshape(-1) for name in names])
    global_comparison = tensor_comparison(reference_flat, actual_flat)

    per_tensor = [(name, tensor_comparison(reference[name], actual[name])) for name in names]
    worst = max(per_tensor, key=lambda item: item[1].relative_l2)
    return global_comparison, worst
