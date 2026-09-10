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

"""RoPE construction helpers for suffix-only TPR forwards."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from .shard import SequenceShard


class _RotaryEmbedding(Protocol):
    def __call__(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: object | None = None,
    ) -> Tensor: ...


class _UnshardedContextParallelGroup:
    """Size-one sentinel that prevents RotaryEmbedding using its bound CP group."""

    @staticmethod
    def size() -> int:
        return 1


_UNSHARDED_CP_GROUP = _UnshardedContextParallelGroup()


def build_suffix_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    prefix_length: int,
    suffix_length: int,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build standard 1D RoPE frequencies for positions ``[P, P+b)``.

    This delegates frequency generation to the model's existing Megatron
    ``RotaryEmbedding`` instance, preserving its rotary dimension, base,
    interpolation, interleaving, and scaling settings.  The MVP deliberately
    requests neither packed-sequence nor context-parallel sharding.
    """

    _validate_length("prefix_length", prefix_length, allow_zero=True)
    _validate_length("suffix_length", suffix_length, allow_zero=False)
    if not callable(rotary_embedding):
        raise TypeError("rotary_embedding must be callable")

    rotary_pos_emb = rotary_embedding(
        suffix_length,
        offset=prefix_length,
        packed_seq=False,
        cp_group=_UNSHARDED_CP_GROUP if disable_context_parallel_sharding else None,
    )
    if not isinstance(rotary_pos_emb, Tensor):
        raise TypeError(
            "TPR RoPE MVP requires standard RotaryEmbedding to return one tensor; "
            f"got {type(rotary_pos_emb).__name__}"
        )
    if rotary_pos_emb.ndim != 4:
        raise ValueError(
            "suffix rotary embedding must have shape [suffix, 1, 1, rotary_dim], "
            f"got {rotary_pos_emb.shape}"
        )
    if rotary_pos_emb.shape[0] != suffix_length:
        raise ValueError(
            f"suffix rotary embedding sequence length must be {suffix_length}, "
            f"got {rotary_pos_emb.shape[0]}"
        )
    if rotary_pos_emb.shape[1] != 1 or rotary_pos_emb.shape[2] != 1:
        raise ValueError(
            "TPR RoPE MVP requires singleton batch/head broadcast dimensions, "
            f"got {rotary_pos_emb.shape}"
        )
    if rotary_pos_emb.shape[3] <= 0:
        raise ValueError("suffix rotary embedding must have a positive rotary dimension")
    return rotary_pos_emb


def build_sharded_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    position_start: int,
    shard: SequenceShard,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build RoPE in the local-token order described by ``shard``."""

    _validate_length("position_start", position_start, allow_zero=True)
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"shard must implement SequenceShard, got {type(shard).__name__}")
    chunks = []
    for physical_start, physical_end in shard.physical_global_ranges:
        valid_end = min(physical_end, shard.global_length)
        valid_length = max(0, valid_end - physical_start)
        physical_length = physical_end - physical_start
        if valid_length:
            chunk = build_suffix_rotary_pos_emb(
                rotary_embedding,
                prefix_length=position_start + physical_start,
                suffix_length=valid_length,
                disable_context_parallel_sharding=disable_context_parallel_sharding,
            )
        else:
            chunk = build_suffix_rotary_pos_emb(
                rotary_embedding,
                prefix_length=position_start + shard.global_length - 1,
                suffix_length=1,
                disable_context_parallel_sharding=disable_context_parallel_sharding,
            )
        if valid_length < physical_length:
            padding = chunk[-1:].expand(physical_length - valid_length, *chunk.shape[1:])
            chunk = torch.cat((chunk if valid_length else chunk[:0], padding), dim=0)
        chunks.append(chunk)
    rotary_pos_emb = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    if rotary_pos_emb.shape[0] != shard.local_length:
        raise RuntimeError(
            f"sharded rotary embedding length must be {shard.local_length}, "
            f"got {rotary_pos_emb.shape[0]}"
        )
    return rotary_pos_emb


def _validate_length(name: str, value: int, *, allow_zero: bool) -> None:
    lower_bound = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")
