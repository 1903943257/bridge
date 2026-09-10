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

"""Backend selection contract for context-parallel TPR segment execution."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..context import TPRAttentionBackend
from ..kv_stack import KVStack
from ..shard import (
    PrefixShard,
    RangeSequenceShard,
    SequenceShard,
    maybe_pad_sequence_shard,
    physical_sequence_shard,
    round_up_sequence_length,
)
from .execution_context import (
    ShardedPastKVAnchors,
    make_anchored_allgather_cp_backend,
    make_cached_allgather_cp_backend,
    resolve_cp_group,
)
from .hybrid_attention import (
    HybridCPTopology,
    make_anchored_hybrid_cp_backend,
    make_cached_hybrid_cp_backend,
    make_hybrid_sequence_shard,
    resolve_mindspeed_hybrid_topology,
)
from .ring_attention import (
    make_anchored_ring_cp_backend,
    make_cached_ring_cp_backend,
    make_ring_sequence_shard,
)
from .ulysses_attention import (
    make_anchored_ulysses_cp_backend,
    make_cached_ulysses_cp_backend,
)

ALLGATHER_CP_BACKEND = "allgather"
MINDSPEED_ALLGATHER_CP_ALGO = "kvallgather_cp_algo"
ULYSSES_CP_BACKEND = "ulysses"
MINDSPEED_ULYSSES_CP_ALGO = "ulysses_cp_algo"
RING_CP_BACKEND = "ring"
MINDSPEED_RING_CP_ALGO = "megatron_cp_algo"
HYBRID_CP_BACKEND = "hybrid"
MINDSPEED_HYBRID_CP_ALGO = "hybrid_cp_algo"


@runtime_checkable
class TPRCPBackend(Protocol):
    """Persistent CP policy used by a :class:`SegmentExecutor`.

    The policy owns sequence placement and creates a per-forward attention
    backend. Prefix-state storage and traversal order remain executor concerns.
    """

    @property
    def backend_name(self) -> str: ...

    @property
    def parallel_size(self) -> int: ...

    @property
    def parallel_rank(self) -> int: ...

    def validate_segment_length(self, global_length: int) -> None: ...

    def make_sequence_shard(self, global_length: int) -> SequenceShard: ...

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: ShardedPastKVAnchors | None,
    ) -> TPRAttentionBackend: ...


class AllGatherCPBackend:
    """Reference CP policy using contiguous shards and KV AllGather."""

    backend_name = ALLGATHER_CP_BACKEND

    def __init__(
        self,
        cp_group: Any,
        *,
        parallel_size: int | None = None,
        parallel_rank: int | None = None,
    ) -> None:
        if parallel_size is None or parallel_rank is None:
            if parallel_size is not None or parallel_rank is not None:
                raise ValueError("parallel_size and parallel_rank must be provided together")
            parallel_size, parallel_rank = resolve_cp_group(cp_group)
        if parallel_size <= 1:
            raise ValueError(f"AllGather CP requires more than one rank, got {parallel_size}")
        if parallel_rank < 0 or parallel_rank >= parallel_size:
            raise ValueError(f"parallel_rank must be in [0, {parallel_size}), got {parallel_rank}")
        self._cp_group = cp_group
        self._parallel_size = parallel_size
        self._parallel_rank = parallel_rank

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    @property
    def parallel_rank(self) -> int:
        return self._parallel_rank

    def validate_segment_length(self, global_length: int) -> None:
        if not isinstance(global_length, int) or isinstance(global_length, bool) or global_length <= 0:
            raise ValueError(f"length must be a positive integer, got {global_length!r}")

    def make_sequence_shard(self, global_length: int) -> SequenceShard:
        self.validate_segment_length(global_length)
        padded_length = round_up_sequence_length(global_length, self.parallel_size)
        physical_shard = PrefixShard.contiguous(
            padded_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
        )
        return maybe_pad_sequence_shard(global_length, physical_shard)

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: ShardedPastKVAnchors | None,
    ) -> TPRAttentionBackend:
        if not isinstance(physical_sequence_shard(current_shard), PrefixShard):
            raise TypeError(
                "AllGather CP requires a contiguous PrefixShard, "
                f"got {type(current_shard).__name__}"
            )
        if past_anchors is None:
            return make_cached_allgather_cp_backend(
                kv_stack,
                expected_layer_numbers=expected_layer_numbers,
                current_shard=current_shard,
                cp_group=self._cp_group,
            )
        return make_anchored_allgather_cp_backend(
            past_anchors,
            expected_layer_numbers=expected_layer_numbers,
            current_shard=current_shard,
            cp_group=self._cp_group,
        )


class UlyssesCPBackend(AllGatherCPBackend):
    """TPR policy using contiguous sequence shards and MindSpeed Ulysses A2A."""

    backend_name = ULYSSES_CP_BACKEND

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: ShardedPastKVAnchors | None,
    ) -> TPRAttentionBackend:
        if not isinstance(physical_sequence_shard(current_shard), PrefixShard):
            raise TypeError(
                "Ulysses CP requires a contiguous PrefixShard, "
                f"got {type(current_shard).__name__}"
            )
        if past_anchors is None:
            return make_cached_ulysses_cp_backend(
                kv_stack,
                expected_layer_numbers=expected_layer_numbers,
                current_shard=current_shard,
                cp_group=self._cp_group,
            )
        return make_anchored_ulysses_cp_backend(
            past_anchors,
            expected_layer_numbers=expected_layer_numbers,
            current_shard=current_shard,
            cp_group=self._cp_group,
        )


class RingCPBackend(AllGatherCPBackend):
    """TPR policy using MindSpeed's symmetric causal Ring layout and P2P."""

    backend_name = RING_CP_BACKEND

    def validate_segment_length(self, global_length: int) -> None:
        make_ring_sequence_shard(
            global_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
        )

    def make_sequence_shard(self, global_length: int) -> SequenceShard:
        return make_ring_sequence_shard(
            global_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
        )

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: ShardedPastKVAnchors | None,
    ) -> TPRAttentionBackend:
        if not isinstance(physical_sequence_shard(current_shard), RangeSequenceShard):
            raise TypeError(
                "Ring CP requires a RangeSequenceShard, "
                f"got {type(current_shard).__name__}"
            )
        if past_anchors is None:
            return make_cached_ring_cp_backend(
                kv_stack,
                expected_layer_numbers=expected_layer_numbers,
                current_shard=current_shard,
                cp_group=self._cp_group,
            )
        return make_anchored_ring_cp_backend(
            past_anchors,
            expected_layer_numbers=expected_layer_numbers,
            current_shard=current_shard,
            cp_group=self._cp_group,
        )


class HybridCPBackend(AllGatherCPBackend):
    """TPR policy composing MindSpeed Ulysses and Ring CP subgroups."""

    backend_name = HYBRID_CP_BACKEND

    def __init__(
        self,
        cp_group: Any,
        *,
        parallel_size: int | None = None,
        parallel_rank: int | None = None,
        topology: HybridCPTopology | None = None,
    ) -> None:
        super().__init__(
            cp_group,
            parallel_size=parallel_size,
            parallel_rank=parallel_rank,
        )
        if topology is None:
            topology = resolve_mindspeed_hybrid_topology(
                cp_group,
                cp_size=self.parallel_size,
                cp_rank=self.parallel_rank,
            )
        elif not isinstance(topology, HybridCPTopology):
            raise TypeError(f"topology must be HybridCPTopology, got {type(topology).__name__}")
        if (topology.cp_size, topology.cp_rank) != (
            self.parallel_size,
            self.parallel_rank,
        ):
            raise ValueError("Hybrid topology coordinates must match the full CP group")
        self._topology = topology

    @property
    def topology(self) -> HybridCPTopology:
        return self._topology

    def validate_segment_length(self, global_length: int) -> None:
        make_hybrid_sequence_shard(
            global_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
            ulysses_degree=self.topology.ulysses_size,
        )

    def make_sequence_shard(self, global_length: int) -> SequenceShard:
        return make_hybrid_sequence_shard(
            global_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
            ulysses_degree=self.topology.ulysses_size,
        )

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: ShardedPastKVAnchors | None,
    ) -> TPRAttentionBackend:
        if not isinstance(physical_sequence_shard(current_shard), RangeSequenceShard):
            raise TypeError(
                "Hybrid CP requires a RangeSequenceShard, "
                f"got {type(current_shard).__name__}"
            )
        if past_anchors is None:
            return make_cached_hybrid_cp_backend(
                kv_stack,
                expected_layer_numbers=expected_layer_numbers,
                current_shard=current_shard,
                topology=self.topology,
            )
        return make_anchored_hybrid_cp_backend(
            past_anchors,
            expected_layer_numbers=expected_layer_numbers,
            current_shard=current_shard,
            topology=self.topology,
        )


def resolve_tpr_cp_backend(
    backend: TPRCPBackend | str | None,
    *,
    cp_group: Any,
    parallel_size: int,
    parallel_rank: int,
) -> TPRCPBackend:
    """Resolve one CP policy without silently falling back between algorithms."""

    if backend is None or backend in (ALLGATHER_CP_BACKEND, MINDSPEED_ALLGATHER_CP_ALGO):
        return AllGatherCPBackend(
            cp_group,
            parallel_size=parallel_size,
            parallel_rank=parallel_rank,
        )
    if backend in (ULYSSES_CP_BACKEND, MINDSPEED_ULYSSES_CP_ALGO):
        return UlyssesCPBackend(
            cp_group,
            parallel_size=parallel_size,
            parallel_rank=parallel_rank,
        )
    if backend in (RING_CP_BACKEND, MINDSPEED_RING_CP_ALGO):
        return RingCPBackend(
            cp_group,
            parallel_size=parallel_size,
            parallel_rank=parallel_rank,
        )
    if backend in (HYBRID_CP_BACKEND, MINDSPEED_HYBRID_CP_ALGO):
        return HybridCPBackend(
            cp_group,
            parallel_size=parallel_size,
            parallel_rank=parallel_rank,
        )
    if isinstance(backend, str):
        raise ValueError(
            f"unknown TPR CP backend {backend!r}; expected one of "
            f"{(ALLGATHER_CP_BACKEND, ULYSSES_CP_BACKEND, RING_CP_BACKEND, HYBRID_CP_BACKEND)}"
        )
    if not isinstance(backend, TPRCPBackend):
        raise TypeError(f"cp_backend must implement TPRCPBackend, got {type(backend).__name__}")
    if (backend.parallel_size, backend.parallel_rank) != (parallel_size, parallel_rank):
        raise ValueError(
            "CP backend coordinates must match cp_group: "
            f"expected rank {parallel_rank}/{parallel_size}, "
            f"got {backend.parallel_rank}/{backend.parallel_size}"
        )
    return backend
