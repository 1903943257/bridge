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

"""Context-parallel attention backends for TPR."""

from .allgather_attention import (
    LocalKVBlock,
    all_gather_sequence,
    allgather_cp_rectangular_attention,
)
from .backend import (
    ALLGATHER_CP_BACKEND,
    HYBRID_CP_BACKEND,
    RING_CP_BACKEND,
    ULYSSES_CP_BACKEND,
    AllGatherCPBackend,
    HybridCPBackend,
    RingCPBackend,
    TPRCPBackend,
    UlyssesCPBackend,
    resolve_tpr_cp_backend,
)
from .hybrid_attention import (
    HybridCPAttentionBackend,
    HybridCPTopology,
    HybridLocalKVBlock,
    hybrid_cp_rectangular_attention,
    make_hybrid_sequence_shard,
    resolve_mindspeed_hybrid_topology,
)
from .ring_attention import (
    RingBlockKind,
    RingCPAttentionBackend,
    RingLocalKVBlock,
    classify_ring_block,
    make_ring_sequence_shard,
    ring_cp_attention,
)
from .ulysses_attention import UlyssesCPAttentionBackend, ulysses_cp_rectangular_attention
from .execution_context import (
    AllGatherCPAttentionBackend,
    ShardedPastKVAnchors,
    accumulate_sharded_past_anchor_gradients,
    build_sharded_past_anchors,
    make_anchored_allgather_cp_backend,
    make_cached_allgather_cp_backend,
    resolve_cp_group,
)

__all__ = [
    "ALLGATHER_CP_BACKEND",
    "HYBRID_CP_BACKEND",
    "RING_CP_BACKEND",
    "ULYSSES_CP_BACKEND",
    "AllGatherCPBackend",
    "AllGatherCPAttentionBackend",
    "HybridCPAttentionBackend",
    "HybridCPBackend",
    "HybridCPTopology",
    "HybridLocalKVBlock",
    "LocalKVBlock",
    "RingBlockKind",
    "RingCPAttentionBackend",
    "RingCPBackend",
    "RingLocalKVBlock",
    "ShardedPastKVAnchors",
    "TPRCPBackend",
    "UlyssesCPAttentionBackend",
    "UlyssesCPBackend",
    "accumulate_sharded_past_anchor_gradients",
    "all_gather_sequence",
    "allgather_cp_rectangular_attention",
    "build_sharded_past_anchors",
    "classify_ring_block",
    "hybrid_cp_rectangular_attention",
    "make_hybrid_sequence_shard",
    "make_ring_sequence_shard",
    "make_anchored_allgather_cp_backend",
    "make_cached_allgather_cp_backend",
    "resolve_cp_group",
    "resolve_mindspeed_hybrid_topology",
    "resolve_tpr_cp_backend",
    "ring_cp_attention",
    "ulysses_cp_rectangular_attention",
]
