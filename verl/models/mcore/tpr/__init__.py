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

"""Model-side primitives for depth-first tree-prefix-reuse training."""

from .attention import TPRSelfAttention
from .context import (
    TPRAttentionBackend,
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)
from .engine_adapter import TPR_REQUEST_KEY, TPRForwardBackwardRequest
from .fixed_topology_scheduler import (
    FixedTopologyScheduler,
    PhysicalExecution,
    PhysicalExecutionKind,
    SchedulerState,
    TreeScheduleResult,
)
from .kv_stack import KVStack, KVStackEntry, PastKVAnchors, PastKVSlice, SegmentKV
from .module_spec import make_tpr_module_spec_provider, replace_self_attention_with_tpr
from .parallel import (
    ALLGATHER_CP_BACKEND,
    HYBRID_CP_BACKEND,
    RING_CP_BACKEND,
    ULYSSES_CP_BACKEND,
    AllGatherCPBackend,
    AllGatherCPAttentionBackend,
    LocalKVBlock,
    RingBlockKind,
    RingCPAttentionBackend,
    RingCPBackend,
    RingLocalKVBlock,
    ShardedPastKVAnchors,
    TPRCPBackend,
    UlyssesCPAttentionBackend,
    UlyssesCPBackend,
    all_gather_sequence,
    allgather_cp_rectangular_attention,
    classify_ring_block,
    make_ring_sequence_shard,
    resolve_tpr_cp_backend,
    ring_cp_attention,
    ulysses_cp_rectangular_attention,
)
from .prefix_state import (
    KVPrefixAnchors,
    KVPrefixState,
    PrefixState,
    PrefixStateEntry,
    PrefixStateStack,
)
from .rectangular_attention import rectangular_causal_attention
from .rope import build_sharded_rotary_pos_emb, build_suffix_rotary_pos_emb
from .shard import PrefixShard, RangeSequenceShard, SequenceShard
from .segment_executor import LeafVisitResult, SegmentBackwardResult, SegmentExecutor, SegmentForwardResult
from .segment_plan import (
    PopSegment,
    PushSegment,
    SegmentEvent,
    SegmentEventKind,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)

__all__ = [
    "ALLGATHER_CP_BACKEND",
    "HYBRID_CP_BACKEND",
    "RING_CP_BACKEND",
    "ULYSSES_CP_BACKEND",
    "TPRSelfAttention",
    "TPRForwardBackwardRequest",
    "TPR_REQUEST_KEY",
    "AllGatherCPAttentionBackend",
    "AllGatherCPBackend",
    "FixedTopologyScheduler",
    "KVStack",
    "KVStackEntry",
    "KVPrefixAnchors",
    "KVPrefixState",
    "LocalKVBlock",
    "LeafVisitResult",
    "PastKVAnchors",
    "PastKVSlice",
    "PhysicalExecution",
    "PhysicalExecutionKind",
    "PrefixShard",
    "RangeSequenceShard",
    "RingBlockKind",
    "RingCPAttentionBackend",
    "RingCPBackend",
    "RingLocalKVBlock",
    "PrefixState",
    "PrefixStateEntry",
    "PrefixStateStack",
    "SegmentKV",
    "ShardedPastKVAnchors",
    "SequenceShard",
    "SchedulerState",
    "TPRAttentionContext",
    "TPRAttentionBackend",
    "TPRCPBackend",
    "UlyssesCPAttentionBackend",
    "UlyssesCPBackend",
    "TreeScheduleResult",
    "PopSegment",
    "PushSegment",
    "SegmentEvent",
    "SegmentEventKind",
    "SegmentBackwardResult",
    "SegmentExecutor",
    "SegmentForwardResult",
    "SegmentLossTerm",
    "SegmentPlan",
    "SegmentSpec",
    "all_gather_sequence",
    "allgather_cp_rectangular_attention",
    "build_sharded_rotary_pos_emb",
    "build_suffix_rotary_pos_emb",
    "classify_ring_block",
    "get_tpr_attention_context",
    "make_tpr_module_spec_provider",
    "make_ring_sequence_shard",
    "rectangular_causal_attention",
    "resolve_tpr_cp_backend",
    "ring_cp_attention",
    "replace_self_attention_with_tpr",
    "use_tpr_attention_context",
    "ulysses_cp_rectangular_attention",
]
