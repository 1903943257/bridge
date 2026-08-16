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

"""Model-side primitives for depth-first tree attention training."""

from .attention import DTASelfAttention
from .context import TreeAttentionContext, get_tree_attention_context, use_tree_attention_context
from .engine_adapter import DTA_REQUEST_KEY, DTAForwardBackwardRequest
from .fixed_topology_scheduler import FixedTopologyScheduler, SchedulerState, TreeScheduleResult
from .kv_stack import KVStack, KVStackEntry, PastKVAnchors, PastKVSlice, SegmentKV
from .module_spec import make_dta_module_spec_provider, replace_self_attention_with_dta
from .rectangular_attention import rectangular_causal_attention
from .rope import build_suffix_rotary_pos_emb
from .segment_executor import SegmentBackwardResult, SegmentExecutor, SegmentForwardResult
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
    "DTASelfAttention",
    "DTAForwardBackwardRequest",
    "DTA_REQUEST_KEY",
    "FixedTopologyScheduler",
    "KVStack",
    "KVStackEntry",
    "PastKVAnchors",
    "PastKVSlice",
    "SegmentKV",
    "SchedulerState",
    "TreeAttentionContext",
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
    "build_suffix_rotary_pos_emb",
    "get_tree_attention_context",
    "make_dta_module_spec_provider",
    "rectangular_causal_attention",
    "replace_self_attention_with_dta",
    "use_tree_attention_context",
]
