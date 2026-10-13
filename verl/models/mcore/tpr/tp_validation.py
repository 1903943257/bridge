# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""TP2 schedule agreement for the existing TPR runner.

TP shards model weights/attention heads, NOT the TPR trajectory or scheduler.
Both TP peers MUST execute the same Push/Visit/Pop graph in the same order
before entering Megatron's native TP projection / vocab-parallel collectives.
Nothing here reimplements TP kernels, scatters KV, or synchronizes gradients.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .segment_plan import SegmentEvent, SegmentPlan
    from .tree_plan_builder import ForestExecutionPlan


def _segment_record(segment) -> tuple:
    # CPU token_ids are part of SegmentSpec's immutable metadata (the spec
    # clones the input); hashing raw bytes avoids ambiguous list separators.
    token_bytes = segment.token_ids.contiguous().numpy().tobytes()
    return (
        segment.segment_id,
        segment.parent_id,
        segment.position_start,
        segment.prefix_length,
        len(token_bytes),
        token_bytes,
        tuple(
            (t.query_offset, t.target_token_id, t.weight, t.sample_id)
            for t in segment.loss_terms
        ),
    )


def _write_plan(
    hasher,
    plan: "SegmentPlan",
    events: Iterable["SegmentEvent"] | None = None,
) -> None:
    # Insertion order is schedule-significant; do not sort segments.
    hasher.update(str((plan.root_id, bool(plan.topology_only))).encode())
    for segment in plan.segments.values():
        seg_id, parent, start, prefix, length, raw, terms = _segment_record(segment)
        hasher.update(json.dumps((seg_id, parent, start, prefix, length, terms),
                                 separators=(",", ":"), ensure_ascii=True).encode())
        hasher.update(raw)
    if events is None:
        events = plan.dfs_events()
    for event in events:
        hasher.update(f"{event.kind.value}:{event.segment_id};".encode())


def digest_segment_plan(
    plan: "SegmentPlan", events: Iterable["SegmentEvent"] | None = None
) -> str:
    hasher = hashlib.sha256()
    _write_plan(hasher, plan, events)
    return hasher.hexdigest()


def digest_forest_plan(forest: "ForestExecutionPlan") -> str:
    hasher = hashlib.sha256()
    hasher.update(f"loss_tokens={forest.logical_loss_tokens};".encode())
    for executable in forest.trees:
        tree = executable.tree
        hasher.update(
            json.dumps((tree.uid, tree.tree_index, list(tree.member_rows), tree.root_id),
                       separators=(",", ":")).encode()
        )
        for node in tree.nodes.values():
            hasher.update(json.dumps((
                node.node_id, node.parent_id, node.segment.row,
                node.segment.start, node.segment.end,
                node.member_rows, node.terminal_rows, node.children
            ), separators=(",", ":")).encode())
        _write_plan(hasher, executable.segment_plan)
        # PPO refs are logical and may differ even if physical tree coincides.
        # Identical refs ensure TP ranks enter vocab-parallel collectives with
        # identical query/target/temperature ownership and call counts.
        for ref in executable.objective_refs:
            hasher.update(json.dumps((
                ref.segment_id, ref.query_offset, ref.target_token_id,
                ref.sample_row, ref.response_offset
            ), separators=(",", ":")).encode())
    return hasher.hexdigest()


def assert_tp_plan_agreement(digest: str, *, tp_size: int) -> None:
    """Collectively reject divergent execution plans on the native TP group.

    Called ONCE per TPR minibatch, *before* model forward or TP collectives.
    Uses only a 32-byte SHA256 signature per rank, not model tensors / KV.
    Every peer receives the same success or mismatch result.
    """
    if tp_size == 1:
        return
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if len(digest) != 64:
        raise ValueError("SHA256 digest must be a 64-character hex string")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TPR TP2 requires an initialized distributed process group")
    group = parallel_state.get_tensor_model_parallel_group()
    if group is None:
        raise RuntimeError("TPR TP2 cannot resolve native Megatron TP group")
    if dist.get_world_size(group=group) != tp_size:
        raise RuntimeError("TPR TP group size differs from Engine TP config")
    # HCCL AllGather supports int64, so store SHA256 as eight int64 words
    # (each word is a non-negative 32-bit unsigned value).
    words = [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]
    device = torch.device("npu", torch.npu.current_device()) if dist.get_backend(group) == "hccl" else torch.device("cpu")
    local = torch.tensor(words, dtype=torch.int64, device=device)
    signatures = [torch.empty_like(local) for _ in range(tp_size)]
    dist.all_gather(signatures, local, group=group)
    if any(not torch.equal(value, local) for value in signatures):
        rank = dist.get_rank(group=group)
        raise RuntimeError(
            f"TPR TP schedule mismatch on tp_rank={rank}: forest, SegmentPlan, "
            "DFS events or PPO refs differ across model-parallel peers"
        )
