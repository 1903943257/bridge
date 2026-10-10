# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Pure TP2 control-plane tests; no multi-NPU requirement."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.tpr.segment_plan import (
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)
from verl.models.mcore.tpr.tp_validation import (
    assert_tp_plan_agreement,
    digest_forest_plan,
    digest_segment_plan,
)
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _plan(left=4, right=5):
    return SegmentPlan(
        (
            SegmentSpec(0, None, torch.tensor([1, 2, 3]), 0, 0,
                        (SegmentLossTerm(2, left), SegmentLossTerm(2, right))),
            SegmentSpec(1, 0, torch.tensor([left, 6]), 3, 3,
                        (SegmentLossTerm(0, 6),)),
            SegmentSpec(2, 0, torch.tensor([right, 7]), 3, 3,
                        (SegmentLossTerm(0, 7),)),
        ), root_id=0,
    )


def test_tp_plan_digest_deterministic_and_sensitive_to_label_and_execution():
    plan = _plan()
    assert digest_segment_plan(plan) == digest_segment_plan(_plan())
    assert digest_segment_plan(plan) != digest_segment_plan(_plan(right=8))
    reordered = (plan.validate_events(plan.dfs_events()))
    # Both plan token ownership and the exact event stream are TP invariants.
    assert len(digest_segment_plan(plan, reordered)) == 64
    assert digest_segment_plan(plan, reordered) == digest_segment_plan(plan)
    assert_tp_plan_agreement(digest_segment_plan(plan), tp_size=1)


def test_tp_forest_digest_sensitive_to_ppo_targets_and_row_order():
    def batch(rows):
        return {
            "input_ids": [torch.tensor(row) for row in rows],
            "response_mask": [torch.tensor([1, 1], dtype=torch.bool)
                              for _ in rows],
        }

    keys = ["alpha_trace_0", "alpha_trace_1"]
    first = build_tree_execution_plans(keys, batch([[1,2,3,4],[1,2,3,5]]))
    second = build_tree_execution_plans(keys, batch([[1,2,3,4],[1,2,3,5]]))
    changed = build_tree_execution_plans(keys, batch([[1,2,3,4],[1,2,3,6]]))
    assert digest_forest_plan(first) == digest_forest_plan(second)
    assert digest_forest_plan(first) != digest_forest_plan(changed)


def test_tp_rejects_missing_or_mismatched_native_group(monkeypatch):
    import torch.distributed as dist
    from megatron.core import parallel_state

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: object())
    with pytest.raises(RuntimeError, match="group size differs"):
        assert_tp_plan_agreement(digest_segment_plan(_plan()), tp_size=2)


def test_tp_plan_gather_catches_divergence_without_entering_forward(monkeypatch):
    import torch.distributed as dist
    from megatron.core import parallel_state

    group = object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda: group)

    def simulated_all_gather(outputs, local, *, group):
        outputs[0].copy_(local)
        outputs[1].copy_(local)
        outputs[1][0] ^= 1

    monkeypatch.setattr(dist, "all_gather", simulated_all_gather)
    with pytest.raises(RuntimeError, match="schedule mismatch"):
        assert_tp_plan_agreement(digest_segment_plan(_plan()), tp_size=2)
