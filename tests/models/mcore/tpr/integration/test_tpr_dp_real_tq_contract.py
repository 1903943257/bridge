# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real 8-trajectory UniAgent TQ, DP-placement-to-existing-Forest acceptance.

No synthetic model or synthetic trajectory data, no changed PPO numerics.
This is CPU planning, NOT a production Trainer DP dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from verl.models.mcore.tpr.dp_placement import plan_tpr_dta_dp
from verl.models.mcore.tpr.tree_plan_builder import _rows, build_tree_execution_plans

_REAL_TQ = Path(os.getenv(
    "TPR_REAL_TQ_BATCH",
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt",
))


def test_dp2_planner_preserves_real_tq_rows_and_existing_forest():
    if not _REAL_TQ.is_file():
        pytest.skip(f"real UniAgent TQ dump not mounted: {_REAL_TQ}")
    dump = torch.load(str(_REAL_TQ), weights_only=False, map_location="cpu")
    batch = dump["tensordict"]
    keys = tuple(dump["keys"])
    rows = _rows(batch, "input_ids")
    masks = _rows(batch, "response_mask")
    assert len(rows) == len(masks) == len(keys) == 8

    sequences = [row.detach().cpu().to(torch.long).tolist() for row in rows]
    # Different DP replicas may recompute the same UID's Prefix; splitting
    # the single 8-rollout UID is *not* a correctness bug, but affects speed.
    # The DTA minimax may legitimately choose uneven row counts, which the
    # native VERL dispatcher cannot yet accept. Keep that result OFFLINE.
    plan = plan_tpr_dta_dp(
        sequences, keys, dp_size=2, enforce_equal_rows=False
    )
    assert sorted(i for p in plan.partitions for i in p) == list(range(8))
    assert plan.global_tree_tokens > 0
    assert plan.duplicated_tree_tokens >= 0

    physical_sum = 0
    owners = set()
    for part in plan.partitions:
        inputs = [rows[i] for i in part]
        local_masks = [masks[i] for i in part]
        local_batch = {
            "input_ids": inputs, "response_mask": local_masks,
        }
        forest = build_tree_execution_plans(
            [keys[i] for i in part], local_batch
        )
        local_cost = sum(
            segment.length
            for tree in forest.trees
            for segment in tree.segment_plan.segments.values()
        )
        physical_sum += local_cost
        for tree in forest.trees:
            for ref in tree.objective_refs:
                logical = (part[ref.sample_row], ref.response_offset)
                assert logical not in owners
                owners.add(logical)
        assert len(forest.trees) > 0

    expected = {
        (idx, offset)
        for idx, mask in enumerate(masks)
        for offset in torch.nonzero(mask.to(bool)).flatten().tolist()
    }
    assert owners == expected
    assert physical_sum == sum(plan.tree_tokens_by_rank)
    assert physical_sum - plan.global_tree_tokens == plan.duplicated_tree_tokens

    if not plan.equal_rows_per_rank:
        with pytest.raises(ValueError, match="unequal per-DP row counts"):
            plan_tpr_dta_dp(
                sequences, keys, dp_size=2, enforce_equal_rows=True
            )

    print(
        "TPR_DP_REAL_TQ_PLACEMENT status=PASS "
        f"rows=8 dp=2 per_dp_rows={[len(p) for p in plan.partitions]} "
        f"tree_costs={list(plan.tree_tokens_by_rank)} "
        f"global_tree_tokens={plan.global_tree_tokens} "
        f"duplication={plan.duplicated_tree_tokens} "
        f"equal_rows={plan.equal_rows_per_rank} "
        "trainer_dispatch=NOT_WIRED",
        flush=True,
    )
