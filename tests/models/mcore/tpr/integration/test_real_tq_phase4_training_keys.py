"""Phase-4 real TQ identity and Forest routing input contract (no toy data).

The TQ dump predates actor PPO preprocessing. It can validate keys, topology,
node scheduling and loss ownership, but not model PPO gradients until advantage
and old-policy fields are present in the actor update batch.
"""

import os
from pathlib import Path

import pytest
import torch

from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
from verl.utils import tensordict_utils as tu

_DEFAULT = Path(
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt"
)


def test_real_tq_keys_survive_engine_side_plan_building():
    path = Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT)))
    if not path.is_file():
        pytest.skip(f"real trajectory fixture not mounted: {path}")
    dump = torch.load(path, weights_only=False, map_location="cpu")
    keys = tuple(dump["keys"])
    batch = dump["tensordict"].clone()
    tu.assign_non_tensor(batch, tpr_trajectory_keys=keys)

    engine_keys = _trajectory_keys_from_minibatch(batch)
    assert engine_keys == keys
    forest = build_tree_execution_plans(engine_keys, batch)
    assert len(forest.trees) == 1
    assert forest.segment_count == 15
    assert set(forest.trees[0].tree.member_rows) == set(range(8))
    assert forest.logical_loss_tokens > 0

    print(
        f"PHASE4 REAL TQ INPUT CONTRACT: PASS "
        f"rows={len(keys)} forest_trees={len(forest.trees)} "
        f"segments={forest.segment_count} logical_loss_tokens={forest.logical_loss_tokens}"
    )
