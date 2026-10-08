import os
from pathlib import Path

import pytest
import torch

from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


_DEFAULT_TQ_DUMP = Path(
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt"
)


def _rows(tensor):
    return list(tensor) if isinstance(tensor, (tuple, list)) else list(tensor.unbind())


def test_real_tq_objective_refs_exactly_cover_response_mask():
    path = Path(os.getenv("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ_DUMP)))
    if not path.is_file():
        pytest.skip(f"real TQ dump not present: {path}")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(dump["keys"])
    batch = dump["tensordict"]
    forest = build_tree_execution_plans(keys, batch)

    input_rows = _rows(batch["input_ids"])
    response_rows = _rows(batch["response_mask"])
    expected = {
        (row, i)
        for row, mask in enumerate(response_rows)
        for i in torch.nonzero(mask.to(bool)).flatten().tolist()
    }
    actual = {
        (ref.sample_row, ref.response_offset)
        for tree in forest.trees
        for ref in tree.objective_refs
    }
    assert len(actual) == forest.logical_loss_tokens
    assert actual == expected

    cross_segment = 0
    for tree in forest.trees:
        for ref in tree.objective_refs:
            segment = tree.segment_plan.get(ref.segment_id)
            absolute_query = segment.position_start + ref.query_offset
            row = ref.sample_row
            target_position = len(input_rows[row]) - len(response_rows[row]) + ref.response_offset
            assert absolute_query == target_position - 1
            assert ref.target_token_id == int(input_rows[row][target_position])
            if absolute_query + 1 == segment.position_end:
                cross_segment += 1

    assert len(forest.trees) == 1
    assert forest.segment_count == 15
    print()
    print(f"real dump: {path}")
    print(f"rows: {len(keys)}")
    print(f"trees: {len(forest.trees)}")
    print(f"segments: {forest.segment_count}")
    print(f"logical PPO response tokens: {forest.logical_loss_tokens}")
    # Native Megatron's denominator uses loss_mask; vanilla PPO uses
    # response_mask. Surface mismatches before turning on e2e training.
    if "loss_mask" in batch:
        loss_mask_tokens = sum(
            int(row.to(torch.bool).sum().item()) for row in _rows(batch["loss_mask"])
        )
        print(f"loss_mask tokens (Megatron denominator): {loss_mask_tokens}")
        print(f"response_mask tokens (PPO objective): {forest.logical_loss_tokens}")
        if loss_mask_tokens != forest.logical_loss_tokens:
            print("WARNING: loss_mask and response_mask token counts differ; "
                  "inspect global PPO normalization before e2e training")
    print(f"objective refs at segment boundaries: {cross_segment}")
    print("REAL TQ OBJECTIVE PLAN CHECK: PASS")
