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

"""CP=2 penetration test for TPR Push/Visit/Pop scheduling.

Run from the verl repository root with two visible NPUs::

    torchrun --standalone --nproc_per_node=2 -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_tpr_cp_segment_push_pop_npu.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from verl.models.mcore.tpr import (
    FixedTopologyScheduler,
    PhysicalExecutionKind,
    SegmentExecutor,
)
from verl.utils.device import is_torch_npu_available

from ._tpr_cp_test_utils import (
    _EXPECTED_WORLD_SIZE,
    _FIRST_SUFFIX_LENGTH,
    _LAYER_COUNT,
    _PREFIX_LENGTH,
    _SECOND_SUFFIX_LENGTH,
    _clone_prefix_gradients,
    _collective_probe,
    _make_model,
    _plan,
    cp_runtime,
)


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


def test_cp2_scheduler_reuses_prefix_across_unequal_siblings_and_pops(cp_runtime):
    runtime = cp_runtime
    torch.manual_seed(260907)
    model = _make_model(runtime)
    plan = _plan()
    executor = SegmentExecutor(
        model,
        plan,
        expected_layer_numbers=tuple(range(1, _LAYER_COUNT + 1)),
        cp_group=runtime.cp_group,
    )
    observations = {}
    original_push = executor.push
    original_visit_leaf = executor.visit_leaf
    original_pop = executor.pop

    def push(segment_id):
        result = original_push(segment_id)
        if segment_id == 0:
            state = executor.kv_stack.top().kv
            observations["prefix_state"] = state
            observations["prefix_local_length"] = state.local_length
            observations["prefix_shard"] = state.shard
        return result

    def visit_leaf(segment_id):
        result = original_visit_leaf(segment_id)
        observations[f"grad_after_{segment_id}"] = _clone_prefix_gradients(executor)
        return result

    def pop(segment_id):
        if segment_id == 0:
            observations["grad_before_root_pop"] = _clone_prefix_gradients(executor)
        return original_pop(segment_id)

    executor.push = push
    executor.visit_leaf = visit_leaf
    executor.pop = pop

    with _collective_probe() as counts:
        result = FixedTopologyScheduler(plan, executor).run()

    assert [execution.kind for execution in result.execution_trace] == [
        PhysicalExecutionKind.PUSH,
        PhysicalExecutionKind.VISIT_LEAF,
        PhysicalExecutionKind.VISIT_LEAF,
        PhysicalExecutionKind.POP,
    ]
    assert [execution.segment_id for execution in result.execution_trace] == [0, 1, 2, 0]
    assert result.direct_leaf_count == 2
    assert result.peak_path_tokens == _PREFIX_LENGTH + _FIRST_SUFFIX_LENGTH
    assert observations["prefix_local_length"] == _PREFIX_LENGTH // _EXPECTED_WORLD_SIZE
    assert observations["prefix_shard"].cp_rank == runtime.rank
    assert observations["prefix_shard"].cp_size == _EXPECTED_WORLD_SIZE
    assert observations["prefix_state"].released
    executor.kv_stack.assert_empty()

    first_gradients = observations["grad_after_1"]
    combined_gradients = observations["grad_after_2"]
    assert first_gradients.keys() == combined_gradients.keys() == {1, 2}
    for layer_number in first_gradients:
        for first, combined in zip(first_gradients[layer_number], combined_gradients[layer_number], strict=True):
            assert torch.isfinite(first).all()
            assert torch.isfinite(combined).all()
            assert torch.count_nonzero(first).item() > 0
            assert torch.count_nonzero(combined).item() > 0
            assert not torch.equal(first, combined)
    for layer_number, pair in observations["grad_before_root_pop"].items():
        for index, gradient in enumerate(pair):
            torch.testing.assert_close(gradient, combined_gradients[layer_number][index])

    parameter_count = 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing parameter gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite parameter gradient: {name}"
        parameter_count += 1
    assert parameter_count > 0

    expected_all_gathers = _LAYER_COUNT * 12
    expected_reduce_scatters = _LAYER_COUNT * 10
    assert counts == {
        "all_gather": expected_all_gathers,
        "reduce_scatter": expected_reduce_scatters,
    }
    local_counts = torch.tensor(
        [counts["all_gather"], counts["reduce_scatter"]],
        device=runtime.device,
        dtype=torch.int64,
    )
    rank_counts = [torch.empty_like(local_counts) for _ in range(_EXPECTED_WORLD_SIZE)]
    dist.all_gather(rank_counts, local_counts, group=runtime.cp_group)
    assert all(torch.equal(item, rank_counts[0]) for item in rank_counts[1:])

    global_loss_sum = result.loss_sum.detach().clone()
    global_normalized_loss = result.normalized_loss.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(global_normalized_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    torch.testing.assert_close(
        global_normalized_loss,
        global_loss_sum / plan.total_loss_weight,
        atol=1e-6,
        rtol=1e-6,
    )
    local_loss_term_count = sum(item.loss_term_count for item in result.backward_results)
    local_loss_term_count += sum(
        item.backward.loss_term_count for item in result.direct_leaf_results
    )
    global_loss_term_count = torch.tensor(local_loss_term_count, device=runtime.device)
    dist.all_reduce(global_loss_term_count, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    expected_loss_term_count = (
        len(plan.get(0).loss_terms)
        + len(plan.get(1).loss_terms)
        + len(plan.get(2).loss_terms)
    )
    assert global_loss_term_count.item() == expected_loss_term_count

    if runtime.rank == 0:
        print(
            "\nTPR CP=2 Push/Visit/Pop penetration passed\n"
            f"  topology: P={_PREFIX_LENGTH}, S1={_FIRST_SUFFIX_LENGTH}, "
            f"S2={_SECOND_SUFFIX_LENGTH}\n"
            f"  layers: {_LAYER_COUNT}\n"
            f"  collectives/rank: AllGather={counts['all_gather']}, "
            f"ReduceScatter={counts['reduce_scatter']}\n"
            f"  global loss: {global_normalized_loss.item():.9f}\n"
            f"  parameter tensors with gradients: {parameter_count}"
        )

    dist.barrier(group=runtime.cp_group)
