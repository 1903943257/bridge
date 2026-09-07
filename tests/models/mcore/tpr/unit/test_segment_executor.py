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

import pytest
import torch
from torch import nn

from verl.models.mcore.tpr import (
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
    get_tpr_attention_context,
)
from verl.models.mcore.tpr.segment_executor import _compact_kv_cache


class _FakeRotaryEmbedding:
    def __call__(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        del offset, packed_seq, cp_group
        return torch.zeros(max_seq_len, 1, 1, 2)


class _FakeTPRModel(nn.Module):
    def __init__(self, *, fail=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.rotary_pos_emb = _FakeRotaryEmbedding()
        self.fail = fail

    def forward(self, *, input_ids, position_ids, attention_mask):
        del position_ids, attention_mask
        if self.fail:
            raise RuntimeError("injected forward failure")
        context = get_tpr_attention_context()
        assert context is not None
        token_signal = input_ids.transpose(0, 1).to(self.scale.dtype).view(-1, 1, 1, 1) * self.scale
        past_signal = self.scale.new_zeros(())
        for layer_number in (1, 2):
            past = context.get_past_kv(layer_number)
            if past is not None:
                past_signal = past_signal + past[0].sum() * (0.001 * layer_number)
                past_signal = past_signal + past[1].sum() * (0.002 * layer_number)
            context.set_new_kv(
                layer_number,
                token_signal * layer_number,
                token_signal * (layer_number + 2),
            )
        class_scale = torch.arange(8, device=input_ids.device, dtype=self.scale.dtype)
        logits = token_signal.view(1, -1, 1) * class_scale.view(1, 1, -1)
        return logits + past_signal * class_scale.view(1, 1, -1)


def _plan():
    root = SegmentSpec(
        0,
        None,
        torch.tensor([1, 2, 3, 4]),
        position_start=0,
        prefix_length=0,
        loss_terms=(
            SegmentLossTerm(0, 2, weight=2.0),
            SegmentLossTerm(3, 5, weight=1.0),
        ),
    )
    child = SegmentSpec(
        1,
        0,
        torch.tensor([5, 6, 7]),
        position_start=4,
        prefix_length=4,
        loss_terms=(SegmentLossTerm(0, 6, weight=3.0),),
    )
    return SegmentPlan([root, child], root_id=0)


def test_compact_kv_cache_owns_exact_graph_free_storage():
    source = torch.arange(4 * 1 * 1 * 8, dtype=torch.float32).reshape(4, 1, 1, 8)
    source.requires_grad_(True)
    key_view = source[..., 1:3]
    value_view = source[..., 5:7]

    compact = _compact_kv_cache({1: (key_view, value_view)})
    key, value = compact[1]

    torch.testing.assert_close(key, key_view)
    torch.testing.assert_close(value, value_view)
    assert key.is_contiguous() and value.is_contiguous()
    assert not key.requires_grad and key.grad_fn is None
    assert not value.requires_grad and value.grad_fn is None
    assert key.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_push_pop_relays_child_kv_gradients_and_empties_stack():
    model = _FakeTPRModel()
    executor = SegmentExecutor(model, _plan(), expected_layer_numbers=(1, 2))

    root_forward = executor.push(0)
    assert root_forward.prefix_length == 0
    assert root_forward.suffix_length == 4
    for key, value in executor.kv_stack.top().kv.key_values.values():
        assert not key.requires_grad and key.grad_fn is None
        assert not value.requires_grad and value.grad_fn is None

    executor.push(1)
    child_backward = executor.pop(1)
    assert child_backward.loss_term_count == 1
    assert child_backward.relayed_layer_count == 0
    assert executor.kv_stack.segment_ids == (0,)
    for key_grad, value_grad in executor.kv_stack.get_new_kv_gradients(0).values():
        assert torch.count_nonzero(key_grad).item() > 0
        assert torch.count_nonzero(value_grad).item() > 0

    grad_after_child = model.scale.grad.detach().clone()
    root_backward = executor.pop(0)
    assert root_backward.loss_term_count == 2
    assert root_backward.relayed_layer_count == 2
    assert not torch.equal(model.scale.grad, grad_after_child)
    executor.kv_stack.assert_empty()


def test_loss_is_normalized_once_by_plan_total_weight():
    executor = SegmentExecutor(_FakeTPRModel(), _plan(), expected_layer_numbers=(1, 2))
    executor.push(0)
    executor.push(1)
    result = executor.pop(1)

    torch.testing.assert_close(result.normalized_loss, result.loss_sum / 6.0)


def test_loss_scale_hook_scales_gradients_without_scaling_reported_loss():
    baseline_model = _FakeTPRModel()
    baseline = SegmentExecutor(baseline_model, _plan(), expected_layer_numbers=(1, 2))
    baseline.push(0)
    baseline.push(1)
    baseline_result = baseline.pop(1)
    baseline_grad = baseline_model.scale.grad.detach().clone()

    scaled_model = _FakeTPRModel()
    scaled = SegmentExecutor(
        scaled_model,
        _plan(),
        expected_layer_numbers=(1, 2),
        loss_scale_func=lambda loss: loss * 8.0,
    )
    scaled.push(0)
    scaled.push(1)
    scaled_result = scaled.pop(1)

    torch.testing.assert_close(scaled_result.normalized_loss, baseline_result.normalized_loss)
    torch.testing.assert_close(scaled_model.scale.grad, baseline_grad * 8.0)


def test_visit_leaf_matches_push_pop_and_does_not_push_leaf_kv():
    baseline_model = _FakeTPRModel()
    baseline = SegmentExecutor(baseline_model, _plan(), expected_layer_numbers=(1, 2))
    baseline.push(0)
    baseline.push(1)
    baseline_result = baseline.pop(1)
    baseline_parameter_grad = baseline_model.scale.grad.detach().clone()
    baseline_parent_grads = {
        layer: (key.clone(), value.clone())
        for layer, (key, value) in baseline.kv_stack.get_new_kv_gradients(0).items()
    }

    direct_model = _FakeTPRModel()
    direct = SegmentExecutor(direct_model, _plan(), expected_layer_numbers=(1, 2))
    direct.push(0)
    direct_result = direct.visit_leaf(1)

    assert direct.kv_stack.segment_ids == (0,)
    assert direct_result.forward.segment_id == direct_result.backward.segment_id == 1
    torch.testing.assert_close(direct_result.backward.normalized_loss, baseline_result.normalized_loss)
    torch.testing.assert_close(direct_model.scale.grad, baseline_parameter_grad)
    direct_parent_grads = direct.kv_stack.get_new_kv_gradients(0)
    assert direct_parent_grads.keys() == baseline_parent_grads.keys()
    for layer, (expected_key, expected_value) in baseline_parent_grads.items():
        actual_key, actual_value = direct_parent_grads[layer]
        torch.testing.assert_close(actual_key, expected_key)
        torch.testing.assert_close(actual_value, expected_value)


def test_visit_leaf_accumulates_sibling_parent_kv_gradients():
    base_plan = _plan()
    second_leaf = SegmentSpec(
        2,
        0,
        torch.tensor([4, 3, 2]),
        position_start=4,
        prefix_length=4,
        loss_terms=(SegmentLossTerm(0, 3, weight=3.0),),
    )
    plan = SegmentPlan([base_plan.get(0), base_plan.get(1), second_leaf], root_id=0)

    def one_leaf_parent_grads(segment_id):
        model = _FakeTPRModel()
        executor = SegmentExecutor(model, plan, expected_layer_numbers=(1, 2))
        executor.push(0)
        executor.visit_leaf(segment_id)
        return {
            layer: (key.clone(), value.clone())
            for layer, (key, value) in executor.kv_stack.get_new_kv_gradients(0).items()
        }

    first = one_leaf_parent_grads(1)
    second = one_leaf_parent_grads(2)

    combined_model = _FakeTPRModel()
    combined = SegmentExecutor(combined_model, plan, expected_layer_numbers=(1, 2))
    combined.push(0)
    combined.visit_leaf(1)
    combined.visit_leaf(2)
    accumulated = combined.kv_stack.get_new_kv_gradients(0)

    for layer, (actual_key, actual_value) in accumulated.items():
        torch.testing.assert_close(actual_key, first[layer][0] + second[layer][0])
        torch.testing.assert_close(actual_value, first[layer][1] + second[layer][1])


def test_wrong_pop_marks_executor_failed_without_popping_stack():
    executor = SegmentExecutor(_FakeTPRModel(), _plan(), expected_layer_numbers=(1, 2))
    executor.push(0)
    executor.push(1)

    with pytest.raises(RuntimeError, match="stack top"):
        executor.pop(0)
    assert executor.failed
    assert executor.kv_stack.segment_ids == (0, 1)
    with pytest.raises(RuntimeError, match="failed"):
        executor.pop(1)


def test_failed_push_does_not_write_kv_stack_and_closes_executor():
    executor = SegmentExecutor(_FakeTPRModel(fail=True), _plan(), expected_layer_numbers=(1, 2))

    with pytest.raises(RuntimeError, match="injected"):
        executor.push(0)
    executor.kv_stack.assert_empty()
    assert executor.failed
    with pytest.raises(RuntimeError, match="failed"):
        executor.push(0)
