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

from verl.models.mcore.dta import KVStack, SegmentLossTerm, SegmentSpec


def _segment(segment_id, parent_id, length, prefix_length):
    return SegmentSpec(
        segment_id,
        parent_id,
        torch.arange(length, dtype=torch.long),
        position_start=prefix_length,
        prefix_length=prefix_length,
        loss_terms=(SegmentLossTerm(0, 1),),
    )


def _kv(length, *, layers=(1, 2), value=1.0, requires_grad=False, dtype=torch.float32):
    return {
        layer: (
            torch.full((length, 1, 2, 4), value + layer, dtype=dtype, requires_grad=requires_grad),
            torch.full((length, 1, 2, 4), value + 10 + layer, dtype=dtype, requires_grad=requires_grad),
        )
        for layer in layers
    }


def test_single_segment_push_pop_and_empty_assertion():
    stack = KVStack()
    root = _segment(0, None, 3, 0)
    entry = stack.push(root, _kv(3))

    assert stack.top() is entry
    assert stack.segment_ids == (0,)
    assert stack.prefix_length == 3
    assert stack.pop(0) is entry
    stack.assert_empty()


def test_multilevel_past_kv_is_concatenated_in_path_order():
    stack = KVStack()
    stack.push(_segment(0, None, 1024, 0), _kv(1024, value=0))
    stack.push(_segment(1, 0, 512, 1024), _kv(512, value=100))

    past = stack.build_past_key_values()
    assert past[1][0].shape[0] == 1536
    torch.testing.assert_close(past[1][0][:1024], _kv(1024, value=0)[1][0])
    torch.testing.assert_close(past[1][0][1024:], _kv(512, value=100)[1][0])


def test_past_anchors_are_leaf_tensors_isolated_from_cached_kv():
    stack = KVStack()
    cached = _kv(2)
    stack.push(_segment(0, None, 2, 0), cached)
    anchors = stack.build_past_anchors()
    key_anchor, value_anchor = anchors.key_values[1]

    assert anchors.prefix_length == 2
    assert key_anchor.requires_grad and key_anchor.is_leaf and key_anchor.grad_fn is None
    assert value_anchor.requires_grad and value_anchor.is_leaf and value_anchor.grad_fn is None
    assert key_anchor is not cached[1][0]
    torch.testing.assert_close(key_anchor, cached[1][0])


def test_full_past_gradient_is_split_across_multilevel_segments():
    stack = KVStack()
    stack.push(_segment(0, None, 1024, 0), _kv(1024))
    stack.push(_segment(1, 0, 512, 1024), _kv(512))
    anchors = stack.build_past_anchors()

    for layer, (key, value) in anchors.key_values.items():
        factor = float(layer)
        (key * factor).sum().backward(retain_graph=True)
        (value * (factor + 10)).sum().backward()
    stack.accumulate_anchor_gradients(anchors)

    for layer in (1, 2):
        root_dk, root_dv = stack.get_new_kv_gradients(0)[layer]
        middle_dk, middle_dv = stack.get_new_kv_gradients(1)[layer]
        assert root_dk.shape[0] == 1024 and middle_dk.shape[0] == 512
        torch.testing.assert_close(root_dk, torch.full_like(root_dk, float(layer)))
        torch.testing.assert_close(middle_dk, torch.full_like(middle_dk, float(layer)))
        torch.testing.assert_close(root_dv, torch.full_like(root_dv, float(layer + 10)))
        torch.testing.assert_close(middle_dv, torch.full_like(middle_dv, float(layer + 10)))


def test_sibling_gradients_accumulate_into_shared_prefix():
    stack = KVStack()
    root = _segment(0, None, 1024, 0)
    child_a = _segment(1, 0, 512, 1024)
    child_b = _segment(2, 0, 256, 1024)
    stack.push(root, _kv(1024))

    for child, factor in ((child_a, 2.0), (child_b, 3.0)):
        stack.push(child, _kv(child.length))
        stack.pop(child.segment_id)
        anchors = stack.build_past_anchors()
        for key, value in anchors.key_values.values():
            (key * factor).sum().backward(retain_graph=True)
            (value * factor).sum().backward()
        stack.accumulate_anchor_gradients(anchors)

    for key_grad, value_grad in stack.get_new_kv_gradients(0).values():
        assert key_grad.shape[0] == 1024
        torch.testing.assert_close(key_grad, torch.full_like(key_grad, 5.0))
        torch.testing.assert_close(value_grad, torch.full_like(value_grad, 5.0))


def test_stale_anchor_and_missing_gradient_are_rejected():
    stack = KVStack()
    stack.push(_segment(0, None, 2, 0), _kv(2))
    anchors = stack.build_past_anchors()
    stack.push(_segment(1, 0, 1, 2), _kv(1))
    with pytest.raises(RuntimeError, match="stale"):
        stack.accumulate_anchor_gradients(anchors)

    stack.pop(1)
    with pytest.raises(RuntimeError, match="gradient is missing"):
        stack.accumulate_anchor_gradients(anchors)


def test_stack_rejects_wrong_parent_position_duplicate_and_non_lifo_pop():
    stack = KVStack()
    root = _segment(0, None, 2, 0)
    stack.push(root, _kv(2))
    with pytest.raises(RuntimeError, match="already"):
        stack.push(root, _kv(2))
    with pytest.raises(ValueError, match="stack prefix length"):
        stack.push(_segment(1, 0, 1, 3), _kv(1))

    stack.push(_segment(1, 0, 1, 2), _kv(1))
    with pytest.raises(RuntimeError, match="stack top"):
        stack.pop(0)


@pytest.mark.parametrize(
    "key_values,match",
    [
        (_kv(2, requires_grad=True), "graph-free"),
        (_kv(3), "sequence length"),
        (_kv(2, layers=(1,)), "KV layers"),
    ],
)
def test_invalid_cached_kv_is_rejected(key_values, match):
    stack = KVStack()
    if tuple(key_values) == (1,):
        stack.push(_segment(0, None, 1, 0), _kv(1))
        segment = _segment(1, 0, 2, 1)
    else:
        segment = _segment(0, None, 2, 0)
    with pytest.raises(ValueError, match=match):
        stack.push(segment, key_values)


def test_pop_releases_child_state_and_empty_stack_has_no_past():
    stack = KVStack()
    assert not stack.build_past_key_values()
    assert not stack.build_past_anchors().key_values
    stack.push(_segment(0, None, 2, 0), _kv(2))
    stack.push(_segment(1, 0, 1, 2), _kv(1))
    child = stack.pop(1)

    assert stack.segment_ids == (0,)
    with pytest.raises(KeyError, match="not on"):
        stack.get(child.segment.segment_id)

