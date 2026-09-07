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

from verl.models.mcore.tpr import (
    KVPrefixState,
    PrefixShard,
    PrefixState,
    PrefixStateStack,
    SegmentLossTerm,
    SegmentSpec,
)


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


def test_full_and_cp2_contiguous_prefix_shards():
    full = PrefixShard.full(1024)
    rank0 = PrefixShard.contiguous(1024, cp_rank=0, cp_size=2)
    rank1 = PrefixShard.contiguous(1024, cp_rank=1, cp_size=2)

    assert full.is_full and full.local_length == 1024
    assert (rank0.local_start, rank0.local_end, rank0.local_length) == (0, 512, 512)
    assert (rank1.local_start, rank1.local_end, rank1.local_length) == (512, 1024, 512)


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: PrefixShard.contiguous(1025, cp_rank=0, cp_size=2), "divisible"),
        (lambda: PrefixShard.contiguous(1024, cp_rank=2, cp_size=2), "cp_rank"),
        (lambda: PrefixShard(1024, 1, 513, cp_rank=0, cp_size=2), "range mismatch"),
    ],
)
def test_invalid_prefix_shards_are_rejected(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_kv_prefix_state_satisfies_protocol_and_owns_local_shard():
    shard = PrefixShard.contiguous(1024, cp_rank=1, cp_size=2)
    state = KVPrefixState(7, 1024, _kv(512), shard=shard)

    assert isinstance(state, PrefixState)
    assert state.state_kind == "kv"
    assert state.segment_id == 7
    assert state.global_length == 1024
    assert state.local_length == 512
    assert state.layer_numbers == (1, 2)
    assert not state.released
    state.validate()


def test_kv_prefix_state_compacts_large_backing_storage():
    backing_key = torch.arange(1024 * 8, dtype=torch.float32).reshape(1024, 1, 2, 4)
    backing_value = backing_key + 1
    key_view = backing_key[:512]
    value_view = backing_value[:512]
    state = KVPrefixState(
        0,
        1024,
        {1: (key_view, value_view)},
        shard=PrefixShard.contiguous(1024, cp_rank=0, cp_size=2),
    )
    cached_key, cached_value = state.key_values[1]

    assert cached_key.untyped_storage().nbytes() == cached_key.numel() * cached_key.element_size()
    assert cached_value.untyped_storage().nbytes() == cached_value.numel() * cached_value.element_size()
    assert cached_key.untyped_storage().data_ptr() != backing_key.untyped_storage().data_ptr()
    assert cached_value.untyped_storage().data_ptr() != backing_value.untyped_storage().data_ptr()


def test_anchor_gradients_accumulate_across_branches_and_are_consumed_once():
    state = KVPrefixState(0, 1024, _kv(1024))

    for factor in (2.0, 3.0):
        anchors = state.make_anchors()
        for key, value in anchors.key_values.values():
            (key * factor).sum().backward(retain_graph=True)
            (value * factor).sum().backward()
        state.accumulate_anchor_gradients(anchors)

    gradients = state.consume_gradients()
    assert not state.gradients
    for key_grad, value_grad in gradients.values():
        torch.testing.assert_close(key_grad, torch.full_like(key_grad, 5.0))
        torch.testing.assert_close(value_grad, torch.full_like(value_grad, 5.0))
    with pytest.raises(RuntimeError, match="stale"):
        state.accumulate_anchor_gradients(anchors)
    assert not state.consume_gradients()


def test_missing_anchor_gradient_and_incomplete_gradient_set_are_rejected():
    state = KVPrefixState(0, 4, _kv(4))
    anchors = state.make_anchors()
    with pytest.raises(RuntimeError, match="gradient is missing"):
        state.accumulate_anchor_gradients(anchors)

    key, value = state.key_values[1]
    state.accumulate_layer_gradients(1, torch.ones_like(key), torch.ones_like(value))
    with pytest.raises(RuntimeError, match="gradient layers"):
        state.consume_gradients()


def test_release_clears_payload_and_rejects_future_use():
    state = KVPrefixState(0, 4, _kv(4))
    key, value = state.key_values[1]
    state.accumulate_layer_gradients(1, torch.ones_like(key), torch.ones_like(value))
    state.release()
    state.release()

    assert state.released
    assert state.layer_numbers == (1, 2)
    for operation in (
        state.validate,
        lambda: state.key_values,
        lambda: state.gradients,
        state.make_anchors,
        state.consume_gradients,
    ):
        with pytest.raises(RuntimeError, match="released"):
            operation()


def test_prefix_state_stack_validates_topology_with_cp_shards():
    stack = PrefixStateStack()
    root = _segment(0, None, 1024, 0)
    child = _segment(1, 0, 512, 1024)
    root_state = KVPrefixState(
        0,
        1024,
        _kv(512),
        shard=PrefixShard.contiguous(1024, cp_rank=0, cp_size=2),
    )
    child_state = KVPrefixState(
        1,
        512,
        _kv(256),
        shard=PrefixShard.contiguous(512, cp_rank=0, cp_size=2),
    )

    root_entry = stack.push_state(root, root_state)
    child_entry = stack.push_state(child, child_state)
    assert stack.segment_ids == (0, 1)
    assert stack.prefix_length == 1536
    assert stack.top() is child_entry
    assert stack.pop_state(1) is child_entry
    assert stack.pop_state(0) is root_entry
    stack.assert_empty()


def test_prefix_state_stack_rejects_mismatched_or_released_state():
    root = _segment(0, None, 4, 0)
    stack = PrefixStateStack()
    wrong_id = KVPrefixState(1, 4, _kv(4))
    with pytest.raises(ValueError, match="segment_id"):
        stack.push_state(root, wrong_id)

    released = KVPrefixState(0, 4, _kv(4))
    released.release()
    with pytest.raises(RuntimeError, match="released"):
        stack.push_state(root, released)


@pytest.mark.parametrize(
    "state_factory,match",
    [
        (lambda: KVPrefixState(0, 1024, _kv(1024, requires_grad=True)), "graph-free"),
        (
            lambda: KVPrefixState(
                0,
                1024,
                _kv(1024),
                shard=PrefixShard.contiguous(1024, cp_rank=0, cp_size=2),
            ),
            "local sequence length",
        ),
    ],
)
def test_invalid_kv_prefix_state_is_rejected(state_factory, match):
    with pytest.raises(ValueError, match=match):
        state_factory()
