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

from verl.models.mcore.tpr import GDNLayerState, GDNPrefixState


def _layer_state(*, value=1.0, requires_grad=False):
    return GDNLayerState(
        conv_state=torch.full((1, 12, 3), value, dtype=torch.float32, requires_grad=requires_grad),
        recurrent_state=torch.full(
            (1, 4, 8, 6), value + 10, dtype=torch.float32, requires_grad=requires_grad
        ),
    )


def test_save_detaches_compacts_and_restores_gdn_layer_states():
    conv_backing = torch.arange(1 * 12 * 7, dtype=torch.float32).reshape(1, 12, 7).requires_grad_()
    recurrent_backing = torch.arange(1 * 4 * 8 * 10, dtype=torch.float32).reshape(1, 4, 8, 10).requires_grad_()
    graph_state = GDNLayerState(
        conv_state=(conv_backing[..., :3] * 2),
        recurrent_state=(recurrent_backing[..., :6] * 3),
    )

    state = GDNPrefixState.save(7, 32, {3: graph_state})
    restored = state.restore(3)

    assert state.state_kind == "gdn"
    assert state.segment_id == 7
    assert state.sequence_length == 32
    assert state.layer_numbers == (3,)
    assert not state.released
    assert not restored.conv_state.requires_grad and restored.conv_state.grad_fn is None
    assert not restored.recurrent_state.requires_grad and restored.recurrent_state.grad_fn is None
    assert restored.conv_state.is_contiguous()
    assert restored.recurrent_state.is_contiguous()
    assert (
        restored.conv_state.untyped_storage().nbytes()
        == restored.conv_state.numel() * restored.conv_state.element_size()
    )
    assert (
        restored.recurrent_state.untyped_storage().nbytes()
        == restored.recurrent_state.numel() * restored.recurrent_state.element_size()
    )
    assert restored.conv_state.untyped_storage().data_ptr() != conv_backing.untyped_storage().data_ptr()
    assert restored.recurrent_state.untyped_storage().data_ptr() != recurrent_backing.untyped_storage().data_ptr()
    state.validate()


def test_anchor_gradients_accumulate_across_siblings_and_are_consumed_once():
    state = GDNPrefixState(0, 16, {1: _layer_state(value=1), 2: _layer_state(value=2)})

    for factor in (2.0, 3.0):
        anchors = state.make_anchors()
        loss = sum(
            (
                layer_state.conv_state.sum() * factor
                + layer_state.recurrent_state.sum() * factor
            )
            for layer_state in anchors.layer_states.values()
        )
        loss.backward()
        state.accumulate_anchor_gradients(anchors)

    gradients = state.consume_gradients()
    assert not state.gradients
    for gradient in gradients.values():
        torch.testing.assert_close(gradient.conv_state, torch.full_like(gradient.conv_state, 5.0))
        torch.testing.assert_close(
            gradient.recurrent_state,
            torch.full_like(gradient.recurrent_state, 5.0),
        )
    with pytest.raises(RuntimeError, match="stale"):
        state.accumulate_anchor_gradients(anchors)
    assert not state.consume_gradients()


def test_saved_state_is_reusable_by_independent_branch_anchors():
    state = GDNPrefixState(0, 8, {1: _layer_state()})
    saved = state.restore(1)
    saved_conv = saved.conv_state.clone()
    saved_recurrent = saved.recurrent_state.clone()

    first = state.make_anchors().layer_states[1]
    second = state.make_anchors().layer_states[1]

    assert first.conv_state.data_ptr() == saved.conv_state.data_ptr()
    assert second.conv_state.data_ptr() == saved.conv_state.data_ptr()
    assert first.recurrent_state.data_ptr() == saved.recurrent_state.data_ptr()
    assert second.recurrent_state.data_ptr() == saved.recurrent_state.data_ptr()
    assert first.conv_state is not second.conv_state
    assert first.recurrent_state is not second.recurrent_state
    torch.testing.assert_close(state.restore(1).conv_state, saved_conv)
    torch.testing.assert_close(state.restore(1).recurrent_state, saved_recurrent)


def test_missing_or_incomplete_gdn_gradients_are_rejected():
    state = GDNPrefixState(0, 4, {1: _layer_state(), 2: _layer_state(value=2)})
    anchors = state.make_anchors()
    anchors.layer_states[1].conv_state.sum().backward()
    with pytest.raises(RuntimeError, match="gradient is missing"):
        state.accumulate_anchor_gradients(anchors)

    layer_state = state.restore(1)
    state.accumulate_layer_gradients(
        1,
        torch.ones_like(layer_state.conv_state),
        torch.ones_like(layer_state.recurrent_state),
    )
    with pytest.raises(RuntimeError, match="gradient layers"):
        state.consume_gradients()


def test_release_clears_gdn_state_and_rejects_future_use():
    state = GDNPrefixState(0, 4, {1: _layer_state()})
    layer_state = state.restore(1)
    state.accumulate_layer_gradients(
        1,
        torch.ones_like(layer_state.conv_state),
        torch.ones_like(layer_state.recurrent_state),
    )
    state.release()
    state.release()

    assert state.released
    assert state.layer_numbers == (1,)
    for operation in (
        state.validate,
        lambda: state.layer_states,
        lambda: state.gradients,
        lambda: state.restore(1),
        state.make_anchors,
        state.consume_gradients,
    ):
        with pytest.raises(RuntimeError, match="released"):
            operation()


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: GDNPrefixState(0, 4, {1: _layer_state(requires_grad=True)}), "graph-free"),
        (
            lambda: GDNLayerState(
                torch.ones(1, 12),
                torch.ones(1, 4, 8, 6),
            ),
            "conv_state",
        ),
        (
            lambda: GDNLayerState(
                torch.ones(1, 12, 3),
                torch.ones(2, 4, 8, 6),
            ),
            "batch size",
        ),
    ],
)
def test_invalid_gdn_prefix_state_is_rejected(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()
