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
    GDNLayerState,
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)


def _kv(sequence_length: int, *, requires_grad: bool = False):
    key = torch.randn(sequence_length, 1, 2, 8, requires_grad=requires_grad)
    value = torch.randn(sequence_length, 1, 2, 8, requires_grad=requires_grad)
    return key, value


def _gdn_state(*, requires_grad: bool = False):
    return GDNLayerState(
        torch.randn(1, 12, 4, requires_grad=requires_grad),
        torch.randn(1, 4, 8, 6, requires_grad=requires_grad),
    )


def test_context_manager_sets_and_restores_context():
    outer = TPRAttentionContext(prefix_length=0, suffix_length=3)
    inner = TPRAttentionContext(prefix_length=0, suffix_length=2)

    assert get_tpr_attention_context() is None
    with use_tpr_attention_context(outer):
        assert get_tpr_attention_context() is outer
        with use_tpr_attention_context(inner):
            assert get_tpr_attention_context() is inner
        assert get_tpr_attention_context() is outer
    assert get_tpr_attention_context() is None


def test_context_manager_restores_context_after_exception():
    context = TPRAttentionContext(prefix_length=0, suffix_length=3)

    with pytest.raises(RuntimeError, match="expected failure"):
        with use_tpr_attention_context(context):
            assert get_tpr_attention_context() is context
            raise RuntimeError("expected failure")
    assert get_tpr_attention_context() is None


def test_past_kv_is_selected_by_layer_number_without_detaching():
    layer_1 = _kv(6, requires_grad=True)
    layer_2 = _kv(6, requires_grad=True)
    context = TPRAttentionContext(
        prefix_length=6,
        suffix_length=3,
        past_key_values={1: layer_1, 2: layer_2},
    )

    assert context.get_past_kv(1) is layer_1
    assert context.get_past_kv(2) is layer_2
    assert context.get_past_kv(1)[0] is layer_1[0]
    assert context.get_past_kv(1)[0].requires_grad


def test_zero_prefix_returns_no_past_kv():
    context = TPRAttentionContext(prefix_length=0, suffix_length=3)
    assert context.get_past_kv(1) is None


def test_missing_past_kv_fails_with_layer_number():
    context = TPRAttentionContext(prefix_length=6, suffix_length=3, past_key_values={1: _kv(6)})
    with pytest.raises(KeyError, match="layer 2"):
        context.get_past_kv(2)


def test_new_kv_collector_preserves_tensor_identity_and_checks_layers():
    context = TPRAttentionContext(prefix_length=6, suffix_length=3, past_key_values={1: _kv(6)})
    new_key, new_value = _kv(3, requires_grad=True)

    context.set_new_kv(1, new_key, new_value)

    assert context.new_key_values[1][0] is new_key
    assert context.new_key_values[1][1] is new_value
    assert context.new_key_values[1][0].requires_grad
    context.assert_new_kv_layers([1])


def test_duplicate_new_kv_is_rejected():
    context = TPRAttentionContext(prefix_length=0, suffix_length=3)
    new_key, new_value = _kv(3)
    context.set_new_kv(1, new_key, new_value)

    with pytest.raises(RuntimeError, match="already recorded"):
        context.set_new_kv(1, new_key, new_value)


def test_gdn_state_is_restored_and_collected_without_detaching():
    initial_state = _gdn_state(requires_grad=True)
    context = TPRAttentionContext(
        prefix_length=6,
        suffix_length=3,
        initial_gdn_states={1: initial_state},
    )
    new_state = _gdn_state(requires_grad=True)

    assert context.get_initial_gdn_state(1) is initial_state
    assert context.get_initial_gdn_state(1).conv_state.requires_grad
    context.set_new_gdn_state(1, new_state)
    assert context.new_gdn_states[1] is new_state
    context.assert_new_gdn_layers([1])


def test_gdn_state_context_enforces_root_missing_and_duplicate_contracts():
    with pytest.raises(ValueError, match="initial_gdn_states"):
        TPRAttentionContext(
            prefix_length=0,
            suffix_length=3,
            initial_gdn_states={1: _gdn_state()},
        )

    root_context = TPRAttentionContext(prefix_length=0, suffix_length=3)
    assert root_context.get_initial_gdn_state(1) is None

    suffix_context = TPRAttentionContext(
        prefix_length=6,
        suffix_length=3,
        initial_gdn_states={1: _gdn_state()},
    )
    with pytest.raises(KeyError, match="layer 2"):
        suffix_context.get_initial_gdn_state(2)
    suffix_context.set_new_gdn_state(1, _gdn_state())
    with pytest.raises(RuntimeError, match="already recorded"):
        suffix_context.set_new_gdn_state(1, _gdn_state())


@pytest.mark.parametrize(
    ("prefix_length", "suffix_length", "match"),
    [(-1, 3, "prefix_length"), (0, 0, "suffix_length"), (True, 3, "prefix_length")],
)
def test_invalid_lengths_are_rejected(prefix_length, suffix_length, match):
    with pytest.raises(ValueError, match=match):
        TPRAttentionContext(prefix_length=prefix_length, suffix_length=suffix_length)


def test_invalid_past_or_new_kv_length_is_rejected():
    with pytest.raises(ValueError, match="past KV sequence length"):
        TPRAttentionContext(prefix_length=6, suffix_length=3, past_key_values={1: _kv(5)})

    context = TPRAttentionContext(prefix_length=0, suffix_length=3)
    with pytest.raises(ValueError, match="new KV sequence length"):
        context.set_new_kv(1, *_kv(2))


def test_collector_reports_missing_and_unexpected_layers():
    context = TPRAttentionContext(prefix_length=0, suffix_length=3)
    context.set_new_kv(2, *_kv(3))

    with pytest.raises(RuntimeError, match=r"missing=\[1\], unexpected=\[2\]"):
        context.assert_new_kv_layers([1])
