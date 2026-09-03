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

from types import SimpleNamespace

import torch

from megatron.core.transformer.attention import SelfAttention
from verl.models.mcore.tpr import TPRSelfAttention, TreeAttentionContext, use_tree_attention_context
from verl.models.mcore.tpr import attention as tpr_attention


class _IdentityProjection(torch.nn.Module):
    def forward(self, tensor):
        return tensor, None


class _StubTPRSelfAttention(TPRSelfAttention):
    """Small fixture that exercises TPR orchestration without a ModuleSpec."""

    def __init__(self, query, key, value):
        torch.nn.Module.__init__(self)
        self.query = query
        self.key = key
        self.value = value
        self.layer_number = 2
        self.config = SimpleNamespace(
            attention_dropout=0.0,
            fused_single_qkv_rope=False,
            attention_output_gate=False,
            flash_decode=False,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
        self.pg_collection = SimpleNamespace(cp=None)
        self.core_attention = SimpleNamespace(softmax_scale=0.25)
        self.linear_proj = _IdentityProjection()
        self.checkpoint_core_attention = False
        self.offload_qkv_linear = False
        self.offload_core_attention = False
        self.offload_attn_proj = False
        self._yarn_concentration_factor = 1.0

    def get_query_key_value_tensors(self, hidden_states, key_value_states, *, split_qkv, output_gate):
        assert key_value_states is None
        assert split_qkv is True
        assert output_gate is False
        return self.query, self.key, self.value


def test_no_context_delegates_to_original_self_attention(monkeypatch):
    sentinel = (object(), object())
    recorded = {}

    def fake_forward(self, hidden_states, attention_mask, **kwargs):
        recorded["hidden_states"] = hidden_states
        recorded["attention_mask"] = attention_mask
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr(SelfAttention, "forward", fake_forward)
    attention = _StubTPRSelfAttention(None, None, None)
    hidden_states = torch.randn(3, 1, 16)
    attention_mask = torch.ones(3, 3, dtype=torch.bool)

    assert attention(hidden_states, attention_mask) is sentinel
    assert recorded["hidden_states"] is hidden_states
    assert recorded["attention_mask"] is attention_mask
    assert recorded["packed_seq_params"] is None


def test_tree_forward_selects_layer_kv_collects_new_kv_and_preserves_prefix_grad(monkeypatch):
    prefix_length, suffix_length = 6, 3
    query = torch.randn(suffix_length, 1, 4, 8, requires_grad=True)
    new_key = torch.randn(suffix_length, 1, 2, 8, requires_grad=True)
    new_value = torch.randn(suffix_length, 1, 2, 8, requires_grad=True)
    past_key = torch.randn(prefix_length, 1, 2, 8, requires_grad=True)
    past_value = torch.randn(prefix_length, 1, 2, 8, requires_grad=True)
    other_layer_kv = (
        torch.randn_like(past_key, requires_grad=True),
        torch.randn_like(past_value, requires_grad=True),
    )
    rope = torch.zeros(suffix_length, 1, 1, 8)
    captured = {}

    monkeypatch.setattr(tpr_attention, "apply_rotary_pos_emb", lambda tensor, *args, **kwargs: tensor)

    def fake_rectangular(query_arg, key_arg, value_arg, **kwargs):
        captured.update(query=query_arg, key=key_arg, value=value_arg, kwargs=kwargs)
        prefix_dependency = key_arg.sum() + value_arg.sum()
        return query_arg.reshape(suffix_length, 1, -1) + prefix_dependency

    monkeypatch.setattr(tpr_attention, "rectangular_causal_attention", fake_rectangular)
    attention = _StubTPRSelfAttention(query, new_key, new_value)
    context = TreeAttentionContext(
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        past_key_values={1: other_layer_kv, 2: (past_key, past_value)},
        suffix_rotary_pos_emb=rope,
    )

    with use_tree_attention_context(context):
        output, bias = attention(torch.randn(suffix_length, 1, 32), attention_mask=None)
    output.sum().backward()

    assert bias is None
    assert captured["query"] is query
    assert torch.equal(captured["key"][:prefix_length], past_key)
    assert torch.equal(captured["value"][:prefix_length], past_value)
    assert captured["kwargs"]["softmax_scale"] == 0.25
    assert context.new_key_values[2][0] is new_key
    assert context.new_key_values[2][1] is new_value
    assert past_key.grad is not None and torch.count_nonzero(past_key.grad)
    assert past_value.grad is not None and torch.count_nonzero(past_value.grad)
    assert other_layer_kv[0].grad is None
    assert other_layer_kv[1].grad is None


def test_tree_forward_with_zero_prefix_uses_only_new_kv(monkeypatch):
    suffix_length = 6
    query = torch.randn(suffix_length, 1, 4, 8, requires_grad=True)
    new_key = torch.randn(suffix_length, 1, 2, 8, requires_grad=True)
    new_value = torch.randn(suffix_length, 1, 2, 8, requires_grad=True)
    rope = torch.zeros(suffix_length, 1, 1, 8)
    captured = {}

    monkeypatch.setattr(tpr_attention, "apply_rotary_pos_emb", lambda tensor, *args, **kwargs: tensor)

    def fake_rectangular(query_arg, key_arg, value_arg, **kwargs):
        captured.update(query=query_arg, key=key_arg, value=value_arg)
        kv_dependency = key_arg.sum() + value_arg.sum()
        return query_arg.reshape(suffix_length, 1, -1) + kv_dependency

    monkeypatch.setattr(tpr_attention, "rectangular_causal_attention", fake_rectangular)
    attention = _StubTPRSelfAttention(query, new_key, new_value)
    context = TreeAttentionContext(
        prefix_length=0,
        suffix_length=suffix_length,
        suffix_rotary_pos_emb=rope,
    )

    with use_tree_attention_context(context):
        output, bias = attention(torch.randn(suffix_length, 1, 32), attention_mask=None)
    output.sum().backward()

    assert bias is None
    assert captured["query"] is query
    assert captured["key"] is new_key
    assert captured["value"] is new_value
    assert captured["key"].shape[0] == suffix_length
    assert context.new_key_values[2][0] is new_key
    assert context.new_key_values[2][1] is new_value
    assert new_key.grad is not None and torch.count_nonzero(new_key.grad)
    assert new_value.grad is not None and torch.count_nonzero(new_value.grad)


def test_tree_forward_rejects_missing_suffix_rope_before_qkv():
    attention = _StubTPRSelfAttention(None, None, None)
    context = TreeAttentionContext(prefix_length=0, suffix_length=3)

    with use_tree_attention_context(context), torch.no_grad():
        try:
            attention(torch.randn(3, 1, 16), attention_mask=None)
        except ValueError as exc:
            assert "suffix_rotary_pos_emb" in str(exc)
        else:
            raise AssertionError("missing suffix RoPE must be rejected")
