# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TPR-only long-context smoke test.

The ordinary Megatron DotProductAttention reference materializes a square
attention matrix and is known to exceed the target NPU memory at this shape.
It is deliberately not executed in this process: an NPU OOM/AICore failure can
poison the device context and invalidate the subsequent TPR run.
"""

import torch
import torch.nn.functional as F

from verl.models.mcore.tpr import (
    TreeAttentionContext,
    build_suffix_rotary_pos_emb,
    use_tree_attention_context,
)

from ..equivalence.test_tpr_gpt_model_equivalence_npu import (
    _VOCAB_SIZE,
    _all_parameter_grads,
    _assert_finite,
    _install_single_rank_test_runtime,
    _make_model,
    _position_ids,
)

_PREFIX_LENGTH = 16384
_SUFFIX_LENGTH = 8192
_EXPECTED_LAYERS = [1, 2]


def _assert_kv_is_finite(context, expected_length, phase):
    context.assert_new_kv_layers(_EXPECTED_LAYERS)
    for layer_number, (key, value) in context.new_key_values.items():
        expected_shape = (expected_length, 1, 2, 32)
        assert key.shape == expected_shape, f"{phase} layer {layer_number} K: {key.shape}"
        assert value.shape == expected_shape, f"{phase} layer {layer_number} V: {value.shape}"
        _assert_finite(key, f"{phase} layer {layer_number} new K")
        _assert_finite(value, f"{phase} layer {layer_number} new V")


def test_tpr_16k_prefix_8k_suffix_forward_backward_is_finite(monkeypatch):
    device = torch.device("npu")
    dtype = torch.bfloat16
    _install_single_rank_test_runtime(monkeypatch, device)
    full_length = _PREFIX_LENGTH + _SUFFIX_LENGTH

    torch.manual_seed(2026)
    model = _make_model(
        device,
        dtype,
        tpr=True,
        max_sequence_length=full_length,
    )
    input_ids = (
        torch.arange(17, 17 + full_length, dtype=torch.long, device=device) % _VOCAB_SIZE
    ).unsqueeze(0)

    prefix_context = TreeAttentionContext(
        prefix_length=0,
        suffix_length=_PREFIX_LENGTH,
        suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
            model.rotary_pos_emb,
            prefix_length=0,
            suffix_length=_PREFIX_LENGTH,
        ),
    )
    with use_tree_attention_context(prefix_context):
        prefix_logits = model(
            input_ids=input_ids[:, :_PREFIX_LENGTH],
            position_ids=_position_ids(0, _PREFIX_LENGTH, device),
            attention_mask=None,
        )
    _assert_finite(prefix_logits, "TPR 16K prefix logits")
    _assert_kv_is_finite(prefix_context, _PREFIX_LENGTH, "prefix")

    past_key_values = prefix_context.new_key_values
    retained_past = {}
    for layer_number, (past_key, past_value) in past_key_values.items():
        past_key.retain_grad()
        past_value.retain_grad()
        retained_past[layer_number] = (past_key, past_value)
    del prefix_logits

    suffix_context = TreeAttentionContext(
        prefix_length=_PREFIX_LENGTH,
        suffix_length=_SUFFIX_LENGTH,
        past_key_values=past_key_values,
        suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
            model.rotary_pos_emb,
            prefix_length=_PREFIX_LENGTH,
            suffix_length=_SUFFIX_LENGTH,
        ),
    )
    with use_tree_attention_context(suffix_context):
        suffix_logits = model(
            input_ids=input_ids[:, _PREFIX_LENGTH:],
            position_ids=_position_ids(_PREFIX_LENGTH, _SUFFIX_LENGTH, device),
            attention_mask=None,
        )
    assert suffix_logits.shape == (1, _SUFFIX_LENGTH, _VOCAB_SIZE)
    _assert_finite(suffix_logits, "TPR 8K suffix logits")
    _assert_kv_is_finite(suffix_context, _SUFFIX_LENGTH, "suffix")

    # A token-mean cross-entropy gives realistic, bounded LM gradients while
    # exercising the entire suffix and its dependency on the external prefix.
    labels = input_ids[:, _PREFIX_LENGTH:].reshape(-1)
    loss = F.cross_entropy(suffix_logits.float().reshape(-1, _VOCAB_SIZE), labels)
    _assert_finite(loss, "TPR long-context loss")
    loss.backward()

    for layer_number, (past_key, past_value) in retained_past.items():
        assert past_key.grad is not None, f"layer {layer_number} past K gradient is None"
        assert past_value.grad is not None, f"layer {layer_number} past V gradient is None"
        _assert_finite(past_key.grad, f"layer {layer_number} past K gradient")
        _assert_finite(past_value.grad, f"layer {layer_number} past V gradient")
        assert torch.count_nonzero(past_key.grad).item() > 0, (
            f"layer {layer_number} past K gradient is all zero"
        )
        assert torch.count_nonzero(past_value.grad).item() > 0, (
            f"layer {layer_number} past V gradient is all zero"
        )

    parameter_grads = _all_parameter_grads(model)
    assert parameter_grads
