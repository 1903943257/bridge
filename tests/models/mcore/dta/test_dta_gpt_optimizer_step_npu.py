# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from verl.models.mcore.dta import (
    TreeAttentionContext,
    build_suffix_rotary_pos_emb,
    use_tree_attention_context,
)

from test_dta_gpt_model_equivalence_npu import (
    _VOCAB_SIZE,
    _causal_mask,
    _install_single_rank_test_runtime,
    _make_model,
    _position_ids,
)

_LEARNING_RATE = 5e-2
_UPDATE_RELATIVE_L2_TOL = 5e-2
_UPDATE_MAX_ABS_RATIO_TOL = 1e-1


def _parameter_snapshot(model):
    return {name: parameter.detach().float().clone() for name, parameter in model.named_parameters()}


def _normalized_backward(logits, upstream_gradient):
    # Keep gradient magnitudes representative of a mean-reduced token loss.
    token_count = upstream_gradient.shape[0] * upstream_gradient.shape[1]
    logits.backward(upstream_gradient / token_count)


def _full_step(model, optimizer, input_ids, prefix_length, suffix_length, upstream_gradient):
    full_length = prefix_length + suffix_length
    logits = model(
        input_ids=input_ids,
        position_ids=_position_ids(0, full_length, input_ids.device),
        attention_mask=_causal_mask(full_length, input_ids.device),
    )
    _normalized_backward(logits[:, prefix_length:, :], upstream_gradient)
    optimizer.step()


def _external_kv_step(model, optimizer, input_ids, prefix_length, suffix_length, upstream_gradient):
    past_key_values = {}
    if prefix_length:
        prefix_context = TreeAttentionContext(
            prefix_length=0,
            suffix_length=prefix_length,
            suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
                model.rotary_pos_emb, prefix_length=0, suffix_length=prefix_length
            ),
        )
        with use_tree_attention_context(prefix_context):
            model(
                input_ids=input_ids[:, :prefix_length],
                position_ids=_position_ids(0, prefix_length, input_ids.device),
                attention_mask=None,
            )
        prefix_context.assert_new_kv_layers([1, 2])
        past_key_values = prefix_context.new_key_values

    suffix_context = TreeAttentionContext(
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        past_key_values=past_key_values,
        suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
            model.rotary_pos_emb,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        ),
    )
    with use_tree_attention_context(suffix_context):
        suffix_logits = model(
            input_ids=input_ids[:, prefix_length:],
            position_ids=_position_ids(prefix_length, suffix_length, input_ids.device),
            attention_mask=None,
        )
    suffix_context.assert_new_kv_layers([1, 2])
    _normalized_backward(suffix_logits, upstream_gradient)
    optimizer.step()


def _assert_update_close(actual, expected, *, name):
    difference = actual - expected
    max_abs = difference.abs().max().item()
    expected_max_abs = expected.abs().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    ).item()
    max_abs_ratio = max_abs / max(expected_max_abs, 1e-12)
    assert relative_l2 <= _UPDATE_RELATIVE_L2_TOL, (
        f"parameter update {name} relative-L2 mismatch: {relative_l2:.6g}"
    )
    assert max_abs_ratio <= _UPDATE_MAX_ABS_RATIO_TOL, (
        f"parameter update {name} max-abs ratio mismatch: {max_abs_ratio:.6g}"
    )


@pytest.mark.parametrize(("prefix_length", "suffix_length"), [(0, 32), (256, 32)])
def test_tiny_gpt_single_optimizer_step_matches_external_kv(
    prefix_length, suffix_length, monkeypatch
):
    device = torch.device("npu")
    dtype = torch.bfloat16
    _install_single_rank_test_runtime(monkeypatch, device)

    torch.manual_seed(2026)
    reference_model = _make_model(device, dtype, dta=True)
    candidate_model = _make_model(device, dtype, dta=True)
    candidate_model.load_state_dict(reference_model.state_dict(), strict=True)
    initial_parameters = _parameter_snapshot(reference_model)

    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=_LEARNING_RATE)
    candidate_optimizer = torch.optim.SGD(candidate_model.parameters(), lr=_LEARNING_RATE)
    full_length = prefix_length + suffix_length
    input_ids = torch.arange(17, 17 + full_length, dtype=torch.long, device=device).unsqueeze(0)
    torch.manual_seed(2027)
    upstream_gradient = torch.randn(
        1, suffix_length, _VOCAB_SIZE, device=device, dtype=dtype
    )

    _full_step(
        reference_model,
        reference_optimizer,
        input_ids,
        prefix_length,
        suffix_length,
        upstream_gradient,
    )
    _external_kv_step(
        candidate_model,
        candidate_optimizer,
        input_ids,
        prefix_length,
        suffix_length,
        upstream_gradient,
    )

    reference_parameters = _parameter_snapshot(reference_model)
    candidate_parameters = _parameter_snapshot(candidate_model)
    assert reference_parameters.keys() == candidate_parameters.keys() == initial_parameters.keys()
    changed_parameter_count = 0
    for name, initial in initial_parameters.items():
        reference_update = reference_parameters[name] - initial
        candidate_update = candidate_parameters[name] - initial
        if torch.count_nonzero(reference_update).item() == 0:
            assert torch.count_nonzero(candidate_update).item() == 0, name
            continue
        changed_parameter_count += 1
        _assert_update_close(candidate_update, reference_update, name=name)
    assert changed_parameter_count > 0
