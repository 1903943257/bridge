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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel import mappings as tensor_parallel_mappings
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from verl.models.mcore.dta import (
    DTASelfAttention,
    TreeAttentionContext,
    build_suffix_rotary_pos_emb,
    replace_self_attention_with_dta,
    use_tree_attention_context,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

_OUTPUT_ATOL = 1e-2
_OUTPUT_RTOL = 1e-2
_GRAD_ATOL = 2e-2
_GRAD_RTOL = 2e-2


class _SingleProcessGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _NpuGlobalMemoryBuffer:
    def __init__(self, device):
        self.device = device
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        required_length = 1
        for dimension in tensor_shape:
            required_length *= dimension
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < required_length:
            self.buffer[key] = torch.empty(
                required_length,
                dtype=dtype,
                device=self.device,
                requires_grad=False,
            )
        return self.buffer[key][:required_length].view(*tensor_shape)


class _ZeroDropoutRngTracker:
    def fork(self, *args, **kwargs):
        return nullcontext()


def _install_single_rank_test_runtime(monkeypatch, device):
    parallel_state._GLOBAL_MEMORY_BUFFER = _NpuGlobalMemoryBuffer(device)
    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", lambda: _ZeroDropoutRngTracker())
    original_reduce = tensor_parallel_mappings._reduce
    monkeypatch.setattr(
        tensor_parallel_mappings,
        "_reduce",
        lambda tensor, group: tensor if group is None else original_reduce(tensor, group),
    )


def _make_block(device, dtype):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=True,
        sequence_parallel=False,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )
    original_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=False,
        pp_rank=0,
    )
    dta_spec = replace_self_attention_with_dta(original_spec)
    assert all(
        layer.submodules.self_attention.module is not DTASelfAttention
        for layer in original_spec.layer_specs
    )

    process_group = _SingleProcessGroup()
    process_groups = SimpleNamespace(
        tp=process_group,
        cp=process_group,
        pp=process_group,
    )
    block = TransformerBlock(
        config=config,
        spec=dta_spec,
        post_layer_norm=True,
        pre_process=True,
        post_process=True,
        pg_collection=process_groups,
    ).to(device=device, dtype=dtype)
    block.train()
    assert len(block.layers) == 2
    assert all(isinstance(layer.self_attention, DTASelfAttention) for layer in block.layers)

    rotary_embedding = RotaryEmbedding(
        kv_channels=config.kv_channels,
        rotary_percent=1.0,
        rotary_interleaved=config.rotary_interleaved,
        use_cpu_initialization=True,
        cp_group=process_group,
    )
    rotary_embedding.inv_freq = rotary_embedding.inv_freq.to(device)
    return block, rotary_embedding


def _causal_mask(sequence_length, device):
    return torch.triu(
        torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def _all_parameter_grads(block):
    gradients = {}
    for name, parameter in block.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        gradients[name] = parameter.grad.detach().float().clone()
    if not gradients:
        raise AssertionError("TransformerBlock has no trainable parameter gradients")
    return gradients


def _assert_close(actual, expected, *, atol, rtol, label):
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    max_abs = difference.abs().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    ).item()
    try:
        torch.testing.assert_close(actual_float, expected_float, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(
            f"{label} mismatch: max_abs={max_abs:.6g}, relative_l2={relative_l2:.6g}"
        ) from exc


def _assert_collected_kv(context, expected_length):
    context.assert_new_kv_layers([1, 2])
    for layer_number, (key, value) in context.new_key_values.items():
        expected_shape = (expected_length, 1, 2, 32)
        assert key.shape == expected_shape, f"layer {layer_number} K shape is {key.shape}"
        assert value.shape == expected_shape, f"layer {layer_number} V shape is {value.shape}"


@pytest.mark.parametrize(
    ("prefix_length", "suffix_length"),
    [
        (0, 64),
        (128, 32),
        (1024, 64),
    ],
)
def test_two_layer_transformer_block_full_vs_external_kv(
    prefix_length,
    suffix_length,
    monkeypatch,
):
    torch.manual_seed(2026)
    device = torch.device("npu")
    dtype = torch.bfloat16
    _install_single_rank_test_runtime(monkeypatch, device)
    block, rotary_embedding = _make_block(device, dtype)
    full_length = prefix_length + suffix_length

    full_hidden = torch.randn(
        full_length,
        1,
        block.config.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    upstream_gradient = torch.randn(
        suffix_length,
        1,
        block.config.hidden_size,
        device=device,
        dtype=dtype,
    )
    full_rope = build_suffix_rotary_pos_emb(
        rotary_embedding,
        prefix_length=0,
        suffix_length=full_length,
    )
    reference_output = block(
        hidden_states=full_hidden,
        attention_mask=_causal_mask(full_length, device),
        rotary_pos_emb=full_rope,
    )
    reference_suffix_output = reference_output[prefix_length:]
    reference_suffix_output.backward(upstream_gradient)
    reference_hidden_grad = full_hidden.grad.detach().float().clone()
    reference_parameter_grads = _all_parameter_grads(block)

    block.zero_grad(set_to_none=True)
    prefix_hidden = None
    past_key_values = {}
    retained_past = {}
    if prefix_length:
        prefix_hidden = full_hidden.detach()[:prefix_length].clone().requires_grad_(True)
        prefix_rope = build_suffix_rotary_pos_emb(
            rotary_embedding,
            prefix_length=0,
            suffix_length=prefix_length,
        )
        prefix_context = TreeAttentionContext(
            prefix_length=0,
            suffix_length=prefix_length,
            suffix_rotary_pos_emb=prefix_rope,
        )
        with use_tree_attention_context(prefix_context):
            block(hidden_states=prefix_hidden, attention_mask=None)
        _assert_collected_kv(prefix_context, prefix_length)
        past_key_values = prefix_context.new_key_values
        for layer_number, (past_key, past_value) in past_key_values.items():
            past_key.retain_grad()
            past_value.retain_grad()
            retained_past[layer_number] = (past_key, past_value)

    suffix_hidden = full_hidden.detach()[prefix_length:].clone().requires_grad_(True)
    suffix_rope = build_suffix_rotary_pos_emb(
        rotary_embedding,
        prefix_length=prefix_length,
        suffix_length=suffix_length,
    )
    suffix_context = TreeAttentionContext(
        prefix_length=prefix_length,
        suffix_length=suffix_length,
        past_key_values=past_key_values,
        suffix_rotary_pos_emb=suffix_rope,
    )
    with use_tree_attention_context(suffix_context):
        dta_suffix_output = block(hidden_states=suffix_hidden, attention_mask=None)
    _assert_collected_kv(suffix_context, suffix_length)
    for layer_number, pair in retained_past.items():
        assert suffix_context.get_past_kv(layer_number) is pair
    dta_suffix_output.backward(upstream_gradient)
    dta_parameter_grads = _all_parameter_grads(block)

    _assert_close(
        dta_suffix_output,
        reference_suffix_output,
        atol=_OUTPUT_ATOL,
        rtol=_OUTPUT_RTOL,
        label="TransformerBlock suffix output",
    )
    _assert_close(
        suffix_hidden.grad,
        reference_hidden_grad[prefix_length:],
        atol=_GRAD_ATOL,
        rtol=_GRAD_RTOL,
        label="suffix hidden gradient",
    )
    if prefix_length:
        _assert_close(
            prefix_hidden.grad,
            reference_hidden_grad[:prefix_length],
            atol=_GRAD_ATOL,
            rtol=_GRAD_RTOL,
            label="prefix hidden gradient",
        )
        for layer_number, (past_key, past_value) in retained_past.items():
            assert past_key.grad is not None and past_key.grad.float().norm() > 0, (
                f"layer {layer_number} past K did not receive a gradient"
            )
            assert past_value.grad is not None and past_value.grad.float().norm() > 0, (
                f"layer {layer_number} past V did not receive a gradient"
            )

    assert dta_parameter_grads.keys() == reference_parameter_grads.keys()
    for name in reference_parameter_grads:
        _assert_close(
            dta_parameter_grads[name],
            reference_parameter_grads[name],
            atol=_GRAD_ATOL,
            rtol=_GRAD_RTOL,
            label=f"parameter gradient {name}",
        )

