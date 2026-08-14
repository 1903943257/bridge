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

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import layers as tensor_parallel_layers
from megatron.core.tensor_parallel import mappings as tensor_parallel_mappings
from megatron.core.transformer.attention import SelfAttention
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

_VOCAB_SIZE = 2048
_MAX_SEQUENCE_LENGTH = 512
_LOGIT_ATOL = 2e-2
_LOGIT_RTOL = 2e-2
_GRAD_RELATIVE_L2_TOL = 1e-2
_GRAD_MAX_ABS_RATIO_TOL = 3e-2


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
    original_gather = tensor_parallel_layers.gather_from_tensor_model_parallel_region
    monkeypatch.setattr(
        tensor_parallel_layers,
        "gather_from_tensor_model_parallel_region",
        lambda tensor, group=None: (
            tensor if group is None or group.size() == 1 else original_gather(tensor, group)
        ),
    )


def _make_config(dtype):
    return TransformerConfig(
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


def _make_process_groups():
    process_group = _SingleProcessGroup()
    return SimpleNamespace(
        tp=process_group,
        cp=process_group,
        pp=process_group,
        embd=None,
    )


def _make_model(device, dtype, *, dta, max_sequence_length=_MAX_SEQUENCE_LENGTH):
    config = _make_config(dtype)
    original_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=False,
        pp_rank=0,
    )
    model_spec = replace_self_attention_with_dta(original_spec) if dta else original_spec
    model = GPTModel(
        config=config,
        transformer_layer_spec=model_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=max_sequence_length,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=_make_process_groups(),
    )
    model = model.to(device=device, dtype=dtype)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    # Megatron's compatibility helper returns None whenever torch.distributed is
    # uninitialized, even when an explicit process group was supplied.  This test
    # intentionally runs as a single process, so restore the rank-one test group
    # on modules which use ``tp_group.size()`` directly in forward.
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = model.pg_collection.tp
    model.train()
    expected_type = DTASelfAttention if dta else SelfAttention
    assert len(model.decoder.layers) == 2
    assert all(isinstance(layer.self_attention, expected_type) for layer in model.decoder.layers)
    return model


def _causal_mask(sequence_length, device):
    return torch.triu(
        torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool, device=device),
        diagonal=1,
    )


def _position_ids(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device).unsqueeze(0)


def _all_parameter_grads(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise AssertionError(f"missing gradient for {name}")
        _assert_finite(parameter.grad, f"parameter gradient {name}")
        gradients[name] = parameter.grad.detach().float().clone()
    if not gradients:
        raise AssertionError("GPTModel has no trainable parameter gradients")
    return gradients


def _assert_finite(tensor, label):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{label} must be a tensor, got {type(tensor)}")
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    nan_count = int(torch.isnan(tensor).sum().item())
    posinf_count = int(torch.isposinf(tensor).sum().item())
    neginf_count = int(torch.isneginf(tensor).sum().item())
    finite_values = tensor.detach()[finite].float()
    max_abs_finite = finite_values.abs().max().item() if finite_values.numel() else float("nan")
    raise AssertionError(
        f"{label} is non-finite: nan={nan_count}, +inf={posinf_count}, "
        f"-inf={neginf_count}, max_abs_finite={max_abs_finite:.6g}"
    )


def _check_finite_tree(value, label):
    if torch.is_tensor(value):
        _assert_finite(value, label)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{label}[{key!r}]")


@contextmanager
def _finite_forward_hooks(model, phase):
    modules = [("embedding", model.embedding)]
    modules.extend(
        (f"decoder.layers.{index}", layer) for index, layer in enumerate(model.decoder.layers)
    )
    final_layernorm = getattr(model.decoder, "final_layernorm", None)
    if final_layernorm is not None:
        modules.append(("decoder.final_layernorm", final_layernorm))
    modules.append(("output_layer", model.output_layer))
    handles = []
    for name, module in modules:
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, module_name=name: _check_finite_tree(
                    output, f"{phase} {module_name} output"
                )
            )
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


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


def _assert_gradient_close(actual, expected, *, label):
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    max_abs = difference.abs().max().item()
    reference_max_abs = expected_float.abs().max().item()
    relative_l2 = (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(expected_float).clamp_min(1e-12)
    ).item()
    max_abs_ratio = max_abs / max(reference_max_abs, 1e-12)
    assert relative_l2 <= _GRAD_RELATIVE_L2_TOL and max_abs_ratio <= _GRAD_MAX_ABS_RATIO_TOL, (
        f"{label} mismatch: max_abs={max_abs:.6g}, "
        f"max_abs_ratio={max_abs_ratio:.6g}, relative_l2={relative_l2:.6g}"
    )


def _assert_collected_kv(context, expected_length):
    context.assert_new_kv_layers([1, 2])
    for layer_number, (key, value) in context.new_key_values.items():
        expected_shape = (expected_length, 1, 2, 32)
        assert key.shape == expected_shape, f"layer {layer_number} K shape is {key.shape}"
        assert value.shape == expected_shape, f"layer {layer_number} V shape is {value.shape}"
        _assert_finite(key, f"layer {layer_number} new K")
        _assert_finite(value, f"layer {layer_number} new V")


@pytest.mark.parametrize(
    ("prefix_length", "suffix_length"),
    [
        (0, 64),
        (128, 32),
        (1024, 64),
        (4096, 1),
        (1024, 128),
        (2048, 128),
        (4096, 128),
        (8192, 128),
        (16384, 128),
        (16384, 1),
    ],
)
def test_tiny_gpt_model_full_vs_external_kv(prefix_length, suffix_length, monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    dtype = torch.bfloat16
    _install_single_rank_test_runtime(monkeypatch, device)
    full_length = prefix_length + suffix_length
    model = _make_model(
        device,
        dtype,
        dta=True,
        max_sequence_length=max(_MAX_SEQUENCE_LENGTH, full_length),
    )
    for name, parameter in model.named_parameters():
        _assert_finite(parameter, f"initial parameter {name}")

    # Keep long-context cases inside the vocabulary. Token uniqueness is not
    # required by this end-to-end equivalence test.
    full_input_ids = (
        torch.arange(17, 17 + full_length, dtype=torch.long, device=device) % _VOCAB_SIZE
    ).unsqueeze(0)
    assert 0 <= full_input_ids.min().item()
    assert full_input_ids.max().item() < _VOCAB_SIZE
    full_position_ids = _position_ids(0, full_length, device)
    upstream_gradient = torch.randn(
        1,
        suffix_length,
        _VOCAB_SIZE,
        device=device,
        dtype=dtype,
    )
    _assert_finite(upstream_gradient, "upstream gradient")
    with _finite_forward_hooks(model, "reference full forward"):
        reference_logits = model(
            input_ids=full_input_ids,
            position_ids=full_position_ids,
            attention_mask=_causal_mask(full_length, device),
        )
    assert reference_logits.shape == (1, full_length, _VOCAB_SIZE)
    _assert_finite(reference_logits, "reference full logits")
    reference_suffix_logits = reference_logits[:, prefix_length:, :]
    _assert_finite(reference_suffix_logits, "reference suffix logits")
    reference_suffix_logits.backward(upstream_gradient)
    reference_parameter_grads = _all_parameter_grads(model)

    model.zero_grad(set_to_none=True)
    past_key_values = {}
    retained_past = {}
    if prefix_length:
        prefix_context = TreeAttentionContext(
            prefix_length=0,
            suffix_length=prefix_length,
            suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
                model.rotary_pos_emb,
                prefix_length=0,
                suffix_length=prefix_length,
            ),
        )
        with _finite_forward_hooks(model, "DTA prefix forward"):
            with use_tree_attention_context(prefix_context):
                prefix_logits = model(
                    input_ids=full_input_ids[:, :prefix_length],
                    position_ids=_position_ids(0, prefix_length, device),
                    attention_mask=None,
                )
        assert prefix_logits.shape == (1, prefix_length, _VOCAB_SIZE)
        _assert_finite(prefix_logits, "DTA prefix logits")
        _assert_collected_kv(prefix_context, prefix_length)
        del prefix_logits
        past_key_values = prefix_context.new_key_values
        for layer_number, (past_key, past_value) in past_key_values.items():
            past_key.retain_grad()
            past_value.retain_grad()
            retained_past[layer_number] = (past_key, past_value)

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
    with _finite_forward_hooks(model, "DTA suffix forward"):
        with use_tree_attention_context(suffix_context):
            dta_suffix_logits = model(
                input_ids=full_input_ids[:, prefix_length:],
                position_ids=_position_ids(prefix_length, suffix_length, device),
                attention_mask=None,
            )
    assert dta_suffix_logits.shape == (1, suffix_length, _VOCAB_SIZE)
    _assert_finite(dta_suffix_logits, "DTA suffix logits")
    _assert_collected_kv(suffix_context, suffix_length)
    for layer_number in retained_past:
        context_key, context_value = suffix_context.get_past_kv(layer_number)
        assert context_key is not None and context_value is not None
    dta_suffix_logits.backward(upstream_gradient)
    dta_parameter_grads = _all_parameter_grads(model)

    _assert_close(
        dta_suffix_logits,
        reference_suffix_logits,
        atol=_LOGIT_ATOL,
        rtol=_LOGIT_RTOL,
        label="suffix logits",
    )
    for layer_number, (past_key, past_value) in retained_past.items():
        assert past_key.grad is not None, f"layer {layer_number} past K gradient is None"
        assert past_value.grad is not None, f"layer {layer_number} past V gradient is None"
        _assert_finite(past_key.grad, f"layer {layer_number} past K gradient")
        _assert_finite(past_value.grad, f"layer {layer_number} past V gradient")
        assert torch.count_nonzero(past_key.grad).item() > 0, (
            f"layer {layer_number} past K gradient is finite but all zero"
        )
        assert torch.count_nonzero(past_value.grad).item() > 0, (
            f"layer {layer_number} past V gradient is finite but all zero"
        )

    assert dta_parameter_grads.keys() == reference_parameter_grads.keys()
    for name in reference_parameter_grads:
        _assert_gradient_close(
            dta_parameter_grads[name],
            reference_parameter_grads[name],
            label=f"parameter gradient {name}",
        )


def test_dta_gpt_model_state_dict_is_strictly_compatible(monkeypatch):
    device = torch.device("npu")
    dtype = torch.bfloat16
    _install_single_rank_test_runtime(monkeypatch, device)
    torch.manual_seed(11)
    ordinary_model = _make_model(device, dtype, dta=False)
    torch.manual_seed(22)
    dta_model = _make_model(device, dtype, dta=True)

    ordinary_state = ordinary_model.state_dict()
    dta_state = dta_model.state_dict()
    assert ordinary_state.keys() == dta_state.keys()
    dta_model.load_state_dict(ordinary_state, strict=True)
    loaded_state = dta_model.state_dict()
    for name, ordinary_tensor in ordinary_state.items():
        loaded_tensor = loaded_state[name]
        if ordinary_tensor is None:
            assert loaded_tensor is None, name
        else:
            assert torch.equal(loaded_tensor, ordinary_tensor), name
