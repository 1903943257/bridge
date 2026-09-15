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

"""Opt-in compatibility checks for TPR with a real Qwen3-0.6B checkpoint."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tensordict import TensorDict
from transformers import AutoConfig

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from ..profiling.test_tpr_engine_profile_npu import (
    _ProfileFusedCausalAttention,
    _profile_reference_loss_function,
    _verify_fused_adapter_path,
)
from ..equivalence.test_tpr_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_plan,
    _reference_data,
)
from ..equivalence.test_segment_push_pop_npu import _parameter_grads
from verl.models.mcore.tpr import (
    TPR_REQUEST_KEY,
    TPRForwardBackwardRequest,
    TPRSelfAttention,
    replace_self_attention_with_tpr,
)
from verl.models.mcore.config_converter import get_hf_rope_theta, hf_to_mcore_config_dense
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_COMPAT") != "1",
    reason="Set TPR_RUN_QWEN_COMPAT=1 for the real Qwen3 compatibility test",
)

QWEN_MODEL_PATH = Path(
    os.getenv("TPR_QWEN_MODEL_PATH", "/workspace/hf_models/Qwen3-0.6B")
)
QWEN_NUM_LAYERS = 28
QWEN_VOCAB_SIZE = 151936
_PREFIX_LENGTH = 128
_SUFFIX_LENGTH = 64
_GLOBAL_GRAD_RELATIVE_L2_TOL = 2e-2
_PER_PARAMETER_GRAD_RELATIVE_L2_TOL = 5e-2
_GLOBAL_GRAD_COSINE_MIN = 0.999


def _initialize_single_rank_megatron():
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29537")
        torch.distributed.init_process_group("hccl", rank=0, world_size=1)
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )


def _validate_checkpoint_files(model_path):
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Qwen config: {config_path}")
    weight_files = tuple(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"no safetensors weights found under {model_path}")


def _load_hf_state_dict(model_path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = [path.name for path in sorted(model_path.glob("*.safetensors"))]
    state_dict = {}
    for shard_name in shard_names:
        state_dict.update(load_file(str(model_path / shard_name), device="cpu"))
    return state_dict


@torch.no_grad()
def _load_qwen3_torch_spec_weights(model, hf_config, model_path):
    state_dict = _load_hf_state_dict(model_path)
    loaded_parameter_ids = set()

    def copy_weight(target, source_name):
        if source_name not in state_dict:
            raise KeyError(f"missing Qwen checkpoint tensor: {source_name}")
        source = state_dict[source_name]
        if tuple(source.shape) != tuple(target.shape):
            raise ValueError(
                f"Qwen weight shape mismatch for {source_name}: "
                f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}"
            )
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        loaded_parameter_ids.add(id(target))

    copy_weight(model.embedding.word_embeddings.weight, "model.embed_tokens.weight")
    if not hf_config.tie_word_embeddings:
        copy_weight(model.output_layer.weight, "lm_head.weight")
    head_dim = hf_config.head_dim
    num_query_heads = hf_config.num_attention_heads
    num_query_groups = hf_config.num_key_value_heads
    queries_per_group = num_query_heads // num_query_groups

    for layer_index, layer in enumerate(model.decoder.layers):
        prefix = f"model.layers.{layer_index}"
        copy_weight(layer.input_layernorm.weight, f"{prefix}.input_layernorm.weight")

        q = state_dict[f"{prefix}.self_attn.q_proj.weight"].view(
            num_query_groups,
            queries_per_group * head_dim,
            hf_config.hidden_size,
        )
        k = state_dict[f"{prefix}.self_attn.k_proj.weight"].view(
            num_query_groups,
            head_dim,
            hf_config.hidden_size,
        )
        v = state_dict[f"{prefix}.self_attn.v_proj.weight"].view(
            num_query_groups,
            head_dim,
            hf_config.hidden_size,
        )
        qkv = torch.cat((q, k, v), dim=1).reshape(-1, hf_config.hidden_size)
        target_qkv = layer.self_attention.linear_qkv.weight
        if tuple(qkv.shape) != tuple(target_qkv.shape):
            raise ValueError(
                f"QKV shape mismatch at layer {layer_index}: "
                f"checkpoint={tuple(qkv.shape)}, model={tuple(target_qkv.shape)}"
            )
        target_qkv.copy_(qkv.to(device=target_qkv.device, dtype=target_qkv.dtype))
        loaded_parameter_ids.add(id(target_qkv))

        copy_weight(
            layer.self_attention.q_layernorm.weight,
            f"{prefix}.self_attn.q_norm.weight",
        )
        copy_weight(
            layer.self_attention.k_layernorm.weight,
            f"{prefix}.self_attn.k_norm.weight",
        )
        copy_weight(
            layer.self_attention.linear_proj.weight,
            f"{prefix}.self_attn.o_proj.weight",
        )
        copy_weight(
            layer.pre_mlp_layernorm.weight,
            f"{prefix}.post_attention_layernorm.weight",
        )

        gate = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        up = state_dict[f"{prefix}.mlp.up_proj.weight"]
        fc1 = torch.cat((gate, up), dim=0)
        target_fc1 = layer.mlp.linear_fc1.weight
        if tuple(fc1.shape) != tuple(target_fc1.shape):
            raise ValueError(
                f"MLP FC1 shape mismatch at layer {layer_index}: "
                f"checkpoint={tuple(fc1.shape)}, model={tuple(target_fc1.shape)}"
            )
        target_fc1.copy_(fc1.to(device=target_fc1.device, dtype=target_fc1.dtype))
        loaded_parameter_ids.add(id(target_fc1))
        copy_weight(layer.mlp.linear_fc2.weight, f"{prefix}.mlp.down_proj.weight")

    copy_weight(model.decoder.final_layernorm.weight, "model.norm.weight")
    missing_parameters = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) not in loaded_parameter_ids
    ]
    if missing_parameters:
        raise RuntimeError(f"Qwen torch-spec loader missed parameters: {missing_parameters}")
    del state_dict


def _replace_core_attention(original):
    def replace_core_attention(spec):
        copied = copy.deepcopy(spec)
        if isinstance(copied, ModuleSpec):
            submodules = copied.submodules
            layer_specs = getattr(submodules, "layer_specs", None)
            if layer_specs is None and hasattr(submodules, "self_attention"):
                layer_specs = [copied]
        else:
            layer_specs = getattr(copied, "layer_specs", None)
        if not layer_specs:
            raise TypeError("Qwen provider did not return TransformerLayer specs")
        for layer_spec in layer_specs:
            attention_spec = layer_spec.submodules.self_attention
            attention_spec.submodules.core_attention = _ProfileFusedCausalAttention
        return copied
    return replace_core_attention(original)


def _make_qwen_model(
    device,
    *,
    tpr,
    max_sequence_length,
    core_attention_module=None,
    model_shape=None,
):
    del max_sequence_length, model_shape
    if core_attention_module not in (None, _ProfileFusedCausalAttention):
        raise ValueError("real Qwen fixture only supports the controlled fused core attention")
    _validate_checkpoint_files(QWEN_MODEL_PATH)
    _initialize_single_rank_megatron()

    hf_config = AutoConfig.from_pretrained(str(QWEN_MODEL_PATH), trust_remote_code=True)
    config = hf_to_mcore_config_dense(
        hf_config,
        torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
    )
    spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=False,
        pp_rank=0,
    )
    spec = _replace_core_attention(spec)
    if tpr:
        spec = replace_self_attention_with_tpr(spec)

    pg_collection = SimpleNamespace(
        tp=mpu.get_tensor_model_parallel_group(),
        cp=mpu.get_context_parallel_group(),
        pp=mpu.get_pipeline_model_parallel_group(),
        embd=mpu.get_embedding_group(),
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
        position_embedding_type="rope",
        rotary_base=get_hf_rope_theta(hf_config),
        pg_collection=pg_collection,
    ).to(device=device, dtype=torch.bfloat16)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    _load_qwen3_torch_spec_weights(model, hf_config, QWEN_MODEL_PATH)
    model.train()
    return model


def _assert_qwen_architecture(model, *, tpr):
    config = model.config
    assert config.num_layers == 28
    assert config.hidden_size == 1024
    assert config.ffn_hidden_size == 3072
    assert config.num_attention_heads == 16
    assert config.num_query_groups == 8
    assert config.kv_channels == 128
    assert config.gated_linear_unit
    assert config.normalization == "RMSNorm"
    assert float(config.attention_dropout) == 0.0
    assert len(model.decoder.layers) == QWEN_NUM_LAYERS

    for layer in model.decoder.layers:
        attention = layer.self_attention
        assert isinstance(attention, TPRSelfAttention) is tpr
        assert hasattr(attention, "q_layernorm")
        assert hasattr(attention, "k_layernorm")
    assert tuple(layer.self_attention.layer_number for layer in model.decoder.layers) == tuple(
        range(1, QWEN_NUM_LAYERS + 1)
    )

    assert model.share_embeddings_and_output_weights
    embedding_weight = model.embedding.word_embeddings.weight
    assert model.shared_embedding_or_output_weight().data_ptr() == embedding_weight.data_ptr()
    assert model.output_layer.weight is None
    assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _tokens(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device) % QWEN_VOCAB_SIZE


def _assert_real_qwen_gradients_close(
    actual,
    expected,
    *,
    global_relative_l2_tol=_GLOBAL_GRAD_RELATIVE_L2_TOL,
    per_parameter_relative_l2_tol=_PER_PARAMETER_GRAD_RELATIVE_L2_TOL,
    enforce_per_parameter_relative_l2=True,
):
    assert actual.keys() == expected.keys()
    difference_square_sum = 0.0
    expected_square_sum = 0.0
    actual_square_sum = 0.0
    dot_sum = 0.0
    per_parameter = []
    for name in actual:
        actual_gradient = actual[name].float()
        expected_gradient = expected[name].float()
        difference = actual_gradient - expected_gradient
        difference_square = torch.sum(difference * difference).item()
        expected_square = torch.sum(expected_gradient * expected_gradient).item()
        actual_square = torch.sum(actual_gradient * actual_gradient).item()
        dot = torch.sum(actual_gradient * expected_gradient).item()
        difference_square_sum += difference_square
        expected_square_sum += expected_square
        actual_square_sum += actual_square
        dot_sum += dot
        relative_l2 = (difference_square / max(expected_square, 1e-24)) ** 0.5
        per_parameter.append(
            (
                relative_l2,
                name,
                difference_square**0.5,
                expected_square**0.5,
                actual_square**0.5,
                actual_gradient.numel(),
            )
        )

    global_relative_l2 = (
        difference_square_sum / max(expected_square_sum, 1e-24)
    ) ** 0.5
    global_cosine = dot_sum / max(
        (actual_square_sum * expected_square_sum) ** 0.5,
        1e-24,
    )
    worst_parameters = sorted(
        per_parameter,
        key=lambda item: item[0],
        reverse=True,
    )[:10]
    should_print = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if should_print:
        print(
            "Qwen gradient comparison: "
            f"global_relative_l2={global_relative_l2:.6g}, "
            f"global_relative_l2_tolerance={global_relative_l2_tol:.6g}, "
            f"global_cosine={global_cosine:.8f}"
        )
        for (
            relative_l2,
            name,
            difference_l2,
            expected_l2,
            actual_l2,
            numel,
        ) in worst_parameters:
            print(
                f"  gradient relative_l2={relative_l2:.6g}, "
                f"difference_l2={difference_l2:.6g}, "
                f"expected_l2={expected_l2:.6g}, actual_l2={actual_l2:.6g}, "
                f"numel={numel}: {name}"
            )

    assert global_relative_l2 <= global_relative_l2_tol, (
        f"Qwen gradient global relative L2 {global_relative_l2:.6g} exceeds "
        f"{global_relative_l2_tol:.6g}"
    )
    assert global_cosine >= _GLOBAL_GRAD_COSINE_MIN
    excessive = [
        (name, relative_l2, difference_l2, expected_l2)
        for relative_l2, name, difference_l2, expected_l2, _, _ in per_parameter
        if relative_l2 > per_parameter_relative_l2_tol
    ]
    if enforce_per_parameter_relative_l2:
        assert not excessive, (
            f"per-parameter Qwen gradient relative L2 exceeds "
            f"{per_parameter_relative_l2_tol}: {excessive[:10]}"
        )
    elif excessive and should_print:
        print(
            "Qwen per-parameter gradient diagnostic outliers "
            f"(non-blocking, tolerance={per_parameter_relative_l2_tol}): "
            f"{excessive[:10]}"
        )


def test_qwen3_real_checkpoint_matches_reference_and_tpr(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    hf_config = AutoConfig.from_pretrained(str(QWEN_MODEL_PATH), trust_remote_code=True)
    assert hf_config.model_type == "qwen3"
    assert hf_config.num_hidden_layers == 28
    assert hf_config.hidden_size == 1024
    assert hf_config.intermediate_size == 3072
    assert hf_config.num_attention_heads == 16
    assert hf_config.num_key_value_heads == 8
    assert hf_config.head_dim == 128
    assert hf_config.vocab_size == QWEN_VOCAB_SIZE
    assert hf_config.tie_word_embeddings
    assert get_hf_rope_theta(hf_config) == 1_000_000

    full_length = _PREFIX_LENGTH + _SUFFIX_LENGTH
    reference_model = _make_qwen_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    tpr_model = _make_qwen_model(
        device,
        tpr=True,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    _assert_qwen_architecture(reference_model, tpr=False)
    _assert_qwen_architecture(tpr_model, tpr=True)
    assert reference_model.state_dict().keys() == tpr_model.state_dict().keys()

    _configure_model_runtime(reference_model)
    _configure_model_runtime(tpr_model)
    reference_engine = _make_engine(reference_model, tpr_enabled=False, monkeypatch=monkeypatch)
    tpr_engine = _make_engine(tpr_model, tpr_enabled=True, monkeypatch=monkeypatch)
    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffixes = (_tokens(700, _SUFFIX_LENGTH, device), _tokens(1300, _SUFFIX_LENGTH, device))

    # TPRSelfAttention must remain a transparent replacement without an active
    # TPRAttentionContext.
    full_tokens = torch.cat((prefix, suffixes[0])).unsqueeze(0)
    positions = torch.arange(full_length, device=device).unsqueeze(0)
    causal_mask = torch.triu(
        torch.ones((1, 1, full_length, full_length), dtype=torch.bool, device=device),
        diagonal=1,
    )
    with torch.no_grad():
        reference_logits = reference_model(full_tokens, positions, causal_mask)
        tpr_normal_logits = tpr_model(full_tokens, positions, causal_mask)
    torch.testing.assert_close(tpr_normal_logits, reference_logits, atol=2e-2, rtol=2e-2)

    observed_microbatches = []
    _install_reference_forward(reference_engine, observed_microbatches, monkeypatch)
    reference_data = _reference_data(
        *(torch.cat((prefix, suffix)) for suffix in suffixes)
    )

    def run_reference():
        return reference_engine.forward_backward_batch(
            reference_data,
            loss_function=_profile_reference_loss_function,
            forward_only=False,
        )

    reference_model.zero_grad(set_to_none=True)
    reference_calls, _ = _verify_fused_adapter_path(
        run_reference,
        lambda: reference_model.zero_grad(set_to_none=True),
        expected_path="reference",
    )
    reference_model.zero_grad(set_to_none=True)
    reference_output = run_reference()
    reference_gradients = _parameter_grads(reference_model)
    assert observed_microbatches == [1, 1, 1, 1]
    assert reference_calls == 2 * QWEN_NUM_LAYERS

    tpr_data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        tpr_data,
        **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(_make_plan(prefix, *suffixes))},
    )

    def run_tpr():
        return tpr_engine.forward_backward_batch(tpr_data, loss_function=None, forward_only=False)

    tpr_model.zero_grad(set_to_none=True)
    tpr_calls, _ = _verify_fused_adapter_path(
        run_tpr,
        lambda: tpr_model.zero_grad(set_to_none=True),
        expected_path="tpr",
    )
    tpr_model.zero_grad(set_to_none=True)
    tpr_output = run_tpr()
    tpr_gradients = _parameter_grads(tpr_model)
    reference_loss = sum(reference_output["loss"])
    torch.testing.assert_close(
        torch.tensor(tpr_output["loss"]),
        torch.tensor(reference_loss),
        atol=2e-2,
        rtol=2e-2,
    )
    _assert_real_qwen_gradients_close(tpr_gradients, reference_gradients)
    assert tpr_calls == 4 * QWEN_NUM_LAYERS
    assert tpr_output["metrics"]["tpr_direct_leaf_count"] == 2
