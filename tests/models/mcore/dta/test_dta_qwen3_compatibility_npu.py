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

"""Opt-in compatibility checks for DTA with a real Qwen3-0.6B checkpoint."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from transformers import AutoConfig

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from test_dta_engine_profile_npu import (
    _ProfileFusedCausalAttention,
    _profile_reference_loss_function,
    _verify_fused_adapter_path,
)
from test_dta_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_plan,
    _reference_data,
)
from test_segment_push_pop_npu import _assert_gradients_close, _parameter_grads
from verl.models.mcore.dta import (
    DTA_REQUEST_KEY,
    DTAForwardBackwardRequest,
    DTASelfAttention,
    replace_self_attention_with_dta,
)
from verl.models.mcore.config_converter import hf_to_mcore_config_dense
from verl.models.mcore.mbridge import AutoBridge
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_QWEN_COMPAT") != "1",
    reason="Set DTA_RUN_QWEN_COMPAT=1 for the real Qwen3 compatibility test",
)

QWEN_MODEL_PATH = Path(
    os.getenv("DTA_QWEN_MODEL_PATH", "/workspace/hf_models/Qwen3-0.6B")
)
QWEN_NUM_LAYERS = 28
QWEN_VOCAB_SIZE = 151936
_PREFIX_LENGTH = 128
_SUFFIX_LENGTH = 64


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
    dta,
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
    if dta:
        spec = replace_self_attention_with_dta(spec)

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
        rotary_base=hf_config.rope_theta,
        pg_collection=pg_collection,
    ).to(device=device, dtype=torch.bfloat16)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    bridge = AutoBridge.from_config(hf_config, dtype=torch.bfloat16)
    bridge.load_weights([model], str(QWEN_MODEL_PATH))
    model.train()
    return model


def _assert_qwen_architecture(model, *, dta):
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
        assert isinstance(attention, DTASelfAttention) is dta
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


def test_qwen3_real_checkpoint_matches_reference_and_dta(monkeypatch):
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
    assert hf_config.rope_theta == 1_000_000

    full_length = _PREFIX_LENGTH + _SUFFIX_LENGTH
    reference_model = _make_qwen_model(
        device,
        dta=False,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    dta_model = _make_qwen_model(
        device,
        dta=True,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    _assert_qwen_architecture(reference_model, dta=False)
    _assert_qwen_architecture(dta_model, dta=True)
    assert reference_model.state_dict().keys() == dta_model.state_dict().keys()

    _configure_model_runtime(reference_model)
    _configure_model_runtime(dta_model)
    reference_engine = _make_engine(reference_model, dta_enabled=False, monkeypatch=monkeypatch)
    dta_engine = _make_engine(dta_model, dta_enabled=True, monkeypatch=monkeypatch)
    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffixes = (_tokens(700, _SUFFIX_LENGTH, device), _tokens(1300, _SUFFIX_LENGTH, device))

    # DTASelfAttention must remain a transparent replacement without an active
    # TreeAttentionContext.
    full_tokens = torch.cat((prefix, suffixes[0])).unsqueeze(0)
    positions = torch.arange(full_length, device=device).unsqueeze(0)
    causal_mask = torch.triu(
        torch.ones((1, 1, full_length, full_length), dtype=torch.bool, device=device),
        diagonal=1,
    )
    with torch.no_grad():
        reference_logits = reference_model(full_tokens, positions, causal_mask)
        dta_normal_logits = dta_model(full_tokens, positions, causal_mask)
    torch.testing.assert_close(dta_normal_logits, reference_logits, atol=2e-2, rtol=2e-2)

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

    dta_data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        dta_data,
        **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(_make_plan(prefix, *suffixes))},
    )

    def run_dta():
        return dta_engine.forward_backward_batch(dta_data, loss_function=None, forward_only=False)

    dta_model.zero_grad(set_to_none=True)
    dta_calls, _ = _verify_fused_adapter_path(
        run_dta,
        lambda: dta_model.zero_grad(set_to_none=True),
        expected_path="dta",
    )
    dta_model.zero_grad(set_to_none=True)
    dta_output = run_dta()
    dta_gradients = _parameter_grads(dta_model)
    reference_loss = sum(reference_output["loss"])
    torch.testing.assert_close(
        torch.tensor(dta_output["loss"]),
        torch.tensor(reference_loss),
        atol=2e-2,
        rtol=2e-2,
    )
    _assert_gradients_close(dta_gradients, reference_gradients)
    assert dta_calls == 4 * QWEN_NUM_LAYERS
    assert dta_output["metrics"]["dta_direct_leaf_count"] == 2
