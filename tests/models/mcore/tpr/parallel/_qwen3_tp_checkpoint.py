# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Load the *real* HF Qwen3-1.7B checkpoint into native Megatron TP2 shards.

The existing CP checkpoint loader is TP1-only. For TP>1 its full QKV,
MLP and embedding tensors MUST be partitioned according to Megatron's
ColumnParallel/RowParallel and GQA-group layouts; blindly loading a full
HF weight into a shard or randomly initializing is NOT a TP correctness test.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from transformers import AutoConfig

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from verl.models.mcore.config_converter import (
    get_hf_rope_theta,
    hf_to_mcore_config_dense,
)
from verl.models.mcore.tpr import replace_self_attention_with_tpr

from ..correctness.test_tpr_qwen3_compatibility_npu import (
    _load_hf_state_dict,
    _validate_checkpoint_files,
)


def load_qwen3_1_7b_config(path: Path):
    _validate_checkpoint_files(path)
    cfg = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    expected = {
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise AssertionError(
                f"expected authentic Qwen3-1.7B {key}={value}, "
                f"got {getattr(cfg, key)!r}"
            )
    if cfg.attention_dropout != 0:
        raise AssertionError("TP2 Qwen checkpoint requires attention_dropout=0")
    return cfg


@torch.no_grad()
def load_hf_qwen3_tp_shards(model, hf, path: Path, *, tp_rank: int, tp_size: int = 2):
    """Load every parameter from checkpoint; assert shape and full coverage."""
    state = _load_hf_state_dict(path)
    if tp_size != 2 or not 0 <= tp_rank < tp_size:
        raise ValueError("Qwen3-1.7B initial TP gate requires TP2")
    if hf.num_key_value_heads % tp_size or hf.vocab_size % tp_size:
        raise AssertionError("invalid Qwen TP group or vocab divisibility")

    loaded_ids = set()

    def put(target, weight, source):
        if tuple(target.shape) != tuple(weight.shape):
            raise ValueError(
                f"TP{tp_size} shard shape mismatch for {source}: "
                f"checkpoint-shard={tuple(weight.shape)}, "
                f"target={tuple(target.shape)}"
            )
        target.copy_(weight.to(device=target.device, dtype=target.dtype))
        loaded_ids.add(id(target))

    def hf_weight(name):
        if name not in state:
            raise KeyError(f"Qwen checkpoint missing {name}")
        return state[name]

    def row_shard(weight):
        if weight.shape[0] % tp_size:
            raise ValueError("TP row shard is not divisible")
        return weight.chunk(tp_size, dim=0)[tp_rank].contiguous()

    def col_shard(weight):
        if weight.shape[1] % tp_size:
            raise ValueError("TP column shard is not divisible")
        return weight.chunk(tp_size, dim=1)[tp_rank].contiguous()

    put(
        model.embedding.word_embeddings.weight,
        row_shard(hf_weight("model.embed_tokens.weight")),
        "model.embed_tokens.weight / vocab-shard",
    )
    head_dim = hf.head_dim
    groups = hf.num_key_value_heads
    q_per_group = hf.num_attention_heads // groups
    groups_per_rank = groups // tp_size

    for i, layer in enumerate(model.decoder.layers):
        prefix = f"model.layers.{i}"
        attn = f"{prefix}.self_attn"
        put(layer.input_layernorm.weight, hf_weight(
            f"{prefix}.input_layernorm.weight"
        ), f"{prefix}.input_layernorm.weight")
        # GQA packed QKV layout: [group, Q-within-group, K, V].
        # Split by *KV groups*, never blindly chunk concatenated Q,K,V.
        q = hf_weight(f"{attn}.q_proj.weight").reshape(
            groups, q_per_group * head_dim, hf.hidden_size
        )
        k = hf_weight(f"{attn}.k_proj.weight").reshape(
            groups, head_dim, hf.hidden_size
        )
        v = hf_weight(f"{attn}.v_proj.weight").reshape(
            groups, head_dim, hf.hidden_size
        )
        start = tp_rank * groups_per_rank
        stop = start + groups_per_rank
        local_qkv = torch.cat((
            q[start:stop], k[start:stop], v[start:stop]
        ), dim=1).reshape(-1, hf.hidden_size).contiguous()
        put(layer.self_attention.linear_qkv.weight, local_qkv, f"{attn}.qkv_group_{tp_rank}")
        # q/k norms act on head_dim and are replicated across TP ranks.
        put(layer.self_attention.q_layernorm.weight,
            hf_weight(f"{attn}.q_norm.weight"), f"{attn}.q_norm.weight")
        put(layer.self_attention.k_layernorm.weight,
            hf_weight(f"{attn}.k_norm.weight"), f"{attn}.k_norm.weight")
        put(layer.self_attention.linear_proj.weight,
            col_shard(hf_weight(f"{attn}.o_proj.weight")),
            f"{attn}.o_proj.weight / col-shard")

        put(layer.pre_mlp_layernorm.weight,
            hf_weight(f"{prefix}.post_attention_layernorm.weight"),
            f"{prefix}.post_attention_layernorm.weight")
        # SwiGLU FC1 packed layout is [local gate, local up], NOT a
        # contiguous TP chunk of [all gate, all up].
        local_fc1 = torch.cat((
            row_shard(hf_weight(f"{prefix}.mlp.gate_proj.weight")),
            row_shard(hf_weight(f"{prefix}.mlp.up_proj.weight")),
        ), dim=0)
        put(layer.mlp.linear_fc1.weight, local_fc1,
            f"{prefix}.mlp.gate+up / TP shard")
        put(layer.mlp.linear_fc2.weight,
            col_shard(hf_weight(f"{prefix}.mlp.down_proj.weight")),
            f"{prefix}.mlp.down_proj.weight / col-shard")

    put(model.decoder.final_layernorm.weight,
        hf_weight("model.norm.weight"), "model.norm.weight")
    missing = [
        name for name, param in model.named_parameters()
        if id(param) not in loaded_ids
    ]
    if missing:
        raise RuntimeError(f"TP2 Qwen checkpoint loader missed: {missing}")
    # Important: tied lm_head does NOT introduce a second trainable weight.
    if model.output_layer.weight is not None:
        raise RuntimeError("Qwen3-1.7B tied embedding expected output_layer.weight=None")
    del state
    gc.collect()


def make_real_qwen3_tp2_model(runtime, *, model_path: Path):
    """Construct a genuine 28-layer Qwen3-1.7B in native Megatron TP2."""
    hf = load_qwen3_1_7b_config(model_path)
    config = hf_to_mcore_config_dense(
        hf, torch.bfloat16,
        tensor_model_parallel_size=2,
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
    config.use_flash_attn = True  # MindSpeed's native reference causal mask.
    spec = replace_self_attention_with_tpr(
        get_gpt_decoder_block_spec(
            config, use_transformer_engine=False, pp_rank=0
        )
    )
    from types import SimpleNamespace
    pg = SimpleNamespace(
        tp=runtime.tp_group, cp=runtime.cp_group,
        pp=runtime.pp_group, embd=None,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=hf.vocab_size,
        max_sequence_length=hf.max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_base=get_hf_rope_theta(hf),
        pg_collection=pg,
    ).to(device=runtime.device, dtype=torch.bfloat16)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if getattr(module, "tp_group", "missing") is None:
            module.tp_group = runtime.tp_group

    load_hf_qwen3_tp_shards(
        model, hf, model_path, tp_rank=runtime.rank, tp_size=2
    )
    assert model.config.num_layers == 28
    assert model.config.tensor_model_parallel_size == 2
    assert model.config.sequence_parallel is False
    assert model.embedding.word_embeddings.weight.shape == (75968, 2048)
    assert model.output_layer.weight is None
    assert len(model.decoder.layers) == 28
    assert model.decoder.layers[0].self_attention.linear_qkv.weight.shape == (2048, 2048)
    model.train()
    return model, hf
