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
import math
from pathlib import Path

import torch
from megatron.core.transformer.module import MegatronModule
from transformers import AutoConfig

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from verl.models.mcore.config_converter import (
    get_hf_rope_theta,
    hf_to_mcore_config_dense,
)
from verl.models.mcore.tpr import replace_self_attention_with_tpr
from verl.models.mcore.tpr.module_spec import _get_layer_specs
from ..profiling.test_tpr_engine_profile_npu import _ProfileFusedCausalAttention

from ..correctness.test_tpr_qwen3_compatibility_npu import (
    _load_hf_state_dict,
    _validate_checkpoint_files,
)


class _RingCPOnlyCoreAttention(MegatronModule):
    """CP>1 TPR test sentinel: actual attention MUST run via Ring CP.

    The CP1 square-CANN reference subclasses DotProductAttention, whose
    Megatron initializer intentionally forbids CP>1. Overriding forward
    does NOT remove that constructor guard. Do not bypass upstream guards
    or silently use single-rank square attention on CP-sharded tokens.
    This lightweight module only carries the native softmax scale consumed
    by TPRSelfAttention._tree_forward's existing Ring CP dispatch.
    """

    def __init__(
        self, config, layer_number, attn_mask_type, attention_type,
        attention_dropout=None, softmax_scale=None, cp_comm_type=None,
        pg_collection=None,
    ):
        super().__init__(config=config)
        if getattr(config, "apply_query_key_layer_scaling", False):
            raise NotImplementedError(
                "CP Ring test sentinel does not cover query-key layer scaling"
            )
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.softmax_scale = (
            1.0 / math.sqrt(config.kv_channels)
            if softmax_scale is None else float(softmax_scale)
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "CP>1 Qwen TPR test requires active Ring CP attention; "
            "the square-causal fallback cannot consume CP-sharded QKV"
        )


_DENSE_PROFILES = {
    "1.7b": (28, 2048, 6144, 16, 8, 128),
    "4b": (36, 2560, 9728, 32, 8, 128),
}


def load_qwen3_dense_config(path: Path, *, profile: str = "1.7b"):
    """Validate exact real HF model profile before loading TP shards."""
    _validate_checkpoint_files(path)
    if profile not in _DENSE_PROFILES:
        raise ValueError(f"unsupported real checkpoint profile {profile!r}")
    cfg = AutoConfig.from_pretrained(
        str(path), trust_remote_code=True, local_files_only=True
    )
    expected = dict(zip(
        ("num_hidden_layers", "hidden_size", "intermediate_size",
         "num_attention_heads", "num_key_value_heads", "head_dim"),
        _DENSE_PROFILES[profile],
    ))
    expected.update(model_type="qwen3", vocab_size=151936,
                    tie_word_embeddings=True)
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise AssertionError(
                f"expected authentic Qwen3-{profile} {key}={value}, "
                f"got {getattr(cfg, key)!r}"
            )
    if cfg.attention_dropout != 0:
        raise AssertionError("Qwen TP checkpoint requires attention_dropout=0")
    return cfg


def load_qwen3_1_7b_config(path: Path):
    return load_qwen3_dense_config(path, profile="1.7b")


@torch.no_grad()
def load_hf_qwen3_tp_shards(model, hf, path: Path, *, tp_rank: int, tp_size: int = 2):
    """Load every parameter from checkpoint; assert shape and full coverage."""
    state = _load_hf_state_dict(path)
    if tp_size not in (1, 2) or not 0 <= tp_rank < tp_size:
        raise ValueError("Qwen3 dense isolated parallel tests require TP=1 or TP=2")
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


def make_real_qwen3_tp_model(
    runtime, *, model_path: Path, tp_size: int = 2, load_weights: bool = True,
    sequence_parallel: bool = False, cp_size: int = 1,
    profile: str = "1.7b",
    native_attention: bool = False,
):
    """Construct real Qwen3 TP shards; optionally use unmodified MCore attention.

    When load_weights=False, the caller MUST load the already-sharded
    state_dict of a checkpoint-loaded peer BEFORE performing any forward.
    """
    hf = load_qwen3_dense_config(model_path, profile=profile)
    config = hf_to_mcore_config_dense(
        hf, torch.bfloat16,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=sequence_parallel,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
    )
    if native_attention:
        # Native Megatron/MindSpeed Ring CP path: no TPR layer replacement.
        # Match the TPR replay: native activation recomputation is OFF by default.
        # Opt-in only for capacity diagnostics; not for the parity comparison.
        import os
        recompute_layers = int(os.getenv("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0"))
        if recompute_layers < 0:
            raise ValueError("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS must be >= 0")
        if recompute_layers:
            config.recompute_granularity = "full"
            config.recompute_method = "uniform"
            config.recompute_num_layers = recompute_layers
        else:
            config.recompute_granularity = None
            config.recompute_method = None
            config.recompute_num_layers = None
        config.use_flash_attn = True
    # IMPORTANT: setting use_flash_attn=True on a TransformerConfig does
    # not replace Megatron's DotProductAttention implementation. The latter
    # still invokes MindSpeed's ScaledMaskedSoftmax and rejects the 2D mask
    # generated for Flash Attention (requires an NPU-operator 4D mask).
    # Use the established real-Qwen NPU reference adapter from the existing
    # TPR Qwen compatibility tests. It executes a full square causal
    # attention with the CANN rectangular kernel, independently of the TPR
    # Prefix/Suffix cache path. This is a *controlled square-causal oracle*,
    # not a claim of equality to MindSpeed's unmodified native FA kernel.
    vanilla_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=False, pp_rank=0
    )
    spec = vanilla_spec if native_attention else replace_self_attention_with_tpr(vanilla_spec)
    # For CP>1, the ordinary square-CANN reference core is invalid because
    # its DotProductAttention parent constructor only supports CP=1.
    # Every reference and TPR segment in the Ring-combination test must
    # execute inside SegmentExecutor's explicit Ring attention context.
    core_attention_cls = None if native_attention else (
        _ProfileFusedCausalAttention if cp_size == 1
        else _RingCPOnlyCoreAttention
    )
    # get_gpt_decoder_block_spec returns TransformerBlockSubmodules
    # (with .layer_specs) in this Megatron version, NOT necessarily a
    # ModuleSpec (with .submodules.layer_specs). Use the same tested
    # resolver as replace_self_attention_with_tpr above.
    layer_specs = _get_layer_specs(spec)
    if not native_attention:
        for layer_spec in layer_specs:
            layer_spec.submodules.self_attention.submodules.core_attention = (
                core_attention_cls
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

    if load_weights:
        load_hf_qwen3_tp_shards(
            model, hf, model_path, tp_rank=runtime.rank, tp_size=tp_size
        )
    assert model.config.num_layers == hf.num_hidden_layers
    # Both the native full-trajectory reference (no active TPR context)
    # and TPR's tree branch must use the expected CANN attention operator.
    # If a future MindSpeed/Megatron spec bypasses our replacement, fail
    # here rather than entering an incompatible ScaledMaskedSoftmax kernel.
    if native_attention:
        from verl.models.mcore.tpr import TPRSelfAttention
        if any(isinstance(m, TPRSelfAttention) for m in model.modules()):
            raise AssertionError("Native Full must never construct a TPR attention module")
    elif not all(
        isinstance(layer.self_attention.core_attention, core_attention_cls)
        for layer in model.decoder.layers
    ):
        raise AssertionError(
            "Qwen attention core is not the expected CP-compatible reference "
            f"adapter for CP={cp_size}"
        )
    assert model.config.tensor_model_parallel_size == tp_size
    assert model.config.context_parallel_size == cp_size
    assert model.config.sequence_parallel is sequence_parallel
    assert model.embedding.word_embeddings.weight.shape == (hf.vocab_size // tp_size, hf.hidden_size)
    assert model.output_layer.weight is None
    assert len(model.decoder.layers) == hf.num_hidden_layers
    assert model.decoder.layers[0].self_attention.linear_qkv.weight.shape == (
        (hf.num_attention_heads + 2 * hf.num_key_value_heads) * hf.head_dim // tp_size,
        hf.hidden_size,
    )
    model.train()
    return model, hf


def make_real_qwen3_tp2_model(
    runtime, *, model_path: Path, load_weights: bool = True,
    sequence_parallel: bool = False,
):
    """Backward-compatible strict TP2 entry for existing correctness gates."""
    return make_real_qwen3_tp_model(
        runtime, model_path=model_path, tp_size=2, load_weights=load_weights,
        sequence_parallel=sequence_parallel,
    )
