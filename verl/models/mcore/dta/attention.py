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

"""Megatron self-attention with an opt-in external-KV training path."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention
from megatron.core.typed_torch import apply_module

from .context import TreeAttentionContext, get_tree_attention_context
from .rectangular_attention import rectangular_causal_attention


class DTASelfAttention(SelfAttention):
    """Self-attention that consumes per-layer external KV in tree mode.

    With no active :class:`TreeAttentionContext`, this class delegates the
    complete call to Megatron's ``SelfAttention.forward``.  The external-KV
    branch is intentionally restricted to the first MVP configuration and
    bypasses only Megatron/MindSpeed's core-attention wrapper.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> tuple[Tensor, Tensor | None]:
        context = get_tree_attention_context()
        if context is None:
            return super().forward(
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
            )

        self._validate_tree_forward(
            context=context,
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            inference_context=inference_context,
            inference_params=inference_params,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return self._tree_forward(hidden_states, context)

    def _tree_forward(
        self,
        hidden_states: Tensor,
        context: TreeAttentionContext,
    ) -> tuple[Tensor, Tensor | None]:
        # Reuse Megatron's projection, GQA reshaping, and optional Q/K norms.
        query, new_key, new_value = self.get_query_key_value_tensors(
            hidden_states,
            key_value_states=None,
            split_qkv=True,
            output_gate=False,
        )

        q_pos_emb, k_pos_emb = _as_rotary_pair(context.suffix_rotary_pos_emb)
        cp_group = getattr(self.pg_collection, "cp", None)
        rope_kwargs = {
            "config": self.config,
            "cu_seqlens": None,
            "mscale": getattr(self, "_yarn_concentration_factor", 1.0),
            "cp_group": cp_group,
        }
        query = apply_rotary_pos_emb(query, q_pos_emb, **rope_kwargs)
        new_key = apply_rotary_pos_emb(new_key, k_pos_emb, **rope_kwargs)

        past_kv = context.get_past_kv(self.layer_number)
        if past_kv is None:
            key = new_key
            value = new_value
        else:
            past_key, past_value = past_kv
            _validate_past_compatibility(self.layer_number, past_key, past_value, new_key, new_value)
            key = _concat_sequence(past_key, new_key)
            value = _concat_sequence(past_value, new_value)

        core_attn_out = rectangular_causal_attention(
            query,
            key,
            value,
            softmax_scale=getattr(self.core_attention, "softmax_scale", None),
            dropout_p=0.0,
        )

        # Collect post-RoPE K and raw V without detach/clone, preserving the
        # graph both for descendants and for gradients into an external prefix.
        context.set_new_kv(self.layer_number, new_key, new_value)
        return apply_module(self.linear_proj)(core_attn_out)

    def _validate_tree_forward(
        self,
        *,
        context: TreeAttentionContext,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor],
        inference_context: Optional[BaseInferenceContext],
        inference_params: Optional[BaseInferenceContext],
        rotary_pos_cos: Optional[Tensor],
        rotary_pos_sin: Optional[Tensor],
        rotary_pos_cos_sin: Optional[Tensor],
        attention_bias: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[int],
    ) -> None:
        if not self.training:
            raise RuntimeError("DTA external-KV mode is restricted to training")
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                "DTA external-KV mode requires hidden_states [suffix, 1, hidden], "
                f"got {hidden_states.shape}"
            )
        if hidden_states.shape[0] != context.suffix_length:
            raise ValueError(
                f"hidden suffix length ({hidden_states.shape[0]}) does not match "
                f"TreeAttentionContext ({context.suffix_length})"
            )
        if context.suffix_rotary_pos_emb is None:
            raise ValueError("TreeAttentionContext.suffix_rotary_pos_emb is required in DTA mode")
        _validate_rotary_sequence_length(context.suffix_rotary_pos_emb, context.suffix_length)

        unsupported = {
            "key_value_states": key_value_states,
            "inference_context": inference_context,
            "inference_params": inference_params,
            "rotary_pos_cos": rotary_pos_cos,
            "rotary_pos_sin": rotary_pos_sin,
            "rotary_pos_cos_sin": rotary_pos_cos_sin,
            "attention_bias": attention_bias,
            "packed_seq_params": packed_seq_params,
            "sequence_len_offset": sequence_len_offset,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise NotImplementedError(f"DTA external-KV MVP does not support: {', '.join(active)}")

        boolean_restrictions = {
            "activation checkpointing": getattr(self, "checkpoint_core_attention", False),
            "fused_single_qkv_rope": getattr(self.config, "fused_single_qkv_rope", False),
            "attention_output_gate": getattr(self.config, "attention_output_gate", False),
            "flash_decode": getattr(self.config, "flash_decode", False),
            "QKV offload": getattr(self, "offload_qkv_linear", False),
            "core-attention offload": getattr(self, "offload_core_attention", False),
            "attention-projection offload": getattr(self, "offload_attn_proj", False),
        }
        enabled = [name for name, value in boolean_restrictions.items() if value]
        if enabled:
            raise NotImplementedError(f"DTA external-KV MVP does not support: {', '.join(enabled)}")

        if float(getattr(self.config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("DTA external-KV MVP requires attention_dropout=0")
        for name in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
        ):
            value = getattr(self.config, name, 1)
            if value != 1:
                raise NotImplementedError(f"DTA external-KV MVP requires {name}=1, got {value}")


def _as_rotary_pair(rotary_pos_emb: Union[Tensor, Tuple[Tensor, Tensor], None]) -> tuple[Tensor, Tensor]:
    if rotary_pos_emb is None:
        raise ValueError("suffix rotary position embedding is required")
    if isinstance(rotary_pos_emb, tuple):
        if len(rotary_pos_emb) != 2 or not all(isinstance(item, Tensor) for item in rotary_pos_emb):
            raise TypeError("suffix rotary position embedding tuple must contain two tensors")
        return rotary_pos_emb
    if not isinstance(rotary_pos_emb, Tensor):
        raise TypeError("suffix rotary position embedding must be a tensor or a pair of tensors")
    return rotary_pos_emb, rotary_pos_emb


def _validate_rotary_sequence_length(
    rotary_pos_emb: Union[Tensor, Tuple[Tensor, Tensor]],
    suffix_length: int,
) -> None:
    for name, tensor in zip(("query", "key"), _as_rotary_pair(rotary_pos_emb), strict=True):
        if tensor.ndim == 0 or tensor.shape[0] != suffix_length:
            raise ValueError(
                f"{name} suffix rotary embedding sequence length must be {suffix_length}, got {tensor.shape}"
            )


def _validate_past_compatibility(
    layer_number: int,
    past_key: Tensor,
    past_value: Tensor,
    new_key: Tensor,
    new_value: Tensor,
) -> None:
    for name, past, new in (("K", past_key, new_key), ("V", past_value, new_value)):
        if past.shape[1:] != new.shape[1:]:
            raise ValueError(
                f"layer {layer_number} past/new {name} shapes are incompatible: "
                f"past={past.shape}, new={new.shape}"
            )
        if past.device != new.device or past.dtype != new.dtype:
            raise ValueError(
                f"layer {layer_number} past/new {name} device and dtype must match, "
                f"past=({past.device}, {past.dtype}), new=({new.device}, {new.dtype})"
            )


def _concat_sequence(prefix: Tensor, suffix: Tensor) -> Tensor:
    # Kept as a tiny seam so tests can assert that external KV remains part of
    # the autograd graph without introducing any storage transformation here.
    return torch.cat((prefix, suffix), dim=0)

