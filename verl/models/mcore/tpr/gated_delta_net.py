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

"""MindSpeed GDN with an opt-in stateful TPR continuation path."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet

from .context import TPRAttentionContext, get_tpr_attention_context
from .prefix_state import GDNLayerState


class TPRGatedDeltaNet(GatedDeltaNet):
    """GDN that exposes causal-conv and recurrent state only in TPR mode.

    The ordinary path delegates to MindSpeed unchanged. The TPR path is the
    CP=1/non-packed seam over the stateful MindSpeed-Ops primitives already
    validated in Stage 1, or the Stage 4 stateful A2A seam at Ring CP=2.
    """

    tpr_state_kind = "gdn"

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ):
        context = get_tpr_attention_context()
        if context is None:
            return super().forward(
                hidden_states,
                attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
                **kwargs,
            )

        self._validate_tpr_forward(
            context=context,
            hidden_states=hidden_states,
            inference_context=inference_context,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        return self._tpr_forward(hidden_states, context)

    def _tpr_forward(
        self,
        hidden_states: Tensor,
        context: TPRAttentionContext,
    ) -> tuple[Tensor, Tensor | None]:
        sequence_length, batch, _ = hidden_states.shape
        initial_state = context.get_initial_gdn_state(self.layer_number)

        if self.cp_size == 2:
            from .parallel.gdn_state import forward_gdn_cp_with_state

            output, final_state = forward_gdn_cp_with_state(
                self, hidden_states, initial_state=initial_state
            )
            context.set_new_gdn_state(self.layer_number, final_state)
            return output

        qkvzba, input_bias = self.in_proj(hidden_states)
        if input_bias is not None:
            qkvzba = qkvzba + input_bias
        qkvzba = qkvzba.transpose(0, 1)
        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                self.qk_dim_local_tp * 2 + self.v_dim_local_tp,
                self.v_dim_local_tp,
                self.num_v_heads_local_tp,
                self.num_v_heads_local_tp,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, sequence_length, self.num_v_heads_local_tp, self.value_head_dim)
        beta = beta.reshape(batch, sequence_length, self.num_v_heads_local_tp)
        alpha = alpha.reshape(batch, sequence_length, self.num_v_heads_local_tp)

        conv_initial_state = None if initial_state is None else initial_state.conv_state
        qkv, final_conv_state = _stage1_causal_conv1d(
            qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
            initial_state=conv_initial_state,
        )
        if final_conv_state is None:
            raise RuntimeError("Stage 1 causal-conv did not return its requested final state")

        query, key, value, gate, beta, alpha = self._prepare_qkv_for_gated_delta_rule(
            qkv,
            gate,
            beta,
            alpha,
            batch,
            sequence_length,
        )
        g, beta = self._compute_g_and_beta(self.A_log, self.dt_bias, alpha, beta)
        recurrent_initial_state = None if initial_state is None else initial_state.recurrent_state
        core_output, final_recurrent_state = _stage1_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_initial_state,
        )
        if final_recurrent_state is None:
            raise RuntimeError("Stage 1 gated-delta rule did not return its requested final state")

        context.set_new_gdn_state(
            self.layer_number,
            GDNLayerState(
                conv_state=final_conv_state,
                recurrent_state=final_recurrent_state,
            ),
        )
        normalized = self._apply_gated_norm(core_output, gate)
        normalized = normalized.reshape(batch, sequence_length, -1)
        normalized = normalized.transpose(0, 1).contiguous()
        return self.out_proj(normalized)

    def _validate_tpr_forward(
        self,
        *,
        context: TPRAttentionContext,
        hidden_states: Tensor,
        inference_context: Optional[BaseInferenceContext],
        inference_params: Optional[BaseInferenceContext],
        packed_seq_params: Optional[PackedSeqParams],
        sequence_len_offset: Optional[int],
    ) -> None:
        if not self.training:
            raise RuntimeError("TPR GDN mode is restricted to training")
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(
                "TPR GDN mode requires hidden_states [sequence, 1, hidden], "
                f"got {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] != context.local_suffix_length:
            raise ValueError(
                f"GDN segment length {hidden_states.shape[0]} does not match "
                f"context local suffix length {context.local_suffix_length}"
            )
        context_cp = 1 if context.attention_backend is None else context.attention_backend.parallel_size
        if context_cp != self.cp_size:
            raise ValueError(f"GDN CP={self.cp_size} does not match context CP={context_cp}")
        if self.cp_size not in (1, 2) or self.tp_size != 1 or self.sp_size != 1:
            raise NotImplementedError(
                "TPR GDN requires CP=1/2 and TP=SP=1, got "
                f"CP={self.cp_size}, TP={self.tp_size}, SP={self.sp_size}"
            )
        if self.cp_size == 2:
            from .parallel.ring_attention import RingCPAttentionBackend

            if not isinstance(context.attention_backend, RingCPAttentionBackend):
                raise NotImplementedError("TPR GDN CP2 requires the native-zigzag Ring backend")
            if (context.attention_backend.parallel_size != 2
                    or context.local_suffix_length * 2 != context.suffix_length
                    or context.local_suffix_length % 2):
                raise NotImplementedError("TPR GDN CP2 requires unpadded, aligned zigzag segments")
        unsupported = {
            "inference_context": inference_context,
            "inference_params": inference_params,
            "packed_seq_params": packed_seq_params,
            "sequence_len_offset": sequence_len_offset,
        }
        active = [name for name, value in unsupported.items() if value is not None]
        if active:
            raise NotImplementedError(f"Stage 3 TPR GDN does not support: {', '.join(active)}")


def _stage1_causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    activation: str | None,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Call the exact stateful CausalConv API validated by Stage 1."""

    from mindspeed_ops.api.triton.convolution import causal_conv1d

    # MindSpeed stores [channels, 1, width]; MindSpeed-Ops consumes [width, channels].
    primitive_weight = weight.transpose(0, 1).contiguous()
    return causal_conv1d(
        x,
        primitive_weight,
        bias=bias,
        initial_state=initial_state,
        activation=activation,
        output_final_state=True,
    )


def _stage1_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    g: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Call the exact stateful GDR API validated by Stage 1."""

    from mindspeed_ops.api.triton.chunk_gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        head_first=False,
    )
