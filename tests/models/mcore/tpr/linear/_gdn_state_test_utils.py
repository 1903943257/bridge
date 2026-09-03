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

"""Stateful single-rank GDN execution used by prefix-continuation tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LinearPrefixState:
    """Minimal state required to continue a GDN sequence."""

    causal_conv_state: Tensor
    recurrent_state: Tensor
    prefix_length: int


@dataclass(frozen=True)
class StatefulGDNResult:
    """Output, next state, and the state references consumed by one segment."""

    output: Tensor
    output_bias: Tensor | None
    state: LinearPrefixState
    causal_conv_backend: str
    consumed_state_ptrs: tuple[int, int] | None


def _canonical_conv_state(
    state: Tensor | None,
    *,
    batch: int,
    channels: int,
    history_length: int,
    like: Tensor,
) -> Tensor:
    if state is None:
        return torch.zeros(
            batch,
            channels,
            history_length,
            dtype=like.dtype,
            device=like.device,
        )
    expected_shape = (batch, channels, history_length)
    if tuple(state.shape) != expected_shape:
        raise ValueError(
            f"causal-conv state must have shape {expected_shape}, got {tuple(state.shape)}"
        )
    if state.device != like.device or state.dtype != like.dtype:
        raise ValueError(
            "causal-conv state dtype/device must match the current segment: "
            f"state={state.dtype}/{state.device}, segment={like.dtype}/{like.device}"
        )
    return state


def _expected_next_conv_state(x: Tensor, initial_state: Tensor | None, width: int) -> Tensor:
    batch, _, channels = x.shape
    history_length = width - 1
    state = _canonical_conv_state(
        initial_state,
        batch=batch,
        channels=channels,
        history_length=history_length,
        like=x,
    )
    if history_length == 0:
        return state
    history_and_current = torch.cat((state, x.transpose(1, 2)), dim=-1)
    return history_and_current[..., -history_length:].contiguous()


def _fallback_stateful_causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    activation: str | None,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Depthwise causal convolution with an explicit left-history state."""

    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            f"expected x=[B,S,D] and weight=[D,W], got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    batch, _, channels = x.shape
    if weight.shape[0] != channels:
        raise ValueError(
            f"causal-conv channels differ: x={channels}, weight={weight.shape[0]}"
        )

    width = weight.shape[1]
    history_length = width - 1
    state = _canonical_conv_state(
        initial_state,
        batch=batch,
        channels=channels,
        history_length=history_length,
        like=x,
    )
    channel_first = x.transpose(1, 2).contiguous()
    convolution_input = torch.cat((state, channel_first), dim=-1)
    output = F.conv1d(
        convolution_input,
        weight.unsqueeze(1),
        bias=bias,
        padding=0,
        groups=channels,
    )
    if activation in ("silu", "swish"):
        output = F.silu(output)
    elif activation is not None:
        raise NotImplementedError(f"unsupported causal-conv activation: {activation}")

    if history_length == 0:
        final_state = state
    else:
        final_state = convolution_input[..., -history_length:].contiguous()
    return output.transpose(1, 2).contiguous(), final_state


def _stateful_causal_conv1d(
    gdn_module: ModuleType,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    *,
    activation: str | None,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor, str]:
    causal_conv = gdn_module.causal_conv1d
    if causal_conv is None:
        output, final_state = _fallback_stateful_causal_conv1d(
            x,
            weight,
            bias,
            activation=activation,
            initial_state=initial_state,
        )
        backend = "torch.nn.functional.conv1d (stateful NPU fallback)"
    else:
        output, final_state = causal_conv(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=None,
        )
        backend = causal_conv.__module__

    if final_state is None:
        raise AssertionError("causal-conv backend did not return its requested final state")
    expected_state = _expected_next_conv_state(x, initial_state, weight.shape[1])
    torch.testing.assert_close(final_state, expected_state, atol=0.0, rtol=0.0)
    return output, final_state, backend


def run_stateful_gdn_segment(
    model: nn.Module,
    hidden_states: Tensor,
    gdn_module: ModuleType,
    *,
    initial_state: LinearPrefixState | None = None,
) -> StatefulGDNResult:
    """Run one CP=1 GDN segment while exposing both continuation states.

    This deliberately reuses the MindSpeed GDN projections, preparation
    helpers, recurrent backend, normalization, and output projection. Only the
    two state arguments hard-coded by ``GatedDeltaNet.forward`` are surfaced.
    """

    if model.cp_size != 1 or model.tp_size != 1 or model.sp_size != 1:
        raise ValueError(
            "stateful GDN capability runner requires CP=TP=SP=1, got "
            f"CP={model.cp_size}, TP={model.tp_size}, SP={model.sp_size}"
        )
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden states must use [S,B,H], got {tuple(hidden_states.shape)}")

    sequence_length, batch, _ = hidden_states.shape
    if sequence_length <= 0:
        raise ValueError("a stateful GDN segment must contain at least one token")
    consumed_state_ptrs = None
    if initial_state is not None:
        consumed_state_ptrs = (
            initial_state.causal_conv_state.data_ptr(),
            initial_state.recurrent_state.data_ptr(),
        )

    qkvzba, input_bias = model.in_proj(hidden_states)
    if input_bias is not None:
        qkvzba = qkvzba + input_bias
    qkvzba = qkvzba.transpose(0, 1)
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            model.qk_dim_local_tp * 2 + model.v_dim_local_tp,
            model.v_dim_local_tp,
            model.num_value_heads,
            model.num_value_heads,
        ],
        dim=-1,
    )
    gate = gate.reshape(batch, sequence_length, -1, model.value_head_dim)
    beta = beta.reshape(batch, sequence_length, -1)
    alpha = alpha.reshape(batch, sequence_length, -1)

    conv_initial_state = None if initial_state is None else initial_state.causal_conv_state
    qkv, causal_conv_final_state, causal_conv_backend = _stateful_causal_conv1d(
        gdn_module,
        qkv,
        model.conv1d.weight.squeeze(1),
        model.conv1d.bias,
        activation=model.activation,
        initial_state=conv_initial_state,
    )

    query, key, value, gate, beta, alpha = model._prepare_qkv_for_gated_delta_rule(
        qkv,
        gate,
        beta,
        alpha,
        batch,
        sequence_length,
    )
    g, beta = model._compute_g_and_beta(model.A_log, model.dt_bias, alpha, beta)
    recurrent_initial_state = None if initial_state is None else initial_state.recurrent_state
    core_attention_output, recurrent_final_state = model.gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
    )
    if recurrent_final_state is None:
        raise AssertionError("GDN recurrent backend did not return its requested final state")

    normalized = model._apply_gated_norm(core_attention_output, gate)
    normalized = normalized.reshape(batch, sequence_length, -1)
    normalized = normalized.transpose(0, 1).contiguous()
    output, output_bias = model.out_proj(normalized)

    previous_length = 0 if initial_state is None else initial_state.prefix_length
    return StatefulGDNResult(
        output=output,
        output_bias=output_bias,
        state=LinearPrefixState(
            causal_conv_state=causal_conv_final_state,
            recurrent_state=recurrent_final_state,
            prefix_length=previous_length + sequence_length,
        ),
        causal_conv_backend=causal_conv_backend,
        consumed_state_ptrs=consumed_state_ptrs,
    )
