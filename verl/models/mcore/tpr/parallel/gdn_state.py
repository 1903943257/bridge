"""Stage 4.1: explicit non-packed stateful GDN forward on native CP/HP shards.

No tree scheduling or implicit TPR-context dispatch is enabled here. Inputs and
outputs use each segment's native zigzag [S/CP,B,H]. States live after CP->HP,
in chronological order with rank-local Q/K/V channels and recurrent heads.
"""

import torch

from ..prefix_state import GDNLayerState


def gdn_cp_state_shapes(layer, batch=1):
    cp = layer.cp_size
    q, v = layer.qk_dim_local_tp, layer.v_dim_local_tp
    heads = layer.num_v_heads_local_tp
    if cp not in (1, 2) or layer.tp_size != 1 or layer.sp_size != 1:
        raise NotImplementedError("Stage 4.1 requires CP=1/2, TP=SP=1")
    if any(size % cp for size in (q, v, heads)):
        raise ValueError("GDN channels/heads must divide CP size")
    return ((batch, (2 * q + v) // cp, layer.conv1d.weight.shape[-1]),
            (batch, heads // cp, layer.key_head_dim, layer.value_head_dim))


def forward_gdn_cp_with_state(
    layer, hidden_states, initial_state=None, *, output_final_state=True
):
    """Return ((output, bias), optional GDNLayerState), retaining requested state gradients.

    Conv state is [B,Q_local+K_local+V_local,W], not a contiguous slice of
    the full concatenated QKV channels. Recurrent state is [B,Hv_local,K,V].
    Callers own segment identity/placement; a state from a different CP rank
    must not be silently used as continuation state.
    """
    import mindspeed.core.ssm.gated_delta_net as native
    from ..gated_delta_net import _stage1_causal_conv1d, _stage1_gated_delta_rule

    if not isinstance(layer, native.GatedDeltaNet):
        raise TypeError("requires the patched MindSpeed GatedDeltaNet")
    if not layer.training:
        raise ValueError("stateful CP seam is restricted to training")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError("expected non-packed [S/CP,1,H] input")
    local_s, batch, _ = hidden_states.shape
    if local_s <= 0:
        raise ValueError("empty GDN segments are not supported")
    cp = layer.cp_size
    shapes = gdn_cp_state_shapes(layer, batch)
    if cp > 1 and local_s % 2:
        raise ValueError("each local zigzag shard must contain two equal chunks")
    if initial_state is not None:
        if not isinstance(initial_state, GDNLayerState):
            raise TypeError("initial_state must be GDNLayerState")
        if (tuple(initial_state.conv_state.shape), tuple(initial_state.recurrent_state.shape)) != shapes:
            raise ValueError(f"wrong rank-local GDN state shape; expected {shapes}")
        if initial_state.conv_state.device != hidden_states.device or initial_state.recurrent_state.device != hidden_states.device:
            raise ValueError("initial state and input must be on the same device")
        if initial_state.conv_state.dtype != hidden_states.dtype or initial_state.recurrent_state.dtype != torch.float32:
            raise ValueError("expected conv state in input dtype and recurrent state in FP32")
    q, v, heads = layer.qk_dim_local_tp, layer.v_dim_local_tp, layer.num_v_heads_local_tp
    projected, bias = layer.in_proj(hidden_states)
    if bias is not None:
        projected = projected + bias
    projected = native.tensor_a2a_cp2hp(
        projected, seq_dim=0, head_dim=-1, cp_group=layer.pg_collection.cp,
        split_sections=[q, q, v, v, heads, heads],
    ).transpose(0, 1)
    sequence = local_s * cp
    qkv, gate, beta, alpha = torch.split(projected, [(2*q+v)//cp, v//cp, heads//cp, heads//cp], dim=-1)
    gate = gate.reshape(batch, sequence, heads//cp, layer.value_head_dim)
    beta, alpha = beta.reshape(batch, sequence, -1), alpha.reshape(batch, sequence, -1)

    def shard(parameter, sections=None):
        return native.get_parameter_local_cp(parameter, dim=0, cp_group=layer.pg_collection.cp,
                                             split_sections=sections)

    weight = shard(layer.conv1d.weight, [q, q, v])
    conv_bias = None if layer.conv1d.bias is None else shard(layer.conv1d.bias, [q, q, v])
    convolved, conv_state = _stage1_causal_conv1d(
        qkv, weight.squeeze(1), conv_bias, activation=layer.activation,
        initial_state=None if initial_state is None else initial_state.conv_state,
        output_final_state=output_final_state,
    )
    query, key, value, gate, beta, alpha = layer._prepare_qkv_for_gated_delta_rule(
        convolved, gate, beta, alpha, batch, sequence,
    )
    g, beta = layer._compute_g_and_beta(shard(layer.A_log), shard(layer.dt_bias), alpha, beta)
    core, recurrent_state = _stage1_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None if initial_state is None else initial_state.recurrent_state,
        output_final_state=output_final_state,
    )
    state = None
    if output_final_state:
        if conv_state is None or recurrent_state is None:
            raise RuntimeError("stateful GDN primitives did not return their requested final states")
        state = GDNLayerState(conv_state, recurrent_state)
        if (tuple(conv_state.shape), tuple(recurrent_state.shape)) != shapes:
            raise RuntimeError(f"primitive returned unexpected state shapes; expected {shapes}")
    normalized = layer._apply_gated_norm(core, gate).reshape(batch, sequence, -1).transpose(0, 1).contiguous()
    normalized = native.tensor_a2a_hp2cp(
        normalized, seq_dim=0, head_dim=-1, cp_group=layer.pg_collection.cp,
    )
    return layer.out_proj(normalized), state
