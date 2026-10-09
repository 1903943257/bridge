"""CPU-only tests for exact-Segment numerical module trace instrumentation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ..correctness._qwen17_weak_e2e_trace import (
    STAGES, capture_query_stages, describe_trace, describe_kv_trace,
    compare_first_attention_replay, compare_root_v_provenance,
)


class _ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = nn.Linear(4, 4, bias=False)
        self.q_layernorm = nn.LayerNorm(4)
        self.k_layernorm = nn.LayerNorm(4)
        self.linear_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        x = self.linear_qkv(x)
        x = self.q_layernorm(x) + self.k_layernorm(x)
        return self.linear_proj(x)


class _ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(4, 4, bias=False)
        self.linear_fc2 = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.linear_fc2(self.linear_fc1(x))


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attention = _ToyAttention()
        self.pre_mlp_layernorm = nn.LayerNorm(4)
        self.mlp = _ToyMLP()

    def forward(self, *, hidden_states, attention_mask=None):
        # Match real Megatron TransformerLayer's keyword-argument call site:
        # the forward-hook receives args=() and kwargs["hidden_states"].
        x = self.self_attention(self.input_layernorm(hidden_states))
        # Real Megatron layers return a tuple (hidden_states, context).
        return self.mlp(self.pre_mlp_layernorm(x)), None


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_ToyLayer(), _ToyLayer()])


def test_stage_capture_same_token_in_native_and_leaf():
    torch.manual_seed(4)
    model = _ToyModel()
    ids = torch.randn(10, 1, 4)
    with capture_query_stages(
        model, query_position=8,
        get_active_span=lambda: (0, 10),
    ) as native:
        x = ids
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    # At the same absolute query, the leaf only contains indices 8 and 9.
    with capture_query_stages(
        model, query_position=8,
        get_active_span=lambda: (8, 10),
    ) as tpr:
        x = ids[8:]
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    rows = describe_trace(native, tpr, layers=2)
    assert len(rows) == 2 * len(STAGES) + 1
    assert "compared_stages=22" in rows[-1]
    # Same tensor/token, but GEMM sees M=10 vs M=2; a CPU float32
    # projection can differ by ~1 ULP (observed 1.19e-7). This is
    # insignificant and MUST NOT be a strict bitwise-parity gate.
    assert "first_material=None" in rows[-1]


def test_trace_does_not_capture_other_segment():
    model = _ToyModel()
    with capture_query_stages(
        model, query_position=8, get_active_span=lambda: None,
    ) as output:
        x = torch.ones(4, 1, 4)
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    assert not output


def test_trace_rejects_missing_layer_stage():
    with pytest.raises(AssertionError, match="incomplete"):
        describe_trace({}, {}, layers=2)


def test_trace_detects_real_perturbation():
    torch.manual_seed(4)
    model = _ToyModel()
    ids = torch.randn(10, 1, 4)
    with capture_query_stages(
        model, query_position=8, get_active_span=lambda: (0, 10),
    ) as native:
        x = ids
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    with capture_query_stages(
        model, query_position=8, get_active_span=lambda: (0, 10),
    ) as tpr:
        x = ids.clone()
        x[8, 0, 0] += 0.5
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    assert "first_material=None" not in describe_trace(native, tpr, layers=2)[-1]


def test_actual_leaf_prefix_kv_oracle_localizes_first_changed_ancestor():
    path = [(0, 0, 4), (1, 4, 7), (5, 7, 10)]
    base_k = torch.arange(20, dtype=torch.float32).reshape(10, 1, 1, 2)
    base_v = -base_k.clone()
    native_kv = {1: (base_k.clone(), base_v.clone())}
    forest_k = base_k.clone()
    forest_k[5, 0, 0, 0] += 0.5
    forest_kv = {1: (forest_k, base_v.clone())}
    rows = describe_kv_trace(native_kv, forest_kv, path, layers=(1,))
    assert len(rows) == 7
    assert any(
        "segment=1[4:7]" in row and
        "field=K_POST_ROPE" in row and
        "differing_tokens=1/3" in row and
        "first_differing_abs=5" in row
        for row in rows
    )
    assert any(
        "field=V_RAW" in row and "max_abs=0" in row
        for row in rows
    )
    assert "rows=6" in rows[-1]
    assert "worst_max_abs=0.5" in rows[-1]


def test_kv_oracle_rejects_broken_ancestor_path():
    kv = {1: (torch.zeros(5, 1, 1, 2), torch.zeros(5, 1, 1, 2))}
    with pytest.raises(AssertionError, match="contiguous"):
        describe_kv_trace(
            kv, kv, [(0, 0, 2), (5, 3, 5)], layers=(1,)
        )


def test_trace_stages_handle_keyword_only_megatron_hidden_states():
    model = _ToyModel()
    with capture_query_stages(
        model, query_position=1, get_active_span=lambda: (0, 3)
    ) as trace:
        x = torch.ones(3, 1, 4)
        for layer in model.decoder.layers:
            x, _context = layer(hidden_states=x, attention_mask=None)
    assert len(trace) == 2 * len(STAGES)
    assert (0, "layer") in trace



def _fake_rectangular_attention(q, k, v, *, softmax_scale=None, dropout_p=0.0):
    """CPU oracle with right-down causal masking for a single Q/KV head."""
    assert dropout_p == 0.0
    assert q.shape[1:3] == k.shape[1:3] == v.shape[1:3] == (1, 1)
    qt, kt, vt = (x[:, 0, 0].float() for x in (q, k, v))
    scores = (qt @ kt.T) * (float(softmax_scale) if softmax_scale else 0.7071067811865476)
    sk, sq = kt.shape[0], qt.shape[0]
    masked = torch.arange(sk)[None, :] > (
        torch.arange(sq)[:, None] + (sk - sq)
    )
    scores = scores.masked_fill(masked, float("-inf"))
    return (scores.softmax(dim=-1) @ vt).unsqueeze(1)


def test_first_attention_replay_swaps_actual_inputs_independently():
    torch.manual_seed(19)
    q = torch.randn(6, 1, 1, 2)
    k = torch.randn(6, 1, 1, 2)
    v = torch.randn(6, 1, 1, 2)
    tq = q[3:].clone()
    tk = k.clone()
    tv = v.clone()
    tq[0, 0, 0, 0] += 0.2
    tk[1, 0, 0, 1] += 0.1
    tv[2, 0, 0, 1] -= 0.15
    baseline = _fake_rectangular_attention(q[3:], k, v)
    actual = _fake_rectangular_attention(tq, tk, tv)
    messages = compare_first_attention_replay(
        (q, k, v, None), (tq, tk, tv, None, actual),
        position_start=3, query_abs=3,
        native_proj_input=baseline[0],
        tpr_proj_input=actual[0],
        device="cpu", attention_fn=_fake_rectangular_attention,
    )
    assert "source=NATIVE_QKV max_abs=0" in messages[0]
    assert "source=TPR_ALL max_abs=0" in messages[1]
    assert "source=TPR_CORE_OUTPUT max_abs=0" in messages[2]
    assert "variant=TPR_Q_ONLY" in messages[3]
    assert "variant=TPR_K_ONLY" in messages[4]
    assert "variant=TPR_V_ONLY" in messages[5]
    assert "variant=TPR_KV" in messages[6]
    assert "variant=TPR_ALL" in messages[7]
    assert "query_M=3 kv_M=6" in messages[-1]


def test_first_attention_replay_rejects_softmax_scale_mismatch():
    q = torch.ones(6, 1, 1, 2)
    k = torch.ones(6, 1, 1, 2)
    v = torch.ones(6, 1, 1, 2)
    p = torch.zeros(1, 2)
    with pytest.raises(AssertionError, match="softmax scales differ"):
        compare_first_attention_replay(
            (q, k, v, 1.0),
            (q[3:], k, v, 0.5, torch.zeros(3, 1, 2)),
            position_start=3, query_abs=3,
            native_proj_input=p, tpr_proj_input=p,
            device="cpu", attention_fn=_fake_rectangular_attention,
        )


def test_first_layer_root_v_provenance_attributes_native_length_effect():
    full_v = torch.arange(12, dtype=torch.float32).reshape(6, 1, 2)
    cutoff_v = full_v[:4].clone()
    cutoff_v[1, 0, 0] += 0.25
    tpr_v = cutoff_v.clone()
    full_in = torch.ones(6, 1, 3)
    root_in = full_in[:4].clone()
    tpr_in = root_in.clone()

    rows = compare_root_v_provenance(
        full_v, cutoff_v, tpr_v,
        full_in, root_in, tpr_in,
    )
    assert len(rows) == 7
    assert any(
        "field=V_RAW pair=FULL_TO_CUTOFF" in row
        and "max_abs=0.25" in row and "changed_tokens=1/4" in row
        for row in rows
    )
    assert any(
        "field=V_RAW pair=CUTOFF_TO_TPR" in row and "max_abs=0" in row
        for row in rows
    )
    assert any(
        "field=QKV_INPUT" in row and "max_abs=0" in row
        for row in rows
    )
    assert "compared_fields=2 compared_pairs=3 root_tokens=4" in rows[-1]


def test_first_layer_root_v_provenance_rejects_mismatched_projection_inputs():
    full_v = torch.ones(6, 1, 2)
    root_v = torch.ones(4, 1, 2)
    with pytest.raises(AssertionError, match="shape mismatch"):
        compare_root_v_provenance(
            full_v, root_v, root_v,
            torch.ones(6, 1, 3),
            torch.ones(4, 1, 3),
            torch.ones(4, 1, 2),
        )
