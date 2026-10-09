"""CPU-only tests for exact-Segment numerical module trace instrumentation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ..correctness._qwen17_weak_e2e_trace import (
    STAGES, capture_query_stages, describe_trace, describe_kv_trace,
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
