"""CPU-only tests for exact-Segment numerical module trace instrumentation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ..correctness._qwen17_weak_e2e_trace import (
    STAGES, capture_query_stages, describe_trace,
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
