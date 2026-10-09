"""CPU contracts for BF16 fixed-M Linear probe on arbitrary segment lengths.

Uses a synthetic tokenwise module; it verifies padding/slicing/autograd only.
It DOES NOT establish NPU GEMM numerics or production training correctness.
"""
from types import SimpleNamespace

import pytest
import torch

from ..correctness._qwen17_fixed_tile_probe import (
    install_test_only_fixed_tile_gemm,
)


class _TokenwiseLinear(torch.nn.Module):
    def forward(self, x):
        return x * 2, None


def _fake_model():
    linear = _TokenwiseLinear()
    layer = SimpleNamespace(
        self_attention=SimpleNamespace(
            layer_number=1, linear_qkv=linear,
            linear_proj=_TokenwiseLinear(),
        ),
        mlp=SimpleNamespace(
            linear_fc1=_TokenwiseLinear(), linear_fc2=_TokenwiseLinear()
        ),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(tensor_model_parallel_size=1),
        decoder=SimpleNamespace(layers=[layer]),
    )
    return model, linear


@pytest.mark.parametrize("seq", [1, 3, 8, 9, 15, 16, 17])
def test_fixed_tile_pad_tail_preserves_tokenwise_values_and_gradients(
    monkeypatch, seq,
):
    model, module = _fake_model()
    install_test_only_fixed_tile_gemm(
        model, monkeypatch, tile_size=8, groups=("qkv",),
    )
    x = torch.arange(seq*4, dtype=torch.float32).reshape(
        seq, 1, 4
    ).to(torch.bfloat16).requires_grad_(True)
    y, bias = module(x)
    assert bias is None
    assert y.shape == x.shape
    assert torch.equal(y, x*2)
    y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


def test_fixed_tile_rejects_unusable_configuration(monkeypatch):
    model, _ = _fake_model()
    with pytest.raises(ValueError, match="tile_size"):
        install_test_only_fixed_tile_gemm(
            model, monkeypatch, tile_size=0,
        )
    model.config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1"):
        install_test_only_fixed_tile_gemm(
            model, monkeypatch, tile_size=8,
        )
