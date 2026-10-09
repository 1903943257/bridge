"""CPU-only checks for per-layer weak TQ gradient/update diagnostics."""
from __future__ import annotations

import pytest
import torch

from ..correctness.analyze_qwen17_weak_e2e_samples import (
    _diagnostics,
    _group_name,
    analyze_samples,
)


def test_layer_grouping_recognizes_qwen_names():
    assert _group_name("decoder.layers.0.mlp.linear_fc1.weight") == "decoder.layers.00"
    assert _group_name("decoder.layers.27.self_attention.linear_qkv.weight") == "decoder.layers.27"
    assert _group_name("embedding.word_embeddings.weight") == "embedding_or_lm_head"
    assert _group_name("decoder.final_layernorm.weight") == "other"


def test_gradient_energy_not_raw_flip_count():
    a = torch.tensor([1.0, 1.0, 1e-9, 1e-9])
    b = torch.tensor([1.0, 1.0, -1e-9, -1e-9])
    x = _diagnostics(a, b)
    assert x["flips"] == 2
    assert x["both_nonzero"] == 4
    assert x["flip_ref_energy"] < 1e-12
    assert x["r2"] == pytest.approx(2.0)
    assert x["weak_count"] == 2


def test_sample_analyzer_prints_per_layer_and_parameter(capsys):
    idx = torch.tensor([0, 5, 9], dtype=torch.int64)
    n = {
        "decoder.layers.0.mlp.linear_fc1.weight": {
            "indices": idx,
            "grad": torch.tensor([1.0, 2.0, -3.0]),
            "update": torch.tensor([-0.1, -0.1, 0.1]),
        }
    }
    t = {
        "decoder.layers.0.mlp.linear_fc1.weight": {
            "indices": idx,
            "grad": torch.tensor([1.0, -2.0, -3.0]),
            "update": torch.tensor([-0.1, 0.1, 0.1]),
        }
    }
    result = "\n".join(analyze_samples(n, t, top=3))
    assert "[GRAD] ALL SAMPLED" in result
    assert "[UPDATE] ALL SAMPLED" in result
    assert "decoder.layers.00" in result
    assert "err_share=" in result
    assert "flipped_ref_energy_share=" in result


def test_sample_analyzer_refuses_different_indices():
    n = {"x": {"indices": torch.tensor([0, 5]),
               "grad": torch.tensor([1., 2.]), "update": torch.tensor([2., 1.])}}
    t = {"x": {"indices": torch.tensor([0, 6]),
               "grad": torch.tensor([1., 2.]), "update": torch.tensor([2., 1.])}}
    with pytest.raises(ValueError, match="sampling locations"):
        analyze_samples(n, t)
