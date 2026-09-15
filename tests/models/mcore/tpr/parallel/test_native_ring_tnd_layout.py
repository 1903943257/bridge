"""CPU layout-shim contracts; real numerical behavior needs the NPU matrix."""

from types import SimpleNamespace

import torch

from ._native_ring_tnd_layout import install_native_tnd_layout


def test_layout_shim_preserves_gqa_stats_and_backward(monkeypatch):
    q = torch.randn(64, 1, 2048)
    k, v = torch.randn(32, 1, 512), torch.randn(32, 1, 512)
    # Distinct head/token values catch an accidental transpose as well as rank.
    stats = torch.arange(64 * 8 * 8, dtype=torch.float32).reshape(64, 8, 8)
    seen = []

    def forward(tq, tk, tv, n, layout, **kwargs):
        assert tq.shape == (64, 8, 256)
        assert tk.shape == tv.shape == (32, 2, 256)
        assert n == 8 and layout == "TND"
        assert kwargs["actual_seq_qlen"] == [64]
        assert kwargs["actual_seq_kvlen"] == [32]
        seen.append("fwd")
        return tq, stats, stats + 1, 11, 12, 13

    def backward(tq, tk, tv, dy, n, layout, **kwargs):
        assert layout == "TND" and dy.shape == tq.shape == (64, 8, 256)
        assert torch.equal(kwargs["attention_in"], tq)
        assert torch.equal(kwargs["softmax_max"], stats)
        assert torch.equal(kwargs["softmax_sum"], stats + 1)
        assert kwargs["softmax_max"].shape == (64, 8, 8)
        assert kwargs["softmax_max"].is_contiguous()
        assert kwargs["seed"] == 12
        seen.append("bwd")
        return tq, tk, tv, None

    api = SimpleNamespace(npu_fusion_attention=forward, npu_fusion_attention_grad=backward)
    install_native_tnd_layout(monkeypatch, api)
    result = api.npu_fusion_attention(q, k, v, 8, "SBH", sparse_mode=0)
    assert torch.equal(result[0], q)
    assert result[1].shape == result[2].shape == (1, 8, 64, 8)
    assert torch.equal(result[1].reshape(-1), stats.reshape(-1))
    assert result[3:] == (11, 12, 13)
    gradients = api.npu_fusion_attention_grad(q, k, v, q, 8, "SBH",
        attention_in=result[0], softmax_max=result[1], softmax_sum=result[2], seed=12)
    for actual, expected in zip(gradients[:3], (q, k, v)):
        assert torch.equal(actual, expected)
    assert seen == ["fwd", "bwd"]
