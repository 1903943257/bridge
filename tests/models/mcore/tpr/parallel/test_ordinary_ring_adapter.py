"""CPU contracts for the whole-only native MindSpeed adapter."""

import sys
from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.tpr.parallel import ring_attention as ring


def test_native_adapter_preserves_gqa_dtype_and_autograd(monkeypatch):
    group = object()
    shard = SimpleNamespace(global_length=128, padded_length=128)
    config = SimpleNamespace(cp_size=2, cp_rank=1, global_ranks=(4, 7),
                             query_heads=8, softmax_scale=0.0625)
    monkeypatch.setattr(ring, "_normalize_inputs", lambda *a, **k: ((), config))
    monkeypatch.setattr(ring.dist, "get_rank", lambda: 7)
    seen = []

    def native(q, k, v, n, cp_para, **kwargs):
        assert q.shape == (64, 1, 2048)
        assert k.shape == v.shape == (64, 1, 512)
        assert q.dtype == k.dtype == v.dtype == torch.float32
        assert n == 8 and cp_para["rank"] == 1
        assert cp_para["cp_group"] is group
        assert cp_para["cp_inner_ranks"] == [7]
        assert cp_para["cp_outer_ranks"] == cp_para["cp_dkv_outer_ranks"] == [4, 7]
        assert cp_para["cache_policy"] is None
        assert cp_para["causal"] and not cp_para["megatron_cp_in_bnsd"]
        assert kwargs == {"softmax_scale": 0.0625, "dropout_p": 0.0}
        seen.append(True)
        return q + k.sum() + v.sum()

    monkeypatch.setitem(sys.modules,
        "mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel",
        SimpleNamespace(ringattn_context_parallel=native))
    inputs = [torch.ones(64, 1, h, 256, requires_grad=True) for h in (8, 2, 2)]
    out = ring.ordinary_ring_cp_attention(*inputs, current_shard=shard, cp_group=group)
    out.sum().backward()
    assert seen == [True]
    assert torch.equal(inputs[0].grad, torch.ones_like(inputs[0]))
    for x in inputs[1:]:
        assert torch.equal(x.grad, torch.full_like(x, out.numel()))


def test_native_adapter_rejects_padding_before_import(monkeypatch):
    monkeypatch.setattr(ring, "_normalize_inputs", lambda *a, **k: ((), None))
    with pytest.raises(ValueError, match="unpadded"):
        ring.ordinary_ring_cp_attention(None, None, None,
            current_shard=SimpleNamespace(global_length=127, padded_length=128), cp_group=None)
