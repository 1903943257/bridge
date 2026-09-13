"""CPU checks for the explicit Stage 4.1 rank-local state shape contract."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.mcore.tpr.parallel.gdn_state import gdn_cp_state_shapes


def _layer(cp):
    return SimpleNamespace(cp_size=cp, tp_size=1, sp_size=1,
                           qk_dim_local_tp=2048, v_dim_local_tp=2048,
                           num_v_heads_local_tp=16, key_head_dim=128, value_head_dim=128,
                           conv1d=SimpleNamespace(weight=torch.empty(6144, 1, 4)))


@pytest.mark.parametrize("cp", [1, 2])
def test_qwen_gdn_cp_state_shapes(cp):
    assert gdn_cp_state_shapes(_layer(cp)) == ((1, 6144//cp, 4), (1, 16//cp, 128, 128))


@pytest.mark.parametrize("field,value", [("cp_size", 4), ("tp_size", 2), ("sp_size", 2)])
def test_unsupported_topology_rejected(field, value):
    layer = _layer(2)
    setattr(layer, field, value)
    with pytest.raises(NotImplementedError):
        gdn_cp_state_shapes(layer)


def test_nondivisible_head_channels_rejected():
    layer = _layer(2)
    layer.num_v_heads_local_tp = 15
    with pytest.raises(ValueError, match="divide"):
        gdn_cp_state_shapes(layer)
