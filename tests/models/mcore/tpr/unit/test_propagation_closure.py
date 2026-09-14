"""CPU contract checks for the propagation diagnostic's state placement."""

import pytest
import torch

from ..parallel._propagation_closure import shard_conv_state


@pytest.mark.parametrize("rank", [0, 1])
def test_closure_conv_state_keeps_independent_qkv_shards(rank):
    full = torch.arange(16 * 4).reshape(1, 16, 4)
    local = shard_conv_state(full, q=4, v=8, rank=rank)
    indices = torch.tensor([0, 1, 4, 5, 8, 9, 10, 11] if rank == 0 else
                           [2, 3, 6, 7, 12, 13, 14, 15])
    assert torch.equal(local, full.index_select(1, indices))
    assert local.is_contiguous() and local.shape == (1, 8, 4)
