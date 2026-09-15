"""CPU layout/autograd checks for the test-only whole-FA gather reference."""

import torch

from ..parallel._whole_fa_allgather import restore_zigzag_gather


def test_restore_zigzag_values_and_vjp():
    rank_order = torch.tensor([0., 1., 6., 7., 2., 3., 4., 5.], requires_grad=True)
    chronological = restore_zigzag_gather(rank_order)
    torch.testing.assert_close(chronological, torch.arange(8.))
    upstream = torch.arange(8.) + 10
    gradient, = torch.autograd.grad(chronological, (rank_order,), upstream)
    torch.testing.assert_close(gradient, torch.tensor([10., 11., 16., 17., 12., 13., 14., 15.]))
