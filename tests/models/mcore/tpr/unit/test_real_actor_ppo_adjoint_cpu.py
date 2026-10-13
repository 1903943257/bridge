"""CPU-only regression tests for PPO upstream derivative diagnostics."""
from __future__ import annotations

import math
import torch

from tests.models.mcore.tpr.parallel.analyze_real_actor_ppo_adjoint import (
    _ppo_loss_and_adjoint, _stats,
)


def test_ppo_upstream_clipped_and_unclipped_positions():
    new = torch.tensor([0., -3., math.log(2.0), math.log(5.0)])
    old = torch.zeros(4)
    advantage = torch.tensor([1., -1., 1., -1.])
    loss, g, clipped, ratios = _ppo_loss_and_adjoint(
        new, old, advantage, denom=4
    )
    assert torch.isfinite(loss)
    # ratio=1, A=+1: unclipped, dL/dlp = -A*ratio/4.
    torch.testing.assert_close(g[0], torch.tensor(-0.25), atol=1e-7, rtol=0)
    # A=-1 and ratio below 0.8: lower PPO clip, no gradient.
    torch.testing.assert_close(g[1], torch.tensor(0.), atol=1e-7, rtol=0)
    # A=+1 and ratio above 1.2: upper PPO clip, no gradient.
    torch.testing.assert_close(g[2], torch.tensor(0.), atol=1e-7, rtol=0)
    # A=-1 and ratio above 3: dual clip, no gradient.
    torch.testing.assert_close(g[3], torch.tensor(0.), atol=1e-7, rtol=0)
    assert bool(torch.all(clipped[1:]))
    assert math.isfinite(_stats(g, g)["rel_l2"])
    assert _stats(g, g)["rel_l2"] == 0.
