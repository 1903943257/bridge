"""CPU-only acceptance sanity checks; no pretrained weights/NPU needed."""
import math

import pytest
import torch

from ..correctness._qwen17_ppo_acceptance import (
    report_ppo_clip_agreement,
    report_sampled_fresh_adamw,
)


def test_advantage_aware_ppo_clip_branches(capsys):
    old = torch.zeros(4)
    native = old.clone()  # current/old ratio is exactly 1
    tpr = torch.tensor([
        math.log(1.5),  # +A: upper clip
        math.log(0.5),  # -A: lower clip
        math.log(1.5),  # -A: not clipped (beneficial)
        math.log(0.5),  # +A: not clipped (unfavorable)
    ])
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])
    metrics = report_ppo_clip_agreement(native, tpr, old, advantages)
    assert metrics["valid"] == 4
    assert metrics["branch_disagree"] == 2
    assert "branch_disagree=2" in capsys.readouterr().out


def test_same_inputs_have_no_clip_disagreement():
    old = torch.zeros(3)
    new = torch.tensor([math.log(1.4), math.log(0.6), 0.0])
    advantages = torch.tensor([1., -1., 1.])
    result = report_ppo_clip_agreement(new, new, old, advantages)
    assert result["branch_disagree"] == 0
    assert result["native_policy_loss"] == pytest.approx(result["tpr_policy_loss"])


def test_sampled_first_adamw_same_gradients_and_opposite_sign():
    before = {"x": torch.tensor([1., -2., 0.5])}
    grad = {"x": torch.tensor([0.1, -0.2, 0.03])}
    max_abs, rel = report_sampled_fresh_adamw(before, before, grad, grad)
    assert max_abs == 0 and rel == 0

    changed = {"x": torch.tensor([-0.1, -0.2, 0.03])}
    max_abs, rel = report_sampled_fresh_adamw(
        before, before, grad, changed
    )
    assert max_abs > 0 and rel > 0


def test_sampled_first_adamw_rejects_different_initial_weights():
    native = {"x": torch.tensor([1.0])}
    tpr = {"x": torch.tensor([1.01])}
    grad = {"x": torch.tensor([0.1])}
    with pytest.raises(AssertionError, match="initial weight mismatch"):
        report_sampled_fresh_adamw(native, tpr, grad, grad)
