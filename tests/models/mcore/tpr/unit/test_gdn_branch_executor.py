"""CPU algebraic lifecycle/multilevel-relay checks (not a CP kernel test)."""

import pytest
import torch
from torch import nn

from verl.models.mcore.tpr.parallel.gdn_tree import GDNCPBranchExecutor
from verl.models.mcore.tpr.prefix_state import GDNLayerState


class _Stack(nn.Module):
    cp_size = 1

    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(0.05*i, dtype=torch.float64))
                                         for i in (1, 2, 3)])

    def forward(self, hidden, initial):
        states = {}
        for number, weight in enumerate(self.weights, 1):
            conv = hidden.new_zeros(1, 1, 4) if not initial else initial[number].conv_state
            recurrent = hidden.new_zeros(1, 1, 1, 1) if not initial else initial[number].recurrent_state
            outputs = []
            for token in hidden:
                conv = torch.cat((conv[:, :, 1:], (token.mean()*weight).reshape(1, 1, 1)), dim=-1)
                recurrent = recurrent*0.7 + conv.sum()*weight
                outputs.append(token + recurrent.reshape(1, 1))
            hidden = torch.stack(outputs)
            states[number] = GDNLayerState(conv, recurrent)
        return hidden, states


def test_multilevel_sibling_relay_loss_ownership_and_release():
    torch.manual_seed(42)
    inputs = {i: torch.randn(4, 1, 2, dtype=torch.float64) for i in range(4)}
    reference, actual = _Stack(), _Stack()
    xs, states, total = {}, {}, 0
    for sid, parent in ((0, None), (1, 0), (2, 1), (3, 1)):
        xs[sid] = inputs[sid].clone().requires_grad_(True)
        output, states[sid] = reference(xs[sid], {} if parent is None else states[parent])
        total = total + output.square().sum() * (2 if sid < 2 else 1) / 48
    total.backward()
    calls = []

    def loss(sid):
        def callback(output):
            calls.append(sid)
            return output.square().sum() * (2 if sid < 2 else 1) / 48
        return callback

    executor = GDNCPBranchExecutor(actual)
    executor.push(0, None, inputs[0], loss(0))
    root = executor.stack[-1].state
    executor.push(1, 0, inputs[1], loss(1))
    middle = executor.stack[-1].state
    assert not calls
    result = {2: executor.visit(2, 1, inputs[2], loss(2)),
              3: executor.visit(3, 1, inputs[3], loss(3))}
    assert not root.gradients and len(middle.gradients) == 3
    result[1] = executor.pop(1)
    assert middle.released and len(root.gradients) == 3
    result[0] = executor.pop(0)
    assert root.released and calls == [2, 3, 1, 0]
    executor.assert_empty()
    torch.testing.assert_close(sum(r.loss for r in result.values()), total.detach(), atol=1e-12, rtol=1e-12)
    for sid in range(4):
        torch.testing.assert_close(result[sid].input_gradient, xs[sid].grad, atol=1e-12, rtol=1e-12)
    for left, right in zip(reference.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)


def test_rejects_wrong_parent_and_marks_failed():
    executor = GDNCPBranchExecutor(_Stack())
    with pytest.raises(ValueError, match="parent"):
        executor.push(0, 99, torch.ones(4, 1, 2))
    with pytest.raises(RuntimeError, match="failed"):
        executor.pop(0)


def test_rejects_parameter_mutation_between_push_and_pop():
    model = _Stack()
    executor = GDNCPBranchExecutor(model)
    executor.push(0, None, torch.ones(4, 1, 2, dtype=torch.float64))
    with torch.no_grad():
        model.weights[0].add_(1)
    with pytest.raises(RuntimeError, match="parameters changed"):
        executor.pop(0)
