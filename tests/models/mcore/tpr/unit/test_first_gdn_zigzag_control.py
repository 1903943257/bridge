"""CPU checks for the test-only permutation and its autograd routing."""

import pytest
import torch

from ..parallel._first_gdn_zigzag_control import zigzag_control_forward


@pytest.mark.parametrize("tuple_output", [False, True])
def test_zigzag_projection_control_restores_values_and_gradients(tuple_output):
    x = torch.arange(128., dtype=torch.float64).view(128, 1, 1).requires_grad_()
    weight = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    calls = []

    def original(value):
        calls.append(value.detach().flatten().tolist())
        output = value * weight
        return (output, None) if tuple_output else output

    counts = {"full128_to_zigzag_2x64": 0}
    result = zigzag_control_forward(original, counts)(x)
    result = result[0] if tuple_output else result
    assert calls == [list(range(32)) + list(range(96, 128)), list(range(32, 96))]
    assert counts["full128_to_zigzag_2x64"] == 1
    torch.testing.assert_close(result, x * weight, rtol=0, atol=0)
    upstream = torch.arange(128., dtype=torch.float64).view_as(x)
    result.backward(upstream)
    torch.testing.assert_close(x.grad, upstream * 2, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, (x.detach() * upstream).sum(), rtol=0, atol=0)
