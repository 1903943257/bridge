"""CPU checks for the test-only permutation and its autograd routing."""

import pytest
import torch

from ..parallel._first_gdn_zigzag_control import first_gdn_zigzag_control, zigzag_control_forward


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


@pytest.mark.parametrize("out_enabled,fc2_enabled", [(False, False), (True, False), (False, True), (True, True)])
def test_projection_switches_and_restoration(monkeypatch, out_enabled, fc2_enabled):
    from types import SimpleNamespace

    monkeypatch.setenv("STAGE43_OUT_PROJ_ZIGZAG64", str(int(out_enabled)))
    monkeypatch.setenv("STAGE43_MLP_FC2_ZIGZAG64", str(int(fc2_enabled)))
    out = SimpleNamespace(forward=lambda x: (x * 2, None))
    fc2 = SimpleNamespace(forward=lambda x: (x * 3, None))
    originals = (out.forward, fc2.forward)
    layer = SimpleNamespace(layer_number=1,
                            self_attention=SimpleNamespace(tpr_state_kind="gdn", out_proj=out),
                            mlp=SimpleNamespace(linear_fc2=fc2))
    model = SimpleNamespace(config=SimpleNamespace(context_parallel_size=1),
                            decoder=SimpleNamespace(layers=[layer]))
    x = torch.arange(128.).view(128, 1, 1)
    with first_gdn_zigzag_control(model, monkeypatch) as (enabled, counts):
        assert enabled == (out_enabled or fc2_enabled)
        torch.testing.assert_close(out.forward(x)[0], x * 2)
        torch.testing.assert_close(fc2.forward(x)[0], x * 3)
        assert set(counts) == {name for name, flag in (("out_proj", out_enabled), ("mlp_fc2", fc2_enabled)) if flag}
        assert all(c["full128_to_zigzag_2x64"] == 1 for c in counts.values())
    assert (out.forward, fc2.forward) == originals
