"""CPU regressions for the opt-in test intervention, not projection numerics."""

from types import SimpleNamespace

import pytest
import torch

from ..linear._first_layer_projection_control import first_layer_projection_control


class _Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
        self.calls = []

    def forward(self, x):
        self.calls.append(x.shape[0])
        return x @ self.weight.t(), None


def _model():
    projection = _Projection()
    layer = SimpleNamespace(layer_number=1, self_attention=SimpleNamespace(
        tpr_state_kind="gdn", out_proj=projection,
    ))
    return SimpleNamespace(decoder=SimpleNamespace(layers=[layer]),
                           config=SimpleNamespace(context_parallel_size=1)), projection


@pytest.mark.parametrize("mode", ["0", "1"])
def test_control_scope_shapes_and_gradients(monkeypatch, mode):
    monkeypatch.setenv("STAGE33_OUT_PROJ_CHUNK64", mode)
    model, projection = _model()
    original = projection.forward
    x = torch.linspace(-1, 1, 384).reshape(128, 1, 3).requires_grad_(True)
    reference_x = x.detach().clone().requires_grad_(True)
    reference = _Projection()
    with first_layer_projection_control(model, monkeypatch):
        full, _ = projection(x)
        short, _ = projection(x[:64])
        (full.square().sum() + short.square().sum()).backward()
    assert projection.forward == original
    assert projection.calls == ([64, 64, 64] if mode == "1" else [128, 64])
    ref_full = (torch.cat([reference(part)[0] for part in reference_x.split(64)], dim=0)
                if mode == "1" else reference(reference_x)[0])
    ref_short = reference(reference_x[:64])[0]
    (ref_full.square().sum() + ref_short.square().sum()).backward()
    torch.testing.assert_close(full, ref_full)
    torch.testing.assert_close(short, ref_short)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(projection.weight.grad, reference.weight.grad)


def test_control_restores_on_exception(monkeypatch):
    monkeypatch.setenv("STAGE33_OUT_PROJ_CHUNK64", "1")
    model, projection = _model()
    original = projection.forward
    with pytest.raises(RuntimeError, match="injected"):
        with first_layer_projection_control(model, monkeypatch):
            raise RuntimeError("injected")
    assert projection.forward == original
