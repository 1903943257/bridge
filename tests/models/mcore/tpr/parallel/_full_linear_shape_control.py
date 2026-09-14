"""Test-only CP1 projection-module shape control for the full Qwen fixture."""

from collections import Counter
from contextlib import contextmanager

from ._first_gdn_zigzag_control import zigzag_control_forward


def projection_targets(model):
    config = model.config
    assert config.context_parallel_size == config.tensor_model_parallel_size == 1
    assert not config.sequence_parallel and not config.add_bias_linear
    layers = tuple(model.decoder.layers)
    assert len(layers) == 24
    targets = {}
    for number, layer in enumerate(layers, 1):
        assert layer.layer_number == number
        attention = layer.self_attention
        is_gdn = getattr(attention, "tpr_state_kind", None) == "gdn"
        assert is_gdn == (number % 4 != 0)
        attrs = ("in_proj", "out_proj") if is_gdn else ("linear_qkv", "linear_proj")
        for name in attrs:
            targets[f"L{number:02d}.attention.{name}"] = getattr(attention, name)
        for name in ("linear_fc1", "linear_fc2"):
            targets[f"L{number:02d}.mlp.{name}"] = getattr(layer.mlp, name)
    targets["output_layer"] = model.output_layer
    assert len(targets) == len({id(module) for module in targets.values()}) == 97
    assert all(callable(module.forward) for module in targets.values())
    return targets


@contextmanager
def full_linear_shape_control(model, monkeypatch, *, enabled, rank):
    if not enabled:
        yield
        return
    targets = projection_targets(model)
    counts = {name: {"full128_to_zigzag_2x64": 0} for name in targets}
    types = Counter(type(module).__name__ for module in targets.values())
    # Scope is the actual projection module. Some implementations include
    # input norm in that module; log types rather than claim pure GEMM control.
    print(f"STAGE-4.4 LINEAR-CONTROL r={rank} CP1-only targets=97 "
          f"GDN=36 FA=12 MLP=48 head=1 types={dict(types)}; "
          "128->[0:32,96:128]+[32:96]->inverse-permutation", flush=True)
    with monkeypatch.context() as patch:
        for name, module in targets.items():
            patch.setattr(module, "forward", zigzag_control_forward(module.forward, counts[name]))
        try:
            yield
        finally:
            bad = {name: value["full128_to_zigzag_2x64"] for name, value in counts.items()
                   if value["full128_to_zigzag_2x64"] != 3}
            print(f"STAGE-4.4 LINEAR-CONTROL r={rank} "
                  f"calls={sum(c['full128_to_zigzag_2x64'] for c in counts.values())}/291 "
                  f"bad-targets={bad}; restoring all forwards", flush=True)
    assert not bad, f"every projection must process P/S1/S2 once: {bad}"
