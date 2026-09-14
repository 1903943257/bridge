"""Opt-in L5/S2 same-input, same-state, same-cotangent GDN replay."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.distributed as dist


class GDNLayerReplay:
    def __init__(self):
        self.record = None

    @contextmanager
    def capture(self, model, sid, monkeypatch):
        if sid != 2:
            yield
            return
        layer = model.decoder.layers[4].self_attention
        original = layer._tpr_forward

        def forward(hidden, context):
            assert self.record is None
            initial = context.get_initial_gdn_state(5)
            assert initial is not None
            self.record = {"input": hidden.detach().cpu().clone(),
                           "conv": initial.conv_state.detach().cpu().clone(),
                           "recurrent": initial.recurrent_state.detach().cpu().clone()}
            return original(hidden, context)

        with monkeypatch.context() as patch:
            patch.setattr(layer, "_tpr_forward", forward)
            yield


def replay_gdn(model, capture, reference, runtime, monkeypatch, metrics, *, cp):
    """No model .grad mutation. CP2 parameter VJPs SUM once; states never SUM."""
    from verl.models.mcore.tpr.prefix_state import GDNLayerState
    from ._first_gdn_zigzag_control import zigzag_control_forward

    layer = model.decoder.layers[4].self_attention
    assert layer.layer_number == 5 and layer.cp_size == cp
    assert capture.record is not None
    rank = runtime.cp_group.rank()
    indices = torch.cat((torch.arange(rank * 32, (rank + 1) * 32),
                         torch.arange((3-rank) * 32, (4-rank) * 32)))
    q, v = layer.qk_dim_local_tp, layer.v_dim_local_tp

    def local(name, value):
        if name in ("input", "output"):
            return value.index_select(0, indices).contiguous()
        if name in ("conv", "final_conv"):
            return torch.cat([part.chunk(2, dim=1)[rank] for part in value.split((q, q, v), dim=1)], dim=1).contiguous()
        return value.chunk(2, dim=1)[rank].contiguous()

    def report(label, left, right):
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].shape == right[key].shape, (key, left[key].shape, right[key].shape)
            assert torch.isfinite(left[key]).all() and torch.isfinite(right[key]).all()
        pair = metrics(left, right)
        a = pair.aggregate
        print(f"STAGE-4.4 GDN-REPLAY r={rank} L05 S2 {label}: "
              f"rel={a.relative_l2:.6e} cos={a.cosine:.9f} ratio={a.norm_ratio:.6f} "
              f"worst={pair.worst_name}:{pair.worst.relative_l2:.6e}", flush=True)

    if cp == 1:
        canonical = {}
        for name, value in capture.record.items():
            tensor = value.to(runtime.device)
            dist.broadcast(tensor, src=0, group=runtime.cp_group)
            canonical[name] = tensor.cpu()
        generator = torch.Generator().manual_seed(445002)
        cotangents = {}
        # Nonzero synthetic dOutput AND dFinalStates isolate continuation VJP;
        # these are not the model's own-upstream gradients.
        shapes = {"output": canonical["input"], "final_conv": canonical["conv"],
                  "final_recurrent": canonical["recurrent"]}
        for name, value in shapes.items():
            cotangents[name] = (torch.randn(value.shape, generator=generator) * 1e-3).to(value.dtype)
        weights = {n: p.detach().cpu().clone() for n, p in layer.named_parameters()}
    else:
        canonical, cotangents, weights = reference["canonical"], reference["cotangents"], reference["weights"]
        for name, parameter in layer.named_parameters():
            assert torch.equal(weights[name], parameter.detach().cpu()), f"L5 parameter mismatch: {name}"
        for name in canonical:
            report("captured-input/" + name, {name: local(name, canonical[name])}, {name: capture.record[name]})

    def run(data):
        inputs = {n: t.to(runtime.device).detach().clone().requires_grad_(True) for n, t in data.items()}
        initial = GDNLayerState(inputs["conv"], inputs["recurrent"])
        final = {}
        def save(number, value):
            assert number == 5
            final["state"] = value
        context = SimpleNamespace(get_initial_gdn_state=lambda number: initial, set_new_gdn_state=save)
        counts = []
        with monkeypatch.context() as patch, torch.enable_grad():
            if cp == 1:
                for module in (layer.in_proj, layer.out_proj):
                    counter = {"full128_to_zigzag_2x64": 0}
                    counts.append(counter)
                    patch.setattr(module, "forward", zigzag_control_forward(module.forward, counter))
            output, bias = layer._tpr_forward(inputs["input"], context)
            assert bias is None
            outputs = {"output": output, "final_conv": final["state"].conv_state,
                       "final_recurrent": final["state"].recurrent_state}
            parameters = dict(layer.named_parameters())
            upstream = [(local(n, cotangents[n]) if cp == 2 else cotangents[n]).to(runtime.device)
                        for n in outputs]
            grads = torch.autograd.grad(tuple(outputs.values()), (*inputs.values(), *parameters.values()),
                                        grad_outputs=upstream)
        assert cp == 2 or all(c["full128_to_zigzag_2x64"] == 1 for c in counts)
        input_grads = dict(zip(inputs, grads[:len(inputs)]))
        parameter_grads = dict(zip(parameters, grads[len(inputs):]))
        if cp == 2:
            for value in parameter_grads.values():
                dist.all_reduce(value, group=runtime.cp_group)
        return {"output": {n: t.detach().cpu() for n, t in outputs.items()},
                "input_grad": {n: t.detach().cpu() for n, t in input_grads.items()},
                "parameter_grad": {n: t.detach().cpu() for n, t in parameter_grads.items()}}

    data = {n: local(n, t) for n, t in canonical.items()} if cp == 2 else canonical
    print(f"STAGE-4.4 GDN-REPLAY r={rank} L05 CP={cp} inputs(shape,stride,dtype)="
          f"{ {n: (tuple(t.shape), t.stride(), str(t.dtype)) for n, t in data.items()} }", flush=True)
    result = run(data)
    repeat = run(data)
    for kind in result:
        report("repeat/" + kind, result[kind], repeat[kind])
    if cp == 2:
        for kind in result:
            left = reference["result"][kind]
            if kind != "parameter_grad":
                left = {n: local(n, t) for n, t in left.items()}
            if kind == "parameter_grad":
                report("same-input/" + kind, left, result[kind])
            else:
                for name in left:
                    report("same-input/" + kind + "/" + name, {name: left[name]}, {name: result[kind][name]})
        own = run(capture.record)
        for kind in result:
            report("CP2-canonical-vs-own-same-upstream/" + kind, result[kind], own[kind])
        print(f"STAGE-4.4 GDN-REPLAY r={rank} L05: parameters exact; CP1 projections zigzag64; "
              "same rank0 input/state; fixed synthetic dOutput+dFinalConv+dFinalRecurrent; "
              "parameter SUM once, state SUM none; diagnostic only", flush=True)
    return dict(canonical=canonical, cotangents=cotangents, weights=weights, result=result)
