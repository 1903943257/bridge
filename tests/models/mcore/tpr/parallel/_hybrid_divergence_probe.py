"""Observational CP1/CP2 token-layout probes; no collectives or tensor replacement.

Backward comparisons use each run's own upstream gradients. A first nonzero
VJP is a localization clue, not proof that that module's backward is wrong.
"""

from contextlib import contextmanager

import torch


class HybridDivergenceProbe:
    def __init__(self, model):
        self.model = model
        self.records = {}
        self.order = {}

    @contextmanager
    def segment(self, sid):
        handles = []
        order = self.order.setdefault(sid, [])

        def capture(name, tensor):
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name}: expected tensor, got {type(tensor)}")
            # Only token-major module boundaries, never flattened gated_norm
            # or HP state tensors. CP2 here is native zigzag [S/CP, B, C].
            if tensor.ndim != 3 or tensor.shape[1] != 1:
                raise AssertionError(f"{name}: unexpected token layout {tensor.shape}")
            key = (sid, name)
            if key in self.records:
                raise AssertionError(f"duplicate connected probe {key}")
            entry = {"value": tensor.detach().cpu().float().clone(), "grad": None}
            self.records[key] = entry
            order.append(name)
            if tensor.requires_grad:
                def gradient(value):
                    value = value.detach().cpu().float().clone()
                    entry["grad"] = value if entry["grad"] is None else entry["grad"] + value
                tensor.register_hook(gradient)

        def watch(module, name, *, inputs=True):
            if inputs:
                def before(_module, args, kwargs):
                    tensor = args[0] if args else kwargs["hidden_states"]
                    capture(name + ".input", tensor)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))

            def after(_module, args, output):
                capture(name + ".output", output)
            handles.append(module.register_forward_hook(after))

        watch(self.model.embedding, "embedding", inputs=False)
        for layer in self.model.decoder.layers:
            base = f"layer{layer.layer_number}"
            watch(layer, base)
            watch(layer.self_attention, base + ".attention")
            # Both GDN and FA projections; include norm and MLP boundaries to
            # distinguish the attention path from residual/MLP amplification.
            for attr in ("input_layernorm", "pre_mlp_layernorm", "mlp"):
                module = getattr(layer, attr, None)
                if module is not None:
                    watch(module, base + "." + attr)
            for attr in ("in_proj", "out_proj", "linear_qkv", "linear_proj"):
                module = getattr(layer.self_attention, attr, None)
                if module is not None:
                    watch(module, base + ".attention." + attr)
            for attr in ("linear_fc1", "linear_fc2"):
                module = getattr(layer.mlp, attr, None)
                if module is not None:
                    watch(module, base + ".mlp." + attr)
        watch(self.model.decoder.final_layernorm, "final_norm")
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def compare(self, actual, *, rank, metrics):
        if self.records.keys() != actual.records.keys() or self.order != actual.order:
            raise AssertionError("CP1/CP2 diagnostic probe sets/order differ")
        for sid, names in self.order.items():
            differences = {"value": {}, "grad": {}}
            for name in names:
                left_entry, right_entry = self.records[(sid, name)], actual.records[(sid, name)]
                for kind in ("value", "grad"):
                    left, right = left_entry[kind], right_entry[kind]
                    if left is None or right is None:
                        raise AssertionError(f"missing connected {sid}/{name}/{kind}")
                    length = left.shape[0]
                    assert length % 4 == 0
                    chunk = length // 4
                    indices = torch.cat((torch.arange(rank * chunk, (rank + 1) * chunk),
                                         torch.arange((3 - rank) * chunk, (4 - rank) * chunk)))
                    left = left.index_select(0, indices)
                    assert left.shape == right.shape, (sid, name, kind, left.shape, right.shape)
                    assert torch.isfinite(left).all() and torch.isfinite(right).all()
                    pair = metrics({name: left}, {name: right}).aggregate
                    exact = torch.equal(left, right)
                    differences[kind][name] = not exact
                    print(f"STAGE-4.3 TRACE rank={rank} segment={sid} {name}/{kind}: "
                          f"exact={exact} {pair} max_abs={(right-left).abs().max().item():.9e}",
                          flush=True)
            first_forward = next((n for n in names if differences["value"][n]), None)
            # Reverse module-boundary order within this segment, NOT wall-clock
            # autograd order across sibling graphs and state edges.
            first_backward = next((n for n in reversed(names) if differences["grad"][n]), None)
            print(f"STAGE-4.3 FIRST rank={rank} segment={sid}: forward={first_forward}; "
                  f"backward(reverse-boundary-order)={first_backward}. "
                  "Own-upstream VJPs; not a same-upstream kernel replay.", flush=True)
