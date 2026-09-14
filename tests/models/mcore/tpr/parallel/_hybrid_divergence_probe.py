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
            entry = {"value": tensor.detach().cpu().float().clone(), "grad": None,
                     "dtype": tensor.dtype, "stride": tuple(tensor.stride())}
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

    def replay_first_out_proj(self, actual, *, rank, metrics):
        """Forward-only replay with connected-mode autograd enabled, no backward.

        Canonical input = CP1's captured full token sequence, made contiguous.
        Both module instances see identical full and identical zigzag inputs.
        Compare raw projection outputs (no deferred bias addition), matching
        the hook. This does not replace any full-model computation.
        """
        reference = self.model.decoder.layers[0].self_attention.out_proj
        target = actual.model.decoder.layers[0].self_attention.out_proj
        left_state, right_state = reference.state_dict(), target.state_dict()
        assert left_state.keys() == right_state.keys()
        for name in left_state:
            left, right = left_state[name], right_state[name]
            equal = torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right
            assert equal, f"out_proj parameter/buffer differs: {name}"
        parameters = list(reference.parameters()) + list(target.parameters())
        versions = [p._version for p in parameters]
        device = next(reference.parameters()).device
        assert reference.training and target.training

        def run(module, value, dtype):
            # Keep the training forward dispatch; do not use no_grad/inference
            # and do not call backward or mutate existing parameter gradients.
            with torch.enable_grad():
                x = value.to(device=device, dtype=dtype).contiguous().requires_grad_(True)
                result = module(x)
                output = result[0] if isinstance(result, tuple) else result
                return output.detach().cpu().float().clone()

        for sid in self.order:
            name = "layer1.attention.out_proj"
            full_entry = self.records[(sid, name + ".input")]
            local_entry = actual.records[(sid, name + ".input")]
            full, own_local = full_entry["value"], local_entry["value"]
            assert full_entry["dtype"] == local_entry["dtype"]
            dtype = full_entry["dtype"]
            length = full.shape[0]
            assert length % 4 == 0
            chunk = length // 4
            indices = torch.cat((torch.arange(rank * chunk, (rank + 1) * chunk),
                                 torch.arange((3 - rank) * chunk, (4 - rank) * chunk)))
            canonical_local = full.index_select(0, indices)
            assert canonical_local.shape == own_local.shape

            def report(label, left, right):
                assert left.shape == right.shape, (label, left.shape, right.shape)
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                pair = metrics({label: left}, {label: right}).aggregate
                print(f"STAGE-4.3 OUT-PROJ-REPLAY rank={rank} segment={sid} {label}: "
                      f"exact={torch.equal(left, right)} {pair} "
                      f"max_abs={(right-left).abs().max().item():.9e}", flush=True)

            print(f"STAGE-4.3 OUT-PROJ-REPLAY rank={rank} segment={sid} "
                  f"dtype={dtype} full/local={tuple(full.shape)}/{tuple(own_local.shape)} "
                  f"captured-strides={full_entry['stride']}/{local_entry['stride']} "
                  "replay-layout=contiguous parameters=exact grad-enabled=True backward=False", flush=True)
            report("captured-input", canonical_local, own_local)
            captured_full = self.records[(sid, name + ".output")]["value"]
            captured_local = actual.records[(sid, name + ".output")]["value"]
            report("captured-output", captured_full.index_select(0, indices), captured_local)

            full_ref = run(reference, full, dtype)
            full_cp = run(target, full, dtype)
            local_ref = run(reference, canonical_local, dtype)
            local_cp = run(target, canonical_local, dtype)
            own_cp = run(target, own_local, dtype)
            report("CP1-own-input-reproduction", captured_full, full_ref)
            report("CP2-own-input-reproduction", captured_local, own_cp)
            report("same-full-input-CP1-vs-CP2-module", full_ref, full_cp)
            report("same-local-input-CP1-vs-CP2-module", local_ref, local_cp)
            report("canonical-full-vs-shard-CP1-module", full_ref.index_select(0, indices), local_ref)
            report("canonical-full-vs-shard-CP2-module", full_cp.index_select(0, indices), local_cp)
            report("CP2-canonical-vs-own-input", local_cp, own_cp)
            report("CP1-full-repeat", full_ref, run(reference, full, dtype))
            report("CP2-local-repeat", local_cp, run(target, canonical_local, dtype))
        assert [p._version for p in parameters] == versions, "replay mutated projection parameters"
