"""Bounded module-boundary CP diagnostics. No replay/control or new gates."""

from contextlib import contextmanager
import hashlib
import json

import torch


def plan_digest(plan):
    payload = [(s.segment_id, s.parent_id, s.position_start, s.prefix_length,
                s.token_ids.tolist(), [(t.query_offset, t.target_token_id, t.weight) for t in s.loss_terms])
               for s in plan.segments.values()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parameter_category(name):
    if "norm" in name:
        return "norm"
    if "embedding" in name or "output_layer" in name:
        return "embedding/output"
    if ".mlp." in name:
        return "MLP"
    if ".self_attention." in name:
        layer = int(name.split("decoder.layers.", 1)[1].split(".", 1)[0]) + 1
        return "FA" if layer % 4 == 0 else "GDN"
    return "other"


def parameter_summary(reference, actual, metrics, rank):
    assert reference.keys() == actual.keys()
    groups = {}
    for name in reference:
        groups.setdefault(parameter_category(name), []).append(name)
    for group, names in sorted(groups.items()):
        pair = metrics({n: reference[n] for n in names}, {n: actual[n] for n in names}).aggregate
        print(f"STAGE-4.4 CATEGORY r={rank} {group} n={len(names)} "
              f"norm={pair.reference_norm:.6e}/{pair.actual_norm:.6e} ratio={pair.norm_ratio:.6f} "
              f"rel={pair.relative_l2:.6e} cos={pair.cosine:.9f}", flush=True)


class LayerDriftProbe:
    """Store only native token-layout boundaries, sliced to this rank on CPU.

    No model references retained after a segment. VJPs use each run's OWN
    upstream gradients; reverse traversal is not a same-upstream kernel test.
    """

    def __init__(self, cp, rank):
        self.cp, self.rank = cp, rank
        self.records = {}
        self.order = {}

    @contextmanager
    def segment(self, model, sid):
        handles = []
        order = self.order.setdefault(sid, [])

        def capture(name, value):
            value = value[0] if isinstance(value, tuple) else value
            assert isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1] == 1
            assert value.shape[0] == 128 // self.cp

            def snapshot(tensor):
                tensor = tensor.detach().cpu().float()
                if self.cp == 1:
                    rank = self.rank
                    tensor = torch.cat((tensor[rank*32:(rank+1)*32], tensor[(3-rank)*32:(4-rank)*32]))
                return tensor.clone()

            key = (sid, name)
            assert key not in self.records
            entry = {"value": snapshot(value), "grad": None}
            self.records[key] = entry
            order.append(name)
            if value.requires_grad:
                def gradient(tensor):
                    tensor = snapshot(tensor)
                    entry["grad"] = tensor if entry["grad"] is None else entry["grad"] + tensor
                value.register_hook(gradient)

        def watch(module, name, inputs=True):
            if inputs:
                def before(module, args, kwargs):
                    capture(name + ".in", args[0] if args else kwargs["hidden_states"])
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))

            def after(module, args, result):
                capture(name + ".out", result)
            handles.append(module.register_forward_hook(after))

        watch(model.embedding, "embedding", inputs=False)
        for layer in model.decoder.layers:
            name = f"L{layer.layer_number:02d}"
            watch(layer, name)
            watch(layer.self_attention, name + ".attn")
            watch(layer.mlp, name + ".mlp")
        watch(model.decoder.final_layernorm, "final_norm")
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def compare(self, actual, metrics):
        assert self.order == actual.order and self.records.keys() == actual.records.keys()
        errors = {}
        for key, left_entry in self.records.items():
            for kind in ("value", "grad"):
                left, right = left_entry[kind], actual.records[key][kind]
                assert left is not None and right is not None, (key, kind)
                assert left.shape == right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
                errors[(*key, kind)] = metrics({"x": left}, {"x": right}).aggregate.relative_l2
        for sid, order in self.order.items():
            first = next((n for n in order if errors[(sid, n, "value")] != 0), None)
            backward = next((n for n in reversed(order) if errors[(sid, n, "grad")] != 0), None)
            print(f"STAGE-4.4 FIRST r={self.rank} s={sid} forward={first} "
                  f"backward(reverse-boundary)={backward}; own-upstream VJP", flush=True)
        # Max across the three segments, not aggregate/mean; each arrow compares
        # errors at different boundaries, NOT an isolated operator error.
        print(f"STAGE-4.4 LAYERS r={self.rank} rel-L2 max(P,S1,S2): "
              "F=forward B=backward; attn/mlp arrows input->output, B output->input", flush=True)
        jumps = []
        for layer in range(1, 25):
            base = f"L{layer:02d}"
            def error(point, kind):
                return max(errors[(sid, point, kind)] for sid in self.order)
            parts = []
            for kind, label in (("value", "F"), ("grad", "B")):
                start, end = ("in", "out") if kind == "value" else ("out", "in")
                parts.append(f"{label}layer={error(base+'.'+start, kind):.4e}->{error(base+'.'+end, kind):.4e}")
                for block in ("attn", "mlp"):
                    start, end = ("in", "out") if kind == "value" else ("out", "in")
                    left, right = error(f"{base}.{block}.{start}", kind), error(f"{base}.{block}.{end}", kind)
                    parts.append(f"{label}{block}={left:.4e}->{right:.4e}")
                    jumps.append((right-left, base, label, block))
            print(f"STAGE-4.4 LAYER r={self.rank} {base} " + " ".join(parts), flush=True)
        for kind in ("F", "B"):
            top = sorted((j for j in jumps if j[2] == kind and j[0] > 0), reverse=True)[:3]
            print(f"STAGE-4.4 TOP-INCREASE r={self.rank} {kind}: {top}; localization only, not kernel attribution", flush=True)
