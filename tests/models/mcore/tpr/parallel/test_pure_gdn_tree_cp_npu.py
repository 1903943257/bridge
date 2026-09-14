"""Stage 4.2: three residual GDN layers, R->P->(S1,S2), CP2 tree relay.

Four controls: CP1 materialized, CP1 connected split, CP2 connected split,
CP2 graph-free tree. No FA/MLP/THD/Engine or projection chunk intervention.
"""

from collections import Counter

import pytest
import torch
import torch.distributed as dist

from .test_stateful_gdn_cp_npu import runtime, _model, _random, _state_shard, _grads  # noqa: F401
from baseline._qwen35_baseline_utils import (
    AllToAllProbe, HIDDEN_SIZE, broadcast_module_state, allreduce_parameter_gradients,
    gather_native_zigzag, zigzag_indices, gradient_map_diagnostics, assert_gradient_maps_close,
)


LENGTH = 128
LAYERS = 3
DENOMINATOR = 6 * LENGTH * HIDDEN_SIZE


def _forward(model, x, initial=None, prefix_length=0):
    if model.cp_size == 2:
        return model(x, initial)
    # CP1 reference retains the Stage 3.2 implementation, not the CP adapter.
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context

    context = TPRAttentionContext(prefix_length=prefix_length,
                                  suffix_length=x.shape[0], initial_gdn_states=initial or {})
    with use_tpr_attention_context(context):
        for layer in model.layers:
            output, bias = layer(x, attention_mask=None)
            assert bias is None
            x = x + output
    context.assert_new_gdn_layers(model.layer_numbers)
    return x, context.new_gdn_states


def _loss(output, target, weight):
    return (output.float() - target.float()).square().sum() * weight / DENOMINATOR


def _boundary(states, *, clone=False):
    from verl.models.mcore.tpr.prefix_state import GDNLayerState

    if not clone:
        return {f"{number}.{kind}": tensor for number, state in states.items()
                for kind, tensor in (("conv", state.conv_state), ("recurrent", state.recurrent_state))}
    result = {number: GDNLayerState(state.conv_state.clone(), state.recurrent_state.clone())
              for number, state in states.items()}
    for tensor in _boundary(result).values():
        tensor.retain_grad()
    return result


def _materialized(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    outputs, dx = {}, {}
    losses = []
    for leaf in (2, 3):
        ids = (0, 1, leaf)
        x = torch.cat([inputs[i] for i in ids]).detach().requires_grad_(True)
        output, _ = _forward(model, x)
        loss = _loss(output, torch.cat([targets[i] for i in ids]), 1)
        loss.backward()
        losses.append(loss.detach())
        for j, sid in enumerate(ids):
            outputs[sid] = output[j*LENGTH:(j+1)*LENGTH].detach().clone()
            grad = x.grad[j*LENGTH:(j+1)*LENGTH].detach().clone()
            dx[sid] = grad if sid not in dx else dx[sid] + grad
    return dict(output=outputs, dx=dx, loss=sum(losses), params=_grads(model))


def _connected(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    xs, outputs, states, boundaries = {}, {}, {}, {}
    total = 0
    for sid, parent in ((0, None), (1, 0), (2, 1), (3, 1)):
        xs[sid] = inputs[sid].detach().clone().requires_grad_(True)
        output, state = _forward(model, xs[sid], None if parent is None else boundaries[parent],
                                 prefix_length=0 if sid == 0 else LENGTH if sid == 1 else 2*LENGTH)
        outputs[sid], states[sid] = output, state
        if sid in (0, 1):
            # Isolate EXTERNAL state VJPs from any internal use of state outputs.
            boundaries[sid] = _boundary(state, clone=True)
        total = total + _loss(output, targets[sid], 2 if sid < 2 else 1)
    total.backward()
    dstate = {}
    for sid, boundary in boundaries.items():
        from verl.models.mcore.tpr.prefix_state import GDNLayerState

        dstate[sid] = {}
        for number, state in boundary.items():
            assert state.conv_state.grad is not None and state.recurrent_state.grad is not None
            dstate[sid][number] = GDNLayerState(state.conv_state.grad.detach().clone(),
                                              state.recurrent_state.grad.detach().clone())
    return dict(output={i: x.detach().clone() for i, x in outputs.items()},
                dx={i: x.grad.detach().clone() for i, x in xs.items()},
                loss=total.detach(), params=None, states=dstate)


def _tree(model, inputs, targets):
    from verl.models.mcore.tpr.parallel.gdn_tree import GDNCPBranchExecutor
    from verl.models.mcore.tpr.prefix_state import GDNLayerState

    model.zero_grad(set_to_none=True)
    executor = GDNCPBranchExecutor(model)
    calls = Counter()

    def own_loss(sid):
        def calculate(output):
            calls[sid] += 1
            return _loss(output, targets[sid], 2 if sid < 2 else 1)
        return calculate

    executor.push(0, None, inputs[0], own_loss(0))
    root = executor.stack[-1].state
    executor.push(1, 0, inputs[1], own_loss(1))
    middle = executor.stack[-1].state
    for state in (root, middle):
        for tensor in _boundary(state.layer_states).values():
            assert not tensor.requires_grad and tensor.grad_fn is None
    assert not calls, "graph-free Push computed owned loss"
    results = {2: executor.visit(2, 1, inputs[2], own_loss(2))}
    first_sibling = {name: value.clone() for name, value in _boundary(middle.gradients).items()}
    results[3] = executor.visit(3, 1, inputs[3], own_loss(3))
    assert not root.gradients, "GDN leaf must relay only to its direct parent"
    def snapshot(state):
        return {n: GDNLayerState(s.conv_state.detach().clone(), s.recurrent_state.detach().clone())
                for n, s in state.gradients.items()}

    states = {1: snapshot(middle)}
    assert set(first_sibling) == set(_boundary(states[1]))
    # Connected control below validates the SUM, not just its existence.
    results[1] = executor.pop(1)
    assert middle.released
    states[0] = snapshot(root)
    assert len(states[0]) == len(states[1]) == LAYERS
    results[0] = executor.pop(0)
    assert root.released and calls == Counter({0: 1, 1: 1, 2: 1, 3: 1})
    executor.assert_empty()
    return dict(output={i: r.output for i, r in results.items()},
                dx={i: r.input_gradient for i, r in results.items()},
                loss=sum(r.loss for r in results.values()), params=None, states=states)


def _finish_cp(model, result, runtime):
    allreduce_parameter_gradients(model, runtime.cp_group)
    result["params"] = _grads(model)
    dist.all_reduce(result["loss"], group=runtime.cp_group)
    for kind in ("output", "dx"):
        # Same collective order on every rank even though tree results are reversed.
        result[kind] = {sid: gather_native_zigzag(result[kind][sid], runtime.cp_group)
                        for sid in range(4)}


def test_three_gdn_cp2_push_branch_pop(runtime):
    import mindspeed.core.ssm.gated_delta_net as native
    from verl.models.mcore.tpr.parallel.gdn_tree import GDNCPStack

    torch.manual_seed(420001)
    ref = GDNCPStack([_model(runtime, 1, i) for i in range(1, LAYERS+1)])
    cp = GDNCPStack([_model(runtime, 2, i) for i in range(1, LAYERS+1)])
    broadcast_module_state(ref)
    cp.load_state_dict(ref.state_dict(), strict=True)
    inputs = {i: _random((LENGTH, 1, HIDDEN_SIZE), runtime.device, 420010+i) for i in range(4)}
    targets = {i: _random((LENGTH, 1, HIDDEN_SIZE), runtime.device, 420020+i) for i in range(4)}
    indices = zigzag_indices(LENGTH, cp_rank=runtime.cp_group.rank(), cp_size=2, device=runtime.device)
    local_x = {i: x.index_select(0, indices) for i, x in inputs.items()}
    local_y = {i: y.index_select(0, indices) for i, y in targets.items()}
    with AllToAllProbe(native) as ref_probe:
        materialized = _materialized(ref, inputs, targets)
        connected_ref = _connected(ref, inputs, targets)
        connected_ref["params"] = _grads(ref)
    with AllToAllProbe(native) as connected_probe:
        connected_cp = _connected(cp, local_x, local_y)
    _finish_cp(cp, connected_cp, runtime)
    with AllToAllProbe(native) as tree_probe:
        tree = _tree(cp, local_x, local_y)
    _finish_cp(cp, tree, runtime)
    assert not ref_probe.calls
    assert connected_probe.count("cp2hp") == 4*LAYERS*6 and connected_probe.count("hp2cp") == 4*LAYERS
    assert tree_probe.count("cp2hp") == 6*LAYERS*6 and tree_probe.count("hp2cp") == 6*LAYERS
    print(f"STAGE-4.2 rank={runtime.rank} CP1 A2A=0; CP2 connected=72/12 tree=108/18; "
          "own loss once; direct-parent relay; both prefixes released", flush=True)
    failures = []

    def gate(label, left, right, tol):
        try:
            for name in left:
                assert torch.isfinite(left[name]).all() and torch.isfinite(right[name]).all(), name
            print(f"STAGE-4.2 {label}: {gradient_map_diagnostics(left, right)}", flush=True)
            assert_gradient_maps_close(left, right, rtol=tol, cosine_min=0.995 if tol == 0.08 else 0.999)
            print(f"STAGE-4.2 GATE PASS {label}", flush=True)
        except AssertionError as error:
            failures.append(label)
            print(f"STAGE-4.2 GATE FAIL {label}: {error}", flush=True)

    for label, left, right, tol in (
        ("CP1-materialized-vs-CP2-tree", materialized, tree, 0.08),
        ("CP1-connected-vs-CP2-connected", connected_ref, connected_cp, 0.02),
        ("CP2-connected-vs-tree", connected_cp, tree, 0.02),
    ):
        print(f"STAGE-4.2 {label} loss={left['loss'].item():.9e}/{right['loss'].item():.9e}", flush=True)
        gate(label+"/loss", {"loss": left["loss"]}, {"loss": right["loss"]}, 0.002)
        gate(label+"/parameters", left["params"], right["params"], tol)
        for sid in range(4):
            gate(f"{label}/output/{sid}", {"output": left["output"][sid]}, {"output": right["output"][sid]}, tol)
            gate(f"{label}/dx/{sid}", {"dx": left["dx"][sid]}, {"dx": right["dx"][sid]}, tol)
    for sid in (0, 1):
        local_ref = {n: _state_shard(s, runtime.cp_group.rank()) for n, s in connected_ref["states"][sid].items()}
        for label, left, right in (
            ("CP1-vs-CP2/dState", local_ref, connected_cp["states"][sid]),
            ("CP2-connected-vs-tree/dState", connected_cp["states"][sid], tree["states"][sid]),
        ):
            lmap, rmap = _boundary(left), _boundary(right)
            for name in lmap:
                gate(f"{label}/segment={sid}/{name}", {name: lmap[name]}, {name: rmap[name]}, 0.02)
    if failures:
        pytest.fail("Stage 4.2 failed gates: " + ", ".join(failures))
    print(f"STAGE-4.2 PASS rank={runtime.rank}: 3 GDN CP2, R->P->(S1,S2)", flush=True)
