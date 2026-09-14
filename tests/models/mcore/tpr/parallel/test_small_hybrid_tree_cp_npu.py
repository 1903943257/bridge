"""Stage 4.3: GDN x3 -> FA, CP2, non-packed, random Qwen dimensions.

CP1 connected -> CP2 connected -> CP2 Push/Visit/Pop. No kernel edits,
projection controls, optimizer, Engine integration or THD claims.
"""

from collections import Counter
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available
from baseline._qwen35_baseline_utils import (
    AllToAllProbe, VOCAB_SIZE, allreduce_parameter_gradients,
    assert_gradient_maps_close, broadcast_module_state, destroy_npu_runtime,
    gather_native_zigzag, gradient_map_diagnostics, initialize_npu_runtime,
    make_qwen35_model, zigzag_indices,
)

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=2)
    yield value
    destroy_npu_runtime(value)


def _plan():
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec, SegmentLossTerm

    pieces = [(torch.arange(128) * step + start) % VOCAB_SIZE
              for step, start in ((17, 23), (19, 101), (23, 307))]
    terms = tuple(SegmentLossTerm(i, int(pieces[0][i + 1]), weight=2.) for i in range(127))
    terms += tuple(SegmentLossTerm(127, int(s[0])) for s in pieces[1:])
    segments = [SegmentSpec(0, None, pieces[0], 0, 0, terms)]
    for sid in (1, 2):
        terms = tuple(SegmentLossTerm(i, int(pieces[sid][i + 1])) for i in range(127))
        segments.append(SegmentSpec(sid, 0, pieces[sid], 128, 128, terms))
    return SegmentPlan(segments, root_id=0)


@contextmanager
def _communication_probe(monkeypatch):
    import mindspeed.core.ssm.gated_delta_net as gdn
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    counts = Counter()
    transport, _ = ring._load_mindspeed_ring_primitives()
    original_attention = ring.ring_cp_attention
    original_send = transport.async_send_recv

    def attention(*args, **kwargs):
        counts["fa_ring"] += 1
        return original_attention(*args, **kwargs)

    def send(*args, **kwargs):
        counts["ring_p2p"] += 1
        return original_send(*args, **kwargs)

    with monkeypatch.context() as patch, AllToAllProbe(gdn) as a2a:
        patch.setattr(ring, "ring_cp_attention", attention)
        patch.setattr(transport, "async_send_recv", send)
        yield counts, a2a


def _snapshot(states, kv, *, gradients=False):
    result = {}
    for layer, state in states.items():
        for kind, tensor in (("conv", state.conv_state), ("recurrent", state.recurrent_state)):
            value = tensor.grad if gradients else tensor
            assert value is not None, (layer, kind)
            result[f"gdn.{layer}.{kind}"] = value.detach().cpu().clone()
    for layer, pair in kv.items():
        for kind, tensor in zip(("key", "value"), pair):
            value = tensor.grad if gradients else tensor
            assert value is not None, (layer, kind)
            result[f"fa.{layer}.{kind}"] = value.detach().cpu().clone()
    assert len(result) == 8
    return result


def _run(model, plan, runtime, monkeypatch, *, cp, tree):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import GDNLayerState, KVPrefixAnchors
    from verl.models.mcore.tpr.parallel.execution_context import ShardedPastKVAnchors

    model.zero_grad(set_to_none=True)
    outputs, input_gradients, own_losses = {}, {}, Counter()

    class ObservedExecutor(SegmentExecutor):
        def _forward(self, segment, **kwargs):
            layer_order = []

            def check_layout(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                assert tuple(hidden.shape) == (128 // cp, 1, 1024), hidden.shape
                layer_order.append(module.layer_number)

            def capture_embedding(module, args, tensor):
                if tensor.requires_grad:
                    def gradient(value):
                        sid = segment.segment_id
                        value = value.detach().clone()
                        input_gradients[sid] = input_gradients.get(sid, 0) + value
                    tensor.register_hook(gradient)

            handle = model.embedding.register_forward_hook(capture_embedding)
            layout_handles = [layer.self_attention.register_forward_pre_hook(check_layout, with_kwargs=True)
                              for layer in model.decoder.layers]
            try:
                context, logits = super()._forward(segment, **kwargs)
            finally:
                handle.remove()
                for layout_handle in layout_handles:
                    layout_handle.remove()
            assert layer_order == [1, 2, 3, 4], layer_order
            context.assert_new_gdn_layers((1, 2, 3))
            assert tuple(context.new_key_values) == (4,)
            for state in context.new_gdn_states.values():
                assert tuple(state.conv_state.shape) == (1, 6144 // cp, 4)
                assert tuple(state.recurrent_state.shape) == (1, 16 // cp, 128, 128)
            probe = logits[0, :, [0, 17, 1024, VOCAB_SIZE - 1]].detach()
            if cp == 2:
                probe = gather_native_zigzag(probe, runtime.cp_group)
            outputs[segment.segment_id] = probe.cpu().float().clone()
            return context, logits

        def _compute_loss(self, segment, logits):
            own_losses[segment.segment_id] += 1
            return super()._compute_loss(segment, logits)

    executor = ObservedExecutor(
        model, plan, expected_layer_numbers=(4,),
        cp_group=runtime.cp_group if cp == 2 else None,
        cp_backend="ring" if cp == 2 else None,
        # Executor normally multiplies by CP for Engine's averaged reduction.
        # This standalone test uses SUM once; cancel only that loss multiplier.
        loss_scale_func=lambda loss: loss / cp,
    )
    with _communication_probe(monkeypatch) as (counts, a2a):
        if tree:
            executor.push(0)
            saved = executor.gdn_states[0]
            assert all(not t.requires_grad and t.grad_fn is None
                       for s in saved.layer_states.values() for t in (s.conv_state, s.recurrent_state))
            assert all(not t.requires_grad and t.grad_fn is None
                       for pair in executor.kv_stack.top().kv.key_values.values() for t in pair)
            loss = sum(executor.visit_leaf(sid).backward.normalized_loss for sid in (1, 2))
            states = _snapshot(saved.gradients, executor.kv_stack.top().gradients)
            loss = loss + executor.pop(0).normalized_loss
            assert saved.released and not executor.gdn_states
            executor.kv_stack.assert_empty()
        else:
            root, logits = executor._forward(plan.get(0), past_key_values={}, no_grad=False)
            loss = executor._compute_loss(plan.get(0), logits)[1]
            # Clone boundaries before branching: exclude root-owned loss from dState.
            states_live = {n: GDNLayerState(s.conv_state.clone(), s.recurrent_state.clone())
                           for n, s in root.new_gdn_states.items()}
            kv = {n: tuple(t.clone() for t in pair) for n, pair in root.new_key_values.items()}
            for s in states_live.values():
                s.conv_state.retain_grad()
                s.recurrent_state.retain_grad()
            for pair in kv.values():
                for t in pair:
                    t.retain_grad()
            anchors = None
            if cp == 2:
                entry = KVPrefixAnchors(0, executor._segment_shard(plan.get(0)), 0, kv)
                anchors = ShardedPastKVAnchors((entry,), (0,))
            for sid in (1, 2):
                _, logits = executor._forward(
                    plan.get(sid), past_key_values=kv if cp == 1 else {}, no_grad=False,
                    initial_gdn_states=states_live, sharded_past_anchors=anchors,
                )
                loss = loss + executor._compute_loss(plan.get(sid), logits)[1]
            loss.backward()
            states = _snapshot(states_live, kv, gradients=True)
            loss = loss.detach()
    assert own_losses == Counter({0: 1, 1: 1, 2: 1}), own_losses
    forwards = 4 if tree else 3
    if cp == 2:
        assert a2a.count("cp2hp") == forwards * 3 * 6
        assert a2a.count("hp2cp") == forwards * 3
        assert counts["fa_ring"] == forwards and counts["ring_p2p"] > 0, counts
        allreduce_parameter_gradients(model, runtime.cp_group)
        dist.all_reduce(loss, group=runtime.cp_group)
    else:
        assert not a2a.calls and not counts, (a2a.calls, counts)
    parameters = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None}
    assert len(parameters) == sum(p.requires_grad for p in model.parameters())
    assert set(input_gradients) == {0, 1, 2}
    for sid in sorted(input_gradients):
        value = input_gradients[sid]
        if cp == 2:
            value = gather_native_zigzag(value, runtime.cp_group)
        input_gradients[sid] = value.cpu().float()
    print(f"STAGE-4.3 CP={cp} tree={tree} loss={loss.item():.9f} "
          f"A2A={a2a.count('cp2hp')}/{a2a.count('hp2cp')} Ring={dict(counts)}", flush=True)
    return dict(loss=loss.cpu(), output=outputs, input=input_gradients, parameters=parameters, state=states)


def _shard_states(states, rank):
    indices = zigzag_indices(128, cp_rank=rank, cp_size=2, device="cpu")
    result = {}
    for name, value in states.items():
        if name.endswith("conv"):
            result[name] = torch.cat([part[:, rank * 1024:(rank + 1) * 1024]
                                      for part in value.split(2048, dim=1)], dim=1)
        elif name.endswith("recurrent"):
            result[name] = value[:, rank * 8:(rank + 1) * 8]
        else:
            result[name] = value.index_select(0, indices)
    return result


def test_small_hybrid_cp2_tree(runtime, monkeypatch):
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.gated_delta_net import TPRGatedDeltaNet

    torch.manual_seed(430001)
    reference = make_qwen35_model(runtime, cp_size=1, tpr=True, num_layers=4)
    broadcast_module_state(reference)
    target = make_qwen35_model(runtime, cp_size=2, tpr=True, num_layers=4)
    target.load_state_dict(reference.state_dict(), strict=True)
    for model in (reference, target):
        assert [type(l.self_attention) for l in model.decoder.layers] == [TPRGatedDeltaNet] * 3 + [TPRSelfAttention]
    plan = _plan()
    cp1 = _run(reference, plan, runtime, monkeypatch, cp=1, tree=False)
    cp2 = _run(target, plan, runtime, monkeypatch, cp=2, tree=False)
    tree = _run(target, plan, runtime, monkeypatch, cp=2, tree=True)
    cp1["state"] = _shard_states(cp1["state"], runtime.cp_group.rank())
    failures = []
    for label, left, right in (("CP1-connected-vs-CP2-connected", cp1, cp2),
                                ("CP2-connected-vs-tree", cp2, tree)):
        for kind in ("output", "input", "parameters", "state"):
            for name, value in left[kind].items():
                assert torch.isfinite(value).all() and torch.isfinite(right[kind][name]).all()
            print(f"STAGE-4.3 {label}/{kind}: {gradient_map_diagnostics(left[kind], right[kind])}", flush=True)
            try:
                # Retain the Stage 4.1/4.2 CP/relay envelope; no new relaxation.
                assert_gradient_maps_close(left[kind], right[kind], rtol=.02, cosine_min=.999)
                if kind == "state":
                    for name in left[kind]:
                        assert_gradient_maps_close({name: left[kind][name]}, {name: right[kind][name]},
                                                   rtol=.02, cosine_min=.999)
            except AssertionError as exc:
                failures.append(f"{label}/{kind}: {exc}")
        try:
            assert torch.isfinite(left["loss"]) and torch.isfinite(right["loss"])
            torch.testing.assert_close(left["loss"], right["loss"], atol=0, rtol=.002)
        except AssertionError as exc:
            failures.append(f"{label}/loss: {exc}")
    if failures:
        pytest.fail("\n".join(failures))
    if runtime.rank == 0:
        print("STAGE-4.3 PASS: GDN x3 + FA, connected CP controls + tree relay; not Full-Qwen/THD")
