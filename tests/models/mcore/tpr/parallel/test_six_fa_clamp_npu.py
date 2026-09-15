"""Six-FA core-output + new-KV clamp; diagnostic forward only, never training.

CP1 Linear-controlled -> CP2 native -> CP2 clamped, identical random weights.
Real Ring still executes before its output is replaced. No hidden/state reset
at GDN boundaries. This deliberately changes the function; no gradient claims.
"""

from collections import Counter
from contextlib import contextmanager
import gc

import torch
import torch.distributed as dist

from .test_full_qwen35_tree_cp_npu import (
    runtime, _plan, _boundary, _cpu, _communication_probe, _shard_states,
    make_qwen35_model, broadcast_module_state, assert_hybrid_architecture,
    gradient_map_diagnostics,
)
from ._full_linear_shape_control import full_linear_shape_control


FA_LAYERS = (4, 8, 12, 16, 20, 24)


@contextmanager
def clamp_segment(model, sid, runtime, monkeypatch, *, cp, reference, store, layers, counts):
    from verl.models.mcore.tpr import attention
    from verl.models.mcore.tpr.context import TPRAttentionContext
    from verl.models.mcore.tpr.parallel import ring_attention as ring
    shard = ring.make_ring_sequence_shard(128, cp_rank=runtime.cp_group.rank(), cp_size=2)
    indices = shard.global_indices(device="cpu")
    active, handles = [], []
    rect, ring_fn, set_kv = attention.rectangular_causal_attention, ring.ring_cp_attention, TPRAttentionContext.set_new_kv

    def exchange(number, kind, value):
        assert not torch.is_grad_enabled(), "clamp cannot participate in training"
        key = (sid, number, kind)
        counts[key] += 1
        assert counts[key] == 1
        if cp == 1:
            # Every rank's CP1 capture must agree with rank0 before using it
            # as a canonical replacement for independently sharded CP2 runs.
            canonical = value.detach().clone()
            dist.broadcast(canonical, src=0, group=runtime.cp_group)
            assert torch.equal(canonical, value), f"CP1 capture differs across ranks: {key}"
            store[key] = canonical.cpu()
        if reference is not None:
            target = reference[key].index_select(0, indices).contiguous().to(value.device)
            assert target.shape == value.shape and target.dtype == value.dtype
            return target
        return value

    def local(q, k, v, **kwargs):
        output = rect(q, k, v, **kwargs)
        return exchange(active[-1], "core", output)

    def distributed(q, k, v, **kwargs):
        output = ring_fn(q, k, v, **kwargs)
        return exchange(active[-1], "core", output)

    def new_kv(ctx, number, key, value):
        assert number in FA_LAYERS and active[-1] == number
        return set_kv(ctx, number, exchange(number, "key", key), exchange(number, "value", value))

    for layer in model.decoder.layers:
        number = layer.layer_number
        def record(module, args, output, number=number):
            value = output[0] if isinstance(output, tuple) else output
            layers[(sid, number)] = value.detach().cpu().clone()
        handles.append(layer.register_forward_hook(record))
        if number in FA_LAYERS:
            def before(module, args, number=number):
                active.append(number)
            def after(module, args, output, number=number):
                assert active.pop() == number
            handles.append(layer.self_attention.register_forward_pre_hook(before))
            handles.append(layer.self_attention.register_forward_hook(after))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(attention, "rectangular_causal_attention", local)
            patch.setattr(ring, "ring_cp_attention", distributed)
            patch.setattr(TPRAttentionContext, "set_new_kv", new_kv)
            yield
    finally:
        for handle in handles:
            handle.remove()


def _forward_paths(model, runtime, monkeypatch, *, cp, reference):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import KVPrefixAnchors
    from verl.models.mcore.tpr.parallel.execution_context import ShardedPastKVAnchors
    plan = _plan()
    executor = SegmentExecutor(model, plan, expected_layer_numbers=FA_LAYERS,
                               cp_group=runtime.cp_group if cp == 2 else None,
                               cp_backend="ring" if cp == 2 else None)
    captures, layers, states, probes, logprobs = {}, {}, {}, {}, {}
    counts = Counter()
    root = None
    with _communication_probe(monkeypatch) as (communication, a2a), torch.no_grad():
        for sid in (0, 1, 2):
            kv = {} if root is None else dict(root.new_key_values)
            gdn = {} if root is None else dict(root.new_gdn_states)
            anchors = None
            if cp == 2 and root is not None:
                anchors = ShardedPastKVAnchors((KVPrefixAnchors(0, executor._segment_shard(plan.get(0)), 0, kv),), (0,))
            with clamp_segment(model, sid, runtime, monkeypatch, cp=cp, reference=reference,
                               store=captures, layers=layers, counts=counts):
                ctx, logits = executor._forward(plan.get(sid), past_key_values=kv if cp == 1 else {},
                    no_grad=True, initial_gdn_states=gdn, sharded_past_anchors=anchors)
            if sid == 0:
                root = ctx
            states[sid] = _cpu(_boundary(ctx.new_gdn_states, ctx.new_key_values))
            probes[sid] = logits[0, :, [0, 17, 1024, logits.shape[-1]-1]].detach().cpu()
            shard = executor._segment_shard(plan.get(sid))
            terms = plan.get(sid).loss_terms
            selected = [t for t in terms if shard.owns(t.query_offset)]
            offsets = torch.tensor([shard.global_to_local(t.query_offset) for t in selected], device=logits.device)
            labels = torch.tensor([t.target_token_id for t in selected], device=logits.device)
            logprobs[sid] = -torch.nn.functional.cross_entropy(logits[0].index_select(0, offsets).float(), labels, reduction="none").cpu()
        assert counts == Counter({(s, n, k): 1 for s in (0, 1, 2) for n in FA_LAYERS for k in ("core", "key", "value")})
        if cp == 2:
            assert a2a.count("cp2hp") == 324 and a2a.count("hp2cp") == 54
            assert communication["fa_ring"] == 18
            assert communication["ring_p2p"] == 30
        else:
            assert not a2a.calls and not communication
    assert all(p.grad is None for p in model.parameters()), "forward diagnostic generated parameter grads"
    print(f"SIX-FA-CLAMP r={runtime.rank} CP={cp} clamp={reference is not None} "
          f"core=18 KV=36 A2A={a2a.count('cp2hp')}/{a2a.count('hp2cp')} Ring={dict(communication)}; NO BACKWARD", flush=True)
    return dict(captures=captures, layers=layers, states=states, probes=probes, logprobs=logprobs)


def _compare(reference, actual, runtime, label):
    from verl.models.mcore.tpr.parallel.ring_attention import make_ring_sequence_shard
    indices = make_ring_sequence_shard(128, cp_rank=runtime.cp_group.rank(), cp_size=2).global_indices(device="cpu")
    def pair(left, right):
        assert left.shape == right.shape and torch.isfinite(left).all() and torch.isfinite(right).all()
        return torch.equal(left, right), gradient_map_diagnostics({"x": left.float()}, {"x": right.float()}).aggregate.relative_l2
    for sid in (0, 1, 2):
        layer_metrics = {n: pair(reference["layers"][(sid, n)].index_select(0, indices), actual["layers"][(sid, n)]) for n in range(1, 25)}
        first = next((n for n, (exact, _) in layer_metrics.items() if not exact), None)
        left_states = _shard_states(reference["states"][sid], runtime.cp_group.rank())
        state_metrics = {n: pair(left_states[n], t) for n, t in actual["states"][sid].items()}
        worst_state = max(state_metrics, key=lambda n: state_metrics[n][1])
        output = pair(reference["probes"][sid].index_select(0, indices), actual["probes"][sid])
        terms = _plan().get(sid).loss_terms
        owned = set(indices.tolist())
        selected = torch.tensor([i for i, term in enumerate(terms) if term.query_offset in owned], dtype=torch.long)
        logprob = pair(reference["logprobs"][sid].index_select(0, selected), actual["logprobs"][sid])
        print(f"SIX-FA-CLAMP r={runtime.rank} {label} S{sid} first_layer_nonexact={first} "
              f"FA-layer-output(exact,rel)={ {n: layer_metrics[n] for n in FA_LAYERS} } "
              f"all_states_exact={all(x[0] for x in state_metrics.values())} "
              f"worst_state={worst_state}:{state_metrics[worst_state]} probe={output} logprob={logprob}", flush=True)


def test_six_fa_forward_state_clamp(runtime, monkeypatch):
    results = []
    initial = None
    for cp, clamp in ((1, False), (2, False), (2, True)):
        torch.manual_seed(440001)
        model = make_qwen35_model(runtime, cp_size=cp, tpr=True)
        assert_hybrid_architecture(model)
        assert model.config.hidden_dropout == model.config.attention_dropout == 0
        if initial is None:
            broadcast_module_state(model, src=0)
            initial = {n: t.detach().cpu().clone() if isinstance(t, torch.Tensor) else t for n, t in model.state_dict().items()}
        else:
            model.load_state_dict(initial, strict=True)
        for name, tensor in model.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(tensor.detach().cpu(), initial[name]), f"initial tensor mismatch: {name}"
        with full_linear_shape_control(model, monkeypatch, enabled=cp == 1, rank=runtime.rank):
            result = _forward_paths(model, runtime, monkeypatch, cp=cp,
                                    reference=results[0]["captures"] if clamp else None)
        results.append(result)
        del model, result
        gc.collect()
        torch.npu.empty_cache()
    _compare(results[0], results[1], runtime, "native")
    _compare(results[0], results[2], runtime, "clamped")
    print("SIX-FA-CLAMP DIAGNOSTIC COMPLETE; interventions=core-output+new-KV only; "
          "not cross-CP correctness/training PASS; thresholds unchanged", flush=True)
