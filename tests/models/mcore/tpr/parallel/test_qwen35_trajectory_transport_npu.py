"""One P64+S64 trajectory, suffix-only objective, whole/segmented FA A/B."""

from contextlib import contextmanager
import gc
import os

import torch
import torch.distributed as dist

from .test_full_qwen35_tree_cp_npu import (
    runtime, make_qwen35_model, broadcast_module_state, assert_hybrid_architecture,
    allreduce_parameter_gradients, gather_native_zigzag, _boundary, _cpu,
)
from .test_fa_transport_matrix_npu import metric
from ._whole_fa_allgather import install_whole_fa_allgather
from .test_small_hybrid_tree_cp_npu import _communication_probe


def plans():
    from verl.models.mcore.tpr.segment_plan import SegmentPlan, SegmentSpec, SegmentLossTerm
    p = torch.arange(64) * 17 + 23
    s = torch.arange(64) * 19 + 101
    terms = lambda shift: tuple(SegmentLossTerm(i+shift, int(s[i+1])) for i in range(63))
    whole = SegmentPlan([SegmentSpec(0, None, torch.cat((p, s)), 0, 0, terms(64))], root_id=0)
    split = SegmentPlan([SegmentSpec(0, None, p, 0, 0, ()), SegmentSpec(1, 0, s, 64, 64, terms(0))], root_id=0)
    assert whole.total_loss_weight == split.total_loss_weight == 63
    return whole, split


@contextmanager
def projection_control(model, patch, enabled):
    """Match CP2 rank-local M for both whole128 and segment64 CP1 calls."""
    counters = []
    if enabled:
        from ._full_linear_shape_control import projection_targets
        for module in projection_targets(model).values():
            original = module.forward
            counter = []
            counters.append(counter)
            def wrapped(x, *args, original=original, counter=counter, **kwargs):
                size = x.shape[0]
                assert size in (64, 128)
                a, b, c, d = x.chunk(4, dim=0)
                left = original(torch.cat((a, d)).contiguous(), *args, **kwargs)
                right = original(torch.cat((b, c)).contiguous(), *args, **kwargs)
                assert isinstance(left, tuple) and left[1] is None and right[1] is None
                l0, l1 = left[0].chunk(2)
                r0, r1 = right[0].chunk(2)
                counter.append(size)
                return torch.cat((l0, r0, r1, l1)), None
            patch.setattr(module, "forward", wrapped)
    yield
    if enabled:
        assert len(counters) == 97 and all(c == counters[0] for c in counters)
        assert counters[0] in ([128], [64, 64])


def execute(model, runtime, *, cp, segmented):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import KVPrefixAnchors
    from verl.models.mcore.tpr.parallel.execution_context import ShardedPastKVAnchors
    from verl.models.mcore.tpr.parallel.ring_attention import make_ring_sequence_shard
    whole, split = plans()
    plan = split if segmented else whole
    executor = SegmentExecutor(model, plan, expected_layer_numbers=(4, 8, 12, 16, 20, 24),
        cp_group=runtime.cp_group if cp == 2 else None, cp_backend="ring" if cp == 2 else None)
    embeddings = []
    def embedding(module, args, output):
        output.retain_grad()
        embeddings.append(output)
    handle = model.embedding.register_forward_hook(embedding)
    boundary = {}
    try:
        ctx, logits = executor._forward(plan.get(0), past_key_values={}, no_grad=False)
        if segmented:
            kv, gdn = dict(ctx.new_key_values), dict(ctx.new_gdn_states)
            boundary = _boundary(gdn, kv)
            for tensor in boundary.values():
                tensor.retain_grad()
            anchors = None if cp == 1 else ShardedPastKVAnchors(
                (KVPrefixAnchors(0, executor._segment_shard(plan.get(0)), 0, kv),), (0,))
            ctx, logits = executor._forward(plan.get(1), past_key_values=kv if cp == 1 else {}, no_grad=False,
                                            initial_gdn_states=gdn, sharded_past_anchors=anchors)
        segment = plan.get(1 if segmented else 0)
        shard = executor._segment_shard(segment)
        terms = [(i, t) for i, t in enumerate(segment.loss_terms) if shard.owns(t.query_offset)]
        offsets = torch.tensor([shard.global_to_local(t.query_offset) for _, t in terms], device=logits.device)
        labels = torch.tensor([t.target_token_id for _, t in terms], device=logits.device)
        per_token = torch.nn.functional.cross_entropy(logits[0].index_select(0, offsets).float(), labels, reduction="none")
        # Explicit global denominator; do not use Executor's CP loss scaling.
        loss = per_token.sum()/63
        logprob = torch.zeros(63, device=logits.device)
        logprob[torch.tensor([i for i, _ in terms], device=logits.device)] = -per_token.detach()
        loss.backward()
        if cp == 2:
            allreduce_parameter_gradients(model, runtime.cp_group)
            dist.all_reduce(logprob, group=runtime.cp_group)
        scalar = loss.detach().clone()
        if cp == 2:
            dist.all_reduce(scalar, group=runtime.cp_group)
        input_grads = [gather_native_zigzag(t.grad, runtime.cp_group).cpu() if cp == 2 else t.grad.cpu() for t in embeddings]
        state = _cpu({n: t.grad for n, t in boundary.items()})
        if cp == 1 and segmented:
            idx = make_ring_sequence_shard(64, cp_rank=runtime.cp_group.rank(), cp_size=2).global_indices(device="cpu")
            state = {n: (torch.cat([s.chunk(2, dim=1)[runtime.cp_group.rank()] for s in t.split(2048, dim=1)], dim=1)
                        if n.endswith("conv") else t.chunk(2, dim=1)[runtime.cp_group.rank()]
                        if n.endswith("recurrent") else t.index_select(0, idx)) for n, t in state.items()}
        return dict(loss=scalar.cpu().reshape(1), logprob=logprob.cpu(), input=torch.cat(input_grads),
                    parameters=_cpu({n: p.grad for n, p in model.named_parameters() if p.requires_grad}), state=state)
    finally:
        handle.remove()


def test_qwen35_whole_segmented_transport(runtime, monkeypatch):
    control = os.getenv("STAGE44_MATRIX_LINEAR_CONTROL", "1")
    assert control in ("0", "1")
    initial, refs = None, {}
    for backend in ("cp1", "allgather", "ring"):
        for segmented in (False, True):
            cp = 1 if backend == "cp1" else 2
            torch.manual_seed(440001)
            model = make_qwen35_model(runtime, cp_size=cp, tpr=True)
            assert_hybrid_architecture(model)
            assert model.config.hidden_dropout == model.config.attention_dropout == 0
            if initial is None:
                broadcast_module_state(model)
                initial = {n: t.detach().cpu().clone() if isinstance(t, torch.Tensor) else t for n, t in model.state_dict().items()}
            else:
                model.load_state_dict(initial)
            for name, value in model.state_dict().items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value.detach().cpu(), initial[name]), name
            label = f"{backend}/{'segmented' if segmented else 'whole'}"
            with monkeypatch.context() as patch:
                ag = install_whole_fa_allgather(patch) if backend == "allgather" else None
                with projection_control(model, patch, cp == 1 and control == "1"), _communication_probe(patch) as (comm, a2a):
                    result = execute(model, runtime, cp=cp, segmented=segmented)
                if cp == 2:
                    assert a2a.count("cp2hp") == (216 if segmented else 108)
                    assert a2a.count("hp2cp") == (36 if segmented else 18)
                    assert comm["fa_ring"] == (12 if segmented else 6)
                    if ag is not None:
                        assert ag["all_gather"] == ag["reduce_scatter"] == (48 if segmented else 18)
                        assert comm["ring_p2p"] == 0
                    else:
                        assert comm["ring_p2p"] == (36 if segmented else 12)
                else:
                    assert not comm and not a2a.calls
            refs[label] = result
            print(f"TRAJECTORY r={runtime.rank} {label} loss={result['loss'].item():.9f} "
                  f"CP1-projection-control={control} objective=S[0:63]->S[1:64]/63 "
                  f"FA-calls={comm['fa_ring']} RingP2P={comm['ring_p2p']} AG={dict(ag or {})} "
                  f"state={'boundary VJP' if segmented else 'N/A: no explicit intermediate GDN state'}", flush=True)
            comparisons = [("cp1/whole", refs["cp1/whole"])]
            if segmented:
                comparisons.append((f"{backend}/whole", refs[f"{backend}/whole"]))
                comparisons.append(("cp1/segmented", refs["cp1/segmented"]))
            for reference_label, ref in dict(comparisons).items():
                for kind in ("loss", "logprob", "input", "parameters"):
                    l = ref[kind] if kind == "parameters" else {kind: ref[kind]}
                    r = result[kind] if kind == "parameters" else {kind: result[kind]}
                    metric(f"{reference_label}-vs-{label}/{kind}", l, r, runtime.rank)
            if segmented:
                metric(f"cp1-segmented-vs-{label}/state", refs["cp1/segmented"]["state"], result["state"], runtime.rank)
                if backend != "cp1":
                    del refs[f"{backend}/whole"], refs[label]
            del model, result
            gc.collect()
            torch.npu.empty_cache()
    print("TRAJECTORY MATRIX COMPLETE: diagnostics only, no threshold relaxation, no training PASS", flush=True)
