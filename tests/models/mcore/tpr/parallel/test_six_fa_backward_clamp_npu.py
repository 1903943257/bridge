"""Stage-4.4 six-FA forward-value clamp with real backward graph.

Purpose
-------
The forward-only six-FA diagnostic showed that replacing the six Full-Attention
core outputs + newly-produced K/V with CP1 canonical values makes the entire
24-layer CP2 forward/state path bitwise exact.

This test asks the next question:

    If the *values* at all six FA boundaries are canonicalized, while keeping
    the real CP2 autograd graph/Jacobian, how much of the CP1-vs-CP2 backward
    gap remains?

The clamp is straight-through with respect to the CP2 graph:

    y = actual + (canonical - actual).detach()

so forward(y) == canonical, but dy/dactual == 1. Ring attention and its
backward still execute normally. This is a diagnostic intervention, not a
training path and not a correctness gate by itself.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .test_full_qwen35_tree_cp_npu import (
    runtime,
    _plan,
    _boundary,
    _communication_probe,
    _shard_states,
    make_qwen35_model,
    broadcast_module_state,
    assert_hybrid_architecture,
    gradient_map_diagnostics,
)
from ._full_linear_shape_control import full_linear_shape_control


FA_LAYERS = (4, 8, 12, 16, 20, 24)
SIDS = (0, 1, 2)
KINDS = ("core", "key", "value")


class _CanonicalValueIdentityGrad(torch.autograd.Function):
    """Forward uses canonical target exactly; backward is identity to actual."""

    @staticmethod
    def forward(ctx, actual, target):
        # clone keeps the exact target bits while avoiding aliasing surprises.
        return target.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Preserve the real CP2 backward graph:
        # d(output) / d(actual) = 1
        # canonical target is diagnostic-only and receives no grad.
        return grad_output, None


def _straight_through(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert actual.shape == target.shape
    assert actual.dtype == target.dtype
    assert actual.device == target.device
    return _CanonicalValueIdentityGrad.apply(actual, target)

@contextmanager
def six_fa_backward_clamp(
    model,
    sid,
    runtime,
    monkeypatch,
    *,
    cp,
    reference,
    store,
    layers,
    counts,
):
    """Capture CP1 or straight-through clamp CP2 at six FA boundaries.

    CP1 (reference=None): capture global canonical core/new-KV tensors.
    CP2 native (reference=None): observe only; do not alter values.
    CP2 clamped (reference!=None): real Ring executes first, then forward values
    are replaced by CP1 local-shard values while preserving the CP2 graph.
    """
    from verl.models.mcore.tpr import attention
    from verl.models.mcore.tpr.context import TPRAttentionContext
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    active = []
    handles = []
    rect = attention.rectangular_causal_attention
    ring_fn = ring.ring_cp_attention
    set_kv = TPRAttentionContext.set_new_kv

    def local_indices(canonical: torch.Tensor) -> torch.Tensor:
        # CP1 capture is in global sequence order.
        shard = ring.make_ring_sequence_shard(
            canonical.shape[0],
            cp_rank=runtime.cp_group.rank(),
            cp_size=2,
        )
        return shard.global_indices(device="cpu")

    def exchange(number, kind, value):
        key = (sid, number, kind)
        counts[key] += 1
        assert counts[key] == 1, f"duplicate six-FA event: {key}"

        if cp == 1:
            canonical = value.detach().clone()

            # Both ranks execute the replicated CP1 path.
            # Rank0 is the canonical copy.
            dist.broadcast(canonical, src=0, group=runtime.cp_group)
            assert torch.equal(canonical, value), (
                f"CP1 capture differs across ranks: {key}"
            )

            store[key] = canonical.cpu()
            return value

        if reference is None:
            # CP2 native diagnostic.
            return value

        canonical = reference[key]
        indices = local_indices(canonical)
        target = (
            canonical
            .index_select(0, indices)
            .contiguous()
            .to(value.device)
        )

        assert target.shape == value.shape, (
            f"shape mismatch {key}: "
            f"target={tuple(target.shape)} actual={tuple(value.shape)}"
        )
        assert target.dtype == value.dtype

        # Forward value becomes CP1 canonical,
        # backward still flows through real CP2 value.
        return _straight_through(value, target)

    def local(q, k, v, **kwargs):
        out = rect(q, k, v, **kwargs)
        assert active
        return exchange(active[-1], "core", out)

    def distributed(q, k, v, **kwargs):
        # Real Ring forward executes first.
        out = ring_fn(q, k, v, **kwargs)
        assert active
        return exchange(active[-1], "core", out)

    def new_kv(ctx, number, key, value):
        assert active
        assert number in FA_LAYERS
        assert active[-1] == number

        key = exchange(number, "key", key)
        value = exchange(number, "value", value)

        return set_kv(ctx, number, key, value)

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
                assert active
                assert active.pop() == number

            handles.append(
                layer.self_attention.register_forward_pre_hook(before)
            )
            handles.append(
                layer.self_attention.register_forward_hook(after)
            )

    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                attention,
                "rectangular_causal_attention",
                local,
            )
            patch.setattr(
                ring,
                "ring_cp_attention",
                distributed,
            )
            patch.setattr(
                TPRAttentionContext,
                "set_new_kv",
                new_kv,
            )
            yield
    finally:
        assert not active, f"unbalanced FA hook stack: {active}"

        for handle in handles:
            handle.remove()


def _parameter_grad_map(model):
    result = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            result[name] = param.grad.detach().cpu().clone()

    return result


def _retain_boundary_grads(ctx):
    boundary = _boundary(
        ctx.new_gdn_states,
        ctx.new_key_values,
    )

    retained = {}

    for name, tensor in boundary.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        assert tensor.requires_grad, (
            f"root boundary unexpectedly detached: {name}"
        )

        tensor.retain_grad()
        retained[name] = tensor

    assert len(retained) == 48, (
        f"expected 48 root boundary tensors, got {len(retained)}"
    )

    return retained


def _boundary_grad_map(retained):
    result = {}
    missing = []

    for name, tensor in retained.items():
        if tensor.grad is None:
            missing.append(name)
        else:
            result[name] = tensor.grad.detach().cpu().clone()

    assert not missing, f"missing root boundary grads: {missing}"
    assert len(result) == 48

    return result


def _local_loss_sum(logits, segment, shard):
    terms = segment.loss_terms
    selected = [
        term
        for term in terms
        if shard.owns(term.query_offset)
    ]

    assert selected, (
        f"rank owns no loss terms for segment "
        f"{getattr(segment, 'segment_id', '?')}"
    )

    offsets = torch.tensor(
        [
            shard.global_to_local(term.query_offset)
            for term in selected
        ],
        dtype=torch.long,
        device=logits.device,
    )

    labels = torch.tensor(
        [
            term.target_token_id
            for term in selected
        ],
        dtype=torch.long,
        device=logits.device,
    )

    loss = F.cross_entropy(
        logits[0].index_select(0, offsets).float(),
        labels,
        reduction="sum",
    )

    return loss, len(selected)


def _global_loss_weight(plan):
    return sum(
        len(plan.get(sid).loss_terms)
        for sid in SIDS
    )


def _sum_parameter_grads_once(model, runtime):
    """CP2 parameter SUM exactly once over the CP group."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                group=runtime.cp_group,
            )


def _run_backward_path(
    model,
    runtime,
    monkeypatch,
    *,
    cp,
    reference,
):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import KVPrefixAnchors
    from verl.models.mcore.tpr.parallel.execution_context import (
        ShardedPastKVAnchors,
    )

    plan = _plan()
    loss_weight = _global_loss_weight(plan)

    executor = SegmentExecutor(
        model,
        plan,
        expected_layer_numbers=FA_LAYERS,
        cp_group=runtime.cp_group if cp == 2 else None,
        cp_backend="ring" if cp == 2 else None,
    )

    captures = {}
    layers = {}
    counts = Counter()

    root = None
    root_boundary = None

    local_loss = None
    local_owned = 0

    model.zero_grad(set_to_none=True)

    with _communication_probe(monkeypatch) as (
        communication,
        a2a,
    ):
        for sid in SIDS:
            segment = plan.get(sid)

            kv = (
                {}
                if root is None
                else dict(root.new_key_values)
            )

            gdn = (
                {}
                if root is None
                else dict(root.new_gdn_states)
            )

            anchors = None

            if cp == 2 and root is not None:
                anchors = ShardedPastKVAnchors(
                    (
                        KVPrefixAnchors(
                            0,
                            executor._segment_shard(plan.get(0)),
                            0,
                            kv,
                        ),
                    ),
                    (0,),
                )

            with six_fa_backward_clamp(
                model,
                sid,
                runtime,
                monkeypatch,
                cp=cp,
                reference=reference,
                store=captures,
                layers=layers,
                counts=counts,
            ):
                ctx, logits = executor._forward(
                    segment,
                    past_key_values=(
                        kv
                        if cp == 1
                        else {}
                    ),
                    no_grad=False,
                    initial_gdn_states=gdn,
                    sharded_past_anchors=anchors,
                )

            if sid == 0:
                root = ctx

                # Root prefix receives gradient contributions
                # from the two child segments.
                root_boundary = _retain_boundary_grads(root)

            shard = executor._segment_shard(segment)

            part, owned = _local_loss_sum(
                logits,
                segment,
                shard,
            )

            local_loss = (
                part
                if local_loss is None
                else local_loss + part
            )

            local_owned += owned

        expected = Counter(
            {
                (sid, layer, kind): 1
                for sid in SIDS
                for layer in FA_LAYERS
                for kind in KINDS
            }
        )

        assert counts == expected
        assert root_boundary is not None

        # Objective:
        #
        # SUM(all globally-owned query losses) / global loss weight
        #
        # CP2 parameter grads are SUMed exactly once below,
        # therefore there is no extra 1 / CP factor here.
        loss = local_loss / loss_weight

        loss.backward()

        if cp == 2:
            _sum_parameter_grads_once(
                model,
                runtime,
            )

        # Report actual global loss without changing autograd.
        loss_sum_report = local_loss.detach().clone()

        if cp == 2:
            dist.all_reduce(
                loss_sum_report,
                op=dist.ReduceOp.SUM,
                group=runtime.cp_group,
            )

        loss_report = (
            loss_sum_report / loss_weight
        ).item()

        if cp == 2:
            assert a2a.count("cp2hp") == 324
            assert a2a.count("hp2cp") == 54

            assert communication["fa_ring"] == 18

            # Full F+B:
            #   FWD Ring P2P = 30
            #   BWD Ring P2P = 30
            assert communication["ring_p2p"] == 60

        else:
            assert not a2a.calls
            assert not communication

    return {
        "captures": captures,
        "layers": layers,
        "loss": loss_report,
        "owned_loss_terms": local_owned,
        "loss_weight": loss_weight,
        "parameter_grads": _parameter_grad_map(model),
        "state_grads": _boundary_grad_map(root_boundary),
        "communication": dict(communication),
        "a2a": (
            a2a.count("cp2hp"),
            a2a.count("hp2cp"),
        ),
    }


def _diag(reference, actual):
    return gradient_map_diagnostics(
        reference,
        actual,
    )


def _print_diag(runtime, label, diag):
    agg = diag.aggregate

    print(
        f"SIX-FA-BWD r={runtime.rank} {label}: "
        f"rel={agg.relative_l2:.6e} "
        f"cos={agg.cosine:.9f} "
        f"ratio={agg.norm_ratio:.6f} "
        f"worst={diag.worst_name}:"
        f"{diag.worst.relative_l2:.6e}",
        flush=True,
    )


def _compare_forward_layers(
    cp1,
    cp2,
    runtime,
    label,
):
    from verl.models.mcore.tpr.parallel.ring_attention import (
        make_ring_sequence_shard,
    )

    all_exact = True
    worst = (0.0, None)
    first = None

    for sid in SIDS:
        for number in range(1, 25):
            ref = cp1["layers"][(sid, number)]
            actual = cp2["layers"][(sid, number)]

            shard = make_ring_sequence_shard(
                ref.shape[0],
                cp_rank=runtime.cp_group.rank(),
                cp_size=2,
            )

            indices = shard.global_indices(
                device="cpu",
            )

            ref_local = ref.index_select(
                0,
                indices,
            )

            exact = torch.equal(
                ref_local,
                actual,
            )

            if not exact:
                all_exact = False

                if first is None:
                    first = (sid, number)

                d = gradient_map_diagnostics(
                    {"x": ref_local.float()},
                    {"x": actual.float()},
                ).aggregate.relative_l2

                if d > worst[0]:
                    worst = (
                        d,
                        (sid, number),
                    )

    print(
        f"SIX-FA-BWD r={runtime.rank} "
        f"{label}/forward: "
        f"all_layers_exact={all_exact} "
        f"first_nonexact={first} "
        f"worst_rel={worst[0]:.6e}@{worst[1]}",
        flush=True,
    )

    return all_exact


def test_six_fa_backward_straight_through_clamp(
    runtime,
    monkeypatch,
):
    """CP1 controlled vs CP2 native vs CP2 six-FA value clamp."""

    results = []
    initial = None

    # 1. CP1 controlled canonical + backward
    # 2. CP2 native + backward
    # 3. CP2 canonical forward values + real CP2 backward graph
    for cp, clamp in (
        (1, False),
        (2, False),
        (2, True),
    ):
        torch.manual_seed(440001)

        model = make_qwen35_model(
            runtime,
            cp_size=cp,
            tpr=True,
        )

        assert_hybrid_architecture(model)

        assert (
            model.config.hidden_dropout
            == model.config.attention_dropout
            == 0
        )

        if initial is None:
            broadcast_module_state(
                model,
                src=0,
            )

            initial = {
                name: (
                    tensor.detach().cpu().clone()
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                )
                for name, tensor
                in model.state_dict().items()
            }

        else:
            model.load_state_dict(
                initial,
                strict=True,
            )

        for name, tensor in model.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                assert torch.equal(
                    tensor.detach().cpu(),
                    initial[name],
                ), (
                    f"initial tensor mismatch: {name}"
                )

        reference = (
            results[0]["captures"]
            if clamp
            else None
        )

        with full_linear_shape_control(
            model,
            monkeypatch,
            enabled=(cp == 1),
            rank=runtime.rank,
        ):
            result = _run_backward_path(
                model,
                runtime,
                monkeypatch,
                cp=cp,
                reference=reference,
            )

        print(
            f"SIX-FA-BWD r={runtime.rank} "
            f"CP={cp} clamp={clamp} "
            f"loss={result['loss']:.9f} "
            f"owned={result['owned_loss_terms']}/"
            f"{result['loss_weight']} "
            f"A2A={result['a2a'][0]}/"
            f"{result['a2a'][1]} "
            f"Ring={result['communication']}",
            flush=True,
        )

        results.append(result)

        del model
        gc.collect()
        torch.npu.empty_cache()

    cp1, native, clamped = results

    #
    # Forward closure
    #

    _compare_forward_layers(
        cp1,
        native,
        runtime,
        "native",
    )

    clamped_forward_exact = _compare_forward_layers(
        cp1,
        clamped,
        runtime,
        "clamped",
    )

    assert clamped_forward_exact, (
        "six-FA straight-through clamp "
        "did not close forward values"
    )

    #
    # Backward diagnostics
    #

    cp1_state_local = _shard_states(
        cp1["state_grads"],
        runtime.cp_group.rank(),
    )

    native_param = _diag(
        cp1["parameter_grads"],
        native["parameter_grads"],
    )

    clamped_param = _diag(
        cp1["parameter_grads"],
        clamped["parameter_grads"],
    )

    native_state = _diag(
        cp1_state_local,
        native["state_grads"],
    )

    clamped_state = _diag(
        cp1_state_local,
        clamped["state_grads"],
    )

    _print_diag(
        runtime,
        "native/parameter",
        native_param,
    )

    _print_diag(
        runtime,
        "clamped/parameter",
        clamped_param,
    )

    _print_diag(
        runtime,
        "native/root-state",
        native_state,
    )

    _print_diag(
        runtime,
        "clamped/root-state",
        clamped_state,
    )

    native_loss_rel = (
        abs(native["loss"] - cp1["loss"])
        / max(abs(cp1["loss"]), 1e-12)
    )

    clamp_loss_rel = (
        abs(clamped["loss"] - cp1["loss"])
        / max(abs(cp1["loss"]), 1e-12)
    )

    print(
        f"SIX-FA-BWD r={runtime.rank} "
        f"loss-rel "
        f"native={native_loss_rel:.6e} "
        f"clamped={clamp_loss_rel:.6e}; "
        f"param-rel "
        f"native={native_param.aggregate.relative_l2:.6e} "
        f"clamped={clamped_param.aggregate.relative_l2:.6e}; "
        f"state-rel "
        f"native={native_state.aggregate.relative_l2:.6e} "
        f"clamped={clamped_state.aggregate.relative_l2:.6e}",
        flush=True,
    )

    print(
        "SIX-FA-BWD DIAGNOSTIC COMPLETE; "
        "forward values canonicalized at six FA "
        "core+new-KV boundaries; "
        "CP2 backward graph retained; "
        "no threshold waived",
        flush=True,
    )