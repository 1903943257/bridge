"""Stage 4.4: full random Qwen3.5 Hybrid, non-packed CP2 Engine tree.

CP1 connected vs CP2 connected measures CP numerical drift separately from
CP2 connected vs Engine tree relay. No shape controls, optimizer or kernels
are changed. An Engine fixture supplies random models and a SUM finalizer;
this does not validate the production distributed optimizer/DDP wrapper.
"""

from collections import Counter
from contextlib import nullcontext
import gc
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available
from baseline._qwen35_baseline_utils import (
    allreduce_parameter_gradients, assert_gradient_maps_close,
    assert_hybrid_architecture, broadcast_module_state, destroy_npu_runtime,
    gather_native_zigzag, gradient_map_diagnostics, initialize_npu_runtime,
    make_qwen35_model, VOCAB_SIZE,
)
from .test_small_hybrid_tree_cp_npu import _plan, _communication_probe, _shard_states

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=2)
    yield value
    destroy_npu_runtime(value)


def _boundary(gdn, kv):
    tensors = {}
    for layer, state in gdn.items():
        tensors[f"gdn.{layer}.conv"] = state.conv_state
        tensors[f"gdn.{layer}.recurrent"] = state.recurrent_state
    for layer, pair in kv.items():
        tensors[f"fa.{layer}.key"], tensors[f"fa.{layer}.value"] = pair
    assert len(tensors) == 48
    return tensors


def _cpu(tensors):
    result = {}
    for name, tensor in tensors.items():
        assert tensor is not None, f"missing tensor/gradient: {name}"
        assert torch.isfinite(tensor).all(), f"nonfinite: {name}"
        result[name] = tensor.detach().cpu().clone()
    return result


def _engine_run(model, plan, executor_type, runtime, monkeypatch):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.models.mcore.tpr.engine_adapter import TPR_REQUEST_KEY, TPRForwardBackwardRequest
    from verl.models.mcore.tpr import megatron_adapter

    # Same thin-entry fixture as Stage 3.3: runtime already installed MindSpeed.
    # Suppress only the unrelated eager engine repatcher during import.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "verl.workers.engine.mindspeed", ModuleType("verl.workers.engine.mindspeed"))
        from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [model]
    engine.engine_config = SimpleNamespace(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=2, expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None, tpr_enabled=True,
        tpr_cp_backend="ring", pad_bshd_to_minibatch_max=False,
        use_remove_padding=False, dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = model.config
    engine.enable_routing_replay = False
    engine._distillation_use_topk_active = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None
    model.config.no_sync_func = None
    model.config.calculate_per_token_loss = False
    # Cancel executor CP multiplication because this fixture SUMs (not averages)
    # parameter gradients. State VJPs now match the connected global objective.
    model.config.grad_scale_func = lambda loss: loss / 2
    finalizations = []

    def finalize(modules, num_tokens, **kwargs):
        assert modules == [model] and num_tokens is None and kwargs["force_all_reduce"]
        assert not finalizations, "duplicate Engine gradient finalization"
        allreduce_parameter_gradients(model, runtime.cp_group)
        finalizations.append(True)

    model.config.finalize_model_grads_func = finalize
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})

    def unexpected_loss(*args, **kwargs):
        pytest.fail("tree request entered ordinary Engine loss path")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(megatron_adapter, "SegmentExecutor", executor_type)
            result = engine.forward_backward_batch(data, loss_function=unexpected_loss, forward_only=False)
        assert len(finalizations) == 1
        assert result["metrics"]["tpr_cp_size"] == 2
        assert result["metrics"]["tpr_cp_backend"] == "ring"
        return torch.tensor(result["loss"], dtype=torch.float32)
    finally:
        # Break the model/config/finalizer closure before releasing NPU storage.
        model.config.finalize_model_grads_func = None


def _run(model, plan, runtime, monkeypatch, *, cp, tree, trace=False):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import GDNLayerState, KVPrefixAnchors
    from verl.models.mcore.tpr.parallel.execution_context import ShardedPastKVAnchors

    gdn_layers, fa_layers = assert_hybrid_architecture(model)
    gdn_numbers = tuple(layer.layer_number for layer in gdn_layers)
    fa_numbers = tuple(layer.layer_number for layer in fa_layers)
    model.zero_grad(set_to_none=True)
    instances, forwards, saved_states = [], [], []
    outputs, logprobs, input_gradients, state_gradients = {}, {}, {}, {}
    loss_calls = Counter()
    drift = None
    if trace:
        from ._full_hybrid_drift_diagnostic import LayerDriftProbe
        drift = LayerDriftProbe(cp, runtime.cp_group.rank())

    class ObservedExecutor(SegmentExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            assert self.cp_size == cp and self.expected_layer_numbers == fa_numbers
            if cp == 2:
                assert self.cp_backend.backend_name == "ring"
                assert self.cp_group is runtime.cp_group
            instances.append(self)

        def _forward(self, segment, **kwargs):
            order = []

            def layout(module, args, kw):
                hidden = args[0] if args else kw["hidden_states"]
                assert tuple(hidden.shape) == (128 // cp, 1, 1024)
                order.append(module.layer_number)

            def embedding(module, args, output):
                if output.requires_grad:
                    def gradient(value):
                        sid = segment.segment_id
                        input_gradients[sid] = input_gradients.get(sid, 0) + value.detach().clone()
                    output.register_hook(gradient)

            handles = [model.embedding.register_forward_hook(embedding)]
            def check_model_inputs(module, args, kw):
                shard = self._segment_shard(segment)
                index = shard.global_indices(device=kw["input_ids"].device)
                if cp == 2:
                    rank = runtime.cp_group.rank()
                    expected = torch.cat((torch.arange(rank*32, (rank+1)*32, device=index.device),
                                          torch.arange((3-rank)*32, (4-rank)*32, device=index.device)))
                    assert torch.equal(index, expected), "FA shard does not match native GDN zigzag"
                assert torch.equal(kw["input_ids"][0], segment.token_ids.to(index.device).index_select(0, index))
                assert torch.equal(kw["position_ids"][0], index + segment.position_start)
            handles.append(model.register_forward_pre_hook(check_model_inputs, with_kwargs=True))
            handles += [layer.self_attention.register_forward_pre_hook(layout, with_kwargs=True)
                        for layer in model.decoder.layers]
            try:
                with drift.segment(model, segment.segment_id) if drift is not None else nullcontext():
                    context, logits = super()._forward(segment, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            assert order == list(range(1, 25))
            context.assert_new_gdn_layers(gdn_numbers)
            context.assert_new_kv_layers(fa_numbers)
            for state in context.new_gdn_states.values():
                assert tuple(state.conv_state.shape) == (1, 6144 // cp, 4)
                assert tuple(state.recurrent_state.shape) == (1, 16 // cp, 128, 128)
            forwards.append((segment.segment_id, kwargs["no_grad"]))
            probe = logits[0, :, [0, 17, 1024, VOCAB_SIZE - 1]].detach()
            if cp == 2:
                probe = gather_native_zigzag(probe, runtime.cp_group)
            outputs[segment.segment_id] = probe.cpu().float().clone()
            return context, logits

        def _compute_loss(self, segment, logits):
            loss_calls[segment.segment_id] += 1
            # Recover ordered logical loss terms, including the two distinct
            # next-token targets owned by the shared prefix's last query.
            shard = self._segment_shard(segment)
            selected = [(i, term) for i, term in enumerate(segment.loss_terms) if shard.owns(term.query_offset)]
            values = torch.zeros(len(segment.loss_terms), device=logits.device, dtype=torch.float32)
            if selected:
                offsets = torch.tensor([shard.global_to_local(t.query_offset) for _, t in selected], device=logits.device)
                targets = torch.tensor([t.target_token_id for _, t in selected], device=logits.device)
                with torch.no_grad():
                    local = -F.cross_entropy(logits[0].index_select(0, offsets).float(), targets, reduction="none")
                values[torch.tensor([i for i, _ in selected], device=logits.device)] = local
            if cp == 2:
                dist.all_reduce(values, group=runtime.cp_group)
            logprobs[segment.segment_id] = values.cpu()
            return super()._compute_loss(segment, logits)

        def push(self, sid):
            result = super().push(sid)
            saved_states.append(self.gdn_states[sid])
            for tensor in _boundary(self.gdn_states[sid].layer_states, self.kv_stack.top().kv.key_values).values():
                assert not tensor.requires_grad and tensor.grad_fn is None
            return result

        def pop(self, sid):
            state_gradients.update(_cpu(_boundary(self.gdn_states[sid].gradients, self.kv_stack.top().gradients)))
            return super().pop(sid)

    with _communication_probe(monkeypatch) as (counts, a2a):
        if tree:
            assert cp == 2
            loss = _engine_run(model, plan, ObservedExecutor, runtime, monkeypatch)
            assert forwards == [(0, True), (1, False), (2, False), (0, False)]
            assert len(saved_states) == 1 and saved_states[0].released
        else:
            executor = ObservedExecutor(model, plan, expected_layer_numbers=fa_numbers,
                                        cp_group=runtime.cp_group if cp == 2 else None,
                                        cp_backend="ring" if cp == 2 else None)
            root, logits = executor._forward(plan.get(0), past_key_values={}, no_grad=False)
            loss = executor._compute_loss(plan.get(0), logits)[1]
            gdn = {n: GDNLayerState(s.conv_state.clone(), s.recurrent_state.clone()) for n, s in root.new_gdn_states.items()}
            kv = {n: tuple(t.clone() for t in pair) for n, pair in root.new_key_values.items()}
            boundary = _boundary(gdn, kv)
            for tensor in boundary.values():
                tensor.retain_grad()
            anchors = None
            if cp == 2:
                anchors = ShardedPastKVAnchors((KVPrefixAnchors(0, executor._segment_shard(plan.get(0)), 0, kv),), (0,))
            for sid in (1, 2):
                _, logits = executor._forward(plan.get(sid), past_key_values=kv if cp == 1 else {},
                    no_grad=False, initial_gdn_states=gdn, sharded_past_anchors=anchors)
                loss = loss + executor._compute_loss(plan.get(sid), logits)[1]
            loss.backward()
            state_gradients = _cpu({n: t.grad for n, t in boundary.items()})
            loss = loss.detach()
            if cp == 2:
                allreduce_parameter_gradients(model, runtime.cp_group)
                dist.all_reduce(loss, group=runtime.cp_group)
            assert forwards == [(0, False), (1, False), (2, False)]
    assert len(instances) == 1
    assert not instances[0].gdn_states
    instances[0].kv_stack.assert_empty()
    assert loss_calls == Counter({0: 1, 1: 1, 2: 1})
    assert len(state_gradients) == 48
    if cp == 2:
        num_forwards = 4 if tree else 3
        assert a2a.count("cp2hp") == num_forwards * 18 * 6
        assert a2a.count("hp2cp") == num_forwards * 18
        assert counts["fa_ring"] == num_forwards * 6 and counts["ring_p2p"] > 0
    else:
        assert not a2a.calls and not counts
    parameters = _cpu({n: p.grad for n, p in model.named_parameters() if p.requires_grad})
    assert set(input_gradients) == {0, 1, 2}
    for sid in sorted(input_gradients):
        tensor = input_gradients[sid]
        if cp == 2:
            tensor = gather_native_zigzag(tensor, runtime.cp_group)
        input_gradients[sid] = tensor.cpu().float()
    print(f"STAGE-4.4 CP={cp} Engine-tree={tree} loss={loss.item():.9f} "
          f"A2A={a2a.count('cp2hp')}/{a2a.count('hp2cp')} Ring={dict(counts)} "
          "boundary-gradients=48 owned-loss-once=True caches-empty=True", flush=True)
    return dict(loss=loss.cpu(), output=outputs, logprob=logprobs, input=input_gradients,
                parameters=parameters, state=state_gradients, drift=drift)


def _brief(pair):
    return (f"norm={pair.reference_norm:.6e}/{pair.actual_norm:.6e} ratio={pair.norm_ratio:.6f} "
            f"abs={pair.absolute_l2:.6e} rel={pair.relative_l2:.6e} cos={pair.cosine:.9f}")


def _compare(label, reference, actual, *, rank):
    failures = []
    state_worst = []
    verbose = os.getenv("STAGE44_VERBOSE", "0") == "1"
    for kind in ("output", "logprob", "input", "parameters", "state"):
        left, right = reference[kind], actual[kind]
        assert left.keys() == right.keys()
        pairs = [(kind, left, right)]
        if kind == "state":
            pairs += [(f"state/{name}", {name: left[name]}, {name: right[name]}) for name in left]
        for item, first, second in pairs:
            try:
                for name in first:
                    assert torch.isfinite(first[name]).all() and torch.isfinite(second[name]).all()
                diagnostic = gradient_map_diagnostics(first, second)
                if item.startswith("state/"):
                    state_worst.append((diagnostic.aggregate.relative_l2, item, diagnostic.aggregate.cosine))
                if verbose or not item.startswith("state/"):
                    print(f"STAGE-4.4 r={rank} {label}/{item}: {_brief(diagnostic.aggregate)}", flush=True)
                # Preserve the Stage 4.3 CP/relay envelope, not an assumed
                # larger allowance based on layer count. Print every failure.
                assert_gradient_maps_close(first, second, rtol=.02, cosine_min=.999)
            except AssertionError as exc:
                failures.append(f"{label}/{item}: {exc}")
    try:
        assert torch.isfinite(reference["loss"]) and torch.isfinite(actual["loss"])
        torch.testing.assert_close(reference["loss"], actual["loss"], atol=0, rtol=.002)
    except AssertionError as exc:
        failures.append(f"{label}/loss: {exc}")
    print(f"STAGE-4.4 r={rank} {label}/state-top3(rel,name,cos): {sorted(state_worst, reverse=True)[:3]}", flush=True)
    print(f"STAGE-4.4 GATE r={rank} {label}: {'FAIL' if failures else 'PASS'} failed_checks={len(failures)}", flush=True)
    return failures


def _relay_diagnostic(label, reference, actual, *, rank):
    """Print repeatability/relay evidence only; never change numerical gates."""
    left_loss, right_loss = reference["loss"].item(), actual["loss"].item()
    print(f"STAGE-4.4 DIAGNOSTIC rank={rank} {label}/loss: "
          f"reference={left_loss:.9f} actual={right_loss:.9f} "
          f"relative_diff={abs(right_loss-left_loss)/max(abs(left_loss), 1e-24):.9e}", flush=True)
    for kind in ("parameters", "state"):
        print(f"STAGE-4.4 DIAGNOSTIC rank={rank} {label}/{kind}: "
              f"{_brief(gradient_map_diagnostics(reference[kind], actual[kind]).aggregate)}", flush=True)
    if os.getenv("STAGE44_VERBOSE", "0") != "1":
        return
    name = "gdn.5.conv"
    left, right = reference["state"][name], actual["state"][name]
    assert left.shape == right.shape == (1, 3072, 4), (left.shape, right.shape)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    print(f"STAGE-4.4 DIAGNOSTIC rank={rank} {label}/{name}: "
          f"shape={tuple(left.shape)} dtype={left.dtype}/{right.dtype}; "
          "slots=last-axis (state storage order)", flush=True)
    for slot in (None, 0, 1, 2, 3):
        first, second = (left, right) if slot is None else (left[..., slot], right[..., slot])
        tag = "all" if slot is None else f"slot{slot}"
        pair = gradient_map_diagnostics({name: first}, {name: second}).aggregate
        print(f"STAGE-4.4 DIAGNOSTIC rank={rank} {label}/{name}/{tag}: "
              f"exact={torch.equal(first, second)} {pair} "
              f"max_abs={(second.float()-first.float()).abs().max().item():.9e}", flush=True)


def test_full_qwen35_cp2_engine_tree(runtime, monkeypatch):
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.gated_delta_net import TPRGatedDeltaNet

    for name in ("STAGE43_OUT_PROJ_ZIGZAG64", "STAGE43_MLP_FC2_ZIGZAG64",
                 "STAGE33_OUT_PROJ_CHUNK64", "STAGE33_MLP_FC2_CHUNK64"):
        assert os.getenv(name, "0") == "0", f"disable {name}: Stage 4.4 is unmodified baseline"
    plan = _plan()
    from ._full_hybrid_drift_diagnostic import plan_digest, parameter_summary
    digest = plan_digest(plan)
    fingerprint = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8, device=runtime.device)
    fingerprints = [torch.empty_like(fingerprint) for _ in range(2)]
    dist.all_gather(fingerprints, fingerprint, group=runtime.cp_group)
    assert all(torch.equal(value, fingerprint) for value in fingerprints), "plan/labels differ across ranks"
    assert plan.total_loss_weight == 510
    initial = None
    results = []
    # Append repeat AFTER the original three runs: preserve their order. Build
    # the same CP2 model from the same initial state/seed, no optimizer update.
    for label, cp, tree in (("CP1-connected", 1, False), ("CP2-connected", 2, False),
                            ("CP2-Engine-tree", 2, True), ("CP2-connected-repeat", 2, False)):
        print(f"STAGE-4.4 RUN rank={runtime.rank} {label}", flush=True)
        torch.manual_seed(440001)
        model = make_qwen35_model(runtime, cp_size=cp, tpr=True, num_layers=24)
        gdn, fa = assert_hybrid_architecture(model)
        assert all(isinstance(l.self_attention, TPRGatedDeltaNet) for l in gdn)
        assert all(isinstance(l.self_attention, TPRSelfAttention) for l in fa)
        assert model.config.hidden_dropout == model.config.attention_dropout == 0
        if initial is None:
            broadcast_module_state(model)
            initial = {n: t.detach().cpu().clone() if isinstance(t, torch.Tensor) else t
                       for n, t in model.state_dict().items()}
        else:
            model.load_state_dict(initial, strict=True)
        # Equality audit includes every loaded parameter/buffer, not just a
        # seed or checksum. Snapshots are temporary CPU copies, one at a time.
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.detach().cpu(), initial[name]), f"initial state mismatch: {name}"
        assert plan_digest(plan) == digest
        print(f"STAGE-4.4 AUDIT r={runtime.rank} {label}: initial_tensors=exact "
              f"plan={digest[:16]} loss_weight=510 CP={cp} "
              f"parameter_SUM={'once' if cp == 2 else 'none'} state_SUM=none "
              f"Engine_loss_scale={'1/2 (cancels CP factor)' if tree else 'global denominator only'}", flush=True)
        results.append(_run(model, plan, runtime, monkeypatch, cp=cp, tree=tree,
                            trace=os.getenv("STAGE44_TRACE", "0") == "1" and label in ("CP1-connected", "CP2-connected")))
        # Drop layer tuples too; they otherwise retain the full decoder.
        del model, gdn, fa
        gc.collect()
        torch.npu.empty_cache()
    cp1, cp2, tree, repeat = results
    parameter_summary(cp1["parameters"], cp2["parameters"], gradient_map_diagnostics, runtime.rank)
    if cp1["drift"] is not None:
        cp1["drift"].compare(cp2["drift"], gradient_map_diagnostics)
        cp1["drift"] = cp2["drift"] = None
    # Print on BOTH ranks before numerical gates can fail. A repeat diagnostic
    # is not subtracted from error and cannot waive a failing relay assertion.
    _relay_diagnostic("CP2-connected-vs-repeat", cp2, repeat, rank=runtime.rank)
    _relay_diagnostic("CP2-connected-vs-Engine-tree", cp2, tree, rank=runtime.rank)
    _relay_diagnostic("CP2-repeat-vs-Engine-tree", repeat, tree, rank=runtime.rank)
    del repeat
    results.pop()
    cp1["state"] = _shard_states(cp1["state"], runtime.cp_group.rank())
    cross = _compare("cross-CP-connected", cp1, cp2, rank=runtime.rank)
    relay = _compare("CP2-connected-vs-Engine-tree", cp2, tree, rank=runtime.rank)
    # All ranks complete all diagnostics before reporting local failures.
    failed = torch.tensor([bool(cross), bool(relay)], device=runtime.device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    print(f"STAGE-4.4 SUMMARY cross-CP={'FAIL' if failed[0].item() else 'PASS'} "
          f"Engine-relay={'FAIL' if failed[1].item() else 'PASS'}; no thresholds waived", flush=True)
    if failed.any().item():
        messages = cross + relay
        if os.getenv("STAGE44_VERBOSE", "0") != "1" and len(messages) > 8:
            # Keep relay failures visible even when all cross-CP states fail.
            messages = cross[:4] + relay[:4] + [f"{len(cross)} cross-CP and {len(relay)} relay checks failed; STAGE44_VERBOSE=1 for full list"]
        pytest.fail("\n".join(messages) or "numerical gate failed on peer rank", pytrace=False)
    if runtime.rank == 0:
        print("STAGE-4.4 PASS: full 18 GDN + 6 FA, CP2 Engine tree, non-packed; training stability not yet evaluated")
