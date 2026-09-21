"""Stage 3.3: full random Qwen3.5-0.8B Hybrid, CP=1, non-packed.

Native materialized P+S1/P+S2 versus Engine Push(P)/Visit(S1,S2)/Pop(P).
A connected segmented control separately checks the conv/recurrent/KV boundary
VJPs; these boundaries are not observable in native full-sequence kernels.
No optimizer, checkpoint loading, CP, THD, or kernel modification is involved.

From the server verl root (after syncing bridge's verl/ and tests/)::

    torchrun --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
      -m pytest -s -v tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py

Same-path CP1 repeatability (native materialized training and TPR separately)::

    torchrun --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
      -m pytest -s -v tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py \
      -k repeatability_without_offload

Six cases cover 4 layers/P1024/S1024 with and without Prefix-owned loss and
24 layers/P64/S64 with Prefix-owned loss, through both native and TPR paths.
Each case performs three runs of one unchanged model, resetting gradients before
each run. Strict rtol=2e-3/atol=2e-4 gates are separate from the original
cross-path numerical envelopes; failures remain failures and all are reported.
Native has no exposed Prefix boundary VJPs; those are checked on the TPR path.
The 4-layer cases use Phase A's token construction and seed. This is a baseline
repeatability check, not an offload-on acceptance result. CPU gradient snapshots
are retained for two runs at a time; allow several GiB of host memory.

The existing runtime validates Git revisions. If using the verified PR169
MindSpeed-Ops checkout, set STAGE34_MINDSPEED_OPS_SHA to its full commit SHA
(and STAGE34_MINDSPEED_OPS_ROOT if not /workspace/MindSpeed-Ops). The default
historical revision check and other dependency checks are unchanged.
"""

from collections import Counter
from contextlib import contextmanager, nullcontext
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ._first_layer_projection_control import first_layer_projection_control

from verl.utils.device import is_torch_npu_available

# pytest collects baseline and tpr as sibling packages in the server tree.
from baseline._qwen35_baseline_utils import (
    AllToAllProbe,
    VOCAB_SIZE,
    assert_gradient_maps_close,
    assert_hybrid_architecture,
    bind_stage1_gdn_primitives,
    destroy_npu_runtime,
    gradient_map_diagnostics,
    initialize_npu_runtime,
    make_qwen35_model,
)

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=1)
    yield value
    destroy_npu_runtime(value)


def _plan(length=64, *, owned=True, offload_tokens=False):
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec

    tokens = [(torch.arange(length) * step + start) % VOCAB_SIZE
              for step, start in ((17, 23), (19, 101), (23, 307))]
    if offload_tokens:
        tokens = [(torch.arange(length) + start) % 2048 for start in (0, 37, 93)]
    prefix, *suffixes = tokens
    root_terms = tuple(SegmentLossTerm(i, int(prefix[i + 1]), weight=2.0) for i in range(length - 1))
    root_terms += tuple(SegmentLossTerm(length - 1, int(s[0]), sample_id=j)
                        for j, s in enumerate(suffixes, 1))
    if not owned:
        root_terms = ()
    segments = [SegmentSpec(0, None, prefix, 0, 0, root_terms)]
    for j, suffix in enumerate(suffixes, 1):
        terms = tuple(SegmentLossTerm(i, int(suffix[i + 1]), sample_id=j) for i in range(length - 1))
        segments.append(SegmentSpec(j, 0, suffix, length, length, terms))
    plan = SegmentPlan(segments, root_id=0)
    assert plan.total_loss_weight == (2 * (2 * length - 1) if owned else 2 * (length - 1))
    return plan


def _parameter_grads(model):
    # Keep snapshots on CPU; never keep multiple full 0.8B models/graphs on NPU.
    result = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"missing gradient: {name}"
            assert torch.isfinite(p.grad).all().item(), f"non-finite gradient: {name}"
            result[name] = p.grad.detach().cpu().clone()
    return result


def _term_logprobs(logits, segment):
    if not segment.loss_terms:
        return logits.new_empty((0,), dtype=torch.float32)
    indices = torch.tensor([term.query_offset for term in segment.loss_terms], device=logits.device)
    targets = torch.tensor([term.target_token_id for term in segment.loss_terms], device=logits.device)
    return -F.cross_entropy(logits[0].index_select(0, indices).float(), targets, reduction="none")


class _LayerProbe:
    """Observe decoder-layer outputs and their VJPs without changing the graph.

    Full paths have separate prefix graphs: SUM their prefix output gradients
    when comparing with the shared-prefix connected graph; never concatenate or
    average those gradients. Forward prefix values use path 1, with repeat
    differences printed separately. All snapshots/diagnostics live on CPU.
    """

    def __init__(self, model):
        self.model = model
        self.tag = None
        self.records = {}
        self.handles = []

    def __enter__(self):
        for layer in self.model.decoder.layers:
            number = layer.layer_number

            def record(module, args, output, *, number=number):
                tensor = output[0] if isinstance(output, tuple) else output
                assert isinstance(tensor, torch.Tensor) and tensor.ndim == 3
                assert tensor.shape[1] == 1 and tensor.requires_grad
                key = (self.tag, number)
                assert self.tag is not None and key not in self.records, key
                entry = {"output": tensor.detach().cpu().float().clone(), "grad": None}
                self.records[key] = entry

                def capture(grad):
                    value = grad.detach().cpu().float().clone()
                    entry["grad"] = value if entry["grad"] is None else entry["grad"] + value

                self.handles.append(tensor.register_hook(capture))

            self.handles.append(layer.register_forward_hook(record))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    @contextmanager
    def segment(self, tag):
        previous, self.tag = self.tag, tag
        try:
            yield
        finally:
            self.tag = previous

    def logical(self, *, full):
        result = {}
        repeat_max = 0.0
        for layer in range(1, 25):
            if full:
                left, right = self.records[("full1", layer)], self.records[("full2", layer)]
                assert left["grad"] is not None and right["grad"] is not None
                repeat_max = max(repeat_max, (left["output"][:64] - right["output"][:64]).abs().max().item())
                result[(layer, "P")] = {
                    "output": left["output"][:64],
                    "grad": left["grad"][:64] + right["grad"][:64],
                }
                for branch, entry in ((1, left), (2, right)):
                    result[(layer, f"S{branch}")] = {k: v[64:] for k, v in entry.items()}
            else:
                for tag in ("P", "S1", "S2"):
                    entry = self.records[(tag, layer)]
                    assert entry["grad"] is not None
                    result[(layer, tag)] = entry
        if full:
            print(f"STAGE-3.3 full-path shared-prefix output repeat max_abs={repeat_max:.6e}", flush=True)
        return result


def _layer_diagnostics(label, reference, actual):
    """No new gate: print every layer; 'first nonzero' is not a bug threshold."""
    first = {"output": None, "grad": None}
    for layer in range(1, 25):
        parts = []
        for region in ("P", "S1", "S2"):
            for kind in ("output", "grad"):
                left, right = reference[(layer, region)][kind], actual[(layer, region)][kind]
                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                metric = gradient_map_diagnostics({"tensor": left}, {"tensor": right}).aggregate
                if metric.absolute_l2 != 0 and first[kind] is None:
                    first[kind] = (layer, region)
                parts.append(f"{region}/{kind}:rel={metric.relative_l2:.6e},cos={metric.cosine:.9f},"
                             f"max_abs={(right - left).abs().max().item():.6e}")
        name = "FA" if layer % 4 == 0 else "GDN"
        print(f"STAGE-3.3 LAYER {label} layer={layer:02d} {name} " + " | ".join(parts), flush=True)
    print(f"STAGE-3.3 {label} first-nonzero (not a correctness threshold): {first}", flush=True)


def _native_reference(model, plan, device, *, tpr_context=False, probe=None):
    model.zero_grad(set_to_none=True)
    losses, outputs = [], {}
    prefix_length = plan.get(0).length
    for leaf_id in (1, 2):
        tokens = torch.cat((plan.get(0).token_ids, plan.get(leaf_id).token_ids)).to(device)
        positions = torch.arange(tokens.numel(), device=device).unsqueeze(0)
        context_manager = nullcontext()
        if tpr_context:
            from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context
            from verl.models.mcore.tpr.rope import build_suffix_rotary_pos_emb

            context = TPRAttentionContext(
                prefix_length=0, suffix_length=tokens.numel(),
                suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
                    model.rotary_pos_emb, prefix_length=0, suffix_length=tokens.numel(),
                ),
            )
            context_manager = use_tpr_attention_context(context)
        # Same full path/loss for both modes; no Push/Pop or anchors.
        with context_manager, probe.segment(f"full{leaf_id}") if probe else nullcontext():
            logits = model(tokens.unsqueeze(0), positions, attention_mask=None)
        if tpr_context:
            context.assert_new_gdn_layers(tuple(i for i in range(1, 25) if i % 4))
            context.assert_new_kv_layers((4, 8, 12, 16, 20, 24))
            # The diagnostic never consumes or backpropagates final-state roots.
            del context, context_manager
        assert logits.shape == (1, tokens.numel(), VOCAB_SIZE)
        loss_start = 0 if plan.get(0).loss_terms else prefix_length
        loss = F.cross_entropy(logits[0, loss_start:-1].float(), tokens[loss_start + 1:],
                               reduction="sum") / plan.total_loss_weight
        outputs[leaf_id] = {
            0: _term_logprobs(logits[:, :prefix_length], plan.get(0)).detach().cpu(),
            leaf_id: _term_logprobs(logits[:, prefix_length:], plan.get(leaf_id)).detach().cpu(),
        }
        losses.append(loss.detach().cpu())
        loss.backward()
        del logits, loss
    # Shared prefix internals have multiplicity 2; branchpoint has two targets.
    root = outputs[1][0].clone()
    if root.numel():
        root[-1] = outputs[2][0][-1]
    return sum(losses), {0: root, 1: outputs[1][1], 2: outputs[2][2]}, _parameter_grads(model)


def _boundary_tensors(gdn, kv):
    result = {}
    for layer, state in gdn.items():
        result[f"gdn.{layer}.conv"] = state.conv_state
        result[f"gdn.{layer}.recurrent"] = state.recurrent_state
    for layer, (key, value) in kv.items():
        result[f"fa.{layer}.key"] = key
        result[f"fa.{layer}.value"] = value
    return result


def _make_engine(model, monkeypatch, finalizations):
    # Match the existing FA Engine fixture: suppress only the unrelated eager
    # MindSpeed-engine re-patcher; the actual Megatron Engine method is executed.
    monkeypatch.setitem(sys.modules, "verl.workers.engine.mindspeed", ModuleType("verl.workers.engine.mindspeed"))
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [model]
    engine.engine_config = SimpleNamespace(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=1, expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None, tpr_enabled=True,
        pad_bshd_to_minibatch_max=False, use_remove_padding=False,
        dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = model.config
    engine.enable_routing_replay = False
    engine._distillation_use_topk_active = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None
    model.config.no_sync_func = None
    model.config.grad_scale_func = lambda loss: loss
    model.config.calculate_per_token_loss = False

    def finalize(modules, num_tokens, **kwargs):
        assert modules == [model] and num_tokens is None
        assert kwargs["force_all_reduce"]
        finalizations.append(True)  # Singleton, no distributed optimizer in this puncture.

    model.config.finalize_model_grads_func = finalize
    return engine


@contextmanager
def _no_cp_probe(monkeypatch):
    import mindspeed.core.ssm.gated_delta_net as gdn
    import mindspeed.core.context_parallel.dot_product_attention as dpa

    ring = []
    # The legacy transformer.dot_product_attention imports megatron.training
    # unconditionally. Core-only installations use context_parallel instead.
    # Observe a legacy binding only if the runtime already loaded that module.
    targets = [dpa]
    legacy = sys.modules.get("mindspeed.core.transformer.dot_product_attention")
    if legacy is not None and legacy is not dpa:
        targets.append(legacy)

    with monkeypatch.context() as patch, AllToAllProbe(gdn) as a2a:
        for module in targets:
            original = module.ringattn_context_parallel

            def traced(*args, _original=original, **kwargs):
                ring.append(True)
                return _original(*args, **kwargs)

            patch.setattr(module, "ringattn_context_parallel", traced)
        yield
    assert not a2a.calls and not ring, f"CP1 entered A2A/Ring: {a2a.calls}/{ring}"


def _run_engine(model, plan, monkeypatch):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest
    from verl.models.mcore.tpr import megatron_adapter
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor

    instances = []

    class ObservedExecutor(SegmentExecutor):
        """Observe the production executor, never replace its math or schedule."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_calls = Counter()
            self.logprobs = {}
            self.forwards = []
            self.boundary_gradients = {}
            self.saved_state = None
            instances.append(self)

        def _forward(self, segment, **kwargs):
            context, logits = super()._forward(segment, **kwargs)
            self._assert_collected_layers(context)
            self.forwards.append((segment.segment_id, kwargs["no_grad"]))
            return context, logits

        def _compute_loss(self, segment, logits):
            self.loss_calls[segment.segment_id] += 1
            self.logprobs[segment.segment_id] = _term_logprobs(logits, segment).detach().cpu()
            return super()._compute_loss(segment, logits)

        def push(self, segment_id):
            result = super().push(segment_id)
            self.saved_state = self.gdn_states[segment_id]
            for t in _boundary_tensors(self.saved_state.layer_states,
                                        self.kv_stack.top().kv.key_values).values():
                assert not t.requires_grad and t.grad_fn is None
            return result

        def pop(self, segment_id):
            tensors = _boundary_tensors(self.gdn_states[segment_id].gradients,
                                       self.kv_stack.top().gradients)
            self.boundary_gradients = {k: v.detach().cpu().clone() for k, v in tensors.items()}
            return super().pop(segment_id)

    finalizations = []
    engine = _make_engine(model, monkeypatch, finalizations)
    model.zero_grad(set_to_none=True)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})

    def unexpected_loss(**kwargs):
        pytest.fail("tree request incorrectly entered ordinary Engine loss path")

    with monkeypatch.context() as patch:
        patch.setattr(megatron_adapter, "SegmentExecutor", ObservedExecutor)
        result = engine.forward_backward_batch(data, loss_function=unexpected_loss, forward_only=False)
    assert len(instances) == len(finalizations) == 1
    executor = instances[0]
    assert executor.forwards == [(0, True), (1, False), (2, False), (0, False)]
    assert executor.loss_calls == Counter({0: 1, 1: 1, 2: 1})
    assert len(executor.boundary_gradients) == model.config.num_layers * 2
    assert executor.saved_state.released
    assert not executor.gdn_states
    executor.kv_stack.assert_empty()
    return result["loss"], executor.logprobs, _parameter_grads(model), executor.boundary_gradients


def _connected_reference(model, plan, *, probe=None):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import GDNLayerState

    model.zero_grad(set_to_none=True)
    executor = SegmentExecutor(model, plan)
    with probe.segment("P") if probe else nullcontext():
        root, logits = executor._forward(plan.get(0), past_key_values={}, no_grad=False)
    logprobs = {0: _term_logprobs(logits, plan.get(0)).detach().cpu()}
    loss = executor._compute_loss(plan.get(0), logits)[1]
    # Identity-connected boundary clones separate external-state VJPs from
    # internal prefix uses of K/V. Retaining raw K/V.grad would count both.
    gdn = {layer: GDNLayerState(state.conv_state.clone(), state.recurrent_state.clone())
           for layer, state in root.new_gdn_states.items()}
    kv = {layer: (key.clone(), value.clone()) for layer, (key, value) in root.new_key_values.items()}
    boundary = _boundary_tensors(gdn, kv)
    for tensor in boundary.values():
        tensor.retain_grad()
    for leaf_id in (1, 2):
        with probe.segment(f"S{leaf_id}") if probe else nullcontext():
            _, logits = executor._forward(plan.get(leaf_id), past_key_values=kv,
                                          initial_gdn_states=gdn, no_grad=False)
        logprobs[leaf_id] = _term_logprobs(logits, plan.get(leaf_id)).detach().cpu()
        loss = loss + executor._compute_loss(plan.get(leaf_id), logits)[1]
    loss.backward()
    gradients = {}
    for name, tensor in boundary.items():
        assert tensor.grad is not None, f"missing connected boundary gradient: {name}"
        gradients[name] = tensor.grad.detach().cpu().clone()
    return _parameter_grads(model), gradients, loss.detach().cpu(), logprobs


def _gate(failures, label, check, *args, **kwargs):
    """Evaluate every numerical assertion, but fail the overall test if ANY fails."""
    try:
        check(*args, **kwargs)
    except AssertionError as error:
        failures.append(label)
        print(f"STAGE-3.3 GATE FAIL {label}: {error}", flush=True)
    else:
        print(f"STAGE-3.3 GATE PASS {label}", flush=True)


def _compare(label, reference, actual):
    for name in reference:
        assert torch.isfinite(reference[name]).all() and torch.isfinite(actual[name]).all(), name
    diag = gradient_map_diagnostics(reference, actual)
    print(f"STAGE-3.3 {label}: {diag}", flush=True)
    return diag


def test_full_qwen35_hybrid_engine_push_branch_pop(runtime, monkeypatch):
    # Imports/build happen only AFTER initialize_npu_runtime installs MindSpeed.
    from verl.models.mcore.tpr.attention import TPRSelfAttention
    from verl.models.mcore.tpr.gated_delta_net import TPRGatedDeltaNet
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    torch.manual_seed(353301)
    model = make_qwen35_model(runtime, cp_size=1, tpr=True)
    gdn, fa = assert_hybrid_architecture(model)
    assert all(isinstance(layer.self_attention, TPRGatedDeltaNet) for layer in gdn)
    assert all(isinstance(layer.self_attention, TPRSelfAttention) for layer in fa)
    assert model.config.attention_output_gate
    assert model.config.attention_dropout == model.config.hidden_dropout == 0
    plan = _plan()
    failures = []
    # All four passes reuse unchanged parameters; no optimizer is constructed.
    with (bind_stage1_gdn_primitives(mindspeed_gdn, model), _no_cp_probe(monkeypatch),
          first_layer_projection_control(model, monkeypatch) as projection_control):
        with _LayerProbe(model) as native_probe:
            ref_loss, ref_logprobs, ref_grads = _native_reference(model, plan, runtime.device, probe=native_probe)
        loss, logprobs, grads, boundary = _run_engine(model, plan, monkeypatch)
        _compare("materialized-vs-TPR/parameters", ref_grads, grads)
        print(f"STAGE-3.3 loss native/TPR: {ref_loss.item():.9f}/{loss:.9f}", flush=True)
        for segment_id in (0, 1, 2):
            delta = (logprobs[segment_id] - ref_logprobs[segment_id]).abs().max().item()
            print(f"STAGE-3.3 segment={segment_id} target-logprob max_abs={delta:.9e}", flush=True)
        with _LayerProbe(model) as split_probe:
            connected_grads, connected_boundary, split_loss, split_logprobs = _connected_reference(
                model, plan, probe=split_probe,
            )
        _compare("connected-vs-TPR/parameters", connected_grads, grads)
        for kind in ("gdn", "fa"):
            left = {k: v for k, v in connected_boundary.items() if k.startswith(kind)}
            right = {k: v for k, v in boundary.items() if k.startswith(kind)}
            _compare(f"connected-vs-TPR/{kind}-state-gradient", left, right)
        with _LayerProbe(model) as full_probe:
            full_loss, full_logprobs, full_grads = _native_reference(
                model, plan, runtime.device, tpr_context=True, probe=full_probe,
            )
        # These two comparisons are diagnostic only: do not create a new
        # acceptance threshold or assume FA alone accounts for any discrepancy.
        for label, left_grads, right_grads, left_loss, right_loss, left_lp, right_lp in (
            ("native-full-vs-TPR-context-full", ref_grads, full_grads,
             ref_loss, full_loss, ref_logprobs, full_logprobs),
            ("TPR-context-full-vs-connected-split", full_grads, connected_grads,
             full_loss, split_loss, full_logprobs, split_logprobs),
        ):
            _compare(f"{label}/parameters", left_grads, right_grads)
            print(f"STAGE-3.3 {label} loss={left_loss.item():.9f}/{right_loss.item():.9f}", flush=True)
            for sid in (0, 1, 2):
                print(f"STAGE-3.3 {label} segment={sid} target-logprob max_abs="
                      f"{(left_lp[sid] - right_lp[sid]).abs().max().item():.9e}", flush=True)
        native_layers = native_probe.logical(full=True)
        full_layers = full_probe.logical(full=True)
        split_layers = split_probe.logical(full=False)
        _layer_diagnostics("native-full-vs-TPR-context-full", native_layers, full_layers)
        _layer_diagnostics("TPR-context-full-vs-connected-split", full_layers, split_layers)
        # Keep every original threshold. Report each gate even when another
        # gate fails, and fail once at the end with all failing gate names.
        _gate(failures, "materialized-vs-TPR/loss", torch.testing.assert_close,
              torch.tensor(loss), ref_loss, atol=2e-2, rtol=2e-2)
        for segment_id in (0, 1, 2):
            _gate(failures, f"materialized-vs-TPR/logprob/segment={segment_id}", torch.testing.assert_close,
                  logprobs[segment_id], ref_logprobs[segment_id], atol=8e-2, rtol=2e-2)
        # Existing 24-layer BF16 Hybrid envelope, not the tighter relay criterion.
        _gate(failures, "materialized-vs-TPR/parameters", assert_gradient_maps_close,
              ref_grads, grads, rtol=0.10, cosine_min=0.995)
        _gate(failures, "connected-vs-TPR/parameters", assert_gradient_maps_close,
              connected_grads, grads, rtol=0.02, cosine_min=0.999)
        assert connected_boundary.keys() == boundary.keys()
        for name in connected_boundary:
            _gate(failures, f"connected-vs-TPR/state/{name}", assert_gradient_maps_close,
                  {name: connected_boundary[name]}, {name: boundary[name]}, rtol=0.02, cosine_min=0.999)
    print("STAGE-3.3 GATE PASS communication CP1 A2A=0/Ring=0", flush=True)
    if failures:
        pytest.fail("Stage 3.3 failed gates (other gates were evaluated): " + ", ".join(failures))
    status = "CONTROLLED PASS (not unmodified baseline PASS)" if projection_control else "PASS"
    print(f"STAGE-3.3 {status}: 24 layers (18 GDN + 6 FA), CP=1, non-packed; "
          "Engine tree request; prefix own loss once at Pop; 48 boundary gradients; "
          "A2A=0/Ring=0; all cached states released", flush=True)


def _repeat_difference(expected, actual):
    """Bound CPU temporary memory when reporting the tied embedding gradient."""
    expected, actual = expected.reshape(-1), actual.reshape(-1)
    squared_error = squared_reference = max_abs = 0.0
    mismatched = 0
    finite = True
    for offset in range(0, expected.numel(), 1024 * 1024):
        left = expected[offset:offset + 1024 * 1024].float()
        right = actual[offset:offset + 1024 * 1024].float()
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        diff = (right - left).abs()
        max_abs = max(max_abs, diff.max().item())
        mismatched += (diff > 2e-4 + 2e-3 * left.abs()).sum().item()
        squared_error += diff.square().sum(dtype=torch.float64).item()
        squared_reference += left.square().sum(dtype=torch.float64).item()
    return dict(max_abs=max_abs, relative_l2=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
                mismatched=mismatched, elements=expected.numel(), finite=finite)


@pytest.mark.parametrize("path", ["native", "tpr"])
@pytest.mark.parametrize("layers,length,owned", [(4, 1024, True), (4, 1024, False), (24, 64, True)],
                         ids=["l4-p1024-s1024-owned", "l4-p1024-s1024-no-owned", "l24-p64-s64-owned"])
def test_qwen35_cp1_repeatability_without_offload(runtime, monkeypatch, path, layers, length, owned):
    """Repeat one unchanged model/plan three times, through the same path.

    Native repeats ordinary materialized P+S training; TPR repeats Engine tree
    requests. Compare repeats within each path, never native against TPR here.
    The 4-layer cases match Phase A's failing model shape, tokens and loss policy.
    No swap hooks, optimizer or first-layer projection controls are installed.
    """
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn
    from verl.models.mcore.tpr.activation_offload import swap_enabled

    torch.manual_seed(123)
    model = make_qwen35_model(runtime, cp_size=1, tpr=path == "tpr", num_layers=layers)
    model.config.swap_attention = False
    assert not swap_enabled(model)
    assert model.config.attention_dropout == model.config.hidden_dropout == 0
    plan = _plan(length, owned=owned, offload_tokens=layers == 4)
    failures = []
    metadata = dict(path=path, layers=layers, prefix=length, suffix=length, owned=owned,
                    offload=False, deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    print("QWEN35_CP1_REPEAT_CONFIG " + json.dumps(metadata), flush=True)

    def run():
        assert not swap_enabled(model)
        if path == "native":
            loss, logprobs, grads = _native_reference(model, plan, runtime.device)
            boundary = {}
        else:
            loss, logprobs, grads, boundary = _run_engine(model, plan, monkeypatch)
        return {"loss": torch.as_tensor(loss).detach().cpu()}, logprobs, grads, boundary

    with bind_stage1_gdn_primitives(mindspeed_gdn, model), _no_cp_probe(monkeypatch):
        reference = run()
        for repeat in (1, 2):
            actual = run()
            for category, expected_map, actual_map in zip(
                ("loss", "logprob", "parameter_grad", "prefix_grad"), reference, actual
            ):
                assert expected_map.keys() == actual_map.keys(), f"{category}: gradient keys changed"
                if category != "prefix_grad" or path == "tpr":
                    assert expected_map, f"{category}: missing measurements"
                count = 0
                for name, expected in expected_map.items():
                    value = actual_map[name]
                    try:
                        torch.testing.assert_close(value, expected, rtol=2e-3, atol=2e-4)
                    except AssertionError as error:
                        count += 1
                        failures.append(f"repeat={repeat}/{category}/{name}")
                        metrics = (_repeat_difference(expected, value) if expected.shape == value.shape
                                   else dict(error=str(error)))
                        print("QWEN35_CP1_REPEAT " + json.dumps(dict(
                            metadata, repeat=repeat, category=category, tensor=str(name),
                            passed=False, **metrics)), flush=True)
                print("QWEN35_CP1_REPEAT " + json.dumps(dict(
                    metadata, repeat=repeat, category=category, checked=len(expected_map), failed=count,
                    applicable=category != "prefix_grad" or path == "tpr")), flush=True)
            del actual
    assert not failures, "CP1 offload-disabled repeatability failures:\n" + "\n".join(failures)
