"""Stage 3.3: full random Qwen3.5-0.8B Hybrid, CP=1, non-packed.

Native materialized P+S1/P+S2 versus Engine Push(P)/Visit(S1,S2)/Pop(P).
A connected segmented control separately checks the conv/recurrent/KV boundary
VJPs; these boundaries are not observable in native full-sequence kernels.
No optimizer, checkpoint loading, CP, THD, or kernel modification is involved.

From the server verl root (after syncing bridge's verl/ and tests/)::

    torchrun --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
      -m pytest -s -v tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py
"""

from collections import Counter
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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


def _plan():
    from verl.models.mcore.tpr.segment_plan import SegmentLossTerm, SegmentPlan, SegmentSpec

    tokens = [(torch.arange(64) * step + start) % VOCAB_SIZE
              for step, start in ((17, 23), (19, 101), (23, 307))]
    prefix, *suffixes = tokens
    root_terms = tuple(SegmentLossTerm(i, int(prefix[i + 1]), weight=2.0) for i in range(63))
    root_terms += tuple(SegmentLossTerm(63, int(s[0]), sample_id=j)
                        for j, s in enumerate(suffixes, 1))
    segments = [SegmentSpec(0, None, prefix, 0, 0, root_terms)]
    for j, suffix in enumerate(suffixes, 1):
        terms = tuple(SegmentLossTerm(i, int(suffix[i + 1]), sample_id=j) for i in range(63))
        segments.append(SegmentSpec(j, 0, suffix, 64, 64, terms))
    plan = SegmentPlan(segments, root_id=0)
    assert plan.total_loss_weight == 254
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
    indices = torch.tensor([term.query_offset for term in segment.loss_terms], device=logits.device)
    targets = torch.tensor([term.target_token_id for term in segment.loss_terms], device=logits.device)
    return -F.cross_entropy(logits[0].index_select(0, indices).float(), targets, reduction="none")


def _native_reference(model, plan, device):
    model.zero_grad(set_to_none=True)
    losses, outputs = [], {}
    for leaf_id in (1, 2):
        tokens = torch.cat((plan.get(0).token_ids, plan.get(leaf_id).token_ids)).to(device)
        positions = torch.arange(128, device=device).unsqueeze(0)
        # No TPR context: both wrapper classes delegate to the native forward.
        logits = model(tokens.unsqueeze(0), positions, attention_mask=None)
        assert logits.shape == (1, 128, VOCAB_SIZE)
        loss = F.cross_entropy(logits[0, :-1].float(), tokens[1:], reduction="sum") / 254
        outputs[leaf_id] = {
            0: _term_logprobs(logits[:, :64], plan.get(0)).detach().cpu(),
            leaf_id: _term_logprobs(logits[:, 64:], plan.get(leaf_id)).detach().cpu(),
        }
        losses.append(loss.detach().cpu())
        loss.backward()
        del logits, loss
    # Shared prefix internals have multiplicity 2; branchpoint has two targets.
    root = outputs[1][0].clone()
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
    import mindspeed.core.transformer.dot_product_attention as dpa

    ring = []
    original = dpa.ringattn_context_parallel

    def traced(*args, **kwargs):
        ring.append(True)
        return original(*args, **kwargs)

    with monkeypatch.context() as patch, AllToAllProbe(gdn) as a2a:
        patch.setattr(dpa, "ringattn_context_parallel", traced)
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
    assert len(executor.boundary_gradients) == 18 * 2 + 6 * 2
    assert executor.saved_state.released
    assert not executor.gdn_states
    executor.kv_stack.assert_empty()
    return result["loss"], executor.logprobs, _parameter_grads(model), executor.boundary_gradients


def _connected_reference(model, plan):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor
    from verl.models.mcore.tpr.prefix_state import GDNLayerState

    model.zero_grad(set_to_none=True)
    executor = SegmentExecutor(model, plan)
    root, logits = executor._forward(plan.get(0), past_key_values={}, no_grad=False)
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
        _, logits = executor._forward(plan.get(leaf_id), past_key_values=kv,
                                      initial_gdn_states=gdn, no_grad=False)
        loss = loss + executor._compute_loss(plan.get(leaf_id), logits)[1]
    loss.backward()
    gradients = {}
    for name, tensor in boundary.items():
        assert tensor.grad is not None, f"missing connected boundary gradient: {name}"
        gradients[name] = tensor.grad.detach().cpu().clone()
    return _parameter_grads(model), gradients


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
    # All three passes reuse unchanged parameters; no optimizer is constructed.
    with bind_stage1_gdn_primitives(mindspeed_gdn, model), _no_cp_probe(monkeypatch):
        ref_loss, ref_logprobs, ref_grads = _native_reference(model, plan, runtime.device)
        loss, logprobs, grads, boundary = _run_engine(model, plan, monkeypatch)
        _compare("materialized-vs-TPR/parameters", ref_grads, grads)
        print(f"STAGE-3.3 loss native/TPR: {ref_loss.item():.9f}/{loss:.9f}", flush=True)
        for segment_id in (0, 1, 2):
            delta = (logprobs[segment_id] - ref_logprobs[segment_id]).abs().max().item()
            print(f"STAGE-3.3 segment={segment_id} target-logprob max_abs={delta:.9e}", flush=True)
        connected_grads, connected_boundary = _connected_reference(model, plan)
        _compare("connected-vs-TPR/parameters", connected_grads, grads)
        for kind in ("gdn", "fa"):
            left = {k: v for k, v in connected_boundary.items() if k.startswith(kind)}
            right = {k: v for k, v in boundary.items() if k.startswith(kind)}
            _compare(f"connected-vs-TPR/{kind}-state-gradient", left, right)
        # Print both reference comparisons before assertions, so a native/split
        # discrepancy cannot hide whether state relay itself is correct.
        torch.testing.assert_close(torch.tensor(loss), ref_loss, atol=2e-2, rtol=2e-2)
        for segment_id in (0, 1, 2):
            torch.testing.assert_close(logprobs[segment_id], ref_logprobs[segment_id], atol=8e-2, rtol=2e-2)
        # Existing 24-layer BF16 Hybrid envelope, not the tighter relay criterion.
        assert_gradient_maps_close(ref_grads, grads, rtol=0.10, cosine_min=0.995)
        assert_gradient_maps_close(connected_grads, grads, rtol=0.02, cosine_min=0.999)
        for name in connected_boundary:
            assert_gradient_maps_close({name: connected_boundary[name]}, {name: boundary[name]},
                                       rtol=0.02, cosine_min=0.999)
    print("STAGE-3.3 PASS: 24 layers (18 GDN + 6 FA), CP=1, non-packed; "
          "Engine tree request; prefix own loss once at Pop; 48 boundary gradients; "
          "A2A=0/Ring=0; all cached states released", flush=True)
