"""Opt-in native-swap correctness and capacity gates; see adjacent Phase A guide."""

from contextlib import nullcontext
from dataclasses import replace
import json
import os
import resource
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ..equivalence.test_tpr_engine_reference_equivalence_npu import _make_model, _make_plan
from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention, _PROFILE_MODEL_SHAPE
from verl.models.mcore.tpr.segment_executor import SegmentExecutor
from verl.models.mcore.tpr.segment_plan import SegmentPlan

pytestmark = pytest.mark.skipif(os.getenv("TPR_RUN_OFFLOAD") != "1", reason="Set TPR_RUN_OFFLOAD=1")


@pytest.fixture(scope="module")
def runtime():
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from megatron.core import parallel_state
    assert int(os.getenv("WORLD_SIZE", "1")) == 1
    torch.npu.set_device(int(os.getenv("LOCAL_RANK", "0")))
    dist.init_process_group(backend="hccl")
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)
    repatch(dict(context_parallel_size=1, experimental_attention_variant="gated_delta_net",
                 use_naive_l2norm=True, use_flash_attn=True, deterministic_mode=False))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1,
                                             pipeline_model_parallel_size=1, context_parallel_size=1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(123)
    try:
        yield SimpleNamespace(device=torch.device("npu", torch.npu.current_device()),
                              tp_group=parallel_state.get_tensor_model_parallel_group(),
                              pp_group=parallel_state.get_pipeline_model_parallel_group())
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.fixture
def native_args(monkeypatch):
    # Only supply the training namespace normally initialized by the launcher.
    # Transfers, streams, pinned buffers and saved-tensor hooks remain real.
    import megatron.training
    from mindspeed.core.memory.swap_attention import prefetch
    args = SimpleNamespace(swap_attention=False, pipeline_model_parallel_size=1,
                           eval_interval=0, curr_iteration=1, noop_layers=None,
                           swap_modules=os.getenv("TPR_SWAP_MODULES", "self_attention,mlp"))
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    monkeypatch.setattr(prefetch, "get_args", lambda: args)
    return args


def _plan(prefix=1024, suffix=1024, owned=True):
    plan = _make_plan(torch.arange(prefix) % 2048,
                      (torch.arange(suffix) + 37) % 2048,
                      (torch.arange(suffix) + 93) % 2048)
    if not owned:
        plan = SegmentPlan([replace(s, loss_terms=()) if s.segment_id == 0 else s
                            for s in plan.segments.values()], root_id=0)
    return plan


def _run(model, plan, *, observe=False, counts=None):
    model.zero_grad(set_to_none=True)
    logs = {}

    class ObservedExecutor(SegmentExecutor):
        def _compute_loss(self, segment, logits):
            if observe and segment.loss_terms:
                indices = torch.tensor([t.query_offset for t in segment.loss_terms], device=logits.device)
                targets = torch.tensor([t.target_token_id for t in segment.loss_terms], device=logits.device)
                logs[segment.segment_id] = -F.cross_entropy(
                    logits[0].index_select(0, indices).float(), targets, reduction="none").detach().cpu()
            return super()._compute_loss(segment, logits)

    executor = ObservedExecutor(model, plan)
    before = None if counts is None else counts.copy()
    executor.push(0)
    if counts is not None:
        assert counts == before, "Push must remain graph-free, with no activation transfers"
    losses = []
    for i in (1, 2):
        before = None if counts is None else counts.copy()
        losses.append(executor.visit_leaf(i).backward.normalized_loss)
        if counts is not None and model.config.swap_attention:
            assert counts["released_bytes"] > before["released_bytes"], "Visit did not release NPU storage"
            assert counts["h2d_bytes"] > before["h2d_bytes"], "Visit did not reload activations"
    boundary = {}
    if observe:
        for layer, pair in executor.kv_stack.top().gradients.items():
            for name, value in zip(("key", "value"), pair):
                boundary[f"fa.{layer}.{name}"] = value.detach().cpu().clone()
        for state in executor.gdn_states.values():
            for layer, gradient in state.gradients.items():
                for name in ("conv_state", "recurrent_state"):
                    boundary[f"gdn.{layer}.{name}"] = getattr(gradient, name).detach().cpu().clone()
    before = None if counts is None else counts.copy()
    losses.append(executor.pop(0).normalized_loss)
    if counts is not None and model.config.swap_attention:
        assert counts["released_bytes"] > before["released_bytes"], "Pop did not release NPU storage"
        assert counts["h2d_bytes"] > before["h2d_bytes"], "Pop did not reload activations"
    executor.kv_stack.assert_empty()
    assert not executor.gdn_states
    if not observe:
        return
    grads = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    return {"loss": torch.stack(losses).sum().cpu()}, logs, grads, boundary


def _probe(monkeypatch):
    from mindspeed.core.memory.swap_attention.prefetch import SwapTensor
    counts = {"released_bytes": 0, "h2d_bytes": 0}
    release, reload = SwapTensor.wait_d2h_finished, SwapTensor.launch_h2d

    def released(item, *args, **kwargs):
        before = item.stat
        result = release(item, *args, **kwargs)
        if before == "d2h" and item.stat == "host":
            assert item.tensor.storage().size() == 0
            assert item.tensor_cpu.is_pinned()
            counts["released_bytes"] += item.storage_size * item.tensor.element_size()
        return result

    def reloaded(item, *args, **kwargs):
        before = item.stat
        result = reload(item, *args, **kwargs)
        if before == "host" and item.stat == "h2d":
            counts["h2d_bytes"] += item.storage_size * item.tensor.element_size()
        return result

    monkeypatch.setattr(SwapTensor, "wait_d2h_finished", released)
    monkeypatch.setattr(SwapTensor, "launch_h2d", reloaded)
    return counts


@pytest.mark.parametrize("kind", ["fa", "hybrid"])
@pytest.mark.parametrize("owned", [True, False])
def test_correctness(runtime, native_args, monkeypatch, kind, owned):
    torch.manual_seed(123)
    if kind == "hybrid":
        from baseline._qwen35_baseline_utils import make_qwen35_model, bind_stage1_gdn_primitives
        import mindspeed.core.ssm.gated_delta_net as gdn
        model = make_qwen35_model(runtime, cp_size=1, tpr=True, num_layers=4)
        binding = bind_stage1_gdn_primitives(gdn, model)
    else:
        model = _make_model(runtime.device, tpr=True, max_sequence_length=2048,
                            core_attention_module=_ProfileFusedCausalAttention,
                            model_shape=dict(hidden_size=512, ffn_hidden_size=2048,
                                             num_attention_heads=8, num_query_groups=4, kv_channels=64))
        binding = nullcontext()
    counts = _probe(monkeypatch)
    plan = _plan(owned=owned)
    with binding:
        model.config.swap_attention = False
        reference = _run(model, plan, observe=True, counts=counts)
        assert counts == {"released_bytes": 0, "h2d_bytes": 0}
        for enabled in (True, True, False):  # repeated native lifecycle, then non-offload regression
            model.config.swap_attention = enabled
            before = counts.copy()
            actual = _run(model, plan, observe=True, counts=counts)
            for expected_map, actual_map in zip(reference, actual):
                assert expected_map.keys() == actual_map.keys()
                assert expected_map
                for name in expected_map:
                    torch.testing.assert_close(actual_map[name], expected_map[name], rtol=2e-3, atol=2e-4,
                                               msg=lambda message: f"{kind}/{name}: {message}")
            if enabled:
                assert counts["released_bytes"] > before["released_bytes"]
                assert counts["h2d_bytes"] > before["h2d_bytes"]
            else:
                assert counts == before
    print("TPR_OFFLOAD_CORRECTNESS " + json.dumps(dict(kind=kind, owned=owned, **counts)), flush=True)


def test_capacity(runtime, native_args):
    if os.getenv("TPR_OFFLOAD_PROFILE") != "1":
        pytest.skip("Set TPR_OFFLOAD_PROFILE=1; run on/off in separate processes")
    prefix = int(os.getenv("TPR_PREFIX", "16384"))
    suffix = int(os.getenv("TPR_SUFFIX", "4096"))
    enabled = os.getenv("TPR_OFFLOAD", "0") == "1"
    torch.manual_seed(123)
    model = _make_model(runtime.device, tpr=True, max_sequence_length=prefix + suffix,
                        core_attention_module=_ProfileFusedCausalAttention, model_shape=_PROFILE_MODEL_SHAPE)
    model.config.swap_attention = enabled
    plan = _plan(prefix, suffix)
    # No CPU gradient/logprob snapshots or telemetry wrappers inside timed runs.
    _run(model, plan)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(3):
        _run(model, plan)
    torch.npu.synchronize()
    print("TPR_OFFLOAD_PROFILE " + json.dumps(dict(
        offload=enabled, prefix=prefix, suffix=suffix, siblings=2,
        model="synthetic-dense-0.6B", latency_seconds=(time.perf_counter() - start) / 3,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        peak_reserved_bytes=torch.npu.max_memory_reserved(),
        cpu_process_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )), flush=True)
