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

# Model/spec helpers must be imported after runtime has installed MindSpeed's
# backend patches. Importing them during collection caches missing TE providers.

pytestmark = pytest.mark.skipif(os.getenv("TPR_RUN_OFFLOAD") != "1", reason="Set TPR_RUN_OFFLOAD=1")


@pytest.fixture(scope="module")
def npu_determinism():
    """Optional native operator-level control; keep the GDN backend unchanged."""
    enabled = os.getenv("TPR_OFFLOAD_DETERMINISTIC", "0")
    if enabled not in ("0", "1"):
        raise ValueError("TPR_OFFLOAD_DETERMINISTIC must be 0 or 1")
    print(f"TPR_OFFLOAD_DETERMINISTIC={enabled}", flush=True)
    if enabled == "0":
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    variables = ("HCCL_DETERMINISTIC", "CLOSE_MATMUL_K_SHIFT", "PYTHONHASHSEED")
    environment = {name: os.environ.get(name) for name in variables}
    try:
        from mindspeed.functional.npu_deterministic.npu_deterministic import extend_seed_all
        extend_seed_all(123)
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="module")
def runtime(npu_determinism):
    import torch.distributed as dist
    import torch_npu  # noqa: F401
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
    from megatron.core import parallel_state
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
def native_args(runtime, monkeypatch):
    # Only supply the training namespace normally initialized by the launcher.
    # Transfers, streams, pinned buffers and saved-tensor hooks remain real.
    from mindspeed import args_utils
    from verl.models.mcore.tpr import activation_offload
    args = SimpleNamespace(**vars(args_utils.get_full_args()))
    vars(args).update(swap_attention=False, pipeline_model_parallel_size=1,
                      eval_interval=0, curr_iteration=1, noop_layers=None,
                      swap_modules=os.getenv("TPR_SWAP_MODULES", "self_attention,mlp"))
    # Feed both supported launchers without requiring Megatron-LM training.
    monkeypatch.setattr(activation_offload, "_args", lambda: args)
    prefetch = activation_offload._native_prefetch()
    monkeypatch.setattr(prefetch, "get_args", lambda: args)
    return args


def _plan(prefix=1024, suffix=1024, owned=True):
    from ..equivalence.test_tpr_engine_reference_equivalence_npu import _make_plan
    from verl.models.mcore.tpr.segment_plan import SegmentPlan

    plan = _make_plan(torch.arange(prefix) % 2048,
                      (torch.arange(suffix) + 37) % 2048,
                      (torch.arange(suffix) + 93) % 2048)
    if not owned:
        plan = SegmentPlan([replace(s, loss_terms=()) if s.segment_id == 0 else s
                            for s in plan.segments.values()], root_id=0)
    return plan


def _run(model, plan, *, observe=False, counts=None):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor

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
        from ..equivalence.test_tpr_engine_reference_equivalence_npu import _make_model
        from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention

        model = _make_model(runtime.device, tpr=True, max_sequence_length=2048,
                            core_attention_module=_ProfileFusedCausalAttention,
                            model_shape=dict(hidden_size=512, ffn_hidden_size=2048,
                                             num_attention_heads=8, num_query_groups=4, kv_channels=64,
                                             experimental_attention_variant=None, linear_attention_freq=None,
                                             transformer_impl="local"))
        binding = nullcontext()
    if kind == "fa":
        assert model.config.experimental_attention_variant is None
        assert all(getattr(layer.self_attention, "tpr_state_kind", None) != "gdn"
                   for layer in model.decoder.layers)
    else:
        assert model.config.linear_attention_freq == 4
        assert not model.config.deterministic_mode, "control must retain the optimized GDN backend"
        assert sum(getattr(layer.self_attention, "tpr_state_kind", None) == "gdn"
                   for layer in model.decoder.layers) == 3
    model.config.swap_modules = native_args.swap_modules
    counts = _probe(monkeypatch)
    plan = _plan(owned=owned)
    with binding:
        model.config.swap_attention = False
        reference = _run(model, plan, observe=True, counts=counts)
        assert counts == {"released_bytes": 0, "h2d_bytes": 0}
        failures = []
        # Establish run-to-run noise before installing swap hooks. Complete all
        # comparisons before failing so one embedding mismatch cannot hide the
        # per-layer or Prefix-state results and post-offload cleanup control.
        for stage, enabled in (("off_repeat", False), ("on_first", True),
                               ("on_repeat", True), ("off_after", False)):
            model.config.swap_attention = enabled
            before = counts.copy()
            actual = _run(model, plan, observe=True, counts=counts)
            failures.extend(_compare_runs(reference, actual, kind=kind, owned=owned, stage=stage))
            del actual
            if enabled:
                assert counts["released_bytes"] > before["released_bytes"]
                assert counts["h2d_bytes"] > before["h2d_bytes"]
            else:
                assert counts == before
        assert not failures, "Offload correctness failures (see TPR_OFFLOAD_COMPARE):\n" + "\n".join(failures)
    print("TPR_OFFLOAD_CORRECTNESS " + json.dumps(dict(kind=kind, owned=owned, **counts)), flush=True)


def _difference_metrics(expected, actual):
    """CPU diagnostics in bounded chunks, including the ~254M-element embedding."""
    expected, actual = expected.reshape(-1), actual.reshape(-1)
    max_abs = squared_error = squared_reference = 0.0
    mismatched = 0
    finite = True
    for offset in range(0, expected.numel(), 1024 * 1024):
        left = expected[offset:offset + 1024 * 1024].float()
        right = actual[offset:offset + 1024 * 1024].float()
        diff = (right - left).abs()
        finite = finite and bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        max_abs = max(max_abs, diff.max().item())
        mismatched += (diff > 2e-4 + 2e-3 * left.abs()).sum().item()
        squared_error += diff.square().sum(dtype=torch.float64).item()
        squared_reference += left.square().sum(dtype=torch.float64).item()
    return dict(max_abs=max_abs, relative_l2=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
                mismatched=mismatched, elements=expected.numel(), finite=finite)


def _compare_runs(reference, actual, *, kind, owned, stage):
    failures = []
    for category, expected_map, actual_map in zip(("loss", "logprob", "parameter_grad", "prefix_grad"),
                                                 reference, actual):
        assert expected_map.keys() == actual_map.keys(), f"{stage}/{category}: tensor keys changed"
        assert expected_map, f"{stage}/{category}: empty comparison"
        category_failures = 0
        for name in expected_map:
            try:
                torch.testing.assert_close(actual_map[name], expected_map[name], rtol=2e-3, atol=2e-4)
            except AssertionError as error:
                category_failures += 1
                label = f"{kind}/owned={owned}/{stage}/{category}/{name}"
                failures.append(label)
                if expected_map[name].shape != actual_map[name].shape:
                    metrics = {"error": str(error)}
                else:
                    metrics = _difference_metrics(expected_map[name], actual_map[name])
                print("TPR_OFFLOAD_COMPARE " + json.dumps(dict(
                    kind=kind, owned=owned, stage=stage, category=category, tensor=str(name),
                    passed=False, **metrics)), flush=True)
        print("TPR_OFFLOAD_COMPARE " + json.dumps(dict(
            kind=kind, owned=owned, stage=stage, category=category,
            checked=len(expected_map), failed=category_failures)), flush=True)
    return failures


def test_capacity(runtime, native_args):
    if os.getenv("TPR_OFFLOAD_PROFILE") != "1":
        pytest.skip("Set TPR_OFFLOAD_PROFILE=1; run on/off in separate processes")
    from ..equivalence.test_tpr_engine_reference_equivalence_npu import _make_model
    from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention, _PROFILE_MODEL_SHAPE

    prefix = int(os.getenv("TPR_PREFIX", "16384"))
    suffix = int(os.getenv("TPR_SUFFIX", "4096"))
    enabled = os.getenv("TPR_OFFLOAD", "0") == "1"
    torch.manual_seed(123)
    model = _make_model(runtime.device, tpr=True, max_sequence_length=prefix + suffix,
                        core_attention_module=_ProfileFusedCausalAttention,
                        model_shape=dict(_PROFILE_MODEL_SHAPE, experimental_attention_variant=None,
                                         linear_attention_freq=None, transformer_impl="local"))
    assert model.config.experimental_attention_variant is None
    model.config.swap_modules = native_args.swap_modules
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
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_allocated_bytes=torch.npu.max_memory_allocated(),
        peak_reserved_bytes=torch.npu.max_memory_reserved(),
        cpu_process_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )), flush=True)
