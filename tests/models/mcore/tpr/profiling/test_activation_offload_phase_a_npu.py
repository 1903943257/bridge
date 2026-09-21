"""FA-only native-swap gates on real Qwen3-1.7B/4B; see Phase A guide."""

from dataclasses import replace
import gc
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
def runtime():
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
    repatch(dict(context_parallel_size=1, experimental_attention_variant=None,
                 use_flash_attn=True))
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


def _make_real_qwen_model(runtime, monkeypatch, *, max_sequence_length, tpr=True):
    from ._qwen3_profile_target import resolve_qwen3_profile_target
    from ..correctness import test_tpr_qwen3_compatibility_npu as qwen_fixture
    from ..correctness.test_tpr_qwen3_compatibility_npu import _make_qwen_model
    from .test_tpr_engine_profile_npu import _ProfileFusedCausalAttention

    target = resolve_qwen3_profile_target()
    monkeypatch.setattr(qwen_fixture, "QWEN_MODEL_PATH", target.path)
    model = _make_qwen_model(
        runtime.device,
        tpr=tpr,
        max_sequence_length=max_sequence_length,
        core_attention_module=_ProfileFusedCausalAttention,
    )
    parameter_count = target.assert_model_scale(model)
    # Match the mainstream Megatron/MindSpeed training loss path used by the
    # Ascend profiles instead of falling back to the unfused default.
    model.config.cross_entropy_loss_fusion = True
    model.config.cross_entropy_fusion_impl = "native"
    assert model.config.experimental_attention_variant is None
    assert all(
        getattr(layer.self_attention, "tpr_state_kind", None) != "gdn"
        for layer in model.decoder.layers
    )
    if tpr:
        assert any(
            getattr(layer.self_attention, "tpr_state_kind", None) == "fa"
            for layer in model.decoder.layers
        )
    else:
        assert all(
            getattr(layer.self_attention, "tpr_state_kind", None) is None
            for layer in model.decoder.layers
        )
    return model, target, parameter_count


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


def _run(model, plan, *, observe=False, counts=None, memory_audit=None):
    from verl.models.mcore.tpr.segment_executor import SegmentExecutor

    model.zero_grad(set_to_none=True)
    logs = {}

    class ObservedExecutor(SegmentExecutor):
        def _phase_call(self, action, segment_id, function):
            if memory_audit is None:
                return function()
            previous_phase = memory_audit.phase
            previous_executor = memory_audit.executor
            memory_audit.phase = f"{action}.{segment_id}"
            memory_audit.executor = self
            memory_audit.sample(memory_audit.phase + ".enter", executor=self)
            try:
                result = function()
            except Exception as error:
                memory_audit.fail_window(memory_audit.phase, error)
                raise
            memory_audit.sample(memory_audit.phase + ".exit", executor=self)
            memory_audit.phase = previous_phase
            memory_audit.executor = previous_executor
            return result

        def push(self, segment_id):
            return self._phase_call("push", segment_id, lambda: super(ObservedExecutor, self).push(segment_id))

        def visit_leaf(self, segment_id):
            return self._phase_call(
                "visit_leaf", segment_id, lambda: super(ObservedExecutor, self).visit_leaf(segment_id)
            )

        def pop(self, segment_id):
            return self._phase_call("pop", segment_id, lambda: super(ObservedExecutor, self).pop(segment_id))

        def _forward(self, segment, **kwargs):
            if memory_audit is None:
                return super()._forward(segment, **kwargs)
            name = memory_audit.phase + ".forward"
            past_key_values = kwargs.get("past_key_values")
            memory_audit.begin_window(
                name,
                executor=self,
                past_key_values=past_key_values,
            )
            try:
                context, logits = super()._forward(segment, **kwargs)
            except Exception as error:
                memory_audit.fail_window(name, error)
                raise
            memory_audit.end_window(
                name,
                executor=self,
                context=context,
                logits=logits,
                past_key_values=past_key_values,
            )
            return context, logits

        def _compute_loss(self, segment, logits):
            if observe and segment.loss_terms:
                indices = torch.tensor([t.query_offset for t in segment.loss_terms], device=logits.device)
                targets = torch.tensor([t.target_token_id for t in segment.loss_terms], device=logits.device)
                logs[segment.segment_id] = -F.cross_entropy(
                    logits[0].index_select(0, indices).float(), targets, reduction="none").detach().cpu()
            if memory_audit is None:
                return super()._compute_loss(segment, logits)
            name = memory_audit.phase + ".loss"
            memory_audit.begin_window(name, executor=self, logits=logits)
            try:
                result = super()._compute_loss(segment, logits)
            except Exception as error:
                memory_audit.fail_window(name, error)
                raise
            memory_audit.end_window(name, executor=self, logits=logits)
            return result

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


def _reference_trajectories(prefix, suffix, *, device):
    prefix_tokens = torch.arange(prefix, dtype=torch.long, device=device) % 2048
    suffixes = (
        (torch.arange(suffix, dtype=torch.long, device=device) + 37) % 2048,
        (torch.arange(suffix, dtype=torch.long, device=device) + 93) % 2048,
    )
    return tuple(torch.cat((prefix_tokens, suffix_tokens)) for suffix_tokens in suffixes)


def _run_reference(model, trajectories, *, total_loss_weight, counts=None):
    from verl.models.mcore.tpr.activation_offload import mindspeed_swap_attention

    model.zero_grad(set_to_none=True)
    loss_sum = model.parameters().__next__().new_zeros((), dtype=torch.float32)
    for trajectory in trajectories:
        positions = torch.arange(
            trajectory.numel(), dtype=torch.long, device=trajectory.device
        ).unsqueeze(0)
        before = None if counts is None else counts.copy()
        with mindspeed_swap_attention(model, cp_size=1):
            logits = model(
                input_ids=trajectory.unsqueeze(0),
                position_ids=positions,
                attention_mask=None,
            )
            labels = trajectory[1:].unsqueeze(0)
            per_token_loss = model.compute_language_model_loss(
                labels,
                logits[0, :-1, :].unsqueeze(1),
            ).reshape(-1)
            normalized = per_token_loss.float().sum() / total_loss_weight
            normalized.backward()
            loss_sum = loss_sum + normalized.detach()
        if counts is not None:
            if model.config.swap_attention:
                assert counts["released_bytes"] > before["released_bytes"], (
                    "Reference microbatch did not release NPU storage"
                )
                assert counts["h2d_bytes"] > before["h2d_bytes"], (
                    "Reference microbatch did not reload activations"
                )
            else:
                assert counts == before
    return loss_sum


def _settle_profile_baseline(model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    return int(torch.npu.memory_allocated()), int(torch.npu.memory_reserved())


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


def _iter_tensors(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, dict) or hasattr(item, "values"):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)


def _npu_storage_metrics(value):
    logical_bytes = 0
    storages = {}
    tensor_count = 0
    for tensor in _iter_tensors(value):
        if tensor.device.type != "npu":
            continue
        tensor_count += 1
        logical_bytes += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        pointer = int(storage.data_ptr())
        if pointer:
            storages[(str(tensor.device), pointer)] = int(storage.nbytes())
    return {
        "tensor_count": tensor_count,
        "logical_bytes": logical_bytes,
        "unique_storage_bytes": sum(storages.values()),
        "_storages": storages,
    }


class _CapacityMemoryAudit:
    """Phase-level NPU memory attribution for one untimed TPR iteration.

    The audit deliberately lives in the test harness. It never changes TPR
    execution semantics; it records allocator state and the payload of live
    tensors already owned by the executor/model.
    """

    def __init__(self, model, *, swap_counts):
        self.model = model
        self.swap_counts = swap_counts
        self.phase = None
        self.executor = None
        self._windows = {}

    def _categories(self, *, executor=None, context=None, logits=None, past_key_values=None):
        categories = {
            "parameters": tuple(self.model.parameters()),
            "parameter_grads": tuple(
                parameter.grad
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ),
        }
        if executor is not None:
            prefix_kv = []
            prefix_dkv = []
            for segment_id in executor.kv_stack.segment_ids:
                entry = executor.kv_stack.get(segment_id)
                prefix_kv.extend(
                    tensor
                    for pair in entry.kv.key_values.values()
                    for tensor in pair
                )
                prefix_dkv.extend(
                    tensor
                    for pair in entry.gradients.values()
                    for tensor in pair
                )
            categories["persistent_prefix_kv"] = tuple(prefix_kv)
            categories["accumulated_prefix_dkv"] = tuple(prefix_dkv)
        if past_key_values is not None:
            categories["past_kv_argument"] = past_key_values
        if context is not None:
            categories["current_segment_kv"] = context.new_key_values
        if logits is not None:
            categories["logits"] = logits
        return categories

    def _emit(
        self,
        stage,
        *,
        executor=None,
        context=None,
        logits=None,
        past_key_values=None,
        sync=True,
        window_start=None,
        window_peak=None,
    ):
        if sync:
            torch.npu.synchronize()
        allocated = int(torch.npu.memory_allocated())
        reserved = int(torch.npu.memory_reserved())
        categories = self._categories(
            executor=executor,
            context=context,
            logits=logits,
            past_key_values=past_key_values,
        )
        category_rows = {}
        all_storages = {}
        for name, value in categories.items():
            metrics = _npu_storage_metrics(value)
            all_storages.update(metrics.pop("_storages"))
            category_rows[name] = metrics
        accounted = sum(all_storages.values())
        row = {
            "stage": stage,
            "phase": self.phase,
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "accounted_unique_storage_bytes": accounted,
            "unclassified_allocated_bytes": max(0, allocated - accounted),
            "swap_released_bytes": self.swap_counts["released_bytes"],
            "swap_h2d_bytes": self.swap_counts["h2d_bytes"],
            "categories": category_rows,
        }
        if window_start is not None and window_peak is not None:
            row.update(
                window_start_allocated_bytes=window_start,
                window_peak_allocated_bytes=window_peak,
                window_peak_increment_bytes=max(0, window_peak - window_start),
            )
        print("TPR_OFFLOAD_MEMORY " + json.dumps(row), flush=True)

    def sample(self, stage, *, executor=None, **kwargs):
        self._emit(stage, executor=executor, **kwargs)

    def begin_window(self, name, *, executor=None, **kwargs):
        torch.npu.synchronize()
        start = int(torch.npu.memory_allocated())
        torch.npu.reset_peak_memory_stats()
        self._windows[name] = start
        self._emit(name + ".begin", executor=executor, sync=False, **kwargs)

    def end_window(self, name, *, executor=None, **kwargs):
        torch.npu.synchronize()
        start = self._windows.pop(name)
        peak = int(torch.npu.max_memory_allocated())
        self._emit(
            name + ".end",
            executor=executor,
            sync=False,
            window_start=start,
            window_peak=peak,
            **kwargs,
        )

    def fail_window(self, name, error):
        print(
            "TPR_OFFLOAD_MEMORY_FAILURE "
            + json.dumps(
                {
                    "stage": name,
                    "phase": self.phase,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            flush=True,
        )

    def install_backward_probe(self, patch):
        original_backward = torch.autograd.backward

        def traced_backward(*args, **kwargs):
            if self.phase is None or self.executor is None:
                return original_backward(*args, **kwargs)
            name = self.phase + ".backward"
            self.begin_window(name, executor=self.executor)
            try:
                result = original_backward(*args, **kwargs)
            except Exception as error:
                self.fail_window(name, error)
                raise
            self.end_window(name, executor=self.executor)
            return result

        patch.setattr(torch.autograd, "backward", traced_backward)


@pytest.mark.parametrize("owned", [True, False], ids=["owned", "no-owned"])
def test_correctness(runtime, native_args, monkeypatch, owned):
    torch.manual_seed(123)
    model, target, parameter_count = _make_real_qwen_model(
        runtime,
        monkeypatch,
        max_sequence_length=2048,
    )
    model.config.swap_modules = native_args.swap_modules
    counts = _probe(monkeypatch)
    plan = _plan(owned=owned)
    model.config.swap_attention = False
    reference = _run(model, plan, observe=True, counts=counts)
    assert counts == {"released_bytes": 0, "h2d_bytes": 0}
    failures = []
    # Establish FA baseline repeatability before installing swap hooks. Complete
    # all comparisons before failing so one tensor mismatch cannot hide the
    # Prefix-dKV result or post-offload cleanup control.
    for stage, enabled in (("off_repeat", False), ("on_first", True),
                           ("on_repeat", True), ("off_after", False)):
        model.config.swap_attention = enabled
        before = counts.copy()
        actual = _run(model, plan, observe=True, counts=counts)
        failures.extend(_compare_runs(reference, actual, owned=owned, stage=stage))
        del actual
        if enabled:
            assert counts["released_bytes"] > before["released_bytes"]
            assert counts["h2d_bytes"] > before["h2d_bytes"]
        else:
            assert counts == before
    assert not failures, "Offload correctness failures (see TPR_OFFLOAD_COMPARE):\n" + "\n".join(failures)
    print("TPR_OFFLOAD_CORRECTNESS " + json.dumps(dict(
        model=target.label,
        checkpoint=str(target.path),
        parameter_count=parameter_count,
        owned=owned,
        **counts,
    )), flush=True)

def _difference_metrics(expected, actual):
    """CPU diagnostics in bounded chunks."""
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


def _compare_runs(reference, actual, *, owned, stage):
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
                label = f"fa/owned={owned}/{stage}/{category}/{name}"
                failures.append(label)
                if expected_map[name].shape != actual_map[name].shape:
                    metrics = {"error": str(error)}
                else:
                    metrics = _difference_metrics(expected_map[name], actual_map[name])
                print("TPR_OFFLOAD_COMPARE " + json.dumps(dict(
                    model="fa", owned=owned, stage=stage, category=category, tensor=str(name),
                    passed=False, **metrics)), flush=True)
        print("TPR_OFFLOAD_COMPARE " + json.dumps(dict(
            model="fa", owned=owned, stage=stage, category=category,
            checked=len(expected_map), failed=category_failures)), flush=True)
    return failures


def test_reference_tpr_offload_matrix(runtime, native_args, monkeypatch):
    """Fair 2x2: Reference/TPR x offload on/off on real Qwen3-1.7B.

    Run each cell in a fresh process. Both paths use the same model family,
    native fused Megatron CE, swap_modules and two logical sibling trajectories.
    """
    if os.getenv("TPR_OFFLOAD_MATRIX") != "1":
        pytest.skip("Set TPR_OFFLOAD_MATRIX=1 for the Reference/TPR x offload matrix")

    path = os.getenv("TPR_OFFLOAD_MATRIX_PATH", "tpr").strip().lower()
    if path not in ("reference", "tpr"):
        raise ValueError("TPR_OFFLOAD_MATRIX_PATH must be reference or tpr")
    enabled = os.getenv("TPR_OFFLOAD", "0") == "1"
    prefix = int(os.getenv("TPR_PREFIX", "8192"))
    suffix = int(os.getenv("TPR_SUFFIX", "8192"))

    # This acceptance matrix is intentionally fixed to the 1.7B target for now.
    requested_size = os.getenv("TPR_QWEN_PROFILE_SIZE", "1.7B")
    if requested_size != "1.7B":
        raise ValueError("Phase A 2x2 matrix currently requires TPR_QWEN_PROFILE_SIZE=1.7B")

    torch.manual_seed(123)
    model, target, parameter_count = _make_real_qwen_model(
        runtime,
        monkeypatch,
        max_sequence_length=prefix + suffix,
        tpr=path == "tpr",
    )
    if target.label != "Qwen3-1.7B":
        raise RuntimeError(f"2x2 matrix expected Qwen3-1.7B, got {target.label}")
    model.config.swap_modules = native_args.swap_modules
    model.config.swap_attention = enabled

    plan = _plan(prefix, suffix)
    if path == "reference":
        trajectories = _reference_trajectories(prefix, suffix, device=runtime.device)

        def run():
            return _run_reference(
                model,
                trajectories,
                total_loss_weight=plan.total_loss_weight,
            )
    else:
        def run():
            return _run(model, plan)

    # Untimed native-transfer gate proves that both Reference and TPR actually
    # exercise the same MindSpeed swap implementation in the ON cells.
    with monkeypatch.context() as probe_patch:
        counts = _probe(probe_patch)
        if path == "reference":
            probe_loss = _run_reference(
                model,
                trajectories,
                total_loss_weight=plan.total_loss_weight,
                counts=counts,
            )
        else:
            _run(model, plan, counts=counts)
            probe_loss = None
        if enabled:
            assert counts["released_bytes"] > 0 and counts["h2d_bytes"] > 0
        else:
            assert counts == {"released_bytes": 0, "h2d_bytes": 0}

    baseline_allocated, baseline_reserved = _settle_profile_baseline(model)

    # One uninstrumented warmup, then settle again before the measured window.
    warmup_loss = run()
    del warmup_loss, probe_loss
    baseline_allocated, baseline_reserved = _settle_profile_baseline(model)
    torch.npu.reset_peak_memory_stats()

    started = time.perf_counter()
    measured_loss = None
    for _ in range(3):
        measured_loss = run()
    torch.npu.synchronize()
    peak_allocated = int(torch.npu.max_memory_allocated())
    peak_reserved = int(torch.npu.max_memory_reserved())

    row = dict(
        path=path,
        offload=enabled,
        prefix=prefix,
        suffix=suffix,
        siblings=2,
        model=target.label,
        checkpoint=str(target.path),
        parameter_count=parameter_count,
        cross_entropy_loss_fusion=bool(model.config.cross_entropy_loss_fusion),
        cross_entropy_fusion_impl=str(model.config.cross_entropy_fusion_impl),
        swap_modules=str(model.config.swap_modules),
        latency_seconds=(time.perf_counter() - started) / 3,
        baseline_allocated_bytes=baseline_allocated,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        incremental_peak_allocated_bytes=max(0, peak_allocated - baseline_allocated),
        incremental_peak_reserved_bytes=max(0, peak_reserved - baseline_reserved),
        cpu_process_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    if measured_loss is not None:
        row["normalized_loss"] = float(measured_loss.detach().cpu())
    print("TPR_OFFLOAD_2X2 " + json.dumps(row), flush=True)


def test_capacity(runtime, native_args, monkeypatch):
    if os.getenv("TPR_OFFLOAD_PROFILE") != "1":
        pytest.skip("Set TPR_OFFLOAD_PROFILE=1; run on/off in separate processes")

    prefix = int(os.getenv("TPR_PREFIX", "16384"))
    suffix = int(os.getenv("TPR_SUFFIX", "4096"))
    enabled = os.getenv("TPR_OFFLOAD", "0") == "1"
    torch.manual_seed(123)
    model, target, parameter_count = _make_real_qwen_model(
        runtime,
        monkeypatch,
        max_sequence_length=prefix + suffix,
    )
    model.config.swap_modules = native_args.swap_modules
    model.config.swap_attention = enabled
    plan = _plan(prefix, suffix)

    # One untimed diagnostic iteration runs before the benchmark. It emits each
    # record immediately, so an OOM still leaves the last completed memory stage.
    if os.getenv("TPR_OFFLOAD_MEMORY_AUDIT", "1") == "1":
        with monkeypatch.context() as audit_patch:
            swap_counts = _probe(audit_patch)
            audit = _CapacityMemoryAudit(model, swap_counts=swap_counts)
            audit.install_backward_probe(audit_patch)
            audit.sample("model_ready")
            _run(model, plan, counts=swap_counts, memory_audit=audit)
            audit.sample("iteration_complete")
        del audit

    # Timed runs remain uninstrumented. Settle gradients/cache first so the
    # reported incremental peak is relative to the resident model baseline.
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    baseline_allocated = int(torch.npu.memory_allocated())
    baseline_reserved = int(torch.npu.memory_reserved())

    _run(model, plan)  # one uninstrumented warmup
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.synchronize()
    baseline_allocated = int(torch.npu.memory_allocated())
    baseline_reserved = int(torch.npu.memory_reserved())
    torch.npu.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(3):
        _run(model, plan)
    torch.npu.synchronize()
    peak_allocated = int(torch.npu.max_memory_allocated())
    peak_reserved = int(torch.npu.max_memory_reserved())
    print("TPR_OFFLOAD_PROFILE " + json.dumps(dict(
        offload=enabled, prefix=prefix, suffix=suffix, siblings=2,
        model=target.label, checkpoint=str(target.path), parameter_count=parameter_count,
        latency_seconds=(time.perf_counter() - start) / 3,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        baseline_allocated_bytes=baseline_allocated,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        incremental_peak_allocated_bytes=max(0, peak_allocated - baseline_allocated),
        incremental_peak_reserved_bytes=max(0, peak_reserved - baseline_reserved),
        cpu_process_highwater_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )), flush=True)
