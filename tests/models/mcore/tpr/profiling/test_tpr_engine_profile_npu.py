# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in controlled fused-reference profile for the TPR Engine MVP."""

import copy
import gc
import os
import statistics
import time
import types
import weakref
from contextlib import ExitStack
from contextvars import ContextVar
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from ..equivalence.test_tpr_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_model,
    _make_plan,
    _reference_data,
)
from ..equivalence.test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest, SegmentExecutor
from verl.models.mcore.tpr import attention as tpr_attention_module
from verl.models.mcore.tpr import rectangular_attention as rectangular_attention_module
from verl.models.mcore.tpr.context import TreeAttentionContext, get_tree_attention_context
from verl.models.mcore.tpr.kv_stack import KVStack, KVStackEntry, SegmentKV
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_PROFILE") != "1",
    reason="Set TPR_RUN_PROFILE=1 for the non-gating NPU profile",
)

_PROFILE_CASES = (
    (16384, 512, 2),
    (16384, 512, 4),
    (16384, 512, 8),
    (8192, 1024, 2),
    (8192, 1024, 4),
    (8192, 1024, 8),
    (8192, 8192, 2),
    (8192, 8192, 4),
    (8192, 8192, 8),
    (1024, 8192, 2),
    (1024, 8192, 4),
    (1024, 8192, 8),
)

# A roughly 0.6B synthetic dense-model proxy.  A deliberately modest vocab
# keeps long-sequence logits from dominating the attention/scheduler profile.
_PROFILE_MODEL_SHAPE = {
    "num_layers": 32,
    "hidden_size": 1280,
    "ffn_hidden_size": 5120,
    "num_attention_heads": 20,
    "num_query_groups": 4,
    "kv_channels": 64,
    "vocab_size": 8192,
}
_PROFILE_NUM_LAYERS = _PROFILE_MODEL_SHAPE["num_layers"]
_PROFILE_VOCAB_SIZE = _PROFILE_MODEL_SHAPE["vocab_size"]
_MIN_PROFILE_PARAMETERS = 500_000_000
_MAX_PROFILE_PARAMETERS = 700_000_000

_WARMUP_RUNS = 3
_MEASURE_RUNS = 10
_BREAKDOWN_WARMUP_RUNS = 1
_BREAKDOWN_MEASURE_RUNS = 3
_RUN_BREAKDOWN = os.getenv("TPR_PROFILE_BREAKDOWN", "1") == "1"
_RUN_MEMORY_BREAKDOWN = os.getenv("TPR_PROFILE_MEMORY_BREAKDOWN", "0") == "1"
_MIB = 1024**2
_GIB = 1024**3
_RECOVERY_TOLERANCE_BYTES = 512 * _MIB
_TRACE_SEGMENT_SCOPE = ContextVar("tpr_profile_segment_scope", default=None)
_TRACE_LAYER = ContextVar("tpr_profile_layer", default=None)


class _ProfileFusedCausalAttention(DotProductAttention):
    """Square causal reference backed by the same CANN adapter as TPR."""

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias=None,
        packed_seq_params: PackedSeqParams | None = None,
    ):
        del attention_mask, attn_mask_type
        if attention_bias is not None:
            raise ValueError("controlled fused reference does not support attention_bias")
        if packed_seq_params is not None:
            raise ValueError("controlled fused reference does not support packed sequences")
        return rectangular_attention_module.rectangular_causal_attention(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            dropout_p=0.0,
        )


class _NPUEventRecorder:
    """Collect asynchronous device durations without synchronizing inside modules."""

    def __init__(self):
        self._stacks = {}
        self._pairs = {}

    def begin(self, category):
        event = torch.npu.Event(enable_timing=True)
        event.record()
        self._stacks.setdefault(category, []).append(event)

    def end(self, category):
        stack = self._stacks.get(category)
        if not stack:
            raise RuntimeError(f"unmatched NPU timing event for {category}")
        start = stack.pop()
        end = torch.npu.Event(enable_timing=True)
        end.record()
        self._pairs.setdefault(category, []).append((start, end))

    def call(self, category, function, *args, **kwargs):
        self.begin(category)
        try:
            return function(*args, **kwargs)
        finally:
            self.end(category)

    def summarize(self):
        if any(self._stacks.values()):
            raise RuntimeError("unclosed NPU timing event")
        return {
            category: {
                "total_ms": sum(start.elapsed_time(end) for start, end in pairs),
                "calls": len(pairs),
            }
            for category, pairs in self._pairs.items()
        }


def _cpu_state_dict(model):
    return {
        name: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for name, value in model.state_dict().items()
    }


def _settled_allocated_samples():
    samples = []
    for _ in range(3):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        samples.append(torch.npu.memory_allocated())
    return samples


def _assert_memory_recovered(*, expected_baseline, samples, path_name):
    spread = max(samples) - min(samples)
    excess = samples[-1] - expected_baseline
    if spread > _RECOVERY_TOLERANCE_BYTES or excess > _RECOVERY_TOLERANCE_BYTES:
        raise RuntimeError(
            f"{path_name} NPU allocations did not recover to a stable baseline: "
            f"expected={expected_baseline / _MIB:.1f} MiB, "
            f"samples={[round(value / _MIB, 1) for value in samples]} MiB. "
            "The single-process profile is not trustworthy; use subprocess isolation."
        )


def _verify_fused_adapter_path(run, zero_grad, *, expected_path):
    import torch_npu

    adapter_calls = {"reference": 0, "tpr": 0}
    fusion_calls = {"reference": 0, "tpr": 0}
    tpr_trace = []
    original_adapter = rectangular_attention_module.rectangular_causal_attention
    original_fusion = torch_npu.npu_fusion_attention
    original_push = SegmentExecutor.push
    original_pop = SegmentExecutor.pop
    original_visit_leaf = SegmentExecutor.visit_leaf
    original_tree_forward = tpr_attention_module.TPRSelfAttention._tree_forward

    def counted_attention(*args, **kwargs):
        path_name = "tpr" if get_tree_attention_context() is not None else "reference"
        adapter_calls[path_name] += 1
        if path_name == "tpr":
            segment_scope = _TRACE_SEGMENT_SCOPE.get()
            layer_number = _TRACE_LAYER.get()
            if segment_scope is None or layer_number is None:
                raise AssertionError(
                    "TPR adapter call occurred outside the segment/layer trace scope"
                )
            segment_id, action = segment_scope
            query, key = args[:2]
            tpr_trace.append(
                {
                    "ordinal": len(tpr_trace) + 1,
                    "segment_id": segment_id,
                    "action": action,
                    "layer_number": layer_number,
                    "query_length": query.shape[0],
                    "kv_length": key.shape[0],
                }
            )
        return original_adapter(*args, **kwargs)

    def counted_fusion_attention(*args, **kwargs):
        path_name = "tpr" if get_tree_attention_context() is not None else "reference"
        fusion_calls[path_name] += 1
        return original_fusion(*args, **kwargs)

    def traced_push(executor, segment_id):
        token = _TRACE_SEGMENT_SCOPE.set((segment_id, "push"))
        try:
            return original_push(executor, segment_id)
        finally:
            _TRACE_SEGMENT_SCOPE.reset(token)

    def traced_pop(executor, segment_id):
        token = _TRACE_SEGMENT_SCOPE.set((segment_id, "pop"))
        try:
            return original_pop(executor, segment_id)
        finally:
            _TRACE_SEGMENT_SCOPE.reset(token)

    def traced_visit_leaf(executor, segment_id):
        token = _TRACE_SEGMENT_SCOPE.set((segment_id, "visit_leaf"))
        try:
            return original_visit_leaf(executor, segment_id)
        finally:
            _TRACE_SEGMENT_SCOPE.reset(token)

    def traced_tree_forward(attention, hidden_states, context):
        token = _TRACE_LAYER.set(attention.layer_number)
        try:
            return original_tree_forward(attention, hidden_states, context)
        finally:
            _TRACE_LAYER.reset(token)

    # Reference resolves through the module; TPR imported the function directly.
    # Patch both bindings only for this non-timed probe.
    with (
        patch.object(rectangular_attention_module, "rectangular_causal_attention", counted_attention),
        patch.object(tpr_attention_module, "rectangular_causal_attention", counted_attention),
        patch.object(torch_npu, "npu_fusion_attention", counted_fusion_attention),
        patch.object(SegmentExecutor, "push", traced_push),
        patch.object(SegmentExecutor, "pop", traced_pop),
        patch.object(SegmentExecutor, "visit_leaf", traced_visit_leaf),
        patch.object(tpr_attention_module.TPRSelfAttention, "_tree_forward", traced_tree_forward),
    ):
        zero_grad()
        torch.npu.synchronize()
        run()
        torch.npu.synchronize()

    assert adapter_calls[expected_path] > 0, f"{expected_path} did not enter rectangular_causal_attention"
    assert fusion_calls[expected_path] == adapter_calls[expected_path], (
        f"{expected_path} adapter/fusion call mismatch: adapter={adapter_calls}, fusion={fusion_calls}"
    )
    unexpected_path = "tpr" if expected_path == "reference" else "reference"
    assert adapter_calls[unexpected_path] == 0, f"unexpected adapter calls: {adapter_calls}"
    assert fusion_calls[unexpected_path] == 0, f"unexpected fusion calls: {fusion_calls}"
    return adapter_calls[expected_path], tuple(tpr_trace)


def _profile_runs(run, zero_grad):
    for _ in range(_WARMUP_RUNS):
        zero_grad()
        torch.npu.synchronize()
        run()
        torch.npu.synchronize()

    zero_grad()
    torch.npu.synchronize()
    baseline_allocated = torch.npu.memory_allocated()
    baseline_reserved = torch.npu.memory_reserved()
    torch.npu.reset_peak_memory_stats()

    latencies_ms = []
    for _ in range(_MEASURE_RUNS):
        zero_grad()
        torch.npu.synchronize()
        started = time.perf_counter()
        run()
        torch.npu.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    peak_allocated = torch.npu.max_memory_allocated()
    peak_reserved = torch.npu.max_memory_reserved()
    return {
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.pstdev(latencies_ms),
        "baseline_allocated": baseline_allocated,
        "baseline_reserved": baseline_reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
        "incremental_peak": max(0, peak_allocated - baseline_allocated),
    }


def _iter_tensors(value):
    """Iterate nested tensors without a recursive closure retaining a collector list."""

    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, dict) or hasattr(item, "values"):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)


def _tensor_bytes(value):
    seen = set()
    total = 0
    for tensor in _iter_tensors(value):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _flatten_tensors(value):
    return tuple(_iter_tensors(value))


def _storage_identity(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))


def _tensor_base_closure(tensors):
    original_ids = {id(tensor) for tensor in tensors}
    closure = {}
    pending = list(tensors)
    while pending:
        tensor = pending.pop()
        if id(tensor) in closure:
            continue
        closure[id(tensor)] = tensor
        base = getattr(tensor, "_base", None)
        if isinstance(base, torch.Tensor):
            pending.append(base)
    return tuple(tensor for tensor_id, tensor in closure.items() if tensor_id not in original_ids)


def _live_storage_audit(categories):
    category_rows = {}
    category_storages = {}
    all_storages = {}
    for name, value in categories.items():
        tensors = tuple(
            tensor
            for tensor in _flatten_tensors(value)
            if str(tensor.device).startswith("npu")
        )
        storages = {_storage_identity(tensor) for tensor in tensors}
        category_storages[name] = storages
        category_rows[name] = {
            "tensor_count": len(tensors),
            "logical_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
            "unique_storage_bytes": sum(item[2] for item in storages),
            "storage_ptrs": tuple(sorted(item[1] for item in storages)),
        }
        all_storages.update({item[:2]: item[2] for item in storages})
    overlaps = []
    names = tuple(category_storages)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            shared = category_storages[left_name] & category_storages[right_name]
            if shared:
                overlaps.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "bytes": sum(item[2] for item in shared),
                        "storage_ptrs": tuple(sorted(item[1] for item in shared)),
                    }
                )
    return {
        "categories": category_rows,
        "accounted_unique_storage_bytes": sum(all_storages.values()),
        "overlaps": tuple(overlaps),
    }


def _gc_object_ownership(object_id, expected_type):
    """Find a slots object by id and summarize its immediate Python owners."""

    objects = gc.get_objects()
    target = next(
        (
            candidate
            for candidate in objects
            if id(candidate) == object_id and isinstance(candidate, expected_type)
        ),
        None,
    )
    if target is None:
        del objects
        return {"alive": False, "referrers": ()}

    referrers = gc.get_referrers(target)
    summaries = []
    for referrer in referrers:
        if referrer is objects or referrer is referrers:
            continue
        if isinstance(referrer, types.FrameType):
            local_names = tuple(
                name for name, value in referrer.f_locals.items() if value is target
            )
            summaries.append(
                f"frame:{referrer.f_code.co_name}:locals={local_names or ('<eval-stack>',)}"
            )
        elif isinstance(referrer, dict):
            keys = tuple(str(key) for key, value in referrer.items() if value is target)
            summaries.append(f"dict(len={len(referrer)},keys={keys})")
        elif isinstance(referrer, list):
            indexes = tuple(
                index for index, value in enumerate(referrer) if value is target
            )
            summaries.append(f"list(len={len(referrer)},indexes={indexes[:4]})")
        elif isinstance(referrer, tuple):
            indexes = tuple(
                index for index, value in enumerate(referrer) if value is target
            )
            summaries.append(f"tuple(len={len(referrer)},indexes={indexes[:4]})")
        elif isinstance(referrer, KVStackEntry):
            summaries.append(
                f"KVStackEntry(segment_id={referrer.segment.segment_id})"
            )
        else:
            summaries.append(type(referrer).__name__)
    result = {"alive": True, "referrers": tuple(sorted(set(summaries)))}
    del referrers, target, objects
    return result


def _collect_root_pop_memory_breakdown(run, zero_grad):
    """Run one synchronized diagnostic iteration and sample root-pop allocations."""

    samples = []
    tensor_bytes = {
        "detached_prefix_kv": 0,
        "kv_grad_buffer": 0,
        "anchor": 0,
    }
    storage_audit = None
    detached_kv_weakrefs = ()
    detached_storage_owner_weakrefs = ()
    detached_kv_storage_ptrs = ()
    release_audit = None
    forward_entry_release_audit = None
    ownership_audits = []
    popped_entry_id = None
    popped_segment_kv_id = None
    pushed_owner_ids = {}
    popped_owner_ids = None
    executor_state = None
    active_root_pop = [False]
    original_pop = SegmentExecutor.pop
    original_build_anchors = KVStack.build_past_anchors
    original_forward = SegmentExecutor._forward
    original_backward = torch.autograd.backward

    def sample(stage):
        torch.npu.synchronize()
        row = {
            "stage": stage,
            "allocated": torch.npu.memory_allocated(),
            "reserved": torch.npu.memory_reserved(),
            "window_peak": torch.npu.max_memory_allocated(),
        }
        samples.append(row)

    def weakref_release_status(stage):
        """Snapshot weak references without extending their lifetime afterwards."""

        alive_tensor_count = 0
        alive_storage_ptrs = set()
        for reference in detached_kv_weakrefs:
            referenced_tensor = reference()
            if referenced_tensor is not None:
                alive_tensor_count += 1
                alive_storage_ptrs.add(_storage_identity(referenced_tensor)[1])
            del referenced_tensor
        alive_storage_owner_count = 0
        alive_owner_storage_ptrs = set()
        for reference in detached_storage_owner_weakrefs:
            referenced_tensor = reference()
            if referenced_tensor is not None:
                alive_storage_owner_count += 1
                alive_owner_storage_ptrs.add(_storage_identity(referenced_tensor)[1])
            del referenced_tensor
        status = {
            "stage": stage,
            "original_storage_ptrs": detached_kv_storage_ptrs,
            "alive_tensor_count": alive_tensor_count,
            "alive_storage_ptrs": tuple(sorted(alive_storage_ptrs)),
            "alive_storage_owner_count": alive_storage_owner_count,
            "alive_owner_storage_ptrs": tuple(sorted(alive_owner_storage_ptrs)),
        }
        return status

    def sample_ownership(stage):
        if popped_entry_id is None or popped_segment_kv_id is None:
            return
        row = {
            "stage": stage,
            "entry": _gc_object_ownership(popped_entry_id, KVStackEntry),
            "segment_kv": _gc_object_ownership(popped_segment_kv_id, SegmentKV),
        }
        if popped_owner_ids is not None:
            row.update(
                {
                    "push_context": _gc_object_ownership(
                        popped_owner_ids["context"], TreeAttentionContext
                    ),
                    "new_kv_dict": _gc_object_ownership(
                        popped_owner_ids["new_kv_dict"], dict
                    ),
                    "kv_pair": _gc_object_ownership(
                        popped_owner_ids["kv_pair"], tuple
                    ),
                    "kv_tensor": _gc_object_ownership(
                        popped_owner_ids["kv_tensor"], torch.Tensor
                    ),
                }
            )
        ownership_audits.append(row)

    def traced_pop(executor, segment_id):
        nonlocal storage_audit, detached_kv_weakrefs, detached_storage_owner_weakrefs
        nonlocal detached_kv_storage_ptrs, executor_state
        nonlocal popped_entry_id, popped_segment_kv_id
        nonlocal popped_owner_ids
        entry = executor.kv_stack.top()
        popped_entry_id = id(entry)
        popped_segment_kv_id = id(entry.kv)
        popped_owner_ids = pushed_owner_ids.get(segment_id)
        tensor_bytes["detached_prefix_kv"] = _tensor_bytes(entry.kv.key_values)
        tensor_bytes["kv_grad_buffer"] = _tensor_bytes(entry.gradients)
        detached_tensors = _flatten_tensors(entry.kv.key_values)
        detached_kv_weakrefs = tuple(weakref.ref(tensor) for tensor in detached_tensors)
        detached_storage_owners = _tensor_base_closure(detached_tensors)
        detached_storage_owner_weakrefs = tuple(
            weakref.ref(tensor) for tensor in detached_storage_owners
        )
        detached_kv_storage_ptrs = tuple(
            sorted({_storage_identity(tensor)[1] for tensor in detached_tensors})
        )
        parameters = tuple(executor.model.parameters())
        parameter_grads = tuple(
            parameter.grad for parameter in parameters if parameter.grad is not None
        )
        parameter_main_grads = tuple(
            parameter.main_grad
            for parameter in parameters
            if getattr(parameter, "main_grad", None) is not None
        )
        storage_audit = _live_storage_audit(
            {
                "model_parameters": parameters,
                "model_buffers": tuple(executor.model.buffers()),
                "parameter_grad": parameter_grads,
                "parameter_main_grad": parameter_main_grads,
                "detached_prefix_kv": detached_tensors,
                "kv_grad_buffer": entry.gradients,
            }
        )
        executor_state = {
            "stack_segment_ids": executor.kv_stack.segment_ids,
            "stack_prefix_length": executor.kv_stack.prefix_length,
            "tree_context_active": get_tree_attention_context() is not None,
        }
        # Do not let the diagnostic wrapper extend the KVStackEntry/KV tensor
        # lifetime across SegmentExecutor.pop(). Only weakrefs and primitive
        # storage metadata remain live after this point.
        del detached_storage_owners, detached_tensors
        del parameter_grads, parameter_main_grads, parameters, entry
        active_root_pop[0] = True
        sample("enter_root_pop")
        try:
            return original_pop(executor, segment_id)
        finally:
            sample_ownership("root_pop_complete")
            sample("root_pop_complete")
            active_root_pop[0] = False

    def traced_build_anchors(stack):
        nonlocal release_audit
        if active_root_pop[0]:
            # This hook runs while the caller is still completing the pop/build
            # expression. CPython may therefore still hold the popped entry as a
            # temporary. Sample it accurately instead of claiming it is released.
            sample("after_stack_pop_before_forward")
            release_audit = weakref_release_status("after_stack_pop_before_forward")
            sample_ownership("after_stack_pop_before_forward")
        anchors = original_build_anchors(stack)
        if active_root_pop[0]:
            tensor_bytes["anchor"] = _tensor_bytes(anchors.key_values)
            sample("anchors_built")
        return anchors

    def traced_forward(executor, segment, *, past_key_values, no_grad):
        nonlocal forward_entry_release_audit
        if active_root_pop[0] and not no_grad:
            # By function entry the caller has advanced past anchor construction,
            # so this is the first reliable point at which the old entry can die.
            sample("prefix_recompute_forward_start")
            forward_entry_release_audit = weakref_release_status(
                "prefix_recompute_forward_start"
            )
            sample_ownership("prefix_recompute_forward_start")
        result = original_forward(
            executor,
            segment,
            past_key_values=past_key_values,
            no_grad=no_grad,
        )
        if no_grad:
            context = result[0]
            first_pair = next(iter(context._new_key_values.values()))
            pushed_owner_ids[segment.segment_id] = {
                "context": id(context),
                "new_kv_dict": id(context._new_key_values),
                "kv_pair": id(first_pair),
                "kv_tensor": id(first_pair[0]),
            }
            del first_pair, context
        if active_root_pop[0] and not no_grad:
            sample("prefix_recompute_forward_complete")
        return result

    def traced_backward(*args, **kwargs):
        if not active_root_pop[0]:
            return original_backward(*args, **kwargs)
        sample("before_dkv_injection")
        result = original_backward(*args, **kwargs)
        torch.npu.synchronize()
        sample_ownership("prefix_backward_complete")
        sample("prefix_backward_complete")
        return result

    zero_grad()
    torch.npu.synchronize()
    baseline_allocated = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    with (
        patch.object(SegmentExecutor, "pop", traced_pop),
        patch.object(KVStack, "build_past_anchors", traced_build_anchors),
        patch.object(SegmentExecutor, "_forward", traced_forward),
        patch.object(torch.autograd, "backward", traced_backward),
    ):
        run()
    torch.npu.synchronize()
    return {
        "baseline_allocated": baseline_allocated,
        "whole_iteration_peak": torch.npu.max_memory_allocated(),
        "samples": tuple(samples),
        "tensor_bytes": tensor_bytes,
        "storage_audit": storage_audit,
        "release_audit": release_audit,
        "forward_entry_release_audit": forward_entry_release_audit,
        "ownership_audits": tuple(ownership_audits),
        "executor_state": executor_state,
    }


def _install_module_timing_hooks(model, recorder):
    handles = []

    def add_hooks(module, category, *, measure_backward=True):
        handles.append(module.register_forward_pre_hook(lambda *_: recorder.begin(f"{category}_forward")))
        handles.append(module.register_forward_hook(lambda *_: recorder.end(f"{category}_forward")))
        if measure_backward:
            handles.append(
                module.register_full_backward_pre_hook(
                    lambda *_: recorder.begin(f"{category}_backward")
                )
            )
            handles.append(
                module.register_full_backward_hook(lambda *_: recorder.end(f"{category}_backward"))
            )

    # A root-module backward hook only measures its boundary callback when the
    # integer token inputs do not require gradients; it does not bracket the
    # complete autograd graph. The real backward entry is timed separately.
    add_hooks(model, "model", measure_backward=False)
    for layer in model.decoder.layers:
        add_hooks(layer.self_attention, "attention")
        add_hooks(layer.mlp, "mlp")
    return handles


def _collect_breakdown(run, zero_grad, model, *, tpr):
    for _ in range(_BREAKDOWN_WARMUP_RUNS):
        zero_grad()
        torch.npu.synchronize()
        run()
        torch.npu.synchronize()

    samples = []
    for _ in range(_BREAKDOWN_MEASURE_RUNS):
        recorder = _NPUEventRecorder()
        handles = _install_module_timing_hooks(model, recorder)
        original_push = SegmentExecutor.push
        original_visit_leaf = SegmentExecutor.visit_leaf
        original_pop = SegmentExecutor.pop
        original_autograd_backward = torch.autograd.backward
        active_tpr_phase = [None]

        def call_in_phase(phase, function, *args, **kwargs):
            previous_phase = active_tpr_phase[0]
            active_tpr_phase[0] = phase
            try:
                return recorder.call(phase, function, *args, **kwargs)
            finally:
                active_tpr_phase[0] = previous_phase

        def timed_push(executor, segment_id):
            return call_in_phase("root_push", original_push, executor, segment_id)

        def timed_visit_leaf(executor, segment_id):
            return call_in_phase("leaf_visit", original_visit_leaf, executor, segment_id)

        def timed_pop(executor, segment_id):
            return call_in_phase("root_pop", original_pop, executor, segment_id)

        def timed_autograd_backward(*args, **kwargs):
            phase = active_tpr_phase[0]
            category = f"{phase}_backward" if phase is not None else "reference_backward"
            return recorder.call(category, original_autograd_backward, *args, **kwargs)

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.autograd, "backward", timed_autograd_backward))
                if tpr:
                    stack.enter_context(patch.object(SegmentExecutor, "push", timed_push))
                    stack.enter_context(
                        patch.object(SegmentExecutor, "visit_leaf", timed_visit_leaf)
                    )
                    stack.enter_context(patch.object(SegmentExecutor, "pop", timed_pop))
                zero_grad()
                torch.npu.synchronize()
                started = time.perf_counter()
                run()
                torch.npu.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000.0
        finally:
            for handle in handles:
                handle.remove()
        samples.append((wall_ms, recorder.summarize()))

    categories = set.intersection(*(set(sample) for _, sample in samples))
    result = {
        category: {
            "median_ms": statistics.median(
                sample[category]["total_ms"] for _, sample in samples
            ),
            "mean_ms": statistics.mean(sample[category]["total_ms"] for _, sample in samples),
            "calls": samples[0][1][category]["calls"],
        }
        for category in sorted(categories)
    }
    for category in categories:
        expected_calls = result[category]["calls"]
        assert all(sample[category]["calls"] == expected_calls for _, sample in samples)
    return {
        "wall_median_ms": statistics.median(wall_ms for wall_ms, _ in samples),
        "categories": result,
    }


def _make_case_tokens(prefix_length, suffix_length, sibling_count, device):
    def tokens(start, length):
        return (
            torch.arange(start, start + length, dtype=torch.long, device=device)
            % _PROFILE_VOCAB_SIZE
        )

    return tokens(17, prefix_length), tuple(
        tokens(700 + sibling_index * 601, suffix_length)
        for sibling_index in range(sibling_count)
    )


def _profile_reference_loss_function(*, model_output, data, dp_group):
    del dp_group
    logits = model_output["logits"]
    tokens = data["input_ids"]
    loss_sum = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        tokens[:, 1:].reshape(-1),
        reduction="sum",
    )
    return loss_sum / data["batch_num_tokens"], {}


def _assert_profile_model_scale(model):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert _MIN_PROFILE_PARAMETERS <= parameter_count <= _MAX_PROFILE_PARAMETERS, (
        f"profile model must remain 0.6B-class, got {parameter_count / 1e9:.3f}B parameters"
    )
    return parameter_count


def _profile_reference(monkeypatch, prefix_length, suffix_length, sibling_count):
    device = torch.device("npu")
    full_length = prefix_length + suffix_length
    model = _make_model(
        device,
        tpr=False,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
        model_shape=_PROFILE_MODEL_SHAPE,
    )
    parameter_count = _assert_profile_model_scale(model)
    initial_state = _cpu_state_dict(model)
    _configure_model_runtime(model)
    engine = _make_engine(model, tpr_enabled=False, monkeypatch=monkeypatch)
    prefix, suffixes = _make_case_tokens(prefix_length, suffix_length, sibling_count, device)
    data = _reference_data(*(torch.cat((prefix, suffix)) for suffix in suffixes))
    observed_microbatches = []
    _install_reference_forward(engine, observed_microbatches, monkeypatch)

    def zero_grad():
        model.zero_grad(set_to_none=True)

    def run():
        engine.forward_backward_batch(data, loss_function=_profile_reference_loss_function, forward_only=False)

    adapter_calls, tpr_trace = _verify_fused_adapter_path(run, zero_grad, expected_path="reference")
    assert not tpr_trace
    stats = _profile_runs(run, zero_grad)
    expected_runs = 1 + _WARMUP_RUNS + _MEASURE_RUNS
    assert observed_microbatches == [1] * sibling_count * expected_runs
    stats["adapter_probe_calls"] = adapter_calls
    stats["parameter_count"] = parameter_count
    if _RUN_BREAKDOWN:
        stats["breakdown"] = _collect_breakdown(run, zero_grad, model, tpr=False)
    return stats, initial_state


def _profile_tpr(monkeypatch, prefix_length, suffix_length, sibling_count, initial_state):
    device = torch.device("npu")
    full_length = prefix_length + suffix_length
    model = _make_model(
        device,
        tpr=True,
        max_sequence_length=full_length,
        core_attention_module=_ProfileFusedCausalAttention,
        model_shape=_PROFILE_MODEL_SHAPE,
    )
    parameter_count = _assert_profile_model_scale(model)
    model.load_state_dict(initial_state, strict=True)
    _configure_model_runtime(model)
    engine = _make_engine(model, tpr_enabled=True, monkeypatch=monkeypatch)
    prefix, suffixes = _make_case_tokens(prefix_length, suffix_length, sibling_count, device)
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        data,
        **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(_make_plan(prefix, *suffixes))},
    )

    def zero_grad():
        model.zero_grad(set_to_none=True)

    def run():
        engine.forward_backward_batch(data, loss_function=None, forward_only=False)

    adapter_calls, tpr_trace = _verify_fused_adapter_path(run, zero_grad, expected_path="tpr")
    expected_segment_actions = (
        (0, "push"),
        *((segment_id, "visit_leaf") for segment_id in range(1, sibling_count + 1)),
        (0, "pop"),
    )
    expected_trace = tuple(
        (segment_id, action, layer_number)
        for segment_id, action in expected_segment_actions
        for layer_number in range(1, _PROFILE_NUM_LAYERS + 1)
    )
    actual_trace = tuple(
        (event["segment_id"], event["action"], event["layer_number"])
        for event in tpr_trace
    )
    assert actual_trace == expected_trace
    for event in tpr_trace:
        if event["segment_id"] == 0:
            assert event["query_length"] == event["kv_length"] == prefix_length
        else:
            assert event["query_length"] == suffix_length
            assert event["kv_length"] == prefix_length + suffix_length
    stats = _profile_runs(run, zero_grad)
    stats["adapter_probe_calls"] = adapter_calls
    stats["adapter_trace"] = tpr_trace
    stats["parameter_count"] = parameter_count
    if _RUN_BREAKDOWN:
        stats["breakdown"] = _collect_breakdown(run, zero_grad, model, tpr=True)
    if _RUN_MEMORY_BREAKDOWN:
        stats["memory_breakdown"] = _collect_root_pop_memory_breakdown(run, zero_grad)
    return stats


def _format_gib(value):
    return f"{value / _GIB:.3f} GiB"


def _print_breakdown(name, breakdown, *, sibling_count):
    categories = breakdown["categories"]
    print(f"{name} diagnostic breakdown (median of {_BREAKDOWN_MEASURE_RUNS} runs):")
    print(f"  synchronized wall: {breakdown['wall_median_ms']:.3f} ms")
    ordered_categories = (
        "root_push",
        "leaf_visit",
        "leaf_visit_backward",
        "root_pop",
        "root_pop_backward",
        "model_forward",
        "reference_backward",
        "attention_forward",
        "attention_backward",
        "mlp_forward",
        "mlp_backward",
    )
    for category in ordered_categories:
        if category not in categories:
            continue
        item = categories[category]
        suffix = ""
        if category == "leaf_visit":
            suffix = f", {item['median_ms'] / sibling_count:.3f} ms/leaf"
        print(
            f"  {category:<20} {item['median_ms']:>10.3f} ms "
            f"({item['calls']} calls{suffix})"
        )
    if "leaf_visit" in categories and "leaf_visit_backward" in categories:
        leaf_non_backward = (
            categories["leaf_visit"]["median_ms"]
            - categories["leaf_visit_backward"]["median_ms"]
        )
        print(f"  leaf non-backward    {leaf_non_backward:>10.3f} ms (derived)")
    if "root_pop" in categories and "root_pop_backward" in categories:
        root_pop_non_backward = (
            categories["root_pop"]["median_ms"]
            - categories["root_pop_backward"]["median_ms"]
        )
        print(f"  root_pop non-backward {root_pop_non_backward:>9.3f} ms (derived)")
    backward_categories = (
        "reference_backward",
        "leaf_visit_backward",
        "root_pop_backward",
    )
    model_device_ms = categories.get("model_forward", {}).get("median_ms", 0.0) + sum(
        categories[category]["median_ms"]
        for category in backward_categories
        if category in categories
    )
    approximate_remainder = breakdown["wall_median_ms"] - model_device_ms
    print(f"  wall - model graph    {approximate_remainder:>10.3f} ms (approximate)")


def _print_memory_breakdown(breakdown):
    def format_ptrs(pointers, limit=6):
        rendered = ",".join(hex(pointer) for pointer in pointers[:limit])
        if len(pointers) > limit:
            rendered += f",... ({len(pointers)} total)"
        return rendered

    print("TPR root-pop memory breakdown (one synchronized diagnostic run):")
    print(f"  diagnostic baseline: {breakdown['baseline_allocated'] / _GIB:.3f} GiB")
    print(f"  whole-window peak:   {breakdown['whole_iteration_peak'] / _GIB:.3f} GiB")
    previous = None
    for sample in breakdown["samples"]:
        delta = 0 if previous is None else sample["allocated"] - previous
        print(
            f"  {sample['stage']:<36} allocated={sample['allocated'] / _GIB:.3f} GiB "
            f"delta={delta / _GIB:+.3f} GiB, "
            f"window peak={sample['window_peak'] / _GIB:.3f} GiB"
        )
        previous = sample["allocated"]
    sizes = breakdown["tensor_bytes"]
    print("  logical tensor payloads (storage overlap is not additive):")
    print(f"    detached prefix KV: {sizes['detached_prefix_kv'] / _GIB:.3f} GiB")
    print(f"    KV-grad buffer:     {sizes['kv_grad_buffer'] / _GIB:.3f} GiB")
    print(f"    anchors:            {sizes['anchor'] / _GIB:.3f} GiB")
    audit = breakdown["storage_audit"]
    allocated_at_entry = breakdown["samples"][0]["allocated"]
    print("  live storage audit at root-pop entry:")
    for name, row in audit["categories"].items():
        pointers = format_ptrs(row["storage_ptrs"])
        print(
            f"    {name:<20} tensors={row['tensor_count']:<4} "
            f"logical={row['logical_bytes'] / _GIB:.3f} GiB "
            f"unique_storage={row['unique_storage_bytes'] / _GIB:.3f} GiB "
            f"data_ptr=[{pointers}]"
        )
    accounted = audit["accounted_unique_storage_bytes"]
    print(f"    accounted unique storage: {accounted / _GIB:.3f} GiB")
    print(f"    unclassified allocated:    {(allocated_at_entry - accounted) / _GIB:.3f} GiB")
    for overlap in audit["overlaps"]:
        print(
            f"    alias overlap {overlap['left']} <-> {overlap['right']}: "
            f"{overlap['bytes'] / _GIB:.3f} GiB, data_ptr="
            + format_ptrs(overlap["storage_ptrs"])
        )
    print(f"    executor state:             {breakdown['executor_state']}")
    print("  detached-KV release audit:")
    for release in (
        breakdown["release_audit"],
        breakdown["forward_entry_release_audit"],
    ):
        print(
            f"    {release['stage']}: tensors={release['alive_tensor_count']}, "
            f"view/base owners={release['alive_storage_owner_count']}, "
            "live data_ptr="
            + format_ptrs(release["alive_storage_ptrs"])
        )
    print("  popped object ownership audit (slots objects tracked by id/GC):")
    for ownership in breakdown["ownership_audits"]:
        entry = ownership["entry"]
        segment_kv = ownership["segment_kv"]
        print(
            f"    {ownership['stage']}: "
            f"KVStackEntry alive={entry['alive']} refs={entry['referrers']}; "
            f"SegmentKV alive={segment_kv['alive']} refs={segment_kv['referrers']}"
        )
        for name in ("push_context", "new_kv_dict", "kv_pair", "kv_tensor"):
            if name not in ownership:
                continue
            owner = ownership[name]
            print(
                f"      {name}: alive={owner['alive']} refs={owner['referrers']}"
            )


def _print_case(prefix_length, suffix_length, sibling_count, reference, tpr):
    speedup = reference["median_ms"] / tpr["median_ms"]
    memory_ratio = reference["incremental_peak"] / max(tpr["incremental_peak"], 1)
    memory_reduction = 1.0 - tpr["incremental_peak"] / max(reference["incremental_peak"], 1)
    print(f"\nP={prefix_length}, S={suffix_length}, N={sibling_count}\n")
    assert reference["parameter_count"] == tpr["parameter_count"]
    print(f"Model parameters: {reference['parameter_count'] / 1e9:.3f}B")
    for name, stats in (("Reference", reference), ("TPR", tpr)):
        print(f"{name}:")
        print(f"  median latency:     {stats['median_ms']:.3f} ms")
        print(f"  mean latency:       {stats['mean_ms']:.3f} ms")
        print(f"  std:                {stats['std_ms']:.3f} ms")
        print(f"  baseline allocated: {_format_gib(stats['baseline_allocated'])}")
        print(f"  peak allocated:     {_format_gib(stats['peak_allocated'])}")
        print(f"  incremental peak:   {_format_gib(stats['incremental_peak'])}")
        print(f"  peak reserved:      {_format_gib(stats['peak_reserved'])} (auxiliary)")
        print(f"  adapter probe calls: {stats['adapter_probe_calls']}")
    traced_layers = {event["layer_number"] for event in tpr["adapter_trace"]}
    traced_phases = {
        (event["segment_id"], event["action"])
        for event in tpr["adapter_trace"]
    }
    print(
        "TPR adapter trace verified: "
        f"{len(traced_layers)} layers, {len(traced_phases)} physical phases, "
        f"{len(tpr['adapter_trace'])} adapter calls"
    )
    if _RUN_BREAKDOWN:
        _print_breakdown("Reference", reference["breakdown"], sibling_count=sibling_count)
        _print_breakdown("TPR", tpr["breakdown"], sibling_count=sibling_count)
    if _RUN_MEMORY_BREAKDOWN:
        _print_memory_breakdown(tpr["memory_breakdown"])
    print(f"Speedup (median): {speedup:.3f}x")
    print(f"Incremental-peak ratio: {memory_ratio:.3f}x")
    print(f"Incremental-peak reduction: {memory_reduction * 100.0:.2f}%")
    return speedup, memory_reduction


def test_tpr_engine_profile_report(monkeypatch):
    """Report controlled fused-kernel latency and memory without performance assertions."""

    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    process_baseline = _settled_allocated_samples()[-1]
    rows = []

    for prefix_length, suffix_length, sibling_count in _PROFILE_CASES:
        reference_stats, initial_state = _profile_reference(
            monkeypatch, prefix_length, suffix_length, sibling_count
        )
        recovery_samples = _settled_allocated_samples()
        _assert_memory_recovered(
            expected_baseline=process_baseline,
            samples=recovery_samples,
            path_name="Reference",
        )

        tpr_stats = _profile_tpr(
            monkeypatch, prefix_length, suffix_length, sibling_count, initial_state
        )
        del initial_state
        recovery_samples = _settled_allocated_samples()
        _assert_memory_recovered(
            expected_baseline=process_baseline,
            samples=recovery_samples,
            path_name="TPR",
        )

        speedup, memory_reduction = _print_case(
            prefix_length,
            suffix_length,
            sibling_count,
            reference_stats,
            tpr_stats,
        )
        rows.append(
            (
                prefix_length,
                suffix_length,
                sibling_count,
                reference_stats,
                tpr_stats,
                speedup,
                memory_reduction,
            )
        )

    print("\nControlled fused-reference summary\n")
    print(
        "| P | S | N | Ref median ms | TPR median ms | Speedup | "
        "Ref incr. GiB | TPR incr. GiB | Incr. reduction |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for prefix_length, suffix_length, sibling_count, reference, tpr, speedup, memory_reduction in rows:
        print(
            f"| {prefix_length} | {suffix_length} | {sibling_count} | "
            f"{reference['median_ms']:.3f} | "
            f"{tpr['median_ms']:.3f} | {speedup:.3f}x | "
            f"{reference['incremental_peak'] / _GIB:.3f} | "
            f"{tpr['incremental_peak'] / _GIB:.3f} | {memory_reduction * 100.0:.2f}% |"
        )
