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

"""Opt-in NPU audit of DTA live-storage and phase-level memory peaks."""

import gc
import os
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

import test_dta_engine_profile_npu as profile
from test_dta_engine_reference_equivalence_npu import (
    _configure_model_runtime,
    _install_reference_forward,
    _make_engine,
    _make_plan,
    _reference_data,
)
from test_segment_push_pop_npu import _install_single_rank_runtime
from verl.models.mcore.dta import DTA_REQUEST_KEY, DTAForwardBackwardRequest, SegmentExecutor
from verl.models.mcore.dta.segment_plan import SegmentPlan, SegmentSpec
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("DTA_RUN_MEMORY_AUDIT") != "1",
    reason="Set DTA_RUN_MEMORY_AUDIT=1 for the NPU memory-lifecycle audit",
)

_PREFIX_LENGTH = 16384
_AUDIT_SUFFIX_LENGTH = 1024
_AUDIT_SIBLING_COUNT = 8
_CROSSOVER_SUFFIXES = (512, 1024, 2048, 4096, 8192)
_CROSSOVER_SIBLING_COUNT = 2


def _gib(value):
    return value / profile._GIB


def _build_model(device, *, dta, max_sequence_length):
    model = profile._make_model(
        device,
        dta=dta,
        max_sequence_length=max_sequence_length,
        core_attention_module=profile._ProfileFusedCausalAttention,
        model_shape=profile._PROFILE_MODEL_SHAPE,
    )
    profile._assert_profile_model_scale(model)
    _configure_model_runtime(model)
    return model


def _make_dta_data(prefix_length, suffix_length, sibling_count, device):
    prefix, suffixes = profile._make_case_tokens(
        prefix_length, suffix_length, sibling_count, device
    )
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(
        data,
        **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(_make_plan(prefix, *suffixes))},
    )
    return data


def _measure_whole_iteration(run, zero_grad):
    zero_grad()
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    run()
    torch.npu.synchronize()
    return {
        "baseline": baseline,
        "peak": torch.npu.max_memory_allocated(),
        "end": torch.npu.memory_allocated(),
    }


def _measure_dta_phases(run, zero_grad):
    rows = []
    originals = {
        "root_push": SegmentExecutor.push,
        "leaf_visit": SegmentExecutor.visit_leaf,
        "root_pop": SegmentExecutor.pop,
    }

    def measured(phase, original):
        def wrapper(executor, segment_id):
            torch.npu.synchronize()
            start = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            result = original(executor, segment_id)
            torch.npu.synchronize()
            rows.append(
                {
                    "phase": phase,
                    "segment_id": segment_id,
                    "start": start,
                    "peak": torch.npu.max_memory_allocated(),
                    "end": torch.npu.memory_allocated(),
                }
            )
            return result

        return wrapper

    zero_grad()
    torch.npu.synchronize()
    with (
        patch.object(SegmentExecutor, "push", measured("root_push", originals["root_push"])),
        patch.object(
            SegmentExecutor,
            "visit_leaf",
            measured("leaf_visit", originals["leaf_visit"]),
        ),
        patch.object(SegmentExecutor, "pop", measured("root_pop", originals["root_pop"])),
    ):
        run()
    return tuple(rows)


def _measure_reference_phases(run, zero_grad, model):
    rows = []
    active_forward = [None]
    original_backward = torch.autograd.backward

    def forward_pre_hook(*_):
        torch.npu.synchronize()
        start = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        active_forward[0] = start

    def forward_hook(*_):
        torch.npu.synchronize()
        rows.append(
            {
                "phase": "reference_forward",
                "start": active_forward[0],
                "peak": torch.npu.max_memory_allocated(),
                "end": torch.npu.memory_allocated(),
            }
        )
        active_forward[0] = None

    def traced_backward(*args, **kwargs):
        torch.npu.synchronize()
        start = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        result = original_backward(*args, **kwargs)
        torch.npu.synchronize()
        rows.append(
            {
                "phase": "reference_backward",
                "start": start,
                "peak": torch.npu.max_memory_allocated(),
                "end": torch.npu.memory_allocated(),
            }
        )
        return result

    pre_handle = model.register_forward_pre_hook(forward_pre_hook)
    post_handle = model.register_forward_hook(forward_hook)
    zero_grad()
    try:
        with patch.object(torch.autograd, "backward", traced_backward):
            run()
    finally:
        pre_handle.remove()
        post_handle.remove()
    return tuple(rows)


def _measure_fresh_prefix_forward(model, prefix_length, device):
    prefix = (
        torch.arange(17, 17 + prefix_length, dtype=torch.long, device=device)
        % profile._PROFILE_VOCAB_SIZE
    )
    segment = SegmentSpec(0, None, prefix.cpu(), 0, 0, ())
    del prefix
    plan = SegmentPlan((segment,), root_id=0, total_loss_weight=1.0)
    executor = SegmentExecutor(model, plan)
    model.zero_grad(set_to_none=True)
    torch.npu.synchronize()
    baseline = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    context, logits = executor._forward(segment, past_key_values={}, no_grad=False)
    torch.npu.synchronize()
    result = {
        "baseline": baseline,
        "allocated_after_forward": torch.npu.memory_allocated(),
        "peak": torch.npu.max_memory_allocated(),
    }
    del logits, context, executor, plan, segment
    gc.collect()
    torch.npu.synchronize()
    result["allocated_after_release"] = torch.npu.memory_allocated()
    return result


def _print_phase_rows(title, rows):
    print(f"\n{title}")
    for index, row in enumerate(rows, start=1):
        print(
            f"  #{index:02d} {row['phase']:<20} "
            f"segment={row.get('segment_id', '-')} "
            f"start={_gib(row['start']):.3f} GiB "
            f"peak={_gib(row['peak']):.3f} GiB "
            f"incremental={_gib(row['peak'] - row['start']):.3f} GiB "
            f"end={_gib(row['end']):.3f} GiB"
        )


def test_root_pop_storage_lifecycle_and_fresh_prefix_forward(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _build_model(
        device,
        dta=True,
        max_sequence_length=_PREFIX_LENGTH + _AUDIT_SUFFIX_LENGTH,
    )
    engine = _make_engine(model, dta_enabled=True, monkeypatch=monkeypatch)
    data = _make_dta_data(
        _PREFIX_LENGTH, _AUDIT_SUFFIX_LENGTH, _AUDIT_SIBLING_COUNT, device
    )

    def zero_grad():
        model.zero_grad(set_to_none=True)

    def run():
        engine.forward_backward_batch(data, loss_function=None, forward_only=False)

    fresh = _measure_fresh_prefix_forward(model, _PREFIX_LENGTH, device)
    audit = profile._collect_root_pop_memory_breakdown(run, zero_grad)
    samples = {sample["stage"]: sample for sample in audit["samples"]}
    recompute_incremental = (
        samples["prefix_recompute_forward_complete"]["allocated"]
        - samples["prefix_recompute_forward_start"]["allocated"]
    )
    fresh_incremental = fresh["allocated_after_forward"] - fresh["baseline"]

    print("\nFresh-prefix versus root-pop recompute")
    print(f"  fresh P={_PREFIX_LENGTH} baseline:          {_gib(fresh['baseline']):.3f} GiB")
    print(f"  fresh forward incremental allocated: {_gib(fresh_incremental):.3f} GiB")
    print(f"  fresh forward peak incremental:      {_gib(fresh['peak'] - fresh['baseline']):.3f} GiB")
    print(f"  allocated after graph release:       {_gib(fresh['allocated_after_release']):.3f} GiB")
    print(f"  root-pop recompute allocated delta:   {_gib(recompute_incremental):.3f} GiB")
    profile._print_memory_breakdown(audit)

    # The build-anchors hook can observe a CPython expression temporary. The
    # forward-entry sample is the meaningful lifetime boundary and is reported
    # rather than made gating, because this file is an opt-in diagnostic audit.
    assert not audit["executor_state"]["tree_context_active"]


def test_reference_forward_backward_phase_memory(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _build_model(
        device,
        dta=False,
        max_sequence_length=_PREFIX_LENGTH + _AUDIT_SUFFIX_LENGTH,
    )
    engine = _make_engine(model, dta_enabled=False, monkeypatch=monkeypatch)
    prefix, suffixes = profile._make_case_tokens(
        _PREFIX_LENGTH, _AUDIT_SUFFIX_LENGTH, _AUDIT_SIBLING_COUNT, device
    )
    data = _reference_data(*(torch.cat((prefix, suffix)) for suffix in suffixes))
    observed = []
    _install_reference_forward(engine, observed, monkeypatch)

    def zero_grad():
        model.zero_grad(set_to_none=True)

    def run():
        engine.forward_backward_batch(
            data,
            loss_function=profile._profile_reference_loss_function,
            forward_only=False,
        )

    whole = _measure_whole_iteration(run, zero_grad)
    phases = _measure_reference_phases(run, zero_grad, model)
    _print_phase_rows("Reference phase-level memory", phases)
    max_forward = max(row["peak"] for row in phases if row["phase"] == "reference_forward")
    max_backward = max(row["peak"] for row in phases if row["phase"] == "reference_backward")
    print("\nReference whole-iteration memory")
    print(f"  baseline: {_gib(whole['baseline']):.3f} GiB")
    print(f"  peak:     {_gib(whole['peak']):.3f} GiB")
    print(f"  end:      {_gib(whole['end']):.3f} GiB")
    print(f"  max forward phase peak:  {_gib(max_forward):.3f} GiB")
    print(f"  max backward phase peak: {_gib(max_backward):.3f} GiB")
    print(f"  peak owner: {'forward' if max_forward >= max_backward else 'backward'}")


def test_leaf_root_memory_crossover_by_suffix_length(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _build_model(
        device,
        dta=True,
        max_sequence_length=_PREFIX_LENGTH + max(_CROSSOVER_SUFFIXES),
    )
    engine = _make_engine(model, dta_enabled=True, monkeypatch=monkeypatch)
    rows = []

    def zero_grad():
        model.zero_grad(set_to_none=True)

    for suffix_length in _CROSSOVER_SUFFIXES:
        data = _make_dta_data(
            _PREFIX_LENGTH, suffix_length, _CROSSOVER_SIBLING_COUNT, device
        )

        def run():
            engine.forward_backward_batch(data, loss_function=None, forward_only=False)

        phases = _measure_dta_phases(run, zero_grad)
        leaf_peak = max(row["peak"] for row in phases if row["phase"] == "leaf_visit")
        root_peak = max(row["peak"] for row in phases if row["phase"] == "root_pop")
        rows.append((suffix_length, leaf_peak, root_peak))

    print("\nDTA leaf/root peak crossover")
    print("| S | Leaf peak GiB | Root-pop peak GiB | Peak owner |")
    print("|---:|---:|---:|:---|")
    for suffix_length, leaf_peak, root_peak in rows:
        owner = "leaf" if leaf_peak > root_peak else "root_pop"
        print(f"| {suffix_length} | {_gib(leaf_peak):.3f} | {_gib(root_peak):.3f} | {owner} |")
    crossover = next(
        (suffix_length for suffix_length, leaf_peak, root_peak in rows if leaf_peak > root_peak),
        None,
    )
    print(f"First measured leaf-dominant suffix: {crossover}")
