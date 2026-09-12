"""Twenty-step optimization stability for Qwen3.5 Hybrid CP on NPU.

Validate short-horizon optimization stability between CP=1 and CP=2 after
accepting bounded BF16 single-step numerical divergence.

Run from the verl repository root with two NPUs::

    torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 \
      -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_hybrid_cp_multistep_npu.py

This is deliberately an optimization-trajectory test rather than a
per-element parameter-equivalence test. Both models use the same random
initialization, fixed logical token batch, labels, and momentum-free SGD.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available

from ._qwen35_baseline_utils import (
    AllToAllProbe,
    SEQUENCE_LENGTH,
    allreduce_parameter_gradients,
    assert_hybrid_architecture,
    broadcast_module_state,
    destroy_npu_runtime,
    full_tokens,
    gradient_map_diagnostics,
    initialize_npu_runtime,
    make_qwen35_model,
    selected_output_and_loss,
    zigzag_indices,
)


_STEPS = 20
_LEARNING_RATE = float(os.getenv("STAGE34_MULTISTEP_LR", "1e-3"))
_UPDATE_CHUNK_ELEMENTS = int(os.getenv("STAGE34_UPDATE_CHUNK_ELEMENTS", "16777216"))
_CHECKPOINT_STEPS = frozenset((1, 5, 10, 20))
_LOSS_RELATIVE_TOL = 2e-2
_NORM_RATIO_MIN = 0.95
_NORM_RATIO_MAX = 1.05
_FINAL_PARAMETER_COSINE_MIN = 0.999


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = [
    pytest.mark.skipif(
        os.getenv("STAGE34_RUN_20_STEP") != "1",
        reason="set STAGE34_RUN_20_STEP=1 for the 20-step Hybrid CP baseline",
    ),
    pytest.mark.skipif(
        int(os.getenv("WORLD_SIZE", "1")) != 2,
        reason="run the 20-step Hybrid CP baseline with torchrun --nproc_per_node=2",
    ),
]


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=2)
    yield value
    destroy_npu_runtime(value)


@contextmanager
def _ring_probe():
    import mindspeed.core.context_parallel.dot_product_attention as dpa

    original = dpa.ringattn_context_parallel
    calls = []

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    dpa.ringattn_context_parallel = traced
    try:
        yield calls
    finally:
        dpa.ringattn_context_parallel = original


def _tensor_map(model, *, gradients):
    values = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            raise AssertionError(f"missing parameter gradient: {name}")
        values[name] = value
    return values


def _broadcast_parameter_gradients(model, *, src):
    """Keep the replicated CP1 trajectory identical on both test ranks."""
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing CP1 parameter gradient: {name}")
        dist.broadcast(parameter.grad, src=src)


def _assert_finite_metric(value, *, label, step):
    if not math.isfinite(value):
        raise AssertionError(f"step {step}: {label} is not finite: {value}")


@torch.no_grad()
def _sgd_step_and_actual_update_norm(model, *, learning_rate):
    """Apply momentum-free SGD and measure the realized BF16 parameter delta."""
    if _UPDATE_CHUNK_ELEMENTS <= 0:
        raise AssertionError("STAGE34_UPDATE_CHUNK_ELEMENTS must be positive")
    total_squared = None
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing parameter gradient during SGD update: {name}")
        if not parameter.is_contiguous():
            raise AssertionError(f"SGD baseline expects a contiguous parameter: {name}")
        flat_parameter = parameter.view(-1)
        flat_gradient = parameter.grad.reshape(-1)
        if total_squared is None:
            total_squared = torch.zeros((), device=parameter.device, dtype=torch.float32)
        for start in range(0, flat_parameter.numel(), _UPDATE_CHUNK_ELEMENTS):
            end = min(start + _UPDATE_CHUNK_ELEMENTS, flat_parameter.numel())
            parameter_chunk = flat_parameter[start:end]
            before = parameter_chunk.clone()
            parameter_chunk.add_(flat_gradient[start:end], alpha=-learning_rate)
            difference = parameter_chunk.float() - before.float()
            total_squared.add_(difference.square().sum())
    if total_squared is None:
        raise AssertionError("model has no parameters to update")
    return math.sqrt(total_squared.item())


def _ratio(actual, reference):
    if reference == 0.0:
        return 1.0 if actual == 0.0 else float("inf")
    return actual / reference


def _relative_difference(actual, reference):
    return abs(actual - reference) / max(abs(reference), 1e-12)


def _assert_step_communication(*, step, reference_a2a, reference_ring, cp_a2a, cp_ring):
    if reference_a2a.calls or reference_ring:
        raise AssertionError(
            f"step {step}: CP1 unexpectedly used communication: "
            f"A2A={len(reference_a2a.calls)}, Ring={len(reference_ring)}"
        )
    cp2hp = cp_a2a.count("cp2hp")
    hp2cp = cp_a2a.count("hp2cp")
    if cp2hp != 108 or hp2cp != 18 or len(cp_ring) != 6:
        raise AssertionError(
            f"step {step}: invalid CP2 communication contract: "
            f"cp2hp={cp2hp}, hp2cp={hp2cp}, Ring={len(cp_ring)}"
        )
    return cp2hp, hp2cp, len(cp_ring)


def test_qwen35_hybrid_cp_twenty_step_optimization_stability(runtime):
    if SEQUENCE_LENGTH % 4:
        raise AssertionError("STAGE34_SEQUENCE_LENGTH must be divisible by 4 for CP=2")
    if _LEARNING_RATE <= 0.0:
        raise AssertionError("STAGE34_MULTISTEP_LR must be positive")

    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    torch.manual_seed(355001)
    reference = make_qwen35_model(runtime, cp_size=1)
    broadcast_module_state(reference, src=0)
    reference_gdn, reference_fa = assert_hybrid_architecture(reference)

    torch.manual_seed(355002)
    actual = make_qwen35_model(runtime, cp_size=2)
    actual.load_state_dict(reference.state_dict(), strict=True)
    actual_gdn, actual_fa = assert_hybrid_architecture(actual)
    if (len(reference_gdn), len(reference_fa), len(actual_gdn), len(actual_fa)) != (18, 6, 18, 6):
        raise AssertionError("expected both models to contain 18 GDN and 6 Full-Attention layers")
    if not all(type(layer.self_attention) is mindspeed_gdn.GatedDeltaNet for layer in actual_gdn):
        raise AssertionError("CP2 model did not use the native MindSpeed GDN implementation")

    tokens, positions, labels, valid = full_tokens(runtime.device)
    indices = zigzag_indices(
        SEQUENCE_LENGTH,
        cp_rank=runtime.rank,
        cp_size=2,
        device=runtime.device,
    )
    local_tokens = tokens.index_select(1, indices)
    local_positions = positions.index_select(1, indices)
    local_labels = labels.index_select(0, indices)
    local_valid = valid.index_select(0, indices)

    reference_losses = []
    actual_losses = []
    gradient_ratios = []
    update_ratios = []
    checkpoint_metrics = {}

    for step in range(1, _STEPS + 1):
        reference.zero_grad(set_to_none=True)
        with (
            AllToAllProbe(mindspeed_gdn) as reference_a2a,
            _ring_probe() as reference_ring,
        ):
            reference_logits = reference(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=None,
            )
            _, reference_loss = selected_output_and_loss(
                reference_logits,
                labels,
                valid,
            )
            reference_loss.backward()
        _broadcast_parameter_gradients(reference, src=0)
        reference_loss_value_tensor = reference_loss.detach().clone()
        dist.broadcast(reference_loss_value_tensor, src=0)

        actual.zero_grad(set_to_none=True)
        with (
            AllToAllProbe(mindspeed_gdn) as cp_a2a,
            _ring_probe() as cp_ring,
        ):
            actual_logits = actual(
                input_ids=local_tokens,
                position_ids=local_positions,
                attention_mask=None,
            )
            _, local_loss = selected_output_and_loss(
                actual_logits,
                local_labels,
                local_valid,
            )
            local_loss.backward()

        cp_loss = local_loss.detach().clone()
        dist.all_reduce(cp_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)
        allreduce_parameter_gradients(actual, runtime.cp_group)
        cp2hp, hp2cp, ring = _assert_step_communication(
            step=step,
            reference_a2a=reference_a2a,
            reference_ring=reference_ring,
            cp_a2a=cp_a2a,
            cp_ring=cp_ring,
        )

        reference_gradient_map = _tensor_map(reference, gradients=True)
        actual_gradient_map = _tensor_map(actual, gradients=True)
        gradient_metrics = gradient_map_diagnostics(
            reference_gradient_map,
            actual_gradient_map,
        ).aggregate
        reference_loss_value = reference_loss_value_tensor.item()
        actual_loss_value = cp_loss.item()
        loss_relative = _relative_difference(actual_loss_value, reference_loss_value)

        for label, value in (
            ("CP1 loss", reference_loss_value),
            ("CP2 loss", actual_loss_value),
            ("CP1 gradient norm", gradient_metrics.reference_norm),
            ("CP2 gradient norm", gradient_metrics.actual_norm),
            ("gradient norm ratio", gradient_metrics.norm_ratio),
        ):
            _assert_finite_metric(value, label=label, step=step)

        reference_update_norm = _sgd_step_and_actual_update_norm(
            reference,
            learning_rate=_LEARNING_RATE,
        )
        actual_update_norm = _sgd_step_and_actual_update_norm(
            actual,
            learning_rate=_LEARNING_RATE,
        )
        update_ratio = _ratio(actual_update_norm, reference_update_norm)
        _assert_finite_metric(reference_update_norm, label="CP1 update norm", step=step)
        _assert_finite_metric(actual_update_norm, label="CP2 update norm", step=step)
        _assert_finite_metric(update_ratio, label="update norm ratio", step=step)

        parameter_metrics = None
        if step in _CHECKPOINT_STEPS:
            parameter_metrics = gradient_map_diagnostics(
                _tensor_map(reference, gradients=False),
                _tensor_map(actual, gradients=False),
            ).aggregate
            for label, value in (
                ("parameter relative L2", parameter_metrics.relative_l2),
                ("parameter cosine", parameter_metrics.cosine),
                ("gradient relative L2", gradient_metrics.relative_l2),
                ("gradient cosine", gradient_metrics.cosine),
            ):
                _assert_finite_metric(value, label=label, step=step)
            checkpoint_metrics[step] = (parameter_metrics, gradient_metrics)

        reference_losses.append(reference_loss_value)
        actual_losses.append(actual_loss_value)
        gradient_ratios.append(gradient_metrics.norm_ratio)
        update_ratios.append(update_ratio)

        if runtime.rank == 0:
            print(
                f"PUNCTURE-3-20STEP step={step:02d}"
                f"\n  loss CP1/CP2/relative: "
                f"{reference_loss_value:.9f}/{actual_loss_value:.9f}/{loss_relative:.6e}"
                f"\n  grad norm CP1/CP2/ratio: "
                f"{gradient_metrics.reference_norm:.6e}/"
                f"{gradient_metrics.actual_norm:.6e}/{gradient_metrics.norm_ratio:.9f}"
                f"\n  update norm CP1/CP2/ratio: "
                f"{reference_update_norm:.6e}/{actual_update_norm:.6e}/{update_ratio:.9f}"
                f"\n  communication CP1 A2A/Ring=0/0; "
                f"CP2 cp2hp/hp2cp/Ring={cp2hp}/{hp2cp}/{ring}"
            )
            if parameter_metrics is not None:
                print(
                    f"  checkpoint {step}: parameter rel-L2/cosine="
                    f"{parameter_metrics.relative_l2:.6e}/{parameter_metrics.cosine:.9f}; "
                    f"gradient rel-L2/cosine="
                    f"{gradient_metrics.relative_l2:.6e}/{gradient_metrics.cosine:.9f}"
                )

    final_loss_relative = _relative_difference(actual_losses[-1], reference_losses[-1])
    final_parameter_metrics, final_gradient_metrics = checkpoint_metrics[_STEPS]
    max_loss_relative = max(
        _relative_difference(actual_loss, reference_loss)
        for reference_loss, actual_loss in zip(reference_losses, actual_losses)
    )
    reference_loss_change = reference_losses[-1] - reference_losses[0]
    actual_loss_change = actual_losses[-1] - actual_losses[0]
    if runtime.rank == 0:
        print(
            "PUNCTURE-3-20STEP RESULTS"
            f"\n  steps/learning-rate: {_STEPS}/{_LEARNING_RATE:.6e}"
            f"\n  final loss CP1/CP2/relative: "
            f"{reference_losses[-1]:.9f}/{actual_losses[-1]:.9f}/{final_loss_relative:.6e}"
            f"\n  loss endpoint change CP1/CP2: "
            f"{reference_loss_change:.6e}/{actual_loss_change:.6e}"
            f"\n  max loss relative difference: {max_loss_relative:.6e}"
            f"\n  gradient norm ratio min/max: "
            f"{min(gradient_ratios):.9f}/{max(gradient_ratios):.9f}"
            f"\n  update norm ratio min/max: "
            f"{min(update_ratios):.9f}/{max(update_ratios):.9f}"
            f"\n  step20 parameter rel-L2/cosine: "
            f"{final_parameter_metrics.relative_l2:.6e}/"
            f"{final_parameter_metrics.cosine:.9f}"
            f"\n  step20 gradient rel-L2/cosine: "
            f"{final_gradient_metrics.relative_l2:.6e}/{final_gradient_metrics.cosine:.9f}"
        )

    if final_loss_relative >= _LOSS_RELATIVE_TOL:
        raise AssertionError(
            f"step 20 loss relative difference {final_loss_relative:.6e} "
            f"exceeds {_LOSS_RELATIVE_TOL:.6e}"
        )
    if reference_loss_change * actual_loss_change < 0.0:
        raise AssertionError("CP1 and CP2 loss curves have opposite endpoint trends")
    for step, ratio in enumerate(gradient_ratios, start=1):
        if not _NORM_RATIO_MIN <= ratio <= _NORM_RATIO_MAX:
            raise AssertionError(
                f"step {step} gradient norm ratio {ratio:.9f} is outside "
                f"[{_NORM_RATIO_MIN}, {_NORM_RATIO_MAX}]"
            )
    for step, ratio in enumerate(update_ratios, start=1):
        if not _NORM_RATIO_MIN <= ratio <= _NORM_RATIO_MAX:
            raise AssertionError(
                f"step {step} update norm ratio {ratio:.9f} is outside "
                f"[{_NORM_RATIO_MIN}, {_NORM_RATIO_MAX}]"
            )
    if final_parameter_metrics.cosine < _FINAL_PARAMETER_COSINE_MIN:
        raise AssertionError(
            f"step 20 parameter cosine {final_parameter_metrics.cosine:.9f} is below "
            f"{_FINAL_PARAMETER_COSINE_MIN:.9f}"
        )

    if runtime.rank == 0:
        print("PUNCTURE-3-20STEP PASS: provisional stability thresholds satisfied")
