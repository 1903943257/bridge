"""Training-trajectory diagnostics for the Qwen3.5 Hybrid CP baseline.

Experiment 1 separates CP numerical error from optimization-trajectory drift.
It reproduces the fixed-batch 20-step BF16-SGD split, then copies the CP=1
step-20 parameters into CP=2 and compares gradients at identical parameters::

    STAGE34_RUN_SAME_PARAMETER=1 \
      torchrun --master_addr=127.0.0.1 --master_port=29558 \
      --nproc_per_node=2 -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_hybrid_cp_training_diagnostics_npu.py \
      -k same_parameter

Experiment 2 uses a different deterministic logical batch at every step and
FP32 master-weight SGD. Its first server run is for threshold calibration: it
hard-fails only on non-finite values or an invalid communication contract::

    STAGE34_RUN_REALISTIC_MULTIBATCH=1 \
      torchrun --master_addr=127.0.0.1 --master_port=29559 \
      --nproc_per_node=2 -m pytest -s -v \
      tests/models/mcore/baseline/test_qwen35_hybrid_cp_training_diagnostics_npu.py \
      -k realistic_multibatch

Neither experiment changes Megatron, MindSpeed, or MindSpeed-Ops kernels.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from ._qwen35_baseline_utils import (
    AllToAllProbe,
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
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
from verl.utils.device import is_torch_npu_available


_STEPS = 20
_LEARNING_RATE = float(
    os.getenv(
        "STAGE34_TRAINING_DIAGNOSTIC_LR",
        os.getenv("STAGE34_MULTISTEP_LR", "1e-3"),
    )
)
_UPDATE_CHUNK_ELEMENTS = int(os.getenv("STAGE34_UPDATE_CHUNK_ELEMENTS", "16777216"))
_CHECKPOINT_STEPS = frozenset((1, 5, 10, 20))


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="run the Hybrid CP training diagnostics with torchrun --nproc_per_node=2",
)


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


def _broadcast_parameter_gradients(model, *, src=0):
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing CP1 parameter gradient: {name}")
        dist.broadcast(parameter.grad, src=src)


def _assert_finite(value, *, label, step):
    if not math.isfinite(value):
        raise AssertionError(f"step {step}: {label} is not finite: {value}")


def _assert_metrics_finite(metrics, *, label, step):
    for field in (
        "reference_norm",
        "actual_norm",
        "norm_ratio",
        "absolute_l2",
        "relative_l2",
        "cosine",
    ):
        _assert_finite(getattr(metrics, field), label=f"{label} {field}", step=step)


def _relative_difference(actual, reference):
    return abs(actual - reference) / max(abs(reference), 1e-12)


def _format_metrics(metrics):
    return (
        f"ref_norm={metrics.reference_norm:.6e}, cp2_norm={metrics.actual_norm:.6e}, "
        f"norm_ratio={metrics.norm_ratio:.9f}, rel_l2={metrics.relative_l2:.6e}, "
        f"cosine={metrics.cosine:.9f}"
    )


def _assert_communication(*, step, reference_a2a, reference_ring, cp_a2a, cp_ring):
    if reference_a2a.calls or reference_ring:
        raise AssertionError(
            f"step {step}: CP1 unexpectedly used communication: "
            f"A2A={len(reference_a2a.calls)}, Ring={len(reference_ring)}"
        )
    cp2hp = cp_a2a.count("cp2hp")
    hp2cp = cp_a2a.count("hp2cp")
    if (cp2hp, hp2cp, len(cp_ring)) != (108, 18, 6):
        raise AssertionError(
            f"step {step}: invalid CP2 communication contract: "
            f"cp2hp={cp2hp}, hp2cp={hp2cp}, Ring={len(cp_ring)}"
        )
    return cp2hp, hp2cp, len(cp_ring)


def _make_model_pair(runtime, *, seed):
    torch.manual_seed(seed)
    reference = make_qwen35_model(runtime, cp_size=1)
    broadcast_module_state(reference, src=0)
    reference_gdn, reference_fa = assert_hybrid_architecture(reference)

    torch.manual_seed(seed + 1)
    actual = make_qwen35_model(runtime, cp_size=2)
    actual.load_state_dict(reference.state_dict(), strict=True)
    actual_gdn, actual_fa = assert_hybrid_architecture(actual)
    if (len(reference_gdn), len(reference_fa), len(actual_gdn), len(actual_fa)) != (18, 6, 18, 6):
        raise AssertionError("expected both models to contain 18 GDN and 6 Full-Attention layers")
    return reference, actual


def _local_cp_batch(runtime, tokens, positions, labels, valid):
    indices = zigzag_indices(
        SEQUENCE_LENGTH,
        cp_rank=runtime.rank,
        cp_size=2,
        device=runtime.device,
    )
    return (
        tokens.index_select(1, indices),
        positions.index_select(1, indices),
        labels.index_select(0, indices),
        valid.index_select(0, indices),
    )


@dataclass(frozen=True)
class PairStepResult:
    reference_loss: float
    actual_loss: float
    loss_relative_difference: float
    gradient_metrics: object
    communication: tuple[int, int, int]


def _forward_backward_pair(runtime, reference, actual, batch, *, step):
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    tokens, positions, labels, valid = batch
    local_tokens, local_positions, local_labels, local_valid = _local_cp_batch(
        runtime,
        tokens,
        positions,
        labels,
        valid,
    )

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
        _, reference_loss = selected_output_and_loss(reference_logits, labels, valid)
        reference_loss.backward()
    _broadcast_parameter_gradients(reference)
    reference_loss_value = reference_loss.detach().clone()
    dist.broadcast(reference_loss_value, src=0)

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
        _, local_loss = selected_output_and_loss(actual_logits, local_labels, local_valid)
        local_loss.backward()

    actual_loss_value = local_loss.detach().clone()
    dist.all_reduce(actual_loss_value, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    allreduce_parameter_gradients(actual, runtime.cp_group)
    communication = _assert_communication(
        step=step,
        reference_a2a=reference_a2a,
        reference_ring=reference_ring,
        cp_a2a=cp_a2a,
        cp_ring=cp_ring,
    )
    gradient_metrics = gradient_map_diagnostics(
        _tensor_map(reference, gradients=True),
        _tensor_map(actual, gradients=True),
    ).aggregate

    reference_loss_float = reference_loss_value.item()
    actual_loss_float = actual_loss_value.item()
    result = PairStepResult(
        reference_loss=reference_loss_float,
        actual_loss=actual_loss_float,
        loss_relative_difference=_relative_difference(actual_loss_float, reference_loss_float),
        gradient_metrics=gradient_metrics,
        communication=communication,
    )
    for label, value in (
        ("CP1 loss", result.reference_loss),
        ("CP2 loss", result.actual_loss),
        ("loss relative difference", result.loss_relative_difference),
        ("CP1 gradient norm", gradient_metrics.reference_norm),
        ("CP2 gradient norm", gradient_metrics.actual_norm),
        ("gradient norm ratio", gradient_metrics.norm_ratio),
        ("gradient relative L2", gradient_metrics.relative_l2),
        ("gradient cosine", gradient_metrics.cosine),
    ):
        _assert_finite(value, label=label, step=step)
    return result


@torch.no_grad()
def _direct_bf16_sgd_step(model, *, learning_rate):
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing gradient during direct BF16 SGD: {name}")
        parameter.add_(parameter.grad, alpha=-learning_rate)


class _FP32MasterSGD:
    """Minimal momentum-free SGD with FP32 master parameters."""

    def __init__(self, model, *, learning_rate):
        if _UPDATE_CHUNK_ELEMENTS <= 0:
            raise AssertionError("STAGE34_UPDATE_CHUNK_ELEMENTS must be positive")
        self.learning_rate = learning_rate
        self.entries = [
            (name, parameter, parameter.detach().float().clone())
            for name, parameter in model.named_parameters()
        ]

    def state_map(self):
        return {name: master for name, _, master in self.entries}

    @torch.no_grad()
    def step(self):
        update_squared = None
        for name, parameter, master in self.entries:
            if parameter.grad is None:
                raise AssertionError(f"missing gradient during FP32 master SGD: {name}")
            if not parameter.is_contiguous() or not master.is_contiguous():
                raise AssertionError(f"FP32 master SGD expects contiguous parameters: {name}")
            parameter_flat = parameter.view(-1)
            master_flat = master.view(-1)
            gradient_flat = parameter.grad.reshape(-1)
            if update_squared is None:
                update_squared = torch.zeros((), device=parameter.device, dtype=torch.float32)
            for start in range(0, parameter_flat.numel(), _UPDATE_CHUNK_ELEMENTS):
                end = min(start + _UPDATE_CHUNK_ELEMENTS, parameter_flat.numel())
                gradient_chunk = gradient_flat[start:end].float()
                update_squared.add_(
                    gradient_chunk.square().sum() * (self.learning_rate**2)
                )
                master_chunk = master_flat[start:end]
                master_chunk.add_(gradient_chunk, alpha=-self.learning_rate)
                parameter_flat[start:end].copy_(master_chunk)
        if update_squared is None:
            raise AssertionError("model has no parameters to update")
        return math.sqrt(update_squared.item())


def _multibatch_tokens(device, *, step):
    positions = torch.arange(SEQUENCE_LENGTH, device=device, dtype=torch.long)
    tokens = (
        positions * 15_485_863 + step * 32_452_843 + 23
    ) % VOCAB_SIZE
    labels = torch.roll(tokens, shifts=-1)
    valid = torch.ones(SEQUENCE_LENGTH, device=device, dtype=torch.bool)
    valid[-1] = False
    return tokens.unsqueeze(0), positions.unsqueeze(0), labels, valid


@pytest.mark.skipif(
    os.getenv("STAGE34_RUN_SAME_PARAMETER") != "1",
    reason="set STAGE34_RUN_SAME_PARAMETER=1 for this 20-step diagnostic",
)
def test_same_parameter_gradient_after_twenty_step_trajectory_split(runtime):
    """Determine whether low step-20 gradient cosine comes from parameter drift."""
    if _LEARNING_RATE <= 0.0:
        raise AssertionError("STAGE34_TRAINING_DIAGNOSTIC_LR must be positive")

    # Match the original fixed-batch 20-step experiment exactly.
    reference, actual = _make_model_pair(runtime, seed=355001)
    batch = full_tokens(runtime.device)
    initial_same_parameter_result = _forward_backward_pair(
        runtime,
        reference,
        actual,
        batch,
        step=0,
    )
    trajectory_result = None
    for step in range(1, _STEPS + 1):
        trajectory_result = _forward_backward_pair(
            runtime,
            reference,
            actual,
            batch,
            step=step,
        )
        _direct_bf16_sgd_step(reference, learning_rate=_LEARNING_RATE)
        _direct_bf16_sgd_step(actual, learning_rate=_LEARNING_RATE)
        if runtime.rank == 0 and step in _CHECKPOINT_STEPS:
            print(
                f"SAME-PARAMETER trajectory step={step:02d}"
                f"\n  loss CP1/CP2/relative: {trajectory_result.reference_loss:.9f}/"
                f"{trajectory_result.actual_loss:.9f}/"
                f"{trajectory_result.loss_relative_difference:.6e}"
                f"\n  gradient: {_format_metrics(trajectory_result.gradient_metrics)}"
            )

    trajectory_parameter_metrics = gradient_map_diagnostics(
        _tensor_map(reference, gradients=False),
        _tensor_map(actual, gradients=False),
    ).aggregate

    # Preserve the trained CP=1 point and evaluate both implementations at
    # exactly that point. This intentionally discards only the CP2 trajectory,
    # after all trajectory diagnostics above have already been captured.
    actual.load_state_dict(reference.state_dict(), strict=True)
    synchronized_parameter_metrics = gradient_map_diagnostics(
        _tensor_map(reference, gradients=False),
        _tensor_map(actual, gradients=False),
    ).aggregate
    _assert_metrics_finite(
        trajectory_parameter_metrics,
        label="trajectory parameters",
        step=_STEPS,
    )
    _assert_metrics_finite(
        synchronized_parameter_metrics,
        label="synchronized parameters",
        step=_STEPS,
    )
    same_parameter_result = _forward_backward_pair(
        runtime,
        reference,
        actual,
        batch,
        step=_STEPS + 1,
    )

    if runtime.rank == 0:
        print(
            "SAME-PARAMETER GRADIENT CHECK RESULTS"
            f"\n  learning-rate/steps: {_LEARNING_RATE:.6e}/{_STEPS}"
            f"\n  initial same-parameter gradient: "
            f"{_format_metrics(initial_same_parameter_result.gradient_metrics)}"
            f"\n  trajectory step20 gradient: "
            f"{_format_metrics(trajectory_result.gradient_metrics)}"
            f"\n  trajectory step20 parameters: "
            f"{_format_metrics(trajectory_parameter_metrics)}"
            f"\n  synchronized parameters: "
            f"{_format_metrics(synchronized_parameter_metrics)}"
            f"\n  same-parameter loss CP1/CP2/relative: "
            f"{same_parameter_result.reference_loss:.9f}/"
            f"{same_parameter_result.actual_loss:.9f}/"
            f"{same_parameter_result.loss_relative_difference:.6e}"
            f"\n  same-parameter gradient: "
            f"{_format_metrics(same_parameter_result.gradient_metrics)}"
            "\n  communication: CP1 A2A/Ring=0/0; CP2 cp2hp/hp2cp/Ring=108/18/6"
        )

    if synchronized_parameter_metrics.relative_l2 != 0.0:
        raise AssertionError(
            "CP2 parameters were not exactly synchronized to the CP1 step-20 point: "
            f"{_format_metrics(synchronized_parameter_metrics)}"
        )


@pytest.mark.skipif(
    os.getenv("STAGE34_RUN_REALISTIC_MULTIBATCH") != "1",
    reason="set STAGE34_RUN_REALISTIC_MULTIBATCH=1 for this 20-step diagnostic",
)
def test_twenty_step_realistic_multibatch_fp32_master_sgd(runtime):
    """Measure CP1/CP2 drift with varied batches and FP32 master weights."""
    if _LEARNING_RATE <= 0.0:
        raise AssertionError("STAGE34_TRAINING_DIAGNOSTIC_LR must be positive")

    reference, actual = _make_model_pair(runtime, seed=358001)
    reference_optimizer = _FP32MasterSGD(reference, learning_rate=_LEARNING_RATE)
    actual_optimizer = _FP32MasterSGD(actual, learning_rate=_LEARNING_RATE)
    loss_relatives = []
    gradient_ratios = []
    update_ratios = []
    checkpoint_metrics = {}
    reference_loss_sum = 0.0
    actual_loss_sum = 0.0

    for step in range(1, _STEPS + 1):
        batch = _multibatch_tokens(runtime.device, step=step)
        result = _forward_backward_pair(
            runtime,
            reference,
            actual,
            batch,
            step=step,
        )
        reference_update_norm = reference_optimizer.step()
        actual_update_norm = actual_optimizer.step()
        update_ratio = actual_update_norm / max(reference_update_norm, 1e-24)
        _assert_finite(reference_update_norm, label="CP1 FP32-master update norm", step=step)
        _assert_finite(actual_update_norm, label="CP2 FP32-master update norm", step=step)
        _assert_finite(update_ratio, label="FP32-master update norm ratio", step=step)

        reference_loss_sum += result.reference_loss
        actual_loss_sum += result.actual_loss
        loss_relatives.append(result.loss_relative_difference)
        gradient_ratios.append(result.gradient_metrics.norm_ratio)
        update_ratios.append(update_ratio)

        parameter_metrics = None
        master_metrics = None
        if step in _CHECKPOINT_STEPS:
            parameter_metrics = gradient_map_diagnostics(
                _tensor_map(reference, gradients=False),
                _tensor_map(actual, gradients=False),
            ).aggregate
            master_metrics = gradient_map_diagnostics(
                reference_optimizer.state_map(),
                actual_optimizer.state_map(),
            ).aggregate
            _assert_metrics_finite(
                parameter_metrics,
                label="BF16 parameters",
                step=step,
            )
            _assert_metrics_finite(
                master_metrics,
                label="FP32 master parameters",
                step=step,
            )
            checkpoint_metrics[step] = (parameter_metrics, master_metrics, result.gradient_metrics)

        if runtime.rank == 0:
            print(
                f"REALISTIC-MULTIBATCH step={step:02d}"
                f"\n  loss CP1/CP2/relative: {result.reference_loss:.9f}/"
                f"{result.actual_loss:.9f}/{result.loss_relative_difference:.6e}"
                f"\n  gradient: {_format_metrics(result.gradient_metrics)}"
                f"\n  FP32-master update norm CP1/CP2/ratio: "
                f"{reference_update_norm:.6e}/{actual_update_norm:.6e}/{update_ratio:.9f}"
                "\n  communication: CP1 A2A/Ring=0/0; "
                "CP2 cp2hp/hp2cp/Ring=108/18/6"
            )
            if parameter_metrics is not None:
                print(
                    f"  checkpoint {step} BF16 parameters: "
                    f"{_format_metrics(parameter_metrics)}"
                    f"\n  checkpoint {step} FP32 masters: "
                    f"{_format_metrics(master_metrics)}"
                )

    final_parameter_metrics, final_master_metrics, final_gradient_metrics = checkpoint_metrics[_STEPS]
    mean_reference_loss = reference_loss_sum / _STEPS
    mean_actual_loss = actual_loss_sum / _STEPS
    mean_loss_relative = _relative_difference(mean_actual_loss, mean_reference_loss)
    if runtime.rank == 0:
        print(
            "REALISTIC-MULTIBATCH 20-STEP RESULTS (THRESHOLD CALIBRATION)"
            f"\n  learning-rate/steps: {_LEARNING_RATE:.6e}/{_STEPS}"
            f"\n  mean loss CP1/CP2/relative: "
            f"{mean_reference_loss:.9f}/{mean_actual_loss:.9f}/{mean_loss_relative:.6e}"
            f"\n  max per-step loss relative difference: {max(loss_relatives):.6e}"
            f"\n  gradient norm ratio min/max: "
            f"{min(gradient_ratios):.9f}/{max(gradient_ratios):.9f}"
            f"\n  update norm ratio min/max: {min(update_ratios):.9f}/{max(update_ratios):.9f}"
            f"\n  step20 BF16 parameters: {_format_metrics(final_parameter_metrics)}"
            f"\n  step20 FP32 masters: {_format_metrics(final_master_metrics)}"
            f"\n  step20 gradient: {_format_metrics(final_gradient_metrics)}"
            "\n  numerical pass thresholds intentionally deferred until this first NPU run"
        )
