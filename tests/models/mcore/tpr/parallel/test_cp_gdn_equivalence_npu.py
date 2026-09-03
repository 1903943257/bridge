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

"""Native MindSpeed GDN equivalence between CP=1 and CP=2.

Run from the verl repository root with two visible NPUs::

    torchrun --standalone --nproc_per_node=2 -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_cp_gdn_equivalence_npu.py

The test deliberately stays below GPTModel and MegatronEngine.  It isolates the
native GDN CP path, proves that both A2A directions execute, and compares the
global output-space objective and its complete gradients.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from verl.utils.device import is_torch_npu_available

from ._cp_gdn_test_utils import (
    AllToAllProbe,
    TensorComparison,
    assert_named_tensors_finite,
    broadcast_module_state,
    clone_parameter_gradients,
    gather_native_zigzag,
    named_tensor_comparison,
    native_zigzag_shard,
    reduce_parameter_gradients,
    tensor_comparison,
)


_EXPECTED_WORLD_SIZE = 2
_DTYPE = torch.bfloat16
_OUTPUT_ATOL = 5e-3
_OUTPUT_RTOL = 5e-3
_LOSS_RELATIVE_TOL = 5e-4
_GRAD_RELATIVE_L2_TOL = 2e-2
_GRAD_COSINE_MIN = 0.999
_PER_PARAMETER_ATOL = 5e-3
_PER_PARAMETER_RTOL = 2e-2
_REF_REPEAT_GRAD_RELATIVE_L2_TOL = 5e-3


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != _EXPECTED_WORLD_SIZE,
    reason=(
        "Run with: torchrun --standalone --nproc_per_node=2 -m pytest -s -v "
        "tests/models/mcore/tpr/parallel/test_cp_gdn_equivalence_npu.py"
    ),
)


@dataclass(frozen=True)
class _DistributedGDNRuntime:
    rank: int
    device: torch.device
    tp_group: dist.ProcessGroup
    cp_group: dist.ProcessGroup
    gdn_module: ModuleType
    gated_delta_net_cls: type[nn.Module]
    gated_delta_net_submodules_cls: type
    process_group_collection_cls: type
    transformer_config_cls: type
    local_spec_provider_cls: type


@dataclass(frozen=True)
class _RunResult:
    output: Tensor
    loss: Tensor
    input_gradient: Tensor
    parameter_gradients: dict[str, Tensor]


def _initialize_runtime() -> _DistributedGDNRuntime:
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
    if dist.get_world_size() != _EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"CP/GDN equivalence requires world_size=2, got {dist.get_world_size()}"
        )

    # MindSpeed parses sys.argv when megatron_adaptor is first imported. Pytest
    # short options such as ``-s`` and ``-v`` are otherwise misparsed as a
    # dynamic TransformerConfig field named "". Pytest has already consumed
    # its CLI by fixture setup, so isolate that import from the runner's argv.
    pytest_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = pytest_argv

    # Be defensive when another collection-time import populated MindSpeed's
    # global argument cache before the isolated import above.
    from mindspeed.args_utils import get_full_args

    vars(get_full_args()).pop("", None)

    # Repatch before importing GatedDeltaNet so its FLA symbols bind to the
    # NPU kernels and the Megatron class is replaced by MindSpeed's CP version.
    repatch(
        {
            "context_parallel_size": _EXPECTED_WORLD_SIZE,
            "context_parallel_algo": "megatron_cp_algo",
            "experimental_attention_variant": "gated_delta_net",
            "use_naive_l2norm": False,
        }
    )

    from megatron.core import parallel_state
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_EXPECTED_WORLD_SIZE,
            expert_model_parallel_size=1,
        )

    # The normal Megatron training bootstrap seeds the model-parallel RNG
    # tracker after process-group initialization. This standalone module test
    # must do the same before constructing ColumnParallelLinear.
    model_parallel_cuda_manual_seed(7300)

    if GatedDeltaNet is not mindspeed_gdn.GatedDeltaNet:
        raise AssertionError(
            "Megatron GatedDeltaNet was not replaced by MindSpeed's CP implementation"
        )

    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    if tp_group.size() != 1 or cp_group.size() != _EXPECTED_WORLD_SIZE:
        raise AssertionError(
            f"unexpected process-group topology: TP={tp_group.size()}, CP={cp_group.size()}"
        )

    return _DistributedGDNRuntime(
        rank=dist.get_rank(),
        device=torch.device("npu", local_rank),
        tp_group=tp_group,
        cp_group=cp_group,
        gdn_module=mindspeed_gdn,
        gated_delta_net_cls=GatedDeltaNet,
        gated_delta_net_submodules_cls=GatedDeltaNetSubmodules,
        process_group_collection_cls=ProcessGroupCollection,
        transformer_config_cls=TransformerConfig,
        local_spec_provider_cls=LocalSpecProvider,
    )


@pytest.fixture(scope="module")
def distributed_gdn_runtime():
    runtime = _initialize_runtime()
    yield runtime

    dist.barrier(group=runtime.cp_group)
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _make_config(runtime: _DistributedGDNRuntime, *, cp_size: int):
    return runtime.transformer_config_cls(
        num_layers=1,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_query_groups=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        activation_func=F.silu,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        use_cpu_initialization=False,
        params_dtype=_DTYPE,
        pipeline_dtype=_DTYPE,
        autocast_dtype=_DTYPE,
        bf16=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        deterministic_mode=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="local",
    )


def _make_gdn(runtime: _DistributedGDNRuntime, *, cp_size: int) -> nn.Module:
    config = _make_config(runtime, cp_size=cp_size)
    backend = runtime.local_spec_provider_cls()
    submodules = runtime.gated_delta_net_submodules_cls(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )
    process_groups = runtime.process_group_collection_cls()
    process_groups.tp = runtime.tp_group
    # With TP=1, the native TP group is the singleton group for this
    # process. It provides a real CP=1 reference inside the CP=2 job.
    process_groups.cp = runtime.tp_group if cp_size == 1 else runtime.cp_group
    model = runtime.gated_delta_net_cls(
        config,
        submodules=submodules,
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=0.1,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=process_groups,
    )
    model = model.to(device=runtime.device, dtype=_DTYPE)
    model.train()
    if model.cp_size != cp_size or model.tp_size != 1:
        raise AssertionError(
            f"GDN topology mismatch: expected TP=1/CP={cp_size}, "
            f"got TP={model.tp_size}/CP={model.cp_size}"
        )
    return model


def _make_full_inputs(
    runtime: _DistributedGDNRuntime, sequence_length: int
) -> tuple[Tensor, Tensor]:
    torch.manual_seed(9100 + sequence_length)
    hidden = torch.randn(
        sequence_length,
        1,
        256,
        device=runtime.device,
        dtype=_DTYPE,
    )
    target = torch.randn(
        sequence_length,
        1,
        256,
        device=runtime.device,
        dtype=_DTYPE,
    )
    dist.broadcast(hidden, src=0)
    dist.broadcast(target, src=0)
    return hidden, target


def _output_loss(output: Tensor, target: Tensor, *, global_numel: int) -> Tensor:
    difference = output.float() - target.float()
    return difference.square().sum() / global_numel


def _run_reference(model: nn.Module, hidden: Tensor, target: Tensor) -> _RunResult:
    model.zero_grad(set_to_none=True)
    model_input = hidden.detach().clone().requires_grad_(True)
    output, output_bias = model(model_input, attention_mask=None)
    if output_bias is not None:
        raise AssertionError("bias-free GDN unexpectedly returned an output bias")
    loss = _output_loss(output, target, global_numel=target.numel())
    loss.backward()
    return _RunResult(
        output=output.detach().clone(),
        loss=loss.detach().clone(),
        input_gradient=model_input.grad.detach().clone(),
        parameter_gradients=clone_parameter_gradients(model),
    )


def _run_cp_target(
    runtime: _DistributedGDNRuntime,
    model: nn.Module,
    hidden: Tensor,
    target: Tensor,
) -> _RunResult:
    model.zero_grad(set_to_none=True)
    local_hidden = native_zigzag_shard(hidden, runtime.cp_group).detach().clone()
    local_hidden.requires_grad_(True)
    local_target = native_zigzag_shard(target, runtime.cp_group)

    local_output, output_bias = model(local_hidden, attention_mask=None)
    if output_bias is not None:
        raise AssertionError("bias-free GDN unexpectedly returned an output bias")
    local_loss = _output_loss(local_output, local_target, global_numel=target.numel())
    local_loss.backward()

    # Every CP rank owns a disjoint token/head contribution. SUM reconstructs
    # the gradient of the globally normalized objective.
    reduce_parameter_gradients(model, runtime.cp_group)
    global_loss = local_loss.detach().clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)

    return _RunResult(
        output=gather_native_zigzag(local_output.detach(), runtime.cp_group),
        loss=global_loss,
        input_gradient=gather_native_zigzag(local_hidden.grad.detach(), runtime.cp_group),
        parameter_gradients=clone_parameter_gradients(model),
    )


def _assert_run_finite(result: _RunResult, *, label: str) -> None:
    tensors = {
        "output": result.output,
        "loss": result.loss,
        "input_gradient": result.input_gradient,
        **{f"parameter_gradient.{name}": value for name, value in result.parameter_gradients.items()},
    }
    assert_named_tensors_finite(tensors, label=label)


def _assert_a2a_contract(runtime: _DistributedGDNRuntime, probe: AllToAllProbe) -> tuple[int, int]:
    cp2hp_count = probe.count("cp2hp")
    hp2cp_count = probe.count("hp2cp")
    if cp2hp_count <= 0 or hp2cp_count <= 0:
        raise AssertionError(
            f"CP=2 must execute both A2A directions, got cp2hp={cp2hp_count}, "
            f"hp2cp={hp2cp_count}"
        )

    for call in probe.calls:
        if call.device.type != "npu" or call.dtype != _DTYPE:
            raise AssertionError(
                f"unexpected A2A tensor placement: direction={call.direction}, "
                f"device={call.device}, dtype={call.dtype}"
            )
        input_sequence, _, input_hidden = call.input_shape
        output_sequence, _, output_hidden = call.output_shape
        if call.direction == "cp2hp":
            expected = (input_sequence * _EXPECTED_WORLD_SIZE, input_hidden // _EXPECTED_WORLD_SIZE)
        else:
            expected = (input_sequence // _EXPECTED_WORLD_SIZE, input_hidden * _EXPECTED_WORLD_SIZE)
        if (output_sequence, output_hidden) != expected:
            raise AssertionError(
                f"invalid {call.direction} shape transform: "
                f"input={call.input_shape}, output={call.output_shape}, expected={expected}"
            )

    counts = torch.tensor([cp2hp_count, hp2cp_count], device=runtime.device, dtype=torch.int64)
    gathered_counts = [torch.empty_like(counts) for _ in range(_EXPECTED_WORLD_SIZE)]
    dist.all_gather(gathered_counts, counts, group=runtime.cp_group)
    if any(not torch.equal(rank_counts, gathered_counts[0]) for rank_counts in gathered_counts[1:]):
        raise AssertionError(
            f"A2A call order/count differs across CP ranks: "
            f"{[rank_counts.tolist() for rank_counts in gathered_counts]}"
        )
    return cp2hp_count, hp2cp_count


def _assert_equivalent(
    reference: _RunResult,
    actual: _RunResult,
) -> tuple[TensorComparison, TensorComparison, TensorComparison, tuple[str, TensorComparison]]:
    torch.testing.assert_close(actual.output, reference.output, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
    torch.testing.assert_close(
        actual.input_gradient,
        reference.input_gradient,
        atol=_OUTPUT_ATOL,
        rtol=_OUTPUT_RTOL,
    )

    output_metrics = tensor_comparison(reference.output, actual.output)
    input_gradient_metrics = tensor_comparison(reference.input_gradient, actual.input_gradient)
    loss_relative = abs(float(actual.loss.item()) - float(reference.loss.item())) / max(
        abs(float(reference.loss.item())), 1e-12
    )
    if loss_relative > _LOSS_RELATIVE_TOL:
        raise AssertionError(
            f"loss relative difference {loss_relative:.6e} exceeds {_LOSS_RELATIVE_TOL:.6e}"
        )

    gradient_metrics, worst_gradient = named_tensor_comparison(
        reference.parameter_gradients, actual.parameter_gradients
    )
    for name in sorted(reference.parameter_gradients):
        torch.testing.assert_close(
            actual.parameter_gradients[name],
            reference.parameter_gradients[name],
            atol=_PER_PARAMETER_ATOL,
            rtol=_PER_PARAMETER_RTOL,
            msg=lambda message, name=name: f"parameter gradient mismatch for {name}: {message}",
        )

    if input_gradient_metrics.relative_l2 > _GRAD_RELATIVE_L2_TOL:
        raise AssertionError(
            f"input-gradient relative L2 {input_gradient_metrics.relative_l2:.6e} "
            f"exceeds {_GRAD_RELATIVE_L2_TOL:.6e}"
        )
    if gradient_metrics.relative_l2 > _GRAD_RELATIVE_L2_TOL:
        raise AssertionError(
            f"parameter-gradient relative L2 {gradient_metrics.relative_l2:.6e} "
            f"exceeds {_GRAD_RELATIVE_L2_TOL:.6e}"
        )
    if input_gradient_metrics.cosine < _GRAD_COSINE_MIN:
        raise AssertionError(
            f"input-gradient cosine {input_gradient_metrics.cosine:.9f} "
            f"is below {_GRAD_COSINE_MIN:.9f}"
        )
    if gradient_metrics.cosine < _GRAD_COSINE_MIN:
        raise AssertionError(
            f"parameter-gradient cosine {gradient_metrics.cosine:.9f} "
            f"is below {_GRAD_COSINE_MIN:.9f}"
        )

    return output_metrics, input_gradient_metrics, gradient_metrics, worst_gradient


@pytest.mark.parametrize("sequence_length", [128, 1024, 4096])
def test_native_mindspeed_gdn_cp2_matches_cp1(
    distributed_gdn_runtime: _DistributedGDNRuntime,
    sequence_length: int,
):
    runtime = distributed_gdn_runtime
    torch.manual_seed(7300)
    reference_model = _make_gdn(runtime, cp_size=1)
    broadcast_module_state(reference_model)

    torch.manual_seed(7400)
    cp_model = _make_gdn(runtime, cp_size=_EXPECTED_WORLD_SIZE)
    cp_model.load_state_dict(reference_model.state_dict(), strict=True)
    cp_state = cp_model.state_dict()
    for name, reference_tensor in reference_model.state_dict().items():
        actual_tensor = cp_state[name]
        if reference_tensor is None or actual_tensor is None:
            if reference_tensor is not None or actual_tensor is not None:
                raise AssertionError(f"CP model non-Tensor state differs before execution: {name}")
            continue
        if not torch.equal(reference_tensor, actual_tensor):
            raise AssertionError(f"CP model state differs from reference before execution: {name}")

    kernel_module = cp_model.gated_delta_rule.__module__
    conv_module = runtime.gdn_module.causal_conv1d.__module__
    if kernel_module != "mindspeed.core.ssm.ops.flash_gated_delta_rule":
        raise AssertionError(f"GDN did not bind the expected NPU recurrent kernel: {kernel_module}")
    if conv_module != "mindspeed.core.ssm.ops.npu_causal_conv1d":
        raise AssertionError(f"GDN did not bind the expected NPU causal-conv kernel: {conv_module}")

    hidden, target = _make_full_inputs(runtime, sequence_length)
    with AllToAllProbe(runtime.gdn_module) as probe:
        probe.clear()
        reference_a = _run_reference(reference_model, hidden, target)
        if probe.calls:
            raise AssertionError(f"CP=1 reference unexpectedly executed {len(probe.calls)} A2A calls")

        reference_b = _run_reference(reference_model, hidden, target)
        if probe.calls:
            raise AssertionError(f"CP=1 repeat unexpectedly executed {len(probe.calls)} A2A calls")

        torch.testing.assert_close(
            reference_b.output,
            reference_a.output,
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        torch.testing.assert_close(
            reference_b.input_gradient,
            reference_a.input_gradient,
            atol=_OUTPUT_ATOL,
            rtol=_OUTPUT_RTOL,
        )
        reference_repeat_loss = abs(
            float(reference_b.loss.item()) - float(reference_a.loss.item())
        ) / max(abs(float(reference_a.loss.item())), 1e-12)
        if reference_repeat_loss > _LOSS_RELATIVE_TOL:
            raise AssertionError(
                f"CP=1 loss repeatability noise {reference_repeat_loss:.6e} "
                f"exceeds {_LOSS_RELATIVE_TOL:.6e}"
            )
        reference_repeat_grad, reference_repeat_worst = named_tensor_comparison(
            reference_a.parameter_gradients,
            reference_b.parameter_gradients,
        )
        if reference_repeat_grad.relative_l2 > _REF_REPEAT_GRAD_RELATIVE_L2_TOL:
            raise AssertionError(
                f"CP=1 gradient repeatability noise {reference_repeat_grad.relative_l2:.6e} "
                f"exceeds {_REF_REPEAT_GRAD_RELATIVE_L2_TOL:.6e}; "
                f"worst tensor={reference_repeat_worst[0]}"
            )

        probe.clear()
        cp_result = _run_cp_target(runtime, cp_model, hidden, target)
        cp2hp_count, hp2cp_count = _assert_a2a_contract(runtime, probe)

    _assert_run_finite(reference_a, label="CP=1 reference")
    _assert_run_finite(reference_b, label="CP=1 repeat")
    _assert_run_finite(cp_result, label="CP=2 target")
    output_metrics, input_grad_metrics, parameter_grad_metrics, worst_gradient = _assert_equivalent(
        reference_a, cp_result
    )

    loss_relative = abs(float(cp_result.loss.item()) - float(reference_a.loss.item())) / max(
        abs(float(reference_a.loss.item())), 1e-12
    )
    if runtime.rank == 0:
        print(
            "\nNative MindSpeed CP/GDN equivalence\n"
            f"  sequence length:              {sequence_length}\n"
            f"  GDN implementation:           {type(cp_model).__module__}.{type(cp_model).__name__}\n"
            f"  recurrent kernel:             {kernel_module}\n"
            f"  causal-conv kernel:           {conv_module}\n"
            f"  CP=2 A2A calls:               cp2hp={cp2hp_count}, hp2cp={hp2cp_count}\n"
            f"  output rel L2/cos/max abs:    {output_metrics.relative_l2:.6e} / "
            f"{output_metrics.cosine:.9f} / {output_metrics.max_abs_diff:.6e}\n"
            f"  loss ref/cp/relative diff:    {reference_a.loss.item():.9f} / "
            f"{cp_result.loss.item():.9f} / {loss_relative:.6e}\n"
            f"  input-grad rel L2/cos/max:    {input_grad_metrics.relative_l2:.6e} / "
            f"{input_grad_metrics.cosine:.9f} / {input_grad_metrics.max_abs_diff:.6e}\n"
            f"  parameter-grad rel/cos/max:   {parameter_grad_metrics.relative_l2:.6e} / "
            f"{parameter_grad_metrics.cosine:.9f} / {parameter_grad_metrics.max_abs_diff:.6e}\n"
            f"  worst parameter gradient:     {worst_gradient[0]} "
            f"({worst_gradient[1].relative_l2:.6e})\n"
            f"  CP=1 repeat gradient rel L2:  {reference_repeat_grad.relative_l2:.6e}"
        )

    dist.barrier(group=runtime.cp_group)
