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

"""Forward continuation of a GDN layer through explicit linear prefix state.

Run from the verl repository root on one NPU::

    pytest -s -v \
        tests/models/mcore/tpr/linear/test_linear_prefix_state_forward_npu.py

The test first proves that the state-aware test runner matches the unchanged
MindSpeed GDN forward. It then splits the same hidden sequence into prefix and
suffix segments and verifies continuation through both causal-convolution and
recurrent states.
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

from ..parallel._cp_gdn_test_utils import TensorComparison, tensor_comparison
from ._gdn_state_test_utils import LinearPrefixState, run_stateful_gdn_segment


_DTYPE = torch.bfloat16
_HIDDEN_SIZE = 256
_OUTPUT_ATOL = 5e-3
_OUTPUT_RTOL = 5e-3
_OUTPUT_RELATIVE_L2_TOL = 5e-3
_STATE_ATOL = 1e-2
_STATE_RTOL = 1e-2
_STATE_RELATIVE_L2_TOL = 1e-2
_COSINE_MIN = 0.999


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@dataclass(frozen=True)
class _GDNRuntime:
    device: torch.device
    gdn_module: ModuleType
    gated_delta_net_cls: type[nn.Module]
    gated_delta_net_submodules_cls: type
    process_group_collection_cls: type
    transformer_config_cls: type
    local_spec_provider_cls: type
    tp_group: dist.ProcessGroup
    owns_process_group: bool
    owns_model_parallel: bool


def _initialize_runtime() -> _GDNRuntime:
    import torch_npu  # noqa: F401

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29543")
        dist.init_process_group(backend="hccl", rank=0, world_size=1)
    if dist.get_world_size() != 1:
        raise RuntimeError(
            "linear prefix-state capability test requires world_size=1, "
            f"got {dist.get_world_size()}"
        )

    pytest_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = pytest_argv

    from mindspeed.args_utils import get_full_args

    vars(get_full_args()).pop("", None)
    repatch(
        {
            "context_parallel_size": 1,
            "context_parallel_algo": "megatron_cp_algo",
            "experimental_attention_variant": "gated_delta_net",
            "use_naive_l2norm": False,
        }
    )

    from megatron.core import parallel_state
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(8300)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    if tp_group.size() != 1:
        raise AssertionError(f"expected singleton TP/CP group, got size={tp_group.size()}")

    return _GDNRuntime(
        device=torch.device("npu", local_rank),
        gdn_module=mindspeed_gdn,
        gated_delta_net_cls=mindspeed_gdn.GatedDeltaNet,
        gated_delta_net_submodules_cls=GatedDeltaNetSubmodules,
        process_group_collection_cls=ProcessGroupCollection,
        transformer_config_cls=TransformerConfig,
        local_spec_provider_cls=LocalSpecProvider,
        tp_group=tp_group,
        owns_process_group=owns_process_group,
        owns_model_parallel=owns_model_parallel,
    )


@pytest.fixture(scope="module")
def gdn_runtime():
    runtime = _initialize_runtime()
    yield runtime

    torch.npu.synchronize()
    if runtime.owns_model_parallel:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
    if runtime.owns_process_group:
        dist.destroy_process_group()


def _make_config(runtime: _GDNRuntime):
    return runtime.transformer_config_cls(
        num_layers=1,
        hidden_size=_HIDDEN_SIZE,
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
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        deterministic_mode=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="local",
    )


def _make_gdn(runtime: _GDNRuntime) -> nn.Module:
    config = _make_config(runtime)
    backend = runtime.local_spec_provider_cls()
    submodules = runtime.gated_delta_net_submodules_cls(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )
    process_groups = runtime.process_group_collection_cls()
    process_groups.tp = runtime.tp_group
    process_groups.cp = runtime.tp_group
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
    model.eval()
    return model


def _make_hidden(runtime: _GDNRuntime, sequence_length: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(9500 + sequence_length)
    hidden = torch.randn(
        sequence_length,
        1,
        _HIDDEN_SIZE,
        dtype=torch.float32,
        generator=generator,
    )
    return hidden.to(device=runtime.device, dtype=_DTYPE)


def _run_native(model: nn.Module, hidden: Tensor) -> Tensor:
    output, output_bias = model(hidden, attention_mask=None)
    if output_bias is not None:
        raise AssertionError("bias-free native GDN unexpectedly returned output bias")
    return output


def _assert_state_contract(
    model: nn.Module,
    state: LinearPrefixState,
    *,
    expected_length: int,
    label: str,
) -> None:
    expected_conv_shape = (
        1,
        model.conv_dim_local_tp,
        model.conv_kernel_dim - 1,
    )
    expected_recurrent_shape = (
        1,
        model.num_value_heads,
        model.key_head_dim,
        model.value_head_dim,
    )
    if tuple(state.causal_conv_state.shape) != expected_conv_shape:
        raise AssertionError(
            f"{label} causal-conv state shape is {tuple(state.causal_conv_state.shape)}, "
            f"expected {expected_conv_shape}"
        )
    if tuple(state.recurrent_state.shape) != expected_recurrent_shape:
        raise AssertionError(
            f"{label} recurrent state shape is {tuple(state.recurrent_state.shape)}, "
            f"expected {expected_recurrent_shape}"
        )
    if state.prefix_length != expected_length:
        raise AssertionError(
            f"{label} prefix length is {state.prefix_length}, expected {expected_length}"
        )
    if state.causal_conv_state.device.type != "npu":
        raise AssertionError(f"{label} causal-conv state is not on NPU")
    if state.recurrent_state.device.type != "npu":
        raise AssertionError(f"{label} recurrent state is not on NPU")
    if state.causal_conv_state.dtype != _DTYPE:
        raise AssertionError(
            f"{label} causal-conv state dtype is {state.causal_conv_state.dtype}, expected {_DTYPE}"
        )
    if not state.recurrent_state.is_floating_point():
        raise AssertionError(f"{label} recurrent state must be floating point")
    if not torch.isfinite(state.causal_conv_state).all():
        raise AssertionError(f"{label} causal-conv state contains NaN or Inf")
    if not torch.isfinite(state.recurrent_state).all():
        raise AssertionError(f"{label} recurrent state contains NaN or Inf")


def _assert_output_close(reference: Tensor, actual: Tensor, *, label: str) -> TensorComparison:
    torch.testing.assert_close(actual, reference, atol=_OUTPUT_ATOL, rtol=_OUTPUT_RTOL)
    metrics = tensor_comparison(reference, actual)
    if metrics.relative_l2 > _OUTPUT_RELATIVE_L2_TOL:
        raise AssertionError(
            f"{label} relative L2 {metrics.relative_l2:.6e} exceeds "
            f"{_OUTPUT_RELATIVE_L2_TOL:.6e}"
        )
    if metrics.cosine < _COSINE_MIN:
        raise AssertionError(
            f"{label} cosine {metrics.cosine:.9f} is below {_COSINE_MIN:.9f}"
        )
    return metrics


def _assert_state_close(reference: Tensor, actual: Tensor, *, label: str) -> TensorComparison:
    torch.testing.assert_close(actual, reference, atol=_STATE_ATOL, rtol=_STATE_RTOL)
    metrics = tensor_comparison(reference, actual)
    if metrics.relative_l2 > _STATE_RELATIVE_L2_TOL:
        raise AssertionError(
            f"{label} relative L2 {metrics.relative_l2:.6e} exceeds "
            f"{_STATE_RELATIVE_L2_TOL:.6e}"
        )
    if metrics.cosine < _COSINE_MIN:
        raise AssertionError(
            f"{label} cosine {metrics.cosine:.9f} is below {_COSINE_MIN:.9f}"
        )
    return metrics


@pytest.mark.parametrize(
    ("prefix_length", "suffix_length"),
    [
        (128, 64),
        (1024, 512),
        (4096, 128),
        (127, 65),
    ],
)
def test_gdn_prefix_state_forward_matches_full_sequence(
    gdn_runtime: _GDNRuntime,
    prefix_length: int,
    suffix_length: int,
):
    runtime = gdn_runtime
    torch.manual_seed(8400)
    model = _make_gdn(runtime)
    hidden = _make_hidden(runtime, prefix_length + suffix_length)

    recurrent_backend = model.gated_delta_rule.__module__
    supported_recurrent_backends = {
        "mindspeed.core.ssm.ops.flash_gated_delta_rule",
        "mindspeed.core.ssm.chunk_gated_delta_rule",
    }
    if recurrent_backend not in supported_recurrent_backends:
        raise AssertionError(f"GDN bound an unexpected recurrent backend: {recurrent_backend}")

    with torch.no_grad():
        native_full_a = _run_native(model, hidden)
        native_full_b = _run_native(model, hidden)
        stateful_full = run_stateful_gdn_segment(model, hidden, runtime.gdn_module)

        prefix_result = run_stateful_gdn_segment(
            model,
            hidden[:prefix_length],
            runtime.gdn_module,
        )
        prefix_conv_before = prefix_result.state.causal_conv_state.clone()
        prefix_recurrent_before = prefix_result.state.recurrent_state.clone()
        suffix_result = run_stateful_gdn_segment(
            model,
            hidden[prefix_length:],
            runtime.gdn_module,
            initial_state=prefix_result.state,
        )
        relayed_output = torch.cat((prefix_result.output, suffix_result.output), dim=0)

    if stateful_full.output_bias is not None:
        raise AssertionError("bias-free stateful full GDN unexpectedly returned output bias")
    if prefix_result.output_bias is not None or suffix_result.output_bias is not None:
        raise AssertionError("bias-free segmented GDN unexpectedly returned output bias")
    if prefix_result.causal_conv_backend != suffix_result.causal_conv_backend:
        raise AssertionError(
            "causal-conv backend changed between prefix and suffix: "
            f"{prefix_result.causal_conv_backend} vs {suffix_result.causal_conv_backend}"
        )

    _assert_state_contract(
        model,
        prefix_result.state,
        expected_length=prefix_length,
        label="prefix",
    )
    _assert_state_contract(
        model,
        stateful_full.state,
        expected_length=prefix_length + suffix_length,
        label="stateful full",
    )
    _assert_state_contract(
        model,
        suffix_result.state,
        expected_length=prefix_length + suffix_length,
        label="relayed suffix",
    )

    expected_consumed_ptrs = (
        prefix_result.state.causal_conv_state.data_ptr(),
        prefix_result.state.recurrent_state.data_ptr(),
    )
    if suffix_result.consumed_state_ptrs != expected_consumed_ptrs:
        raise AssertionError(
            "suffix did not consume the exact causal-conv and recurrent state objects "
            "produced by the prefix"
        )
    if not torch.equal(prefix_result.state.causal_conv_state, prefix_conv_before):
        raise AssertionError("suffix execution mutated the reusable prefix causal-conv state")
    if not torch.equal(prefix_result.state.recurrent_state, prefix_recurrent_before):
        raise AssertionError("suffix execution mutated the reusable prefix recurrent state")

    native_repeat_metrics = _assert_output_close(
        native_full_a,
        native_full_b,
        label="native full repeat",
    )
    stateful_full_metrics = _assert_output_close(
        native_full_a,
        stateful_full.output,
        label="stateful full versus native full",
    )
    relay_metrics = _assert_output_close(
        native_full_a,
        relayed_output,
        label="prefix/suffix relay versus native full",
    )
    suffix_metrics = _assert_output_close(
        native_full_a[prefix_length:],
        suffix_result.output,
        label="relayed suffix versus native suffix",
    )
    conv_state_metrics = _assert_state_close(
        stateful_full.state.causal_conv_state,
        suffix_result.state.causal_conv_state,
        label="final causal-conv state",
    )
    recurrent_state_metrics = _assert_state_close(
        stateful_full.state.recurrent_state,
        suffix_result.state.recurrent_state,
        label="final recurrent state",
    )

    print(
        "\nLinear GDN prefix-state forward continuation\n"
        f"  P/S:                          {prefix_length}/{suffix_length}\n"
        f"  GDN implementation:           {type(model).__module__}.{type(model).__name__}\n"
        f"  recurrent backend:            {recurrent_backend}\n"
        f"  causal-conv backend:          {prefix_result.causal_conv_backend}\n"
        f"  native repeat rel/cos/max:    {native_repeat_metrics.relative_l2:.6e} / "
        f"{native_repeat_metrics.cosine:.9f} / {native_repeat_metrics.max_abs_diff:.6e}\n"
        f"  stateful/native rel/cos/max:  {stateful_full_metrics.relative_l2:.6e} / "
        f"{stateful_full_metrics.cosine:.9f} / {stateful_full_metrics.max_abs_diff:.6e}\n"
        f"  relay/full rel/cos/max:       {relay_metrics.relative_l2:.6e} / "
        f"{relay_metrics.cosine:.9f} / {relay_metrics.max_abs_diff:.6e}\n"
        f"  suffix rel/cos/max:           {suffix_metrics.relative_l2:.6e} / "
        f"{suffix_metrics.cosine:.9f} / {suffix_metrics.max_abs_diff:.6e}\n"
        f"  final conv-state rel/cos/max: {conv_state_metrics.relative_l2:.6e} / "
        f"{conv_state_metrics.cosine:.9f} / {conv_state_metrics.max_abs_diff:.6e}\n"
        f"  final recurrent rel/cos/max:  {recurrent_state_metrics.relative_l2:.6e} / "
        f"{recurrent_state_metrics.cosine:.9f} / "
        f"{recurrent_state_metrics.max_abs_diff:.6e}"
    )
