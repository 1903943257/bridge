"""Punctures 1 and 2: CP=1 THD GDN and Stage 1 state primitive regression.

Run with one NPU from the verl repository root::

    torchrun --standalone --nproc_per_node=1 -m pytest -s -v \
      tests/models/mcore/baseline/test_gdn_thd_and_conv_state_npu.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from ._qwen35_baseline_utils import (
    DTYPE,
    STAGE1_CAUSAL_CONV1D_BWD_IMPL,
    STAGE1_CAUSAL_CONV1D_FWD_IMPL,
    assert_gradient_maps_close,
    bind_stage1_gdn_primitives,
    clone_parameter_gradients,
    destroy_npu_runtime,
    initialize_npu_runtime,
    process_groups,
)
from verl.utils.device import is_torch_npu_available


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    int(os.getenv("WORLD_SIZE", "1")) != 1,
    reason="run punctures 1/2 with torchrun --nproc_per_node=1",
)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=1)
    yield value
    destroy_npu_runtime(value)


def _make_gdn(runtime):
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    if GatedDeltaNet is not mindspeed_gdn.GatedDeltaNet:
        raise AssertionError(
            f"MindSpeed GDN patch is inactive: bound={GatedDeltaNet.__module__}.{GatedDeltaNet.__name__}"
        )
    config = TransformerConfig(
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
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        autocast_dtype=DTYPE,
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
    backend = LocalSpecProvider()
    module = GatedDeltaNet(
        config,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_linear(),
            out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        layer_number=1,
        bias=False,
        conv_bias=False,
        conv_init=0.1,
        use_qk_l2norm=True,
        A_init_range=(1, 16),
        pg_collection=process_groups(runtime, cp_size=1),
    ).to(device=runtime.device, dtype=DTYPE)
    module.train()
    return module, mindspeed_gdn


def test_gdn_thd_cp1_uses_patched_class_and_preserves_segment_boundaries(runtime):
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(351001)
    model, gdn_module = _make_gdn(runtime)
    cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=runtime.device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=64,
        max_seqlen_kv=64,
    )
    hidden = torch.randn(128, 1, 256, device=runtime.device, dtype=DTYPE)

    with bind_stage1_gdn_primitives(gdn_module, model) as binding:
        model.zero_grad(set_to_none=True)
        packed_input = hidden.detach().clone().requires_grad_(True)
        packed_output, _ = model(packed_input, attention_mask=None, packed_seq_params=packed)
        packed_loss = packed_output.float().square().mean()
        packed_loss.backward()
        packed_input_grad = packed_input.grad.detach().clone()
        packed_parameter_gradients = clone_parameter_gradients(model)

        model.zero_grad(set_to_none=True)
        separate_input = hidden.detach().clone().requires_grad_(True)
        first, _ = model(separate_input[:64], attention_mask=None)
        second, _ = model(separate_input[64:], attention_mask=None)
        separate_output = torch.cat((first, second), dim=0)
        separate_output.float().square().mean().backward()
        separate_parameter_gradients = clone_parameter_gradients(model)

    torch.testing.assert_close(packed_output, separate_output, atol=7e-3, rtol=1e-2)
    torch.testing.assert_close(packed_input_grad, separate_input.grad, atol=8e-3, rtol=2e-2)
    parameter_metrics = assert_gradient_maps_close(
        packed_parameter_gradients,
        separate_parameter_gradients,
        rtol=3e-2,
        cosine_min=0.999,
    )
    print(
        "PUNCTURE-1 PASS"
        f"\n  patched GDN: {type(model).__module__}.{type(model).__name__}"
        f"\n  causal-conv backend: {binding['causal_conv']}"
        f"\n  gated-delta backend: {binding['gated_delta_rule']}"
        "\n  packed segments: [64, 64]"
        f"\n  parameter-grad rel/cos: {parameter_metrics[0]:.6e}/{parameter_metrics[1]:.9f}"
    )


def test_causal_conv_prefix_state_continuation_and_initial_state_gradient(runtime):
    # Mirror Stage 1's test_state_continuity call exactly: direct arch32
    # forward primitive, FP32, [B,T,D] input and [W,D] weight, no fused options.
    torch.manual_seed(42)
    batch, dim, width, prefix_length, suffix_length = 1, 256, 4, 64, 64
    x = torch.randn(
        batch,
        prefix_length + suffix_length,
        dim,
        device=runtime.device,
        dtype=torch.float32,
    )
    weight = torch.randn(width, dim, device=runtime.device, dtype=torch.float32)

    with torch.no_grad():
        full_output, _ = STAGE1_CAUSAL_CONV1D_FWD_IMPL(
            x=x,
            weight=weight,
            bias=None,
            residual=None,
            output_final_state=False,
        )
        prefix_output, prefix_final_state = STAGE1_CAUSAL_CONV1D_FWD_IMPL(
            x=x[:, :prefix_length],
            weight=weight,
            bias=None,
            residual=None,
            output_final_state=True,
        )
        suffix_output, _ = STAGE1_CAUSAL_CONV1D_FWD_IMPL(
            x=x[:, prefix_length:],
            weight=weight,
            bias=None,
            residual=None,
            initial_state=prefix_final_state,
            output_final_state=False,
        )
    if prefix_final_state is None or prefix_final_state.shape != (batch, dim, width):
        raise AssertionError(
            f"unexpected prefix conv state: {None if prefix_final_state is None else prefix_final_state.shape}"
        )
    torch.testing.assert_close(
        torch.cat((prefix_output, suffix_output), dim=1),
        full_output,
        atol=1e-5,
        rtol=1e-5,
    )

    # Mirror Stage 1's test_grad_with_initial_state case: direct arch32
    # forward/backward primitives, B=2, T=128, FP32, state [B,D,W].
    gradient_batch, gradient_length = 2, 128
    gradient_x = torch.randn(
        gradient_batch,
        gradient_length,
        dim,
        device=runtime.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    gradient_weight = torch.randn(
        width,
        dim,
        device=runtime.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    initial_state = torch.randn(
        gradient_batch,
        dim,
        width,
        device=runtime.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    gradient_output, _ = STAGE1_CAUSAL_CONV1D_FWD_IMPL(
        x=gradient_x,
        weight=gradient_weight,
        bias=None,
        residual=None,
        initial_state=initial_state,
        activation=None,
        output_final_state=False,
    )
    output_gradient = torch.randn_like(gradient_output)
    _, _, _, _, initial_state_gradient = STAGE1_CAUSAL_CONV1D_BWD_IMPL(
        x=gradient_x.detach(),
        dy=output_gradient,
        dht=None,
        weight=gradient_weight.detach(),
        bias=None,
        residual=None,
        initial_state=initial_state.detach(),
        activation=None,
    )
    if initial_state_gradient is None:
        raise AssertionError("MindSpeed-Ops CausalConv did not return dh0")
    if not torch.isfinite(initial_state_gradient).all().item():
        raise AssertionError("initial_conv_state gradient contains NaN/Inf")
    if torch.count_nonzero(initial_state_gradient).item() == 0:
        raise AssertionError("initial_conv_state gradient is unexpectedly all zero")
    print(
        "PUNCTURE-2 PASS"
        "\n  continuation backend: mindspeed_ops.arch32.triton.convolution.causal_conv1d_fwd_impl"
        "\n  gradient backend: mindspeed_ops.arch32.triton.convolution.causal_conv1d_bwd_impl"
        f"\n  continuation: {prefix_length}+{suffix_length} tokens"
        f"\n  initial-state grad norm: {initial_state_gradient.float().norm().item():.6e}"
    )
