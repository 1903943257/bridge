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

import pytest
import torch

from verl.models.mcore.tpr import rectangular_causal_attention
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


def _reference(query, key, value, prefix_length, scale):
    query_fp32 = query.squeeze(1).float()
    key_fp32 = key.squeeze(1).float()
    value_fp32 = value.squeeze(1).float()
    repeats = query_fp32.shape[1] // key_fp32.shape[1]
    key_fp32 = key_fp32.repeat_interleave(repeats, dim=1)
    value_fp32 = value_fp32.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query_fp32, key_fp32) * scale
    query_positions = torch.arange(query.shape[0], device=query.device) + prefix_length
    key_positions = torch.arange(key.shape[0], device=query.device)
    scores.masked_fill_(key_positions[None, None, :] > query_positions[None, :, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value_fp32)
    return output.reshape(query.shape[0], 1, -1)


def _right_down_mask(query_length, kv_length, device):
    prefix_length = kv_length - query_length
    query_positions = torch.arange(query_length, device=device) + prefix_length
    key_positions = torch.arange(kv_length, device=device)
    return key_positions.unsqueeze(0) > query_positions.unsqueeze(1)


def _relative_l2(actual, expected):
    return (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    )


@pytest.mark.parametrize(("prefix_length", "suffix_length"), [(0, 3), (6, 3), (32, 1)])
def test_rectangular_attention_matches_reference_forward_and_backward(prefix_length, suffix_length):
    torch.manual_seed(1234)
    device = torch.device("npu")
    dtype = torch.bfloat16
    heads = 2
    head_dim = 64
    kv_length = prefix_length + suffix_length
    scale = head_dim**-0.5

    query = torch.randn(suffix_length, 1, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(kv_length, 1, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    grad_output = torch.randn(suffix_length, 1, heads * head_dim, device=device, dtype=dtype)

    actual = rectangular_causal_attention(query, key, value, softmax_scale=scale)
    actual.backward(grad_output)
    actual_grads = tuple(tensor.grad.detach().float().clone() for tensor in (query, key, value))

    reference_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in (query, key, value))
    expected = _reference(*reference_inputs, prefix_length=prefix_length, scale=scale)
    expected.backward(grad_output.float())
    expected_grads = tuple(tensor.grad.detach().float() for tensor in reference_inputs)

    torch.testing.assert_close(actual.float(), expected, atol=5e-3, rtol=5e-3)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=5e-3, rtol=5e-3)

    if prefix_length:
        assert key.grad[:prefix_length].float().norm() > 0
        assert value.grad[:prefix_length].float().norm() > 0


def test_explicit_causal_mask_matches_sparse_mode_3_forward_and_backward():
    """Quantify CANN mode-0/mode-3 drift for identical causal semantics."""

    torch.manual_seed(2026)
    device = torch.device("npu")
    dtype = torch.bfloat16
    query_length, kv_length = 63, 127
    heads, head_dim = 4, 64
    scale = head_dim**-0.5

    sparse_inputs = (
        torch.randn(query_length, 1, heads, head_dim, device=device, dtype=dtype, requires_grad=True),
        torch.randn(kv_length, 1, heads, head_dim, device=device, dtype=dtype, requires_grad=True),
        torch.randn(kv_length, 1, heads, head_dim, device=device, dtype=dtype, requires_grad=True),
    )
    masked_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in sparse_inputs)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in sparse_inputs)
    grad_output = torch.randn(query_length, 1, heads * head_dim, device=device, dtype=dtype)

    sparse_output = rectangular_causal_attention(*sparse_inputs, softmax_scale=scale)
    masked_output = rectangular_causal_attention(
        *masked_inputs,
        softmax_scale=scale,
        attention_mask=_right_down_mask(query_length, kv_length, device),
    )
    reference_output = _reference(
        *reference_inputs,
        prefix_length=kv_length - query_length,
        scale=scale,
    )
    sparse_output.backward(grad_output)
    masked_output.backward(grad_output)
    reference_output.backward(grad_output.float())

    output_relative_l2 = _relative_l2(masked_output, sparse_output)
    gradient_relative_l2 = tuple(
        _relative_l2(masked.grad, sparse.grad)
        for masked, sparse in zip(masked_inputs, sparse_inputs, strict=True)
    )
    print(
        "\nCANN explicit-mask mode 0 vs sparse mode 3: "
        f"output_relative_l2={output_relative_l2.item():.6e}, "
        "gradient_relative_l2="
        f"{tuple(value.item() for value in gradient_relative_l2)}"
    )

    # Both kernels must implement the same mathematical attention.  Their
    # direct BF16 difference is diagnostic only: it is precisely the quantity
    # that a deep model can accumulate even when both match the FP32 oracle.
    for output in (sparse_output, masked_output):
        torch.testing.assert_close(output.float(), reference_output, atol=6e-3, rtol=1e-2)
    for inputs in (sparse_inputs, masked_inputs):
        for actual, expected in zip(inputs, reference_inputs, strict=True):
            torch.testing.assert_close(
                actual.grad.float(),
                expected.grad.float(),
                atol=6e-3,
                rtol=2e-2,
            )


def test_physical_padding_mask_matches_logical_custom_mask_forward_and_backward():
    """Measure the numerical effect of one masked physical Q/KV row."""

    torch.manual_seed(2027)
    device = torch.device("npu")
    dtype = torch.bfloat16
    logical_query_length, logical_kv_length = 63, 127
    query_heads, kv_heads, head_dim = 4, 2, 64
    scale = head_dim**-0.5

    logical_inputs = (
        torch.randn(
            logical_query_length,
            1,
            query_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            logical_kv_length,
            1,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
        torch.randn(
            logical_kv_length,
            1,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        ),
    )
    physical_inputs = tuple(
        torch.cat((tensor.detach(), torch.randn_like(tensor[:1])), dim=0).requires_grad_(True)
        for tensor in logical_inputs
    )
    invalid_row_inputs = tuple(
        tensor.detach().clone().requires_grad_(True) for tensor in physical_inputs
    )
    logical_mask = _right_down_mask(logical_query_length, logical_kv_length, device)
    query_positions = torch.arange(64, 128, device=device).clamp_max(126)
    key_positions = torch.arange(128, device=device).clamp_max(126)
    key_valid = torch.arange(128, device=device) < logical_kv_length
    physical_mask = torch.logical_or(
        ~key_valid.unsqueeze(0),
        key_positions.unsqueeze(0) > query_positions.unsqueeze(1),
    )
    invalid_row_mask = torch.logical_or(
        physical_mask,
        (torch.arange(64, device=device) >= logical_query_length).unsqueeze(1),
    )
    grad_output = torch.randn(
        logical_query_length,
        1,
        query_heads * head_dim,
        device=device,
        dtype=dtype,
    )
    physical_grad_output = torch.cat((grad_output, torch.zeros_like(grad_output[:1])), dim=0)

    logical_output = rectangular_causal_attention(
        *logical_inputs,
        softmax_scale=scale,
        attention_mask=logical_mask,
    )
    physical_output = rectangular_causal_attention(
        *physical_inputs,
        softmax_scale=scale,
        attention_mask=physical_mask,
    )
    invalid_row_output = rectangular_causal_attention(
        *invalid_row_inputs,
        softmax_scale=scale,
        attention_mask=invalid_row_mask,
        inner_precise=2,
    )
    logical_output.backward(grad_output)
    physical_output.backward(physical_grad_output)
    invalid_row_output.backward(physical_grad_output)

    output_relative_l2 = _relative_l2(
        physical_output[:logical_query_length],
        logical_output,
    )
    gradient_relative_l2 = tuple(
        _relative_l2(physical.grad[: logical.shape[0]], logical.grad)
        for physical, logical in zip(physical_inputs, logical_inputs, strict=True)
    )
    invalid_row_output_relative_l2 = _relative_l2(
        invalid_row_output[:logical_query_length],
        logical_output,
    )
    invalid_row_gradient_relative_l2 = tuple(
        _relative_l2(physical.grad[: logical.shape[0]], logical.grad)
        for physical, logical in zip(invalid_row_inputs, logical_inputs, strict=True)
    )
    print(
        "\nCANN physical padding vs logical custom mask: "
        f"output_relative_l2={output_relative_l2.item():.6e}, "
        "gradient_relative_l2="
        f"{tuple(value.item() for value in gradient_relative_l2)}\n"
        "CANN fully masked padding row with inner_precise=2: "
        f"output_relative_l2={invalid_row_output_relative_l2.item():.6e}, "
        "gradient_relative_l2="
        f"{tuple(value.item() for value in invalid_row_gradient_relative_l2)}"
    )

    torch.testing.assert_close(
        physical_output[:logical_query_length].float(),
        logical_output.float(),
        atol=6e-3,
        rtol=1e-2,
    )
    for inputs in (physical_inputs, invalid_row_inputs):
        for physical, logical in zip(inputs, logical_inputs, strict=True):
            torch.testing.assert_close(
                physical.grad[: logical.shape[0]].float(),
                logical.grad.float(),
                atol=6e-3,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                physical.grad[logical.shape[0] :],
                torch.zeros_like(physical.grad[logical.shape[0] :]),
            )
    torch.testing.assert_close(
        invalid_row_output[logical_query_length:],
        torch.zeros_like(invalid_row_output[logical_query_length:]),
    )
