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

from verl.models.mcore.dta import rectangular_causal_attention
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


def _reference(query, key, value, prefix_length, scale):
    query_fp32 = query.squeeze(1).float()
    key_fp32 = key.squeeze(1).float()
    value_fp32 = value.squeeze(1).float()
    scores = torch.einsum("qhd,khd->hqk", query_fp32, key_fp32) * scale
    query_positions = torch.arange(query.shape[0], device=query.device) + prefix_length
    key_positions = torch.arange(key.shape[0], device=query.device)
    scores.masked_fill_(key_positions[None, None, :] > query_positions[None, :, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probabilities, value_fp32)
    return output.reshape(query.shape[0], 1, -1)


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
