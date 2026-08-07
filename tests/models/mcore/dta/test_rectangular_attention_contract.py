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

from verl.models.mcore.dta.rectangular_attention import _validate_inputs, rectangular_causal_attention


def _qkv(query_length=3, kv_length=9, query_heads=4, kv_heads=2, head_dim=8):
    query = torch.randn(query_length, 1, query_heads, head_dim)
    key = torch.randn(kv_length, 1, kv_heads, head_dim)
    value = torch.randn_like(key)
    return query, key, value


def test_contract_accepts_rectangular_gqa_shapes():
    query, key, value = _qkv()
    assert _validate_inputs(query, key, value, dropout_p=0.0) == (3, 9, 4, 8)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda q, k, v: (q[:, :, :, :-1], k, v), "head dimensions"),
        (lambda q, k, v: (q, k[:, :, :1], v), "identical shapes"),
        (lambda q, k, v: (q.repeat(1, 2, 1, 1), k, v), "batch size 1"),
        (lambda q, k, v: (q, k.to(torch.float64), v), "dtypes must match"),
    ],
)
def test_contract_rejects_incompatible_inputs(mutation, match):
    query, key, value = mutation(*_qkv())
    with pytest.raises(ValueError, match=match):
        _validate_inputs(query, key, value, dropout_p=0.0)


def test_contract_rejects_attention_dropout():
    with pytest.raises(ValueError, match="dropout_p=0"):
        _validate_inputs(*_qkv(), dropout_p=0.1)


def test_public_adapter_fails_clearly_off_npu():
    with pytest.raises(RuntimeError, match="requires an NPU tensor"):
        rectangular_causal_attention(*_qkv())
