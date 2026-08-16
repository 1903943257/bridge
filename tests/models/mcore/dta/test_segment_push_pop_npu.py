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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel import layers as tensor_parallel_layers
from megatron.core.tensor_parallel import mappings as tensor_parallel_mappings
from megatron.core.transformer.transformer_config import TransformerConfig
from verl.models.mcore.dta import (
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
    replace_self_attention_with_dta,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

_PREFIX_LENGTH = 1024
_SUFFIX_LENGTH = 512
_FULL_LENGTH = _PREFIX_LENGTH + _SUFFIX_LENGTH
_VOCAB_SIZE = 2048
_GRAD_RELATIVE_L2_TOL = 2e-2


class _SingleProcessGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


class _NpuGlobalMemoryBuffer:
    def __init__(self, device):
        self.device = device
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, mem_alloc_context=None):
        del mem_alloc_context
        required_length = 1
        for dimension in tensor_shape:
            required_length *= dimension
        key = (name, dtype)
        if key not in self.buffer or self.buffer[key].numel() < required_length:
            self.buffer[key] = torch.empty(required_length, dtype=dtype, device=self.device)
        return self.buffer[key][:required_length].view(*tensor_shape)


class _ZeroDropoutRngTracker:
    def fork(self, *args, **kwargs):
        del args, kwargs
        return nullcontext()


def _install_single_rank_runtime(monkeypatch, device):
    parallel_state._GLOBAL_MEMORY_BUFFER = _NpuGlobalMemoryBuffer(device)
    monkeypatch.setattr(tensor_parallel, "get_cuda_rng_tracker", lambda: _ZeroDropoutRngTracker())
    original_reduce = tensor_parallel_mappings._reduce
    monkeypatch.setattr(
        tensor_parallel_mappings,
        "_reduce",
        lambda tensor, group: tensor if group is None else original_reduce(tensor, group),
    )
    original_gather = tensor_parallel_layers.gather_from_tensor_model_parallel_region
    monkeypatch.setattr(
        tensor_parallel_layers,
        "gather_from_tensor_model_parallel_region",
        lambda tensor, group=None: (
            tensor if group is None or group.size() == 1 else original_gather(tensor, group)
        ),
    )


def _make_model(device):
    dtype = torch.bfloat16
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=True,
        sequence_parallel=False,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )
    process_group = _SingleProcessGroup()
    pg_collection = SimpleNamespace(tp=process_group, cp=process_group, pp=process_group, embd=None)
    spec = replace_self_attention_with_dta(
        get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_FULL_LENGTH,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=pg_collection,
    ).to(device=device, dtype=dtype)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = process_group
    model.train()
    return model


def _plan(token_ids):
    all_tokens = token_ids.cpu()
    prefix_tokens = all_tokens[:_PREFIX_LENGTH]
    suffix_tokens = all_tokens[_PREFIX_LENGTH:]
    prefix_terms = tuple(
        SegmentLossTerm(query_offset=index, target_token_id=int(all_tokens[index + 1]))
        for index in range(_PREFIX_LENGTH)
    )
    suffix_terms = tuple(
        SegmentLossTerm(query_offset=index, target_token_id=int(all_tokens[_PREFIX_LENGTH + index + 1]))
        for index in range(_SUFFIX_LENGTH - 1)
    )
    return SegmentPlan(
        [
            SegmentSpec(0, None, prefix_tokens, 0, 0, prefix_terms),
            SegmentSpec(1, 0, suffix_tokens, _PREFIX_LENGTH, _PREFIX_LENGTH, suffix_terms),
        ],
        root_id=0,
    )


def _parameter_grads(model):
    result = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all().item(), f"non-finite gradient: {name}"
        result[name] = parameter.grad.detach().float().clone()
    return result


def _assert_gradients_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        difference = actual[name] - expected[name]
        relative_l2 = (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected[name]).clamp_min(1e-12)
        ).item()
        assert relative_l2 <= _GRAD_RELATIVE_L2_TOL, f"{name}: relative_l2={relative_l2:.6g}"


def test_segment_push_pop_matches_full_causal_backward(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _make_model(device)
    token_ids = torch.arange(17, 17 + _FULL_LENGTH, device=device) % _VOCAB_SIZE
    position_ids = torch.arange(_FULL_LENGTH, device=device).unsqueeze(0)
    causal_mask = torch.triu(
        torch.ones((1, 1, _FULL_LENGTH, _FULL_LENGTH), dtype=torch.bool, device=device),
        diagonal=1,
    )

    reference_logits = model(token_ids.unsqueeze(0), position_ids, causal_mask)
    reference_loss = F.cross_entropy(
        reference_logits[:, :-1, :].float().reshape(-1, _VOCAB_SIZE),
        token_ids[1:].reshape(-1),
        reduction="mean",
    )
    reference_loss.backward()
    reference_gradients = _parameter_grads(model)

    model.zero_grad(set_to_none=True)
    executor = SegmentExecutor(model, _plan(token_ids), expected_layer_numbers=(1, 2))
    executor.push(0)
    executor.push(1)
    suffix_result = executor.pop(1)
    prefix_entry = executor.kv_stack.top()
    assert set(prefix_entry.gradients) == {1, 2}
    assert all(torch.count_nonzero(grad).item() > 0 for pair in prefix_entry.gradients.values() for grad in pair)
    prefix_result = executor.pop(0)
    executor.kv_stack.assert_empty()
    dta_gradients = _parameter_grads(model)
    dta_loss = suffix_result.normalized_loss + prefix_result.normalized_loss

    torch.testing.assert_close(dta_loss.float(), reference_loss.detach().float(), atol=2e-2, rtol=2e-2)
    _assert_gradients_close(dta_gradients, reference_gradients)


