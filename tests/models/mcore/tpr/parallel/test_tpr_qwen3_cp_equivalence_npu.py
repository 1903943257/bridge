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

"""Opt-in CP=2 correctness for real Qwen3-0.6B and Qwen3-1.7B checkpoints.

Run from the verl repository root with two visible NPUs::

    TPR_RUN_QWEN_CP=1 torchrun --nproc_per_node=2 \
        --master_addr=127.0.0.1 --master_port=29523 \
        -m pytest -s -v \
        tests/models/mcore/tpr/parallel/test_tpr_qwen3_cp_equivalence_npu.py

The independent reference executes each complete trajectory separately through
AllGather CP.  The actual path enters MegatronEngine once, caches the shared
prefix, and executes the two suffixes with the selected TPR CP backend.  The
non-divisible lengths deliberately exercise physical padding and loss ownership.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, PretrainedConfig

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from verl.models.mcore.config_converter import get_hf_rope_theta, hf_to_mcore_config_dense
from verl.models.mcore.tpr import (
    PhysicalExecutionKind,
    SegmentExecutor,
    SegmentPlan,
    TPRSelfAttention,
    replace_self_attention_with_tpr,
)
from verl.utils.device import is_torch_npu_available

from ..correctness.test_tpr_qwen3_compatibility_npu import (
    _assert_real_qwen_gradients_close,
    _load_qwen3_torch_spec_weights,
    _validate_checkpoint_files,
)
from ._tpr_cp_test_utils import (
    _equivalence_tpr_plan,
    _logical_logprob_indices,
    _run_independent_cp_reference,
    cp_runtime,
)
from .test_megatron_engine_tpr_cp_entry_npu import _run_engine_tpr


if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN_CP") != "1",
    reason="Set TPR_RUN_QWEN_CP=1 for real Qwen3 CP correctness",
)

_PREFIX_LENGTH = 127
_FIRST_SUFFIX_LENGTH = 63
_SECOND_SUFFIX_LENGTH = 31
_LOSS_ATOL = 2e-2
_LOSS_RTOL = 2e-2
_LOGPROB_DIAGNOSTIC_ATOL = 5e-2
_LOGPROB_DIAGNOSTIC_RTOL = 5e-3
_LOGPROB_RELATIVE_L2_TOL = 5e-3
_LOGPROB_COSINE_MIN = 0.9999
_CP_PER_PARAMETER_GRAD_RELATIVE_L2_TOL = 7e-2


@dataclass(frozen=True, slots=True)
class _QwenModelCase:
    name: str
    path: Path
    hidden_size: int
    intermediate_size: int


@dataclass(frozen=True, slots=True)
class _QwenCPReference:
    normalized_loss: torch.Tensor
    target_logprobs: torch.Tensor
    parameter_gradients: dict[str, torch.Tensor]
    execution_trace: tuple[tuple[PhysicalExecutionKind, int], ...]


@dataclass(slots=True)
class _LoadedQwenCPCase:
    spec: _QwenModelCase
    hf_config: PretrainedConfig
    model: GPTModel
    plan: SegmentPlan
    logical_indices: dict[tuple[int, int], int]
    logical_count: int
    reference: _QwenCPReference


_QWEN_MODEL_CASES = (
    _QwenModelCase(
        "qwen3_0_6b",
        Path(
            os.getenv(
                "TPR_QWEN_0_6B_PATH",
                os.getenv("TPR_QWEN_MODEL_PATH", "/workspace/hf_models/Qwen3-0.6B"),
            )
        ),
        hidden_size=1024,
        intermediate_size=3072,
    ),
    _QwenModelCase(
        "qwen3_1_7b",
        Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B")),
        hidden_size=2048,
        intermediate_size=6144,
    ),
)


def _load_hf_config(model_case: _QwenModelCase) -> PretrainedConfig:
    _validate_checkpoint_files(model_case.path)
    config = AutoConfig.from_pretrained(
        str(model_case.path),
        trust_remote_code=True,
        local_files_only=True,
    )
    assert config.model_type == "qwen3"
    assert config.architectures == ["Qwen3ForCausalLM"]
    assert config.num_hidden_layers == 28
    assert config.hidden_size == model_case.hidden_size
    assert config.intermediate_size == model_case.intermediate_size
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.vocab_size == 151936
    assert config.tie_word_embeddings
    assert config.attention_dropout == 0.0
    assert config.rms_norm_eps == 1e-6
    assert get_hf_rope_theta(config) == 1_000_000
    return config


def _make_qwen_cp_model(
    runtime: Any,
    model_case: _QwenModelCase,
    hf_config: PretrainedConfig,
) -> GPTModel:
    config = hf_to_mcore_config_dense(
        hf_config,
        torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=runtime.cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
        use_cpu_initialization=True,
    )
    spec = replace_self_attention_with_tpr(
        get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    )
    pg_collection = SimpleNamespace(
        tp=runtime.tp_group,
        cp=runtime.cp_group,
        pp=runtime.pp_group,
        embd=None,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
        position_embedding_type="rope",
        rotary_base=get_hf_rope_theta(hf_config),
        pg_collection=pg_collection,
    ).to(device=runtime.device, dtype=torch.bfloat16)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = runtime.tp_group
    _load_qwen3_torch_spec_weights(model, hf_config, model_case.path)
    model.train()
    return model


def _assert_qwen_cp_architecture(model, runtime, model_case, hf_config):
    config = model.config
    assert config.context_parallel_size == runtime.cp_size == 2
    assert config.num_layers == hf_config.num_hidden_layers == 28
    assert config.hidden_size == hf_config.hidden_size == model_case.hidden_size
    assert config.ffn_hidden_size == hf_config.intermediate_size == model_case.intermediate_size
    assert config.num_attention_heads == hf_config.num_attention_heads == 16
    assert config.num_query_groups == hf_config.num_key_value_heads == 8
    assert config.kv_channels == hf_config.head_dim == 128
    assert config.gated_linear_unit
    assert config.normalization == "RMSNorm"
    assert len(model.decoder.layers) == hf_config.num_hidden_layers
    assert all(isinstance(layer.self_attention, TPRSelfAttention) for layer in model.decoder.layers)
    assert tuple(layer.self_attention.layer_number for layer in model.decoder.layers) == tuple(
        range(1, hf_config.num_hidden_layers + 1)
    )
    assert model.share_embeddings_and_output_weights
    assert model.output_layer.weight is None
    assert (
        model.shared_embedding_or_output_weight().data_ptr()
        == model.embedding.word_embeddings.weight.data_ptr()
    )


def _move_reference_to_cpu(result) -> _QwenCPReference:
    gradients = result.parameter_gradients
    for name in tuple(gradients):
        gradients[name] = gradients[name].cpu()
    return _QwenCPReference(
        normalized_loss=result.normalized_loss.cpu(),
        target_logprobs=result.target_logprobs.cpu(),
        parameter_gradients=gradients,
        execution_trace=result.execution_trace,
    )


def _clear_model_gradients(model):
    model.zero_grad(set_to_none=True)


def _tokens(start: int, length: int, vocab_size: int) -> torch.Tensor:
    return torch.arange(start, start + length, dtype=torch.long) % vocab_size


@pytest.fixture(scope="module", params=_QWEN_MODEL_CASES, ids=lambda case: case.name)
def real_qwen_cp_case(request, cp_runtime):
    runtime = cp_runtime
    model_case = request.param
    hf_config = _load_hf_config(model_case)
    prefix = _tokens(17, _PREFIX_LENGTH, hf_config.vocab_size)
    first = _tokens(70001, _FIRST_SUFFIX_LENGTH, hf_config.vocab_size)
    second = _tokens(130001, _SECOND_SUFFIX_LENGTH, hf_config.vocab_size)
    first_trajectory = torch.cat((prefix, first))
    second_trajectory = torch.cat((prefix, second))
    plan = _equivalence_tpr_plan(prefix, first, second)
    logical_indices, logical_count = _logical_logprob_indices(first_trajectory, second_trajectory)
    assert plan.total_loss_weight == logical_count

    reference_model = _make_qwen_cp_model(runtime, model_case, hf_config)
    _assert_qwen_cp_architecture(reference_model, runtime, model_case, hf_config)
    reference_result = _run_independent_cp_reference(
        reference_model,
        first_trajectory,
        second_trajectory,
        runtime,
        logical_indices,
        logical_count,
        expected_layer_numbers=tuple(range(1, hf_config.num_hidden_layers + 1)),
    )
    reference = _move_reference_to_cpu(reference_result)
    del reference_result
    _clear_model_gradients(reference_model)
    del reference_model
    gc.collect()
    torch.npu.empty_cache()

    actual_model = _make_qwen_cp_model(runtime, model_case, hf_config)
    _assert_qwen_cp_architecture(actual_model, runtime, model_case, hf_config)
    loaded = _LoadedQwenCPCase(
        model_case,
        hf_config,
        actual_model,
        plan,
        logical_indices,
        logical_count,
        reference,
    )
    try:
        yield loaded
    finally:
        _clear_model_gradients(actual_model)
        gc.collect()
        torch.npu.empty_cache()


@pytest.mark.parametrize("backend", ("allgather", "ulysses", "ring"))
def test_real_qwen3_engine_tpr_cp_matches_independent_cp(
    real_qwen_cp_case,
    cp_runtime,
    backend,
    monkeypatch,
):
    case = real_qwen_cp_case
    runtime = cp_runtime
    actual_logprobs = torch.zeros(
        case.logical_count,
        dtype=torch.float32,
        device=runtime.device,
    )
    owner_counts = torch.zeros(
        case.logical_count,
        dtype=torch.int64,
        device=runtime.device,
    )
    original_compute_loss = SegmentExecutor._compute_loss

    def capture_loss(executor, segment, logits):
        if executor.model is case.model:
            owned_terms = executor._owned_loss_terms(segment)
            if owned_terms:
                if any(term.sample_id is None for term in owned_terms):
                    raise AssertionError("Qwen equivalence loss terms must carry sample_id")
                shard = executor._segment_shard(segment)
                local_offsets = tuple(
                    shard.global_to_local(term.query_offset) for term in owned_terms
                )
                if any(offset is None for offset in local_offsets):
                    raise AssertionError("owned Qwen loss term is absent from its CP shard")
                query_offsets = torch.tensor(
                    local_offsets,
                    dtype=torch.long,
                    device=runtime.device,
                )
                targets = torch.tensor(
                    [term.target_token_id for term in owned_terms],
                    dtype=torch.long,
                    device=runtime.device,
                )
                with torch.no_grad():
                    selected = F.log_softmax(
                        logits[0].index_select(0, query_offsets).float(),
                        dim=-1,
                    ).gather(1, targets.unsqueeze(1)).squeeze(1)
                logical_offsets = torch.tensor(
                    [
                        case.logical_indices[
                            (term.sample_id, segment.position_start + term.query_offset)
                        ]
                        for term in owned_terms
                    ],
                    dtype=torch.long,
                    device=runtime.device,
                )
                actual_logprobs.index_copy_(0, logical_offsets, selected)
                owner_counts.index_add_(
                    0,
                    logical_offsets,
                    torch.ones_like(logical_offsets, dtype=owner_counts.dtype),
                )
        return original_compute_loss(executor, segment, logits)

    monkeypatch.setattr(SegmentExecutor, "_compute_loss", capture_loss)
    dist.barrier(group=runtime.cp_group)
    output, actual_gradients = _run_engine_tpr(
        case.model,
        case.plan,
        runtime,
        expected_backend=backend,
    )
    dist.all_reduce(actual_logprobs, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(owner_counts, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    if not torch.all(owner_counts == 1).item():
        bad = torch.nonzero(owner_counts != 1).flatten().cpu().tolist()
        raise AssertionError(
            f"every Qwen logical logprob must have one CP owner; bad indexes={bad[:16]}"
        )
    actual_logprobs = actual_logprobs.cpu()
    for name in tuple(actual_gradients):
        actual_gradients[name] = actual_gradients[name].cpu()

    reference = case.reference
    assert reference.execution_trace == (
        (PhysicalExecutionKind.VISIT_LEAF, 1),
        (PhysicalExecutionKind.VISIT_LEAF, 2),
    )
    torch.testing.assert_close(
        torch.tensor(output["loss"]),
        reference.normalized_loss,
        atol=_LOSS_ATOL,
        rtol=_LOSS_RTOL,
    )
    difference = actual_logprobs - reference.target_logprobs
    absolute_difference = difference.abs()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        reference.target_logprobs
    ).clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(
        actual_logprobs,
        reference.target_logprobs,
        dim=0,
    )
    diagnostic_limit = _LOGPROB_DIAGNOSTIC_ATOL + (
        _LOGPROB_DIAGNOSTIC_RTOL * reference.target_logprobs.abs()
    )
    pointwise_outliers = int(torch.count_nonzero(absolute_difference > diagnostic_limit).item())
    max_abs_difference = float(absolute_difference.max().item())
    if runtime.rank == 0:
        print(
            "Qwen logprob comparison: "
            f"relative_l2={relative_l2.item():.6e}, "
            f"cosine={cosine.item():.9f}, "
            f"max_abs={max_abs_difference:.6e}, "
            f"pointwise_outliers={pointwise_outliers}/{case.logical_count}"
        )
        reverse_indices = {index: key for key, index in case.logical_indices.items()}
        worst_count = min(10, case.logical_count)
        _, worst_indices = torch.topk(absolute_difference, worst_count)
        for index in worst_indices.tolist():
            sample_id, query_position = reverse_indices[index]
            region = "prefix" if query_position < _PREFIX_LENGTH else "suffix"
            print(
                f"  sample={sample_id} query_position={query_position} region={region} "
                f"reference={reference.target_logprobs[index].item():.7f} "
                f"actual={actual_logprobs[index].item():.7f} "
                f"abs_diff={absolute_difference[index].item():.7f}"
            )
    assert relative_l2.item() <= _LOGPROB_RELATIVE_L2_TOL, (
        f"Qwen logprob relative L2 {relative_l2.item():.6e} exceeds "
        f"{_LOGPROB_RELATIVE_L2_TOL:.6e}"
    )
    assert cosine.item() >= _LOGPROB_COSINE_MIN, (
        f"Qwen logprob cosine {cosine.item():.9f} is below {_LOGPROB_COSINE_MIN:.9f}"
    )
    _assert_real_qwen_gradients_close(
        actual_gradients,
        reference.parameter_gradients,
        per_parameter_relative_l2_tol=_CP_PER_PARAMETER_GRAD_RELATIVE_L2_TOL,
    )
    assert output["metrics"]["tpr_cp_size"] == 2
    assert output["metrics"]["tpr_cp_backend"] == backend
    assert output["metrics"]["tpr_peak_path_tokens"] == (
        _PREFIX_LENGTH + _FIRST_SUFFIX_LENGTH
    )
    assert output["metrics"]["tpr_direct_leaf_count"] == 2

    if runtime.rank == 0:
        print(
            f"\n{case.spec.name} real-checkpoint TPR CP correctness passed\n"
            f"  checkpoint: {case.spec.path}\n"
            f"  backend: {backend}\n"
            f"  topology: P={_PREFIX_LENGTH}, S1={_FIRST_SUFFIX_LENGTH}, "
            f"S2={_SECOND_SUFFIX_LENGTH}\n"
            f"  loss reference/TPR: {reference.normalized_loss.item():.9f} / "
            f"{output['loss']:.9f}\n"
            f"  logprob relative L2: {relative_l2.item():.6e}\n"
            f"  logprob cosine: {cosine.item():.9f}\n"
            f"  logprob max abs diff: {max_abs_difference:.6e}\n"
            f"  pointwise diagnostic outliers: {pointwise_outliers}/{case.logical_count}\n"
            f"  gradient tensors: {len(actual_gradients)}"
        )
    del actual_gradients, actual_logprobs
    case.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.npu.empty_cache()
    dist.barrier(group=runtime.cp_group)
