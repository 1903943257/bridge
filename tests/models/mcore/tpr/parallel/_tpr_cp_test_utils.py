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

"""Shared fixtures and helpers for the TPR CP Push/Visit/Pop NPU tests."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
import verl.models.mcore.tpr.parallel.allgather_attention as cp_attention
import verl.models.mcore.tpr.parallel.ring_attention as ring_attention
import verl.models.mcore.tpr.parallel.ulysses_attention as ulysses_attention
from verl.models.mcore.tpr import (
    FixedTopologyScheduler,
    PhysicalExecutionKind,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
    replace_self_attention_with_tpr,
)

from ._cp_gdn_test_utils import (
    assert_named_tensors_finite,
    broadcast_module_state,
    clone_parameter_gradients,
    named_tensor_comparison,
    tensor_comparison,
)


_EXPECTED_WORLD_SIZE = 2
_HYBRID_WORLD_SIZE = 4
_HYBRID_ULYSSES_SIZE = 2
_PREFIX_LENGTH = 1024
_FIRST_SUFFIX_LENGTH = 512
_SECOND_SUFFIX_LENGTH = 256
_VOCAB_SIZE = 2048
_LAYER_COUNT = 2
_DTYPE = torch.bfloat16
_EQUIVALENCE_CASES = (
    (128, 64, 32),
    (1024, 512, 256),
    (2048, 128, 512),
    (127, 63, 31),
    (1025, 511, 257),
)
_HYBRID_EQUIVALENCE_CASES = (
    (128, 64, 32),
    (1024, 512, 256),
    (127, 63, 31),
    (1025, 511, 257),
)
_LOSS_RELATIVE_TOL = 5e-4
_LOGPROB_ATOL = 5e-2
_LOGPROB_RTOL = 5e-3
_LOGPROB_RELATIVE_L2_TOL = 5e-3
_LOGPROB_COSINE_MIN = 0.9999
_GRAD_ATOL = 5e-3
_GRAD_RTOL = 5e-2
_GRAD_RELATIVE_L2_TOL = 3e-2
_GRAD_COSINE_MIN = 0.999


@pytest.fixture(scope="module")
def cp_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != _EXPECTED_WORLD_SIZE:
        pytest.skip("CP=2 cases require torchrun --nproc_per_node=2")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    # Match the production NPU bootstrap before Megatron initializes its
    # model-parallel RNG tracker. MindSpeed also redirects the legacy
    # ``torch.cuda`` RNG calls in this Megatron revision to torch_npu.
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
            "context_parallel_size": _EXPECTED_WORLD_SIZE,
            "context_parallel_algo": "megatron_cp_algo",
        }
    )

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_EXPECTED_WORLD_SIZE,
            expert_model_parallel_size=1,
        )

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(260907)

    cp_group = parallel_state.get_context_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    if cp_group.size() != 2 or tp_group.size() != 1 or pp_group.size() != 1:
        raise AssertionError(
            f"unexpected topology: TP={tp_group.size()}, PP={pp_group.size()}, CP={cp_group.size()}"
        )
    yield SimpleNamespace(
        rank=dist.get_rank(cp_group),
        cp_size=cp_group.size(),
        device=torch.device("npu", local_rank),
        cp_group=cp_group,
        tp_group=tp_group,
        pp_group=pp_group,
    )

    dist.barrier(group=cp_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def hybrid_cp_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != _HYBRID_WORLD_SIZE:
        pytest.skip("Hybrid CP cases require torchrun --nproc_per_node=4")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

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
            "context_parallel_size": _HYBRID_WORLD_SIZE,
            "context_parallel_algo": "hybrid_cp_algo",
            "ulysses_degree_in_cp": _HYBRID_ULYSSES_SIZE,
        }
    )

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_HYBRID_WORLD_SIZE,
            expert_model_parallel_size=1,
        )

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from mindspeed.core.context_parallel.model_parallel_utils import (
        get_context_parallel_group_for_hybrid_ring,
        get_context_parallel_group_for_hybrid_ulysses,
    )

    model_parallel_cuda_manual_seed(260908)
    cp_group = parallel_state.get_context_parallel_group()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
    ring_group = get_context_parallel_group_for_hybrid_ring()
    if (
        cp_group.size() != _HYBRID_WORLD_SIZE
        or ulysses_group.size() != _HYBRID_ULYSSES_SIZE
        or ring_group.size() != _HYBRID_WORLD_SIZE // _HYBRID_ULYSSES_SIZE
        or tp_group.size() != 1
        or pp_group.size() != 1
    ):
        raise AssertionError(
            "unexpected Hybrid topology: "
            f"TP={tp_group.size()}, PP={pp_group.size()}, CP={cp_group.size()}, "
            f"U={ulysses_group.size()}, R={ring_group.size()}"
        )
    yield SimpleNamespace(
        rank=dist.get_rank(cp_group),
        cp_size=cp_group.size(),
        device=torch.device("npu", local_rank),
        cp_group=cp_group,
        tp_group=tp_group,
        pp_group=pp_group,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
    )

    dist.barrier(group=cp_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def _make_model(runtime, *, max_sequence_length=None):
    if max_sequence_length is None:
        max_sequence_length = _PREFIX_LENGTH + _FIRST_SUFFIX_LENGTH
    config = TransformerConfig(
        num_layers=_LAYER_COUNT,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=_DTYPE,
        pipeline_dtype=_DTYPE,
        autocast_dtype=_DTYPE,
        bf16=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=runtime.cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )
    pg_collection = SimpleNamespace(
        tp=runtime.tp_group,
        cp=runtime.cp_group,
        pp=runtime.pp_group,
        embd=None,
    )
    spec = replace_self_attention_with_tpr(
        get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=max_sequence_length,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=pg_collection,
    ).to(device=runtime.device, dtype=_DTYPE)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = runtime.tp_group
    model.train()
    broadcast_module_state(model)
    return model


def _tokens(start, length):
    return (torch.arange(start, start + length, dtype=torch.long) % _VOCAB_SIZE).contiguous()


def _plan():
    prefix = _tokens(17, _PREFIX_LENGTH)
    first = _tokens(211, _FIRST_SUFFIX_LENGTH)
    second = _tokens(997, _SECOND_SUFFIX_LENGTH)
    prefix_terms = tuple(
        SegmentLossTerm(index, int(prefix[index + 1]), weight=2.0)
        for index in range(_PREFIX_LENGTH - 1)
    ) + (
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(first[0])),
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(second[0])),
    )
    first_terms = tuple(
        SegmentLossTerm(index, int(first[index + 1]))
        for index in range(_FIRST_SUFFIX_LENGTH - 1)
    )
    second_terms = tuple(
        SegmentLossTerm(index, int(second[index + 1]))
        for index in range(_SECOND_SUFFIX_LENGTH - 1)
    )
    return SegmentPlan(
        (
            SegmentSpec(0, None, prefix, 0, 0, prefix_terms),
            SegmentSpec(1, 0, first, _PREFIX_LENGTH, _PREFIX_LENGTH, first_terms),
            SegmentSpec(2, 0, second, _PREFIX_LENGTH, _PREFIX_LENGTH, second_terms),
        ),
        root_id=0,
    )


@dataclass(frozen=True)
class _EquivalenceRun:
    normalized_loss: Tensor
    target_logprobs: Tensor
    parameter_gradients: dict[str, Tensor]
    execution_trace: tuple[tuple[PhysicalExecutionKind, int], ...]


def _sample_internal_terms(tokens, *, sample_id):
    return tuple(
        SegmentLossTerm(index, int(tokens[index + 1]), sample_id=sample_id)
        for index in range(tokens.numel() - 1)
    )


def _equivalence_tpr_plan(prefix, first, second):
    prefix_length = prefix.numel()
    prefix_terms = tuple(
        SegmentLossTerm(
            index,
            int(prefix[index + 1]),
            sample_id=sample_id,
        )
        for sample_id in (1, 2)
        for index in range(prefix_length - 1)
    ) + (
        SegmentLossTerm(prefix_length - 1, int(first[0]), sample_id=1),
        SegmentLossTerm(prefix_length - 1, int(second[0]), sample_id=2),
    )
    plan = SegmentPlan(
        (
            SegmentSpec(0, None, prefix, 0, 0, prefix_terms),
            SegmentSpec(
                1,
                0,
                first,
                prefix_length,
                prefix_length,
                _sample_internal_terms(first, sample_id=1),
            ),
            SegmentSpec(
                2,
                0,
                second,
                prefix_length,
                prefix_length,
                _sample_internal_terms(second, sample_id=2),
            ),
        ),
        root_id=0,
    )
    expected_weight = (prefix_length + first.numel() - 1) + (
        prefix_length + second.numel() - 1
    )
    if plan.total_loss_weight != expected_weight:
        raise AssertionError(
            f"TPR total loss weight must be {expected_weight}, got {plan.total_loss_weight}"
        )
    branch_terms = plan.get(0).loss_terms[-2:]
    if tuple(term.query_offset for term in branch_terms) != (
        prefix_length - 1,
        prefix_length - 1,
    ):
        raise AssertionError("branch-point losses must be owned by the final Prefix query")
    if tuple(term.target_token_id for term in branch_terms) != (int(first[0]), int(second[0])):
        raise AssertionError("branch-point labels do not match the first Suffix tokens")
    return plan


def _independent_plan(trajectory, *, sample_id, total_loss_weight):
    return SegmentPlan(
        (
            SegmentSpec(
                sample_id,
                None,
                trajectory,
                0,
                0,
                _sample_internal_terms(trajectory, sample_id=sample_id),
            ),
        ),
        root_id=sample_id,
        total_loss_weight=total_loss_weight,
    )


def _logical_logprob_indices(first_trajectory, second_trajectory):
    indices = {}
    cursor = 0
    for sample_id, trajectory in ((1, first_trajectory), (2, second_trajectory)):
        for query_position in range(trajectory.numel() - 1):
            indices[(sample_id, query_position)] = cursor
            cursor += 1
    return indices, cursor


def _execute_and_capture(
    model,
    plan,
    runtime,
    logical_indices,
    logprob_values,
    owner_counts,
    *,
    cp_backend=None,
    expected_layer_numbers=None,
):
    executor = SegmentExecutor(
        model,
        plan,
        expected_layer_numbers=expected_layer_numbers,
        cp_group=runtime.cp_group,
        cp_backend=cp_backend,
    )
    original_compute_loss = executor._compute_loss

    def capture_loss(segment, logits):
        owned_terms = executor._owned_loss_terms(segment)
        if owned_terms:
            shard = executor._segment_shard(segment)
            local_offsets = tuple(shard.global_to_local(term.query_offset) for term in owned_terms)
            if any(offset is None for offset in local_offsets):
                raise AssertionError("owned loss term is not present in the local sequence shard")
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
            selected_logits = logits[0].index_select(0, query_offsets).float()
            selected_logprobs = F.log_softmax(selected_logits, dim=-1).gather(
                1, targets.unsqueeze(1)
            ).squeeze(1)
            logical_offsets = []
            for term in owned_terms:
                if term.sample_id is None:
                    raise AssertionError("equivalence loss terms must carry sample_id")
                logical_offsets.append(
                    logical_indices[(term.sample_id, segment.position_start + term.query_offset)]
                )
            logical_offsets = torch.tensor(
                logical_offsets,
                dtype=torch.long,
                device=runtime.device,
            )
            logprob_values.index_copy_(0, logical_offsets, selected_logprobs.detach())
            owner_counts.index_add_(
                0,
                logical_offsets,
                torch.ones_like(logical_offsets, dtype=owner_counts.dtype),
            )
        return original_compute_loss(segment, logits)

    executor._compute_loss = capture_loss
    try:
        return FixedTopologyScheduler(plan, executor).run()
    finally:
        executor._compute_loss = original_compute_loss


def _aggregate_loss_and_logprobs(loss, logprobs, owner_counts, runtime):
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(logprobs, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    dist.all_reduce(owner_counts, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    if not torch.all(owner_counts == 1).item():
        bad = torch.nonzero(owner_counts != 1).flatten().cpu().tolist()
        raise AssertionError(
            f"every logical logprob must have exactly one CP owner; bad indexes={bad[:16]}"
        )
    return global_loss, logprobs


def _finalize_cp_parameter_gradients(model, runtime):
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"missing parameter gradient before CP finalize: {name}")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
        parameter.grad.div_(runtime.cp_size)
    gradients = clone_parameter_gradients(model)
    assert_named_tensors_finite(gradients, label="finalized CP parameter gradient")
    return gradients


def _run_independent_cp_reference(
    model,
    first_trajectory,
    second_trajectory,
    runtime,
    logical_indices,
    logical_count,
    *,
    expected_layer_numbers=None,
    force_explicit_attention_mask=False,
):
    """Run two complete trajectories through independent AllGather CP.

    ``force_explicit_attention_mask`` keeps the independent execution and its
    full-trajectory topology unchanged, but selects the same standard CANN
    custom-mask path required by physically padded TPR segments.  This avoids
    treating sparse-mode 0 versus sparse-mode 3 kernel drift as a padding
    correctness error when the complete trajectories happen to be divisible.
    """

    model.zero_grad(set_to_none=True)
    logprobs = torch.zeros(logical_count, dtype=torch.float32, device=runtime.device)
    owner_counts = torch.zeros(logical_count, dtype=torch.int64, device=runtime.device)
    total_loss_weight = float(
        first_trajectory.numel() + second_trajectory.numel() - 2
    )
    local_loss = torch.zeros((), dtype=torch.float32, device=runtime.device)
    trace = []
    explicit_mask_context = (
        _force_explicit_allgather_attention_mask()
        if force_explicit_attention_mask
        else nullcontext()
    )
    with explicit_mask_context:
        for sample_id, trajectory in ((1, first_trajectory), (2, second_trajectory)):
            plan = _independent_plan(
                trajectory,
                sample_id=sample_id,
                total_loss_weight=total_loss_weight,
            )
            result = _execute_and_capture(
                model,
                plan,
                runtime,
                logical_indices,
                logprobs,
                owner_counts,
                expected_layer_numbers=expected_layer_numbers,
            )
            local_loss = local_loss + result.normalized_loss
            trace.extend((execution.kind, execution.segment_id) for execution in result.execution_trace)
    global_loss, global_logprobs = _aggregate_loss_and_logprobs(
        local_loss, logprobs, owner_counts, runtime
    )
    return _EquivalenceRun(
        normalized_loss=global_loss,
        target_logprobs=global_logprobs,
        parameter_gradients=_finalize_cp_parameter_gradients(model, runtime),
        execution_trace=tuple(trace),
    )


@contextmanager
def _force_explicit_allgather_attention_mask():
    """Select and verify AllGather's explicit-mask path for a reference run."""

    original_has_padding = cp_attention._has_padding
    original_attention = cp_attention.rectangular_causal_attention
    call_count = 0

    def explicit_mask_required(shard):
        del shard
        return True

    def checked_attention(*args, **kwargs):
        nonlocal call_count
        if kwargs.get("attention_mask") is None:
            raise AssertionError("explicit-mask AllGather reference reached sparse mode 3")
        call_count += 1
        return original_attention(*args, **kwargs)

    cp_attention._has_padding = explicit_mask_required
    cp_attention.rectangular_causal_attention = checked_attention
    try:
        yield
        if call_count == 0:
            raise AssertionError("explicit-mask AllGather reference executed no attention calls")
    finally:
        cp_attention._has_padding = original_has_padding
        cp_attention.rectangular_causal_attention = original_attention


def _run_tpr_cp(
    model,
    plan,
    runtime,
    logical_indices,
    logical_count,
    *,
    cp_backend="allgather",
    expected_layer_numbers=None,
):
    model.zero_grad(set_to_none=True)
    logprobs = torch.zeros(logical_count, dtype=torch.float32, device=runtime.device)
    owner_counts = torch.zeros(logical_count, dtype=torch.int64, device=runtime.device)
    result = _execute_and_capture(
        model,
        plan,
        runtime,
        logical_indices,
        logprobs,
        owner_counts,
        cp_backend=cp_backend,
        expected_layer_numbers=expected_layer_numbers,
    )
    global_loss, global_logprobs = _aggregate_loss_and_logprobs(
        result.normalized_loss, logprobs, owner_counts, runtime
    )
    return _EquivalenceRun(
        normalized_loss=global_loss,
        target_logprobs=global_logprobs,
        parameter_gradients=_finalize_cp_parameter_gradients(model, runtime),
        execution_trace=tuple(
            (execution.kind, execution.segment_id) for execution in result.execution_trace
        ),
    )


@contextmanager
def _collective_probe():
    original_all_gather = cp_attention._all_gather_into_tensor
    original_reduce_scatter = cp_attention._reduce_scatter_tensor
    counts = {"all_gather": 0, "reduce_scatter": 0}

    def all_gather(output, input_, group):
        counts["all_gather"] += 1
        return original_all_gather(output, input_, group)

    def reduce_scatter(output, input_, group):
        counts["reduce_scatter"] += 1
        return original_reduce_scatter(output, input_, group)

    cp_attention._all_gather_into_tensor = all_gather
    cp_attention._reduce_scatter_tensor = reduce_scatter
    try:
        yield counts
    finally:
        cp_attention._all_gather_into_tensor = original_all_gather
        cp_attention._reduce_scatter_tensor = original_reduce_scatter


@contextmanager
def _hybrid_communication_probe():
    original_all_to_all = ulysses_attention._mindspeed_all_to_all
    original_ring_forward = ring_attention._block_attention_forward
    original_ring_backward = ring_attention._block_attention_backward
    counts = {"all_to_all": 0, "ring_forward": 0, "ring_backward": 0}

    def all_to_all(*args, **kwargs):
        counts["all_to_all"] += 1
        return original_all_to_all(*args, **kwargs)

    def ring_forward(*args, **kwargs):
        counts["ring_forward"] += 1
        return original_ring_forward(*args, **kwargs)

    def ring_backward(*args, **kwargs):
        counts["ring_backward"] += 1
        return original_ring_backward(*args, **kwargs)

    ulysses_attention._mindspeed_all_to_all = all_to_all
    ring_attention._block_attention_forward = ring_forward
    ring_attention._block_attention_backward = ring_backward
    try:
        yield counts
    finally:
        ulysses_attention._mindspeed_all_to_all = original_all_to_all
        ring_attention._block_attention_forward = original_ring_forward
        ring_attention._block_attention_backward = original_ring_backward


def _clone_prefix_gradients(executor):
    return {
        layer_number: (key.clone(), value.clone())
        for layer_number, (key, value) in executor.kv_stack.get_new_kv_gradients(0).items()
    }


def _run_controlled_equivalence(
    runtime,
    prefix_length,
    first_suffix_length,
    second_suffix_length,
    tpr_backend,
):
    dist.barrier(group=runtime.cp_group)
    torch.manual_seed(261000 + prefix_length + first_suffix_length + second_suffix_length)

    max_sequence_length = prefix_length + max(first_suffix_length, second_suffix_length)
    reference_model = _make_model(
        runtime,
        max_sequence_length=max_sequence_length,
    )
    tpr_model = _make_model(
        runtime,
        max_sequence_length=max_sequence_length,
    )
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)

    prefix = _tokens(17, prefix_length)
    first = _tokens(701, first_suffix_length)
    second = _tokens(1301, second_suffix_length)
    first_trajectory = torch.cat((prefix, first))
    second_trajectory = torch.cat((prefix, second))
    tpr_plan = _equivalence_tpr_plan(prefix, first, second)
    logical_indices, logical_count = _logical_logprob_indices(
        first_trajectory,
        second_trajectory,
    )
    if tpr_plan.total_loss_weight != logical_count:
        raise AssertionError(
            f"logical logprob count {logical_count} does not match "
            f"loss weight {tpr_plan.total_loss_weight}"
        )

    reference = _run_independent_cp_reference(
        reference_model,
        first_trajectory,
        second_trajectory,
        runtime,
        logical_indices,
        logical_count,
    )
    probe = _hybrid_communication_probe() if tpr_backend == "hybrid" else nullcontext(None)
    with probe as hybrid_counts:
        actual = _run_tpr_cp(
            tpr_model,
            tpr_plan,
            runtime,
            logical_indices,
            logical_count,
            cp_backend=tpr_backend,
        )

    if hybrid_counts is not None:
        local_counts = torch.tensor(
            tuple(hybrid_counts.values()),
            dtype=torch.int64,
            device=runtime.device,
        )
        rank_counts = [torch.empty_like(local_counts) for _ in range(runtime.cp_size)]
        dist.all_gather(rank_counts, local_counts, group=runtime.cp_group)
        if any(not torch.equal(counts, rank_counts[0]) for counts in rank_counts[1:]):
            raise AssertionError(f"Hybrid collective counts differ across CP ranks: {rank_counts}")
        if torch.any(local_counts <= 0).item():
            raise AssertionError(f"Hybrid path did not execute every communication phase: {hybrid_counts}")

    assert reference.execution_trace == (
        (PhysicalExecutionKind.VISIT_LEAF, 1),
        (PhysicalExecutionKind.VISIT_LEAF, 2),
    )
    assert actual.execution_trace == (
        (PhysicalExecutionKind.PUSH, 0),
        (PhysicalExecutionKind.VISIT_LEAF, 1),
        (PhysicalExecutionKind.VISIT_LEAF, 2),
        (PhysicalExecutionKind.POP, 0),
    )
    if not torch.isfinite(reference.normalized_loss).item():
        raise AssertionError("Independent CP reference loss is non-finite")
    if not torch.isfinite(actual.normalized_loss).item():
        raise AssertionError("TPR CP loss is non-finite")
    if not torch.isfinite(reference.target_logprobs).all().item():
        raise AssertionError("Independent CP reference logprobs are non-finite")
    if not torch.isfinite(actual.target_logprobs).all().item():
        raise AssertionError("TPR CP logprobs are non-finite")

    reference_loss = float(reference.normalized_loss.item())
    actual_loss = float(actual.normalized_loss.item())
    loss_relative = abs(actual_loss - reference_loss) / max(abs(reference_loss), 1e-12)
    if loss_relative > _LOSS_RELATIVE_TOL:
        raise AssertionError(
            f"loss relative difference {loss_relative:.6e} exceeds {_LOSS_RELATIVE_TOL:.6e}"
        )

    torch.testing.assert_close(
        actual.target_logprobs,
        reference.target_logprobs,
        atol=_LOGPROB_ATOL,
        rtol=_LOGPROB_RTOL,
    )
    logprob_metrics = tensor_comparison(
        reference.target_logprobs,
        actual.target_logprobs,
    )
    if logprob_metrics.relative_l2 > _LOGPROB_RELATIVE_L2_TOL:
        raise AssertionError(
            f"logprob relative L2 {logprob_metrics.relative_l2:.6e} "
            f"exceeds {_LOGPROB_RELATIVE_L2_TOL:.6e}"
        )
    if logprob_metrics.cosine < _LOGPROB_COSINE_MIN:
        raise AssertionError(
            f"logprob cosine {logprob_metrics.cosine:.9f} is below {_LOGPROB_COSINE_MIN:.9f}"
        )

    gradient_metrics, worst_gradient = named_tensor_comparison(
        reference.parameter_gradients,
        actual.parameter_gradients,
    )
    for name in sorted(reference.parameter_gradients):
        torch.testing.assert_close(
            actual.parameter_gradients[name],
            reference.parameter_gradients[name],
            atol=_GRAD_ATOL,
            rtol=_GRAD_RTOL,
            msg=lambda message, name=name: f"parameter gradient mismatch for {name}: {message}",
        )
    if gradient_metrics.relative_l2 > _GRAD_RELATIVE_L2_TOL:
        raise AssertionError(
            f"parameter-gradient relative L2 {gradient_metrics.relative_l2:.6e} "
            f"exceeds {_GRAD_RELATIVE_L2_TOL:.6e}"
        )
    if gradient_metrics.cosine < _GRAD_COSINE_MIN:
        raise AssertionError(
            f"parameter-gradient cosine {gradient_metrics.cosine:.9f} "
            f"is below {_GRAD_COSINE_MIN:.9f}"
        )

    if runtime.rank == 0:
        print(
            f"\nTPR CP={runtime.cp_size} {tpr_backend} controlled equivalence passed\n"
            f"  topology: P={prefix_length}, S1={first_suffix_length}, "
            f"S2={second_suffix_length}\n"
            f"  target logprobs: {logical_count}\n"
            f"  loss reference/TPR: {reference_loss:.9f} / {actual_loss:.9f}\n"
            f"  loss relative diff: {loss_relative:.6e}\n"
            f"  logprob relative L2: {logprob_metrics.relative_l2:.6e}\n"
            f"  logprob cosine: {logprob_metrics.cosine:.9f}\n"
            f"  gradient tensors: {len(reference.parameter_gradients)}\n"
            f"  gradient relative L2: {gradient_metrics.relative_l2:.6e}\n"
            f"  gradient cosine: {gradient_metrics.cosine:.9f}\n"
            f"  worst gradient: {worst_gradient[0]} "
            f"({worst_gradient[1].relative_l2:.6e})"
        )

    dist.barrier(group=runtime.cp_group)
