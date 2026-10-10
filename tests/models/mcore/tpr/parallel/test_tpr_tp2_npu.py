# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Real native Megatron TP2 vs TPR TP2 on pretrained Qwen3-1.7B, Ascend NPU.

Run (two visible NPUs):
    TPR_RUN_TP2=1 torchrun --standalone --nproc_per_node=2 \
      -m pytest -vv -s \
      tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py

This does NOT claim PPO E2E, TP+SP, TP+CP or TP+DP correctness.
It verifies TPR's *thin adapter* TP2 schedule, local vocab-sharded native
CE and PPO logprobs, gradients, and an optimizer step against independent
full-trajectory forwards. Full VERL MegatronEngine interception and
complete PPO/GRPO E2E are NOT exercised by this test. Attention uses the established *controlled CANN
square-causal reference* (not unmodified MindSpeed ScaledMaskedSoftmax)
with authentic TP2-sharded HF weights.
The token trajectories are deterministic test data, but the model is the
real 28-layer Qwen3-1.7B checkpoint, not a synthetic/random Tiny GPT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

from megatron.core import parallel_state
from verl.models.mcore.tpr import (
    TPRForwardBackwardRequest,
)
from verl.utils import tensordict_utils as tu

from ._tpr_cp_test_utils import _equivalence_tpr_plan
from ._qwen3_tp_checkpoint import make_real_qwen3_tp2_model


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_TP2") != "1",
    reason="set TPR_RUN_TP2=1 for two-rank NPU TP2 correctness",
)


@pytest.fixture(scope="module")
def tp2_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires torchrun --nproc_per_node=2")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    # Same bootstrap convention as existing real Engine CP NPU tests, only
    # model parallel dimensions differ. Do NOT enable Megatron SP.
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv

    repatch({
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(260910)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    assert dist.get_world_size(group=tp_group) == 2
    assert dist.get_world_size(group=cp_group) == 1
    runtime = SimpleNamespace(
        rank=dist.get_rank(group=tp_group),
        device=torch.device("npu", local_rank),
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=pp_group,
    )
    yield runtime
    dist.barrier(group=tp_group)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


_QWEN3_1_7B_PATH = Path(
    os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B")
)


def _real_model(runtime, *, load_weights=True):
    model, hf = make_real_qwen3_tp2_model(
        runtime, model_path=_QWEN3_1_7B_PATH, load_weights=load_weights
    )
    assert hf.vocab_size == 151936
    assert model.config.num_layers == 28
    assert model.embedding.word_embeddings.weight.shape == (75968, 2048)
    return model


def _parameter_gradients(model):
    return {
        name: param.grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _full_reference(model, first, second):
    model.zero_grad(set_to_none=True)
    denominator = first.numel() + second.numel() - 2
    loss = None
    for seq in (first, second):
        token_ids = seq.to(device=next(model.parameters()).device)
        seq_len = token_ids.numel()
        logits = model(
            input_ids=token_ids.unsqueeze(0),
            position_ids=torch.arange(
                seq_len, device=token_ids.device
            ).unsqueeze(0),
            attention_mask=None,
        )
        # Full (not TP-sharded) sequence, TP-sharded vocabulary. CE is native.
        assert logits.shape[:2] == (1, seq_len)
        assert logits.shape[-1] == 75968
        per_token = model.compute_language_model_loss(
            token_ids[1:].unsqueeze(0),
            logits[0, :-1, :].unsqueeze(1),
        ).reshape(-1)
        term = per_token.float().sum() / denominator
        term.backward()
        loss = term.detach() if loss is None else loss + term.detach()
    return loss, _parameter_gradients(model)


def _tpr_engine_run(model, runtime, plan):
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine

    calls = {"finalize": 0}

    def finalize(chunks, tokens, *, force_all_reduce=False, **kwargs):
        calls["finalize"] += 1
        assert chunks == [model]
        assert tokens is None and force_all_reduce is True

    model.zero_grad(set_to_none=True)
    model.config.no_sync_func = nullcontext
    model.config.grad_scale_func = lambda tensor: tensor
    model.config.finalize_model_grads_func = finalize
    model.config.calculate_per_token_loss = False

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.module = [model]
    engine.tf_config = model.config
    engine.engine_config = SimpleNamespace(
        tpr_enabled=True,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        dynamic_context_parallel=False,
        tpr_cp_backend=None,
        override_transformer_config={},
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1
    request = TPRForwardBackwardRequest(plan)
    # The deployed VERL MegatronEngine currently reads data["loss_mask"]
    # before it can route a TPR request. A request-only TensorDict therefore
    # fails in the *native* batching preamble before reaching our adapter.
    # This test is specifically the TP2 correctness gate for the explicit
    # TPR thin entry. Full Engine interception is a SEPARATE integration
    # gate; do not mislabel this as an E2E engine-routing success.
    from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward

    output = run_tpr_forward_backward(
        engine, request, forward_only=False
    )
    assert calls == {"finalize": 1}
    return output, _parameter_gradients(model)


def test_two_rank_qwen3_tp2_full_trajectory_vs_tpr_thin_entry(tp2_runtime):
    runtime = tp2_runtime
    seed = 918423
    torch.manual_seed(seed)
    reference_model = _real_model(runtime)
    torch.manual_seed(seed)
    tpr_model = _real_model(runtime, load_weights=False)
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)

    prefix = torch.arange(17, 81, dtype=torch.long)
    # Authentic 151936-word Qwen vocab; second TP vocab shard begins at
    # global token ID 75968. This specifically tests native TP CE labels.
    first_suffix = torch.arange(80101, 80133, dtype=torch.long)
    second_suffix = torch.arange(100201, 100225, dtype=torch.long)
    plan = _equivalence_tpr_plan(prefix, first_suffix, second_suffix)
    first = torch.cat((prefix, first_suffix))
    second = torch.cat((prefix, second_suffix))

    dist.barrier(group=runtime.tp_group)
    reference_loss, ref_grad = _full_reference(reference_model, first, second)
    output, tpr_grad = _tpr_engine_run(tpr_model, runtime, plan)
    tpr_loss = output["loss"]

    assert abs(tpr_loss - reference_loss.item()) < 0.02
    assert ref_grad.keys() == tpr_grad.keys()
    total_diff_sq = sum(
        (ref_grad[name] - tpr_grad[name]).square().sum().item()
        for name in ref_grad
    )
    total_ref_sq = sum(
        ref_grad[name].square().sum().item() for name in ref_grad
    )
    relative_l2 = (total_diff_sq / max(total_ref_sq, 1e-24)) ** 0.5
    assert relative_l2 < 0.05, f"native TP2 vs TPR gradient relative_l2={relative_l2}"

    # One complete optimizer step on the *same rank-local TP parameter shards*.
    before_ref = {
        name: param.detach().float().clone()
        for name, param in reference_model.named_parameters()
    }
    before_tpr = {
        name: param.detach().float().clone()
        for name, param in tpr_model.named_parameters()
    }
    assert before_ref.keys() == before_tpr.keys()
    optimizer_ref = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    optimizer_tpr = torch.optim.SGD(tpr_model.parameters(), lr=0.05)
    optimizer_ref.step()
    optimizer_tpr.step()
    update_diff_sq = 0.0
    update_ref_sq = 0.0
    for name, ref_param in reference_model.named_parameters():
        ref_delta = ref_param.detach().float() - before_ref[name]
        tpr_delta = dict(tpr_model.named_parameters())[name].detach().float() - before_tpr[name]
        update_diff_sq += (ref_delta - tpr_delta).square().sum().item()
        update_ref_sq += ref_delta.square().sum().item()
    assert (update_diff_sq / max(update_ref_sq, 1e-24)) ** 0.5 < 0.1

    if runtime.rank == 0:
        print(
            f"TPR_TP2_NPU native_loss={reference_loss.item():.7f} "
            f"tpr_loss={tpr_loss:.7f} grad_rel_l2={relative_l2:.6g} "
            f"param_tensors={len(ref_grad)}",
            flush=True,
        )
    dist.barrier(group=runtime.tp_group)


def _native_response_nll_reference(model, prefix, first_suffix, second_suffix):
    """Native TP2 local-vocab logprobs, full independent trajectories."""
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    model.zero_grad(set_to_none=True)
    p = prefix.numel()
    total = first_suffix.numel() + second_suffix.numel()
    objective = None
    for suffix in (first_suffix, second_suffix):
        sequence = torch.cat((prefix, suffix)).to(next(model.parameters()).device)
        logits = model(
            input_ids=sequence.unsqueeze(0),
            position_ids=torch.arange(
                sequence.numel(), device=sequence.device
            ).unsqueeze(0),
            attention_mask=None,
        )
        assert logits.shape[-1] == 75968, "expected Qwen3-1.7B TP2 vocab shard"
        positions = torch.arange(
            p - 1, sequence.numel() - 1, device=sequence.device
        )
        logp = vocab_parallel_log_probs_from_logits(
            logits[0].index_select(0, positions),
            sequence[p:],
        )
        term = -logp.float().sum() / total
        term.backward()
        objective = term.detach() if objective is None else objective + term.detach()
    return objective, _parameter_gradients(model)


def _tpr_ppo_nll_run(model, runtime, prefix, first_suffix, second_suffix):
    """Exercise the real PPO Forest adapter/TP2 dispatch with a known NLL loss."""
    from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch

    rows = [
        torch.cat((prefix, first_suffix)),
        torch.cat((prefix, second_suffix)),
    ]
    response_len = first_suffix.numel()
    assert response_len == second_suffix.numel()
    device = runtime.device
    data = TensorDict(
        {
            "input_ids": torch.stack(rows).to(device),
            "responses": torch.stack(
                (first_suffix, second_suffix)
            ).to(device),
            "response_mask": torch.ones(
                (2, response_len), dtype=torch.bool, device=device
            ),
            "old_log_probs": torch.zeros(
                (2, response_len), dtype=torch.float32, device=device
            ),
            "advantages": torch.ones(
                (2, response_len), dtype=torch.float32, device=device
            ),
        },
        batch_size=[2], device=device,
    )
    tu.assign_non_tensor(
        data, tpr_trajectory_keys=["task_trace_0", "task_trace_1"],
        dp_size=1, batch_num_tokens=2 * response_len,
    )
    # Independently verify the real PPO input contract *before* executing
    # the Radix/Forest parser. Both rows are constructed by cat(prefix, suffix);
    # the response suffix is necessarily identical. Check on CPU so a
    # corrupted/asynchronously failed NPU context is not mistaken for a tree
    # parser bug after a prior device-side attention-kernel failure.
    for row, suffix in enumerate((first_suffix, second_suffix)):
        actual_input_tail = data["input_ids"][row, -response_len:].detach().cpu()
        actual_response = data["responses"][row].detach().cpu()
        assert torch.equal(actual_response, suffix.cpu()), (
            f"TP2 test input construction lost response row {row}"
        )
        assert torch.equal(actual_input_tail, actual_response), (
            f"TP2 test data disagrees before Forest: row={row}, "
            f"input_tail={actual_input_tail[:8].tolist()}, "
            f"responses={actual_response[:8].tolist()}"
        )

    # Test only TP orchestration and PPO token/grad ownership here; this is
    # the token-mean negative-logprob specialization, not clipped PPO training.
    def loss_function(*, model_output, data, dp_group):
        packed = model_output["log_probs"]
        active = data["response_mask"].sum(dim=1).tolist()
        cursor = 0
        total = packed.new_zeros((), dtype=torch.float32)
        for row, count in enumerate(active):
            total = total - (
                packed[cursor:cursor + count].float()
                * data["advantages"][row, :count].float()
            ).sum()
            cursor += count + 1  # One unused dummy logit for each fake row.
        assert cursor == packed.numel()
        normalizer = tu.get_non_tensor_data(
            data, key="batch_num_tokens", default=None
        )
        if normalizer is None or normalizer <= 0:
            raise ValueError("TP2 PPO test requires a positive batch_num_tokens")
        return total / normalizer, {}

    calls = {"finalize": 0}
    def finalize(chunks, tokens, *, force_all_reduce=False, **kwargs):
        calls["finalize"] += 1
        assert chunks == [model] and tokens is None and force_all_reduce

    model.config.no_sync_func = nullcontext
    model.config.grad_scale_func = lambda value: value
    model.config.finalize_model_grads_func = finalize
    model.config.calculate_per_token_loss = False

    from verl.workers.engine.megatron.transformer_impl import MegatronEngine
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.module = [model]
    engine.tf_config = model.config
    engine.engine_config = SimpleNamespace(
        tpr_enabled=True,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        dynamic_context_parallel=False,
        tpr_cp_backend=None,
        override_transformer_config={},
        use_fused_kernels=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None

    model.zero_grad(set_to_none=True)
    output = run_tpr_forward_backward_batch(
        engine, data, loss_function, forward_only=False
    )
    assert calls["finalize"] == 1
    assert "tpr/forest_trees" in output["metrics"]
    return output, _parameter_gradients(model)


def test_tp2_real_ppo_forest_native_vocab_logprobs_and_gradients(tp2_runtime):
    runtime = tp2_runtime
    torch.manual_seed(926034)
    reference = _real_model(runtime)
    torch.manual_seed(926034)
    candidate = _real_model(runtime, load_weights=False)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    prefix = torch.arange(21, 85, dtype=torch.long)
    suffix1 = torch.arange(80101, 80133, dtype=torch.long)
    suffix2 = torch.arange(100201, 100233, dtype=torch.long)
    dist.barrier(group=runtime.tp_group)
    reference_loss, reference_grad = _native_response_nll_reference(
        reference, prefix, suffix1, suffix2
    )
    output, tpr_grad = _tpr_ppo_nll_run(
        candidate, runtime, prefix, suffix1, suffix2
    )
    # VERL's postprocess contract uses list-valued loss contributions, even
    # when the TPR Forest adapter has a single *logical* minibatch loss.
    # Compare the sole scalar contribution; do NOT change the production
    # adapter to return float and break downstream VERL metric processing.
    loss_contributions = output["loss"]
    assert isinstance(loss_contributions, list), (
        f"expected VERL loss contributions list, got {type(loss_contributions).__name__}"
    )
    assert len(loss_contributions) == 1, (
        f"TP2 PPO Forest must return one logical minibatch loss, got "
        f"{len(loss_contributions)} contributions"
    )
    actual = float(loss_contributions[0])
    assert abs(actual - reference_loss.item()) < 0.02, (
        f"TP2 PPO Forest NLL mismatch: TPR={actual:.8f}, "
        f"full_reference={reference_loss.item():.8f}"
    )
    assert reference_grad.keys() == tpr_grad.keys()
    numerator = sum(
        (reference_grad[key] - tpr_grad[key]).square().sum().item()
        for key in reference_grad
    )
    denominator = sum(
        ref.square().sum().item() for ref in reference_grad.values()
    )
    relative_l2 = (numerator / max(denominator, 1e-24)) ** 0.5
    assert relative_l2 < 0.05, (
        f"TP2 PPO Forest vs native rank-local vocab gradient error: {relative_l2}"
    )
    if runtime.rank == 0:
        print(
            f"TPR_TP2_PPO native_nll={reference_loss.item():.7f} "
            f"tpr_nll={actual:.7f} grad_rel_l2={relative_l2:.6g}",
            flush=True,
        )
    dist.barrier(group=runtime.tp_group)
