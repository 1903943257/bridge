# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Native Megatron TP2+SP layout and TPR tree correctness on real Qwen3-1.7B.

Run on two Ascend NPUs with the real HF checkpoint:
    TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B TPR_RUN_TP2_SP=1 \
      torchrun --nproc_per_node=2 --master_addr=127.0.0.1 \
      --master_port=29535 -m pytest -vv -s \
      tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_npu.py

This exercises native Megatron SP collectives, NOT custom sequence sharding.
It does not certify TP+DP, CP, production PPO/GRPO, or BF16 training stability.
The full-sequence baseline uses the existing controlled CANN square-causal
core attention, not an unmodified MindSpeed Flash Attention kernel.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from ._qwen3_tp_checkpoint import make_real_qwen3_tp2_model
from ._tpr_cp_test_utils import _equivalence_tpr_plan
from .test_tpr_tp2_npu import _full_reference, _tpr_engine_run


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_TP2_SP") != "1",
    reason="set TPR_RUN_TP2_SP=1 for isolated 2-NPU native TP2+SP tests",
)
_QWEN_PATH = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))


@pytest.fixture(scope="module")
def sp_runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 2:
        pytest.skip("TP2+SP requires torchrun --nproc_per_node=2")
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")
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
        "sequence_parallel": True,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(261011)
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


def _model(runtime, *, load_weights=True):
    model, hf = make_real_qwen3_tp2_model(
        runtime, model_path=_QWEN_PATH,
        load_weights=load_weights, sequence_parallel=True,
    )
    assert model.config.sequence_parallel is True
    assert model.config.tensor_model_parallel_size == 2
    assert hf.vocab_size == 151936
    return model


def _first_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def test_native_qwen3_tp2_sp_attention_layout_and_backward(sp_runtime):
    """Phase A: SP shards attention input, native QKV gathers, proj scatters."""
    model = _model(sp_runtime)
    attention = model.decoder.layers[0].self_attention
    shapes = {"attention_input": [], "qkv_output": [], "proj_output": []}

    def record_input(module, args):
        shapes["attention_input"].append(tuple(args[0].shape))

    def record_qkv(module, args, output):
        shapes["qkv_output"].append(tuple(_first_tensor(output).shape))

    def record_proj(module, args, output):
        shapes["proj_output"].append(tuple(_first_tensor(output).shape))

    hooks = [
        attention.register_forward_pre_hook(record_input),
        attention.linear_qkv.register_forward_hook(record_qkv),
        attention.linear_proj.register_forward_hook(record_proj),
    ]
    seq_length = 96  # divisible by TP=2; no padding in Phase A/B
    tokens = torch.arange(17, 17 + seq_length, device=sp_runtime.device)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(
            input_ids=tokens.unsqueeze(0),
            position_ids=torch.arange(seq_length, device=tokens.device).unsqueeze(0),
            attention_mask=None,
        )
        assert tuple(logits.shape) == (1, seq_length, 75968), (
            "native GPT output layer must restore full sequence, retain TP vocab shard"
        )
        per_token = model.compute_language_model_loss(
            tokens[1:].unsqueeze(0), logits[0, :-1, :].unsqueeze(1),
        )
        loss = per_token.float().mean()
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()

    assert shapes["attention_input"] == [(seq_length // 2, 1, 2048)]
    assert len(shapes["qkv_output"]) == 1
    assert shapes["qkv_output"][0][:2] == (seq_length, 1), shapes
    assert shapes["proj_output"] == [(seq_length // 2, 1, 2048)], shapes
    assert bool(torch.isfinite(loss).item())
    for name in ("linear_qkv", "linear_proj"):
        grad = getattr(attention, name).weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all().item())
        assert bool(grad.float().abs().sum().gt(0).item())
    if sp_runtime.rank == 0:
        print(f"TPR_TP2_SP_NATIVE status=PASS layouts={shapes} loss={loss.item():.7f}", flush=True)
    dist.barrier(group=sp_runtime.tp_group)


def test_qwen3_tp2_sp_full_trajectory_vs_tpr_tree(sp_runtime):
    """Phase B: same SP topology, pretrained weights, loss/logits/local gradients."""
    seed = 918423
    torch.manual_seed(seed)
    reference = _model(sp_runtime)
    torch.manual_seed(seed)
    candidate = _model(sp_runtime, load_weights=False)
    candidate.load_state_dict(reference.state_dict(), strict=True)

    prefix = torch.arange(17, 81, dtype=torch.long)
    suffix1 = torch.arange(80101, 80133, dtype=torch.long)
    suffix2 = torch.arange(100201, 100233, dtype=torch.long)
    first = torch.cat((prefix, suffix1))
    second = torch.cat((prefix, suffix2))
    plan = _equivalence_tpr_plan(prefix, suffix1, suffix2)

    # Capture only the last eight suffix logits to avoid retaining huge
    # vocabulary-sharded output tensors or interfering with autograd.
    native_tail, tree_tail = [], []

    def capture_into(destination):
        def capture(module, args, output):
            tensor = _first_tensor(output)
            assert tensor.ndim == 3 and tensor.shape[-1] == 75968
            destination.append(tensor[0, -8:, :].detach().float().cpu())
        return capture

    dist.barrier(group=sp_runtime.tp_group)
    hook = reference.register_forward_hook(capture_into(native_tail))
    try:
        reference_loss, reference_grad = _full_reference(reference, first, second)
    finally:
        hook.remove()
    hook = candidate.register_forward_hook(capture_into(tree_tail))
    try:
        output, tpr_grad = _tpr_engine_run(candidate, sp_runtime, plan)
    finally:
        hook.remove()

    assert len(native_tail) == 2
    # DFS makes four model calls, not three: Push(P) runs no_grad to cache KV,
    # Visit(S1) and Visit(S2) run with gradients, and Pop(P) recomputes the
    # root with gradients to receive the accumulated Prefix KV gradients.
    # Only the two leaf calls represent the independently executed trajectories.
    assert len(tree_tail) == 4, (
        "TPR must execute Push(P), Visit(S1), Visit(S2), Pop(P) exactly once each"
    )
    suffix_tail = tree_tail[1:3]
    logit_diff_sq = sum(
        (ref - actual).square().sum().item()
        for ref, actual in zip(native_tail, suffix_tail, strict=True)
    )
    logit_ref_sq = sum(item.square().sum().item() for item in native_tail)
    logit_rel_l2 = (logit_diff_sq / max(logit_ref_sq, 1e-24)) ** 0.5
    assert logit_rel_l2 < 0.05, f"TP2+SP suffix logit rel-L2={logit_rel_l2}"

    tpr_loss = float(output["loss"])
    assert abs(tpr_loss - reference_loss.item()) < 0.02
    assert reference_grad.keys() == tpr_grad.keys()
    grad_diff_sq = sum(
        (reference_grad[name] - tpr_grad[name]).square().sum().item()
        for name in reference_grad
    )
    grad_ref_sq = sum(grad.square().sum().item() for grad in reference_grad.values())
    grad_rel_l2 = (grad_diff_sq / max(grad_ref_sq, 1e-24)) ** 0.5
    assert grad_rel_l2 < 0.05, f"TP2+SP rank-local param grad rel-L2={grad_rel_l2}"
    if sp_runtime.rank == 0:
        print(
            f"TPR_TP2_SP_TPR native_loss={reference_loss.item():.7f} "
            f"tpr_loss={tpr_loss:.7f} logit_rel_l2={logit_rel_l2:.6g} "
            f"grad_rel_l2={grad_rel_l2:.6g}",
            flush=True,
        )
    dist.barrier(group=sp_runtime.tp_group)
