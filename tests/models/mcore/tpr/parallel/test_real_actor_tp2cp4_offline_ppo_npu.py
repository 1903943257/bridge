"""Offline recorded Qwen3-4B true-PPO: full independent Ring vs shared TPR.

Each invocation runs ONE mode on eight NPU ranks, no Ray/vLLM.
"full" means two unshared, COMPLETE causal trajectories using the SAME
TPR-capable Qwen/Ring backend; it does NOT claim native MindSpeed FA parity.
The original captured advantages, old_log_probs, masks and temperature are used.
Default SGD step is a low-memory diagnostic, NOT production AdamW.

Use run_real_actor_tp2cp4_offline_ppo.sh to run and compare both modes.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.utils import tensordict_utils as tu
from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_REAL_ACTOR_TP2CP4") != "1",
    reason="Set TPR_RUN_REAL_ACTOR_TP2CP4=1 under torchrun with 8 NPUs",
)


class _PPO:
    policy_loss = {"loss_mode": "vanilla"}
    loss_agg_mode = "token-mean"
    clip_ratio = 0.2
    clip_ratio_low = None
    clip_ratio_high = None
    entropy_coeff = 0.0
    use_kl_loss = False
    loss_scale_factor = None

    def __init__(self):
        self.global_batch_info = {}

    def get(self, key, default=None):
        return getattr(self, key, default)


def _runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 8:
        raise AssertionError("offline replay requires exactly TP2 x RingCP4 x DP1 (8 ranks)")
    import torch_npu  # noqa: F401
    from mindspeed.args_utils import get_full_args
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    old_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = old_argv
    repatch({
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "context_parallel_algo": "megatron_cp_algo",
        "sequence_parallel": False,
        "use_flash_attn": True,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            context_parallel_size=4,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    args = get_full_args()
    assert args.context_parallel_algo == "megatron_cp_algo"
    tp_group = parallel_state.get_tensor_model_parallel_group()
    cp_group = parallel_state.get_context_parallel_group()
    dp_group = parallel_state.get_data_parallel_group()
    assert dist.get_world_size(tp_group) == 2
    assert dist.get_world_size(cp_group) == 4
    assert dist.get_world_size(dp_group) == 1
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(261013)
    return SimpleNamespace(
        tp_group=tp_group, cp_group=cp_group, dp_group=dp_group,
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(),
        rank=parallel_state.get_tensor_model_parallel_rank(),
        device=torch.device("npu", local_rank),
    )


def _load_real_pair(path: Path, rows: tuple[int, int], mode: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Real PPO requires original actor-update capture, not TQ rollout")
    if any(name not in raw["tensordict"] for name in (
        "input_ids", "prompts", "responses", "response_mask", "loss_mask",
        "advantages", "old_log_probs", "temperature"
    )):
        raise AssertionError("Missing original PPO input")
    if len(set(rows)) != 2 or min(rows) < 0 or max(rows) >= len(raw["keys"]):
        raise ValueError("Select two distinct existing capture rows")
    data = tu.index_select_tensor_dict(raw["tensordict"], list(rows))
    original_keys = tuple(raw["keys"][i] for i in rows)
    if mode == "tpr":
        keys = original_keys
    else:
        # Force two independent one-root forests: physically full trajectories,
        # with no shared cached Prefix KV. Do NOT modify any PPO tensor values.
        keys = tuple(f"offline_full_{i}_tpr_0" for i in range(2))
    token_count = sum(int(m.bool().sum()) for m in data["response_mask"].unbind(0))
    assert token_count > 0
    for ids, p, r, mask, adv, old in zip(
        data["input_ids"].unbind(0), data["prompts"].unbind(0),
        data["responses"].unbind(0), data["response_mask"].unbind(0),
        data["advantages"].unbind(0), data["old_log_probs"].unbind(0),
        strict=True,
    ):
        if not torch.equal(ids, torch.cat((p, r))) or mask.numel() != r.numel():
            raise AssertionError("Captured full trajectory does not match prompt + response")
        if adv.numel() != r.numel() or old.numel() != r.numel():
            raise AssertionError("Original PPO fields do not align")
    tu.assign_non_tensor(
        data, tpr_trajectory_keys=keys, dp_size=1,
        batch_num_tokens=token_count, global_batch_size=2,
        tpr_capture_log_probs=True,
    )
    digest = hashlib.sha256()
    for key in ("input_ids", "response_mask", "old_log_probs", "advantages"):
        for row in data[key].unbind(0):
            digest.update(row.contiguous().view(torch.uint8).numpy().tobytes())
    return data, original_keys, token_count, digest.hexdigest()


def _witness(model, *, grads: bool):
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
    result = {}
    for name, p in model.named_parameters():
        x = p.grad if grads else p.detach()
        if x is None:
            raise RuntimeError(f"Missing gradient for {name}")
        flat = x.detach().reshape(-1)
        indices = torch.tensor(
            exact_sample_indices(flat.numel(), 32),
            dtype=torch.long, device=flat.device
        )
        result[name] = {
            "numel": flat.numel(),
            "shape": tuple(x.shape),
            "indices": indices.cpu(),
            "values": flat.index_select(0, indices).float().cpu(),
        }
    return result


def test_recorded_two_sibling_full_vs_tpr_one_step():
    mode = os.environ.get("TPR_OFFLINE_MODE")
    if mode not in ("full", "tpr"):
        raise AssertionError("Set TPR_OFFLINE_MODE=full or tpr")
    rows = tuple(int(z) for z in os.getenv("TPR_OFFLINE_ROWS", "4,6").split(","))
    if len(rows) != 2:
        raise ValueError("TPR_OFFLINE_ROWS must contain exactly two row IDs")
    checkpoint = Path(os.environ.get(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
    ))
    source = Path(os.environ["TPR_REAL_ACTOR_MINIBATCH"])
    destination = Path(os.getenv(
        "TPR_OFFLINE_OUTPUT_DIR", "/workspace/tpr_offline_qwen4b_tp2cp4"
    ))
    runtime = _runtime()
    try:
        from verl.workers.utils.losses import ppo_loss
        batch, real_keys, count, data_sha = _load_real_pair(source, rows, mode)
        from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
        from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
        forest = build_tree_execution_plans(_trajectory_keys_from_minibatch(batch), batch)
        if mode == "full" and (len(forest.trees) != 2 or forest.segment_count != 2):
            raise AssertionError("Full control accidentally reused a prefix")
        if mode == "tpr" and (len(forest.trees) != 1 or forest.segment_count <= 2):
            raise AssertionError("TPR mode did not create a shared tree")
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=4,
            sequence_parallel=False, profile="4b",
        )
        if hf.num_hidden_layers != 36:
            raise AssertionError("Not the real Qwen3-4B")
        model.config.no_sync_func = None
        model.config.grad_scale_func = None
        model.config.calculate_per_token_loss = False
        model.config.recompute_granularity = None
        finalize_sec = [0.0]

        def finalize(chunks, num_tokens, *, force_all_reduce=False, **kwargs):
            assert chunks == [model] and num_tokens is None and force_all_reduce
            torch.npu.synchronize()
            start = time.perf_counter()
            for name, p in model.named_parameters():
                if p.grad is None:
                    raise RuntimeError(f"No gradient for {name}")
                # Megatron CP gradient semantics: SegmentExecutor uses CP size
                # scaling locally, then CP-average across replicated weights.
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
                p.grad.div_(4)
            torch.npu.synchronize()
            finalize_sec[0] += time.perf_counter() - start

        model.config.finalize_model_grads_func = finalize
        engine = SimpleNamespace(
            module=[model],
            engine_config=SimpleNamespace(
                tpr_enabled=True,
                tensor_model_parallel_size=2, context_parallel_size=4,
                pipeline_model_parallel_size=1, expert_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                dynamic_context_parallel=False, tpr_cp_backend="ring",
                use_fused_kernels=False,
                override_transformer_config={},
            ),
            model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False)),
            enable_routing_replay=False,
            get_data_parallel_size=lambda: 1,
            get_data_parallel_group=lambda: runtime.dp_group,
        )
        model.zero_grad(set_to_none=True)
        dist.barrier()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        start_alloc = torch.npu.memory_allocated()
        start_reserved = torch.npu.memory_reserved()
        before = _witness(model, grads=False)
        before_time = time.perf_counter()
        out = run_tpr_forward_backward_batch(
            engine, batch, partial(ppo_loss, config=_PPO()), forward_only=False
        )
        torch.npu.synchronize()
        fwd_bwd_sec = time.perf_counter() - before_time
        grad = _witness(model, grads=True)
        peak_alloc = torch.npu.max_memory_allocated()
        peak_reserved = torch.npu.max_memory_reserved()
        # Optional controlled single SGD step: no extra AdamW FP32 states.
        lr = float(os.getenv("TPR_OFFLINE_SGD_LR", "1e-5"))
        if lr < 0:
            raise ValueError("SGD LR must be >= 0")
        start_step = time.perf_counter()
        with torch.no_grad():
            if lr:
                for param in model.parameters():
                    param.add_(param.grad, alpha=-lr)
        torch.npu.synchronize()
        step_sec = time.perf_counter() - start_step
        after = _witness(model, grads=False)
        payload = {
            "mode": mode, "rank": dist.get_rank(),
            "tp_rank": runtime.tp_rank, "cp_rank": runtime.cp_rank,
            "world": 8, "tp": 2, "cp": 4, "dp": 1,
            "model": "Qwen3-4B", "checkpoint": str(checkpoint),
            "source_capture": str(source), "source_rows": rows,
            "source_keys": real_keys, "data_sha256": data_sha,
            "objective": "original_captured_vanilla_PPO_token_mean",
            "reference": "full_unshared_same_Ring_backend" if mode == "full"
                else "TPR_shared_Ring",
            "optimizer": f"diagnostic_bf16_sgd_lr_{lr}",
            "logical_ppo_tokens": count, "forest_trees": len(forest.trees),
            "forest_segments": forest.segment_count,
            "loss": float(out["loss"][0]),
            "metrics": out["metrics"],
            "logprobs": engine._tpr_captured_log_probs,
            "gradient_samples": grad,
            "param_before": before, "param_after": after,
            "time_forward_backward_finalize_s": fwd_bwd_sec,
            "time_finalize_s": finalize_sec[0],
            "time_sgd_s": step_sec,
            "allocated_before_bytes": start_alloc,
            "reserved_before_bytes": start_reserved,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
        }
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / f"{mode}_rank{dist.get_rank()}.pt"
        if output_path.exists():
            raise FileExistsError(output_path)
        torch.save(payload, output_path)
        print(
            f"TPR_OFFLINE_REPLAY mode={mode} rank={dist.get_rank()} "
            f"rows={rows} loss={payload['loss']:.9g} "
            f"fwd_bwd_sec={fwd_bwd_sec:.3f} "
            f"peak_alloc_gib={peak_alloc / 2**30:.3f} "
            f"peak_reserved_gib={peak_reserved / 2**30:.3f} "
            f"saved={output_path}",
            flush=True,
        )
        dist.barrier()
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
