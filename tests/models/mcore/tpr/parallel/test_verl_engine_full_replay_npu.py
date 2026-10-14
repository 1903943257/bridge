"""Isolated real-VERL/MindSpeed Engine Full replay for original Actor PPO captures.

DOES NOT invoke native_full_megatron_schedule.py, TPR SegmentExecutor,
or the hand-written CPU-master optimizer.  Model, DDP, microbatch scheduling,
CP Ring attention, PPO loss and optimizer are owned by VERL.

This is a new experiment family with a separate "verl_full_" output prefix.
Old "full_" / "tpr_" snapshots are never modified or treated as comparable
optimizer trajectories.  Run a single step first using the wrapper shell.
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from verl.utils import tensordict_utils as tu
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _load_real_pair


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_VERL_ENGINE_FULL_REPLAY") != "1",
    reason="8-Ascend-NPU opt-in genuine VERL/MindSpeed Engine PPO replay",
)


def _construct_real_engine(checkpoint: Path, *, lr: float, clip_norm: float, steps: int):
    # On NPU this class (not the generic CUDA MegatronEngine) owns the
    # MindSpeed repatch and parallel-state initialization.
    from verl.workers.engine.mindspeed.transformer_impl import MindspeedEngineWithLMHead
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.trainer.config import CheckpointConfig

    model_cfg = HFModelConfig(
        path=str(checkpoint),
        enable_gradient_checkpointing=False,
        enable_activation_offload=False,
        use_remove_padding=True,
    )
    # Use EXACTLY the bridge backend proven by the online UniAgent E2E:
    #   use_mbridge=True, vanilla_mbridge=True -> verl.models.mcore.mbridge
    # The non-vanilla Megatron-Bridge path imports NVIDIA TransformerEngine
    # even for Dense Qwen without LoRA; it is NOT installed on this Ascend stack.
    import inspect
    if "vanilla_mbridge" not in inspect.signature(McoreEngineConfig).parameters:
        raise RuntimeError(
            "Installed VERL McoreEngineConfig lacks vanilla_mbridge. "
            "This is not the VERL configuration used by the successful E2E "
            "script (use_mbridge=True, vanilla_mbridge=True); synchronize "
            "the ORIGINAL working VERL code, do not install TransformerEngine."
        )
    engine_cfg = McoreEngineConfig(
        dtype="bfloat16",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=4,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        dynamic_context_parallel=False,
        use_remove_padding=True,
        use_mbridge=True,
        vanilla_mbridge=True,
        use_fused_kernels=False,
        forward_only=False,
        use_distributed_optimizer=True,
        param_offload=False,
        optimizer_offload=False,
        pad_to_length=False,
        override_transformer_config={
            "recompute_granularity": None,
            "calculate_per_token_loss": False,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
        },
    )
    opt_cfg = McoreOptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        lr_warmup_init=lr,
        lr_decay_steps=steps,
        lr_decay_style="constant",
        clip_grad=clip_norm,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        use_precision_aware_optimizer=True,
        main_grads_dtype="bf16",
        exp_avg_dtype="fp32",
        exp_avg_sq_dtype="fp32",
        override_optimizer_config={
            "use_precision_aware_optimizer": True,
            "optimizer_cpu_offload": True,
            "optimizer_offload_fraction": 1.0,
            "overlap_cpu_optimizer_d2h_h2d": True,
        },
    )
    # MindSpeed may inspect sys.argv through repatch/get_full_args.  Do not
    # parse the torchrun/pytest flags as Megatron launcher options.
    old_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        engine = MindspeedEngineWithLMHead(
            model_cfg, engine_cfg, opt_cfg, CheckpointConfig(strict=False)
        )
    finally:
        sys.argv[:] = old_argv
    if not engine.engine_config.vanilla_mbridge:
        raise AssertionError("Offline Full must follow online E2E vanilla_mbridge=True")
    print("VERL_FULL_BRIDGE_SELECTED backend=legacy_mbridge vanilla_mbridge=True", flush=True)
    engine.initialize()

    from megatron.core import parallel_state as mpu
    sizes = {
        "TP": mpu.get_tensor_model_parallel_world_size(),
        "CP": mpu.get_context_parallel_world_size(),
        "DP": mpu.get_data_parallel_world_size(),
    }
    if sizes != {"TP": 2, "CP": 4, "DP": 1}:
        raise AssertionError(f"VERL Full wrong mesh: {sizes}")
    if engine.engine_config.use_fused_kernels:
        raise AssertionError("VERL Full must use the same non-fused PPO objective")
    if engine.tf_config.recompute_granularity is not None:
        raise AssertionError("VERL Full recompute must be disabled in parity replay")
    if engine.tf_config.sequence_parallel:
        raise AssertionError("VERL Full TP2/CP4 parity expects SP=False")
    if engine.tf_config.calculate_per_token_loss:
        raise AssertionError(
            "Dense VERL Full per-token regime unexpectedly enabled; "
            "TPR's PPO objective uses vanilla global-token-mean loss"
        )
    from verl.models.mcore.tpr import TPRSelfAttention
    if any(isinstance(x, TPRSelfAttention) for module in engine.module for x in module.modules()):
        raise AssertionError("Full Engine unexpectedly installed TPRSelfAttention")
    # Verify entrypoint is the unmodified native VERL schedule, not any
    # test-generated adapter.  This check is version-agnostic enough for bridge.
    import inspect
    source = inspect.getsource(engine.forward_backward_batch)
    if "prepare_micro_batches" not in source or "get_forward_backward_func" not in source:
        raise AssertionError("This Engine has no native VERL microbatch schedule")
    print(
        "VERL_FULL_ENGINE_INIT_PASS "
        f"engine={type(engine).__name__} mesh={sizes} "
        f"recompute={engine.tf_config.recompute_granularity} "
        f"per_token_loss={engine.tf_config.calculate_per_token_loss}",
        flush=True,
    )
    return engine


def _load_step(item: dict, *, row_count: int = 8):
    capture = Path(item["capture"])
    rows = tuple(item["rows"])
    if len(rows) != row_count or len(set(rows)) != row_count:
        raise AssertionError("VERL Full requires the exact eight Session rows")
    batch, original_keys, ppo_tokens, digest = _load_real_pair(
        capture, rows, "full"
    )
    # The native Engine denominator is computed from original loss_mask,
    # whereas the original TPR replay used response_mask.  Never silently
    # compare two different token-mean objectives.
    loss_mask = batch["loss_mask"]
    real_native_tokens = int((loss_mask.values() if loss_mask.is_nested else loss_mask).bool().sum().item())
    if real_native_tokens != ppo_tokens:
        raise AssertionError(
            f"Capture loss_mask={real_native_tokens}, response_mask={ppo_tokens}; "
            "reconcile masks before native/TPR parity comparison"
        )
    if ppo_tokens != int(item["ppo_tokens"]):
        raise AssertionError("Original Actor capture PPO tokens mismatch manifest")
    # Force one original trajectory per *native* microbatch; no dynamic
    # token packing or reordering.  The Engine still owns the actual split.
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        force_group_size=1,
        global_batch_size=row_count,
    )
    return batch, original_keys, ppo_tokens, digest, rows, capture


def _cpu_metrics(metrics):
    from verl.utils.metric import Metric
    out = {}
    for key, values in metrics.items():
        if not isinstance(values, (list, tuple)):
            values = [values]
        result = []
        for value in values:
            if isinstance(value, Metric):
                result.append(float(value.aggregate()))
            elif isinstance(value, (int, float)) or (
                isinstance(value, torch.Tensor) and value.numel() == 1
            ):
                result.append(float(value))
        out[key] = result
    return out


def _capture_logprobs(output: dict, batch, row_count: int):
    # Native model_output comes from VERL's postprocess_batch_func.
    # It contains detached, full-sequence token logprobs, not TPR segment
    # logits.  Apply VERL's own response shift to recover logical tokens.
    from verl.workers.utils.padding import no_padding_2_padding
    model_output = output.get("model_output", {})
    if "log_probs" not in model_output:
        raise AssertionError("VERL Full native output missing original log_probs")
    shifted = no_padding_2_padding(model_output["log_probs"], batch).detach()
    masks = batch["response_mask"]
    if shifted.shape[0] != row_count:
        raise AssertionError("Native VERL Full logprob row count changed")
    result = {}
    for row in range(row_count):
        mask = masks[row].bool()
        values = shifted[row, :mask.numel()].detach().float().cpu()
        for offset in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            result[(row, offset)] = float(values[offset])
    return result


def _native_witness(engine, *, grads: bool):
    """Same bounded parameter-sample schema as existing TPR comparisons.

    Wrapped Megatron DDP stores finalized gradients in main_grad rather than
    param.grad; using the latter would silently report missing gradients.
    """
    from verl.utils.megatron_utils import unwrap_model
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices

    unwrapped = unwrap_model(engine.module[0])
    result = {}
    for name, param in unwrapped.named_parameters():
        if grads:
            source = getattr(param, "main_grad", None)
            if source is None:
                source = param.grad
            if source is None:
                raise AssertionError(f"Native Megatron main_grad missing: {name}")
        else:
            source = param.detach()
        flat = source.detach().reshape(-1)
        idx = torch.as_tensor(
            exact_sample_indices(flat.numel(), 32),
            dtype=torch.long, device=flat.device,
        )
        result[name] = {
            "shape": tuple(flat.shape) if grads else tuple(param.shape),
            "indices": idx.cpu(),
            "values": flat.index_select(0, idx).float().cpu(),
        }
        # The logical param shape, not any flat DDP buffer shape, identifies
        # comparable TP shards.
        result[name]["shape"] = tuple(param.shape)
    return result


def test_verl_engine_full_replay():
    if int(os.getenv("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("Native VERL Full replay requires exactly 8 ranks TP2*CP4*DP1")
    import torch_npu  # noqa: F401
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        from verl.workers.utils.losses import ppo_loss
        from megatron.core import parallel_state as mpu

        manifest_path = Path(os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"])
        output_dir = Path(os.environ["TPR_OFFLINE_OUTPUT_DIR"])
        checkpoint = Path(os.getenv("TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"))
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
                or len(manifest.get("items", [])) != 20
                or manifest.get("expected_sessions") != 8):
            raise AssertionError("Requires original 20-step, eight-Session v3 manifest")
        run_steps = int(os.getenv("TPR_VERL_FULL_REPLAY_STEPS", "1"))
        if run_steps < 1 or run_steps > 20:
            raise ValueError("TPR_VERL_FULL_REPLAY_STEPS must be 1..20")
        lr = float(os.getenv("TPR_OFFLINE_ADAMW_LR", "2e-5"))
        clip_norm = float(os.getenv("TPR_OFFLINE_ADAMW_CLIP_NORM", "1.0"))
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = os.getenv("TPR_VERL_FULL_REPLAY_PREFIX", "verl_full_smoke")
        if not prefix.startswith("verl_full") or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in prefix
        ):
            raise ValueError(f"Invalid Full snapshot prefix {prefix!r}")
        for step in range(1, run_steps + 1):
            if (output_dir / f"{prefix}_step{step}_rank{dist.get_rank()}.pt").exists():
                raise FileExistsError(f"Do not overwrite VERL Full step {step} rank {dist.get_rank()}")
        engine = _construct_real_engine(checkpoint, lr=lr, clip_norm=clip_norm, steps=20)
        ppo_config = _PPO()
        initial_param_witness = _native_witness(engine, grads=False)
        with engine.train_mode():
            for step, item in enumerate(manifest["items"][:run_steps], start=1):
                batch, original_keys, tokens, digest, rows, capture = _load_step(item)
                # Start timer only after load & graph construction for parity.
                engine.optimizer_zero_grad()
                dist.barrier()
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                init_alloc = torch.npu.memory_allocated()
                t0 = time.perf_counter()
                outputs = engine.forward_backward_batch(
                    batch, partial(ppo_loss, config=ppo_config), forward_only=False
                )
                torch.npu.synchronize()
                fb_seconds = time.perf_counter() - t0
                losses = outputs.get("loss", []) if outputs is not None else []
                if engine.is_mp_src_rank_with_outputs() and len(losses) != 8:
                    raise AssertionError(f"Native schedule expected 8 microbatches, got {len(losses)}")
                logprobs = (
                    _capture_logprobs(outputs, batch, len(rows))
                    if engine.is_mp_src_rank_with_outputs() else {}
                )
                grad_witness = _native_witness(engine, grads=True)
                t1 = time.perf_counter()
                grad_norm = engine.optimizer_step()
                engine.lr_scheduler_step()
                torch.npu.synchronize()
                optimizer_seconds = time.perf_counter() - t1
                param_after = _native_witness(engine, grads=False)
                snapshot = {
                    "execution_backend": "verl_mindspeed_megatron_engine_full",
                    "mode": "verl_full",
                    "run_prefix": prefix,
                    "step": step,
                    "steps": run_steps,
                    "rank": dist.get_rank(),
                    "tp_rank": mpu.get_tensor_model_parallel_rank(),
                    "cp_rank": mpu.get_context_parallel_rank(),
                    "checkpoint": str(checkpoint),
                    "source_capture": str(capture),
                    "source_rows": rows,
                    "source_keys": original_keys,
                    "source_online_global_step": item.get("source_global_step", step),
                    "source_capture_index": item.get("source_capture_index"),
                    "data_sha256": digest,
                    "capture_manifest": str(manifest_path),
                    "tokens": tokens,
                    "native_microbatches": 8,
                    "recompute_granularity": None,
                    "loss": [float(x) for x in losses],
                    "metrics": _cpu_metrics(outputs.get("metrics", {})) if outputs is not None else {},
                    "logprobs": logprobs,
                    "param_initial": initial_param_witness,
                    "param_after": param_after,
                    "gradient_samples": grad_witness,
                    "grad_norm_preclip": float(grad_norm) if grad_norm is not None else None,
                    "fwd_bwd_seconds": fb_seconds,
                    "optimizer_seconds": optimizer_seconds,
                    "peak_allocated_bytes": int(torch.npu.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.npu.max_memory_reserved()),
                    "allocated_before_bytes": int(init_alloc),
                    "optimizer": "verl_native_megatron_precision_aware_cpu_offload_adam",
                    "optimizer_lr": lr,
                    "optimizer_clip_norm": clip_norm,
                }
                target = output_dir / f"{prefix}_step{step}_rank{dist.get_rank()}.pt"
                if target.exists():
                    raise FileExistsError(target)
                torch.save(snapshot, target)
                if dist.get_rank() == 0:
                    print(
                        f"VERL_ENGINE_FULL step={step} loss_sum={sum(snapshot['loss']):.9g} "
                        f"tokens={tokens} fb_s={fb_seconds:.3f} opt_s={optimizer_seconds:.3f} "
                        f"grad_norm={grad_norm} "
                        f"peak_gib={snapshot['peak_allocated_bytes']/2**30:.3f} "
                        f"logprobs={len(logprobs)}",
                        flush=True,
                    )
                dist.barrier()
    finally:
        if dist.is_initialized():
            from megatron.core import parallel_state as mpu
            if mpu.model_parallel_is_initialized():
                mpu.destroy_model_parallel()
            dist.destroy_process_group()
