"""Recorded, off-policy 50-step PPO replay on real Qwen3-4B TP2 RingCP4.

Two jobs load identical HF weights, original real PPO tokens/labels. Native Full
uses Megatron/MindSpeed attention and Megatron forward/backward schedule,
never TPRSelfAttention or SegmentExecutor; TPR retains shared Prefix KV.
This is FIXED-DATA PPO objective drift, not on-policy rollout convergence.

Optimizer: CP leader only, CPU FP32-master AdamW with CPU FP32 moments;
each CP leader broadcasts BF16 updated weights to its CP peers. Extra host
memory is substantial (~3x local TP-shard parameters on TWO CP leaders).
No GPU FP32 optimizer states and no BF16 direct-SGD false stability.
"""
from __future__ import annotations

import math
import os
import json
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
from .native_full_megatron_schedule import run_native_full_step
from .test_real_actor_tp2cp4_offline_ppo_npu import (
    _PPO, _runtime, _load_real_pair, _witness,
    _global_bf16_gradient_norm, _save_complete_grads,
    _install_offline_full_ring_backend,
)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_REAL_ACTOR_TP2CP4_MULTISTEP") != "1",
    reason="opt-in 8-NPU offline fixed-data true-PPO 50-step drift",
)


class _CPLeaderCpuMasterAdamW:
    """Real FP32 master/m/v only on CP rank 0 of each TP group."""

    def __init__(self, model, runtime, lr: float, clip_norm: float):
        self.runtime = runtime
        self.lr = lr
        self.clip_norm = clip_norm
        self.names = list(model.named_parameters())
        self.state = {}
        if runtime.cp_rank == 0:
            for name, param in self.names:
                master = param.detach().to("cpu", dtype=torch.float32).clone()
                self.state[name] = (
                    master, torch.zeros_like(master), torch.zeros_like(master)
                )

    def sample_master(self):
        """Bounded FP32-master witness, only on CP group leader."""
        from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
        if self.runtime.cp_rank != 0:
            return {}
        result = {}
        for name, (weight, _, _) in self.state.items():
            ids = torch.tensor(
                exact_sample_indices(weight.numel(), 32),
                dtype=torch.long, device="cpu",
            )
            result[name] = {
                "shape": tuple(weight.shape),
                "indices": ids,
                "values": weight.flatten().index_select(0, ids).clone(),
            }
        return result

    def step(self, step: int, global_grad_norm: float):
        if not math.isfinite(global_grad_norm) or global_grad_norm < 0:
            raise FloatingPointError(f"Invalid global gradient norm {global_grad_norm}")
        # Real GRPO groups may have all-zero advantages (every rollout got
        # the same reward). Zero new grads are valid: AdamW's prior momentum
        # may still update weights. Never terminate a 20-step overnight run.
        scale = (1.0 if global_grad_norm == 0 else
                 min(1.0, self.clip_norm / global_grad_norm))
        beta1, beta2, eps = 0.9, 0.95, 1e-8
        with torch.no_grad():
            if self.runtime.cp_rank == 0:
                for name, param in self.names:
                    if param.grad is None:
                        raise RuntimeError(f"Missing gradient for {name}")
                    weight, m, v = self.state[name]
                    grad = param.grad.detach().to("cpu", dtype=torch.float32)
                    grad.mul_(scale)
                    m.mul_(beta1).add_(grad, alpha=1-beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    denom = (v / (1-beta2**step)).sqrt_().add_(eps)
                    weight.addcdiv_(
                        m, denom, value=-self.lr/(1-beta1**step)
                    )
                    param.copy_(weight.to(device=param.device, dtype=param.dtype))
                    del grad, denom
            leader = dist.get_global_rank(self.runtime.cp_group, 0)
            for _, param in self.names:
                # CP ranks hold identical parameters. Only CP leader has CPU
                # AdamW state; model weights must agree before next forward.
                dist.broadcast(param.data, src=leader, group=self.runtime.cp_group)
        return scale


def _make_engine(model, runtime, finalize_time):
    model.config.no_sync_func = None
    model.config.grad_scale_func = None
    model.config.calculate_per_token_loss = False
    model.config.recompute_granularity = None

    def finalize(chunks, num_tokens, *, force_all_reduce=False, **kwargs):
        assert chunks == [model] and num_tokens is None and force_all_reduce
        torch.npu.synchronize()
        start = time.perf_counter()
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"No gradient for {name}")
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM,
                            group=runtime.cp_group)
            param.grad.div_(4)
        torch.npu.synchronize()
        finalize_time[0] += time.perf_counter()-start

    model.config.finalize_model_grads_func = finalize
    return SimpleNamespace(
        module=[model],
        engine_config=SimpleNamespace(
            tpr_enabled=True, tensor_model_parallel_size=2,
            context_parallel_size=4, pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            dynamic_context_parallel=False, tpr_cp_backend="ring",
            use_fused_kernels=False, override_transformer_config={},
        ),
        model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False)),
        enable_routing_replay=False,
        get_data_parallel_size=lambda: 1,
        get_data_parallel_group=lambda: runtime.dp_group,
    )


def test_recorded_fixed_ppo_optimizer_drift():
    mode = os.getenv("TPR_OFFLINE_MODE")
    if mode not in ("full", "tpr"):
        raise ValueError("TPR_OFFLINE_MODE must be full or tpr")
    steps = int(os.getenv("TPR_OFFLINE_STEPS", "50"))
    if not 1 <= steps <= 1000:
        raise ValueError("Allowed offline replay optimizer updates: 1..1000")
    raw_rows = os.getenv("TPR_OFFLINE_ROWS", "0,1")
    capture = Path(os.environ["TPR_REAL_ACTOR_MINIBATCH"])
    if raw_rows == "all":
        original = torch.load(capture, map_location="cpu", weights_only=False)
        rows = tuple(range(len(original["keys"])))
        del original
    else:
        rows = tuple(int(x) for x in raw_rows.split(","))
    output = Path(os.environ.get(
        "TPR_OFFLINE_OUTPUT_DIR", "/workspace/tpr_offline_qwen4b_tp2cp4_50steps"
    ))
    checkpoint = Path(os.environ.get(
        "TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"
    ))
    lr = float(os.getenv("TPR_OFFLINE_ADAMW_LR", "2e-5"))
    clip_norm = float(os.getenv("TPR_OFFLINE_ADAMW_CLIP_NORM", "1.0"))
    if not 0 < lr < 0.1 or not clip_norm > 0:
        raise ValueError("Invalid FP32 AdamW hyperparameters")

    all_milestones = (1, 2, 5, 10, 20, 30, 40, 50)
    milestones = (tuple(range(1, steps+1))
                  if os.getenv("TPR_OFFLINE_CAPTURE_MANIFEST") else
                  tuple(x for x in all_milestones if x <= steps))
    full_grad_steps = {
        int(x) for x in os.getenv("TPR_OFFLINE_FULL_GRAD_STEPS", "").split(",") if x
    }
    if full_grad_steps and (not full_grad_steps.issubset(set(milestones))
                            or os.getenv("TPR_OFFLINE_SAVE_FULL_GRADS") != "1"):
        raise ValueError("Full gradient steps must be milestones and explicitly opt-in")

    manifest_path = os.getenv("TPR_OFFLINE_CAPTURE_MANIFEST")
    manifest_items = None
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if (manifest.get("protocol") not in (
                "real_actor_20_distinct_captures_v1",
                "real_actor_session_first_multicapture_v2",
                "real_actor_session_first_8merged_v3",
            ) or manifest.get("steps") != steps
                or len(manifest.get("items", [])) != steps):
            raise ValueError("offline manifest must contain exactly N valid real PPO updates")
        manifest_items = manifest["items"]
        if [item["step"] for item in manifest_items] != list(range(1, steps+1)):
            raise ValueError("offline replay optimizer update ordinals must be 1..N")
        if manifest.get("protocol") in (
            "real_actor_session_first_multicapture_v2",
            "real_actor_session_first_8merged_v3",
        ):
            online_steps = [int(item["source_global_step"]) for item in manifest_items]
            if (online_steps != sorted(online_steps)
                    or set(online_steps) != set(range(1,manifest["online_steps"]+1))):
                raise ValueError("session-first manifest must cover all original online steps")
            if manifest.get("historical_ppo_labels_modified") is not False:
                raise ValueError("historical PPO advantages and old logprobs must remain immutable")
            if manifest.get("protocol") == "real_actor_session_first_8merged_v3":
                if manifest.get("expected_sessions") != 8 or steps != manifest["online_steps"]:
                    raise ValueError("merged PPO replay must have exactly eight Sessions for every online step")
                if any(
                    len(item["rows"]) != 8 or item.get("session_count") != 8
                    or not item.get("source_row_provenance")
                    for item in manifest_items
                ):
                    raise ValueError("merged PPO replay requires all eight original Session rows")

    runtime = _runtime()
    try:
        from verl.workers.utils.losses import ppo_loss
        from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
        from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
        first = manifest_items[0] if manifest_items is not None else None
        if first is not None:
            capture = Path(first["capture"])
            rows = tuple(first["rows"])
        batch, original_keys, token_count, sha = _load_real_pair(capture, rows, mode)
        if mode == "tpr":
            forest = build_tree_execution_plans(
                _trajectory_keys_from_minibatch(batch), batch,
                tree_policy=os.getenv("TPR_TREE_POLICY", "exact"),
                turn_min_segment_tokens=int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS", "128")),
            )
            if os.getenv("TPR_TREE_POLICY", "exact") == "exact" and len(forest.trees) != 1:
                raise AssertionError("Exact TPR must have one shared tree")
        else:
            forest = SimpleNamespace(trees=(None,) * len(rows), segment_count=len(rows))

        ring_impl = (
            "mindspeed_native_megatron_schedule_thd_zigzag" if mode == "full"
            else _install_offline_full_ring_backend(mode)
        )
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=4,
            sequence_parallel=False, profile="4b",
            native_attention=(mode == "full"),
        )
        assert hf.num_hidden_layers == 36
        finalize_seconds = [0.0]
        engine = (
            SimpleNamespace(_tpr_captured_log_probs={})
            if mode == "full"
            else _make_engine(model, runtime, finalize_seconds)
        )
        before = _witness(model, grads=False)
        optimizer = _CPLeaderCpuMasterAdamW(model, runtime, lr, clip_norm)
        initial_master = optimizer.sample_master()
        output.mkdir(parents=True, exist_ok=True)
        for step in range(1, steps + 1):
            if step in milestones and (output / f"{mode}_step{step}_rank{dist.get_rank()}.pt").exists():
                raise FileExistsError("No overwrite of existing fixed replay snapshot")

        for step in range(1, steps+1):
            if manifest_items is not None:
                item = manifest_items[step-1]
                capture = Path(item["capture"])
                rows = tuple(item["rows"])
                batch, original_keys, token_count, sha = _load_real_pair(
                    capture, rows, mode
                )
                if token_count != item["ppo_tokens"]:
                    raise AssertionError(
                        f"step {step}: PPO captured token count differs from manifest"
                    )
                # Trace the precise physical topology of THIS real step.
                if mode == "tpr":
                    forest = build_tree_execution_plans(
                        _trajectory_keys_from_minibatch(batch), batch,
                        tree_policy=os.getenv("TPR_TREE_POLICY", "exact"),
                        turn_min_segment_tokens=int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS", "128")),
                    )
                else:
                    forest = SimpleNamespace(trees=(None,) * len(rows), segment_count=len(rows))
            model.zero_grad(set_to_none=True)
            capture_lp = step in milestones
            tu.assign_non_tensor(batch, tpr_capture_log_probs=capture_lp)
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start_alloc = torch.npu.memory_allocated()
            fwd_start = time.perf_counter()
            prior_finalize_seconds = finalize_seconds[0]
            if mode == "full":
                result = run_native_full_step(
                    model, batch, runtime, partial(ppo_loss, config=_PPO()),
                    capture_logprobs=capture_lp,
                )
                finalize_seconds[0] += result.finalize_seconds
                engine._tpr_captured_log_probs = result.logprobs
                out = {"loss": (result.loss,), "metrics": result.metrics}
            else:
                out = run_tpr_forward_backward_batch(
                    engine, batch, partial(ppo_loss, config=_PPO()),
                    forward_only=False,
                )
            torch.npu.synchronize()
            fwd_seconds = time.perf_counter() - fwd_start
            profile = (
                getattr(engine, "_tpr_segment_event_profiler", None)
                if os.getenv("TPR_EXEC_PROFILE") == "1" else None
            )
            peak = max(
                int(torch.npu.max_memory_allocated()),
                profile.max_allocated_bytes if profile is not None else 0,
            )
            reserved = max(
                int(torch.npu.max_memory_reserved()),
                profile.max_reserved_bytes if profile is not None else 0,
            )
            segment_events = profile.materialize() if profile is not None else []
            stage_profile = dict(getattr(engine, "_tpr_stage_profile", {})) if profile is not None else {}
            grad_norm = _global_bf16_gradient_norm(model, runtime)
            if not math.isfinite(grad_norm):
                raise FloatingPointError(f"step={step}: gradient norm invalid")
            grad_witness = _witness(model, grads=True) if capture_lp else {}
            full_grad_dir = None
            if step in full_grad_steps:
                full_grad_dir = _save_complete_grads(
                    model, runtime, output, mode, step
                )
            update_start = time.perf_counter()
            scale = optimizer.step(step, grad_norm)
            torch.npu.synchronize()
            update_seconds = time.perf_counter() - update_start
            post = _witness(model, grads=False) if capture_lp else {}
            master_after = optimizer.sample_master() if capture_lp else {}
            if capture_lp:
                snap = {
                    "mode":mode, "step":step, "steps":steps,
                    "rank":dist.get_rank(), "tp_rank":runtime.tp_rank,
                    "cp_rank":runtime.cp_rank, "checkpoint":str(checkpoint),
                    "source_capture":str(capture), "source_rows":rows,
                    "source_online_global_step": (
                        manifest_items[step-1].get("source_global_step",step)
                        if manifest_items is not None else None
                    ),
                    "source_capture_index": (
                        manifest_items[step-1].get("source_capture_index")
                        if manifest_items is not None else None
                    ),
                    "source_keys":original_keys, "data_sha256":sha,
                    "optimizer":"cpu_fp32_master_adamw_0.9_0.95_eps1e-8_wd0",
                    "lr":lr, "clip_norm":clip_norm, "grad_clip_factor":scale,
                    "tokens":token_count, "loss":float(out["loss"][0]),
                    "metrics":out["metrics"],
                    "logprobs":dict(engine._tpr_captured_log_probs),
                    "param_initial":before,
                    "param_after":post,
                    "master_initial":initial_master,
                    "master_after":master_after,
                    "gradient_samples":grad_witness,
                    "global_grad_norm_preclip":grad_norm,
                    "full_grad_dir":full_grad_dir,
                    "peak_allocated_bytes":peak,
                    "peak_reserved_bytes":reserved,
                    "allocated_before_bytes":start_alloc,
                    "fwd_bwd_seconds":fwd_seconds,
                    "finalize_seconds":finalize_seconds[0]-prior_finalize_seconds,
                    "cpu_optimizer_seconds":update_seconds,
                    "forest_trees":len(forest.trees),
                    "forest_segments":forest.segment_count,
                    "ring_attention_impl":ring_impl,
                    "execution_backend": (
                        "native_megatron_schedule" if mode == "full"
                        else "tpr_segment_executor"
                    ),
                    "native_recompute_layers": (
                        int(os.getenv("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0"))
                        if mode == "full" else None
                    ),
                    "native_loss_scaling": (
                        "megatron_cp_once_global_token_mean_v2"
                        if mode == "full" else None
                    ),
                    "run_kind":(
                        "SEQUENTIAL_20_DISTINCT_REAL_CAPTURES"
                        if manifest_items is not None
                        else "REPEATED_FIXED_PPO_DATA_NOT_ON_POLICY"
                    ),
                    "capture_manifest":str(manifest_path) if manifest_path else None,
                    "segment_events":segment_events,
                    "stage_profile":stage_profile,
                }
                target = output / f"{mode}_step{step}_rank{dist.get_rank()}.pt"
                torch.save(snap, target)
            if dist.get_rank() == 0:
                print(
                    f"TPR_FIXED_PPO step={step} mode={mode} "
                    f"loss={float(out['loss'][0]):.9g} gradnorm={grad_norm:.9g} "
                    f"clip_scale={scale:.6g} fwd_bwd_s={fwd_seconds:.3f} "
                    f"cpu_adamw_s={update_seconds:.3f} peak_gib={peak/2**30:.3f} "
                    f"source_step={manifest_items[step-1].get('source_global_step',step) if manifest_items is not None else 'fixed'} "
                    f"segments={forest.segment_count} ring_impl={ring_impl}",
                    flush=True,
                )
            dist.barrier()
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
