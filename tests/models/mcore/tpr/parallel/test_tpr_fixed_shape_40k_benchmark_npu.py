"""Isolated warm-shape Qwen3-4B Native Full vs TPR, same 40K capture.

For each original run5 online step: load the exact already-selected 8-Session
40K merged capture; load the SAME HF checkpoint independently per mode; run
1+ warmup F/B passes then repeated F/B with NO AdamW/weight updates.
Never use these as sequential PPO training steps or online reward evidence.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch
from verl.utils import tensordict_utils as tu
from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .native_full_megatron_schedule import run_native_full_step
from .test_real_actor_tp2cp4_multistep_npu import _make_engine
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _runtime, _load_real_pair

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_FIXED_SHAPE_40K_BENCH") != "1",
    reason="Explicit opt-in: 8-NPU fixed-capture forward/backward benchmark",
)


def test_fixed_shape_40k_one_mode():
    mode = os.environ["TPR_OFFLINE_MODE"]
    if mode not in ("full", "tpr"):
        raise ValueError("TPR_OFFLINE_MODE must be full or tpr")
    step = int(os.environ["TPR_BENCH_SOURCE_STEP"])
    manifest_path = Path(os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"])
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
            or manifest.get("steps") != 20
            or manifest.get("expected_sessions") != 8
            or manifest.get("max_seq_len") != 40960
            or not 1 <= step <= 20):
        raise ValueError("Requires existing, already truncated, session-first 40K/20-step manifest")
    if (manifest.get("only_step_derived") not in (None, step)
            or manifest.get("experimental_sequence_cap", 40960) !=
                manifest["items"][step - 1].get("effective_sequence_cap", 40960)):
        raise ValueError("Refuse to replay an unrelated or mismatched 30K derived step")
    item = manifest["items"][step - 1]
    if (item["step"] != step or item["source_global_step"] != step
            or item["rows"] != list(range(8)) or item.get("session_count") != 8
            or item["max_sequence_length"] > 40960):
        raise AssertionError("Source step or original eight-session selection changed")
    capture = Path(item["capture"])
    if not capture.is_file():
        raise FileNotFoundError(capture)
    warmups = int(os.environ.get("TPR_BENCH_WARMUPS", "1"))
    repeats = int(os.environ.get("TPR_BENCH_REPEATS", "3"))
    capture_logprobs = os.environ.get("TPR_BENCH_CAPTURE_LOGPROBS", "1") == "1"
    if not 0 <= warmups <= 10 or not 1 <= repeats <= 20:
        raise ValueError("Require 0..10 warmups and 1..20 measured repeats")
    if os.environ.get("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0") != "0":
        raise ValueError("Native Full recompute must be OFF")
    output_dir = Path(os.environ["TPR_BENCH_OUTPUT_DIR"])
    checkpoint = Path(os.environ.get("TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"))
    runtime = _runtime()
    try:
        from verl.workers.utils.losses import ppo_loss
        # Reuse existing original PPO-capture loader. Its original source_keys
        # are preserved: NO selection, no re-truncation, no reward re-labeling.
        batch, source_keys, tokens, data_sha = _load_real_pair(
            capture, tuple(range(8)), mode,
        )
        if tokens != item["ppo_tokens"]:
            raise AssertionError("Capture/manifest PPO token count mismatch")
        if max(int(x.numel()) for x in batch["input_ids"].unbind(0)) > int(
                item.get("effective_sequence_cap", 40960)):
            raise AssertionError("Capture exceeds its declared experimental sequence cap")
        if any(int(row.numel()) > 40960 for row in batch["input_ids"].unbind(0)):
            raise AssertionError("Encountered an untruncated row")
        if not source_keys or len(source_keys) != 8:
            raise AssertionError("Expected eight selected Sessions")
        tu.assign_non_tensor(batch, tpr_capture_log_probs=capture_logprobs)
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=4,
            sequence_parallel=False, profile="4b", native_attention=(mode == "full"),
        )
        assert hf.num_hidden_layers == 36
        model.config.recompute_granularity = None
        finalize_time = [0.0]
        engine = _make_engine(model, runtime, finalize_time) if mode == "tpr" else None
        if os.getenv("TPR_EXEC_PROFILE", "0") not in ("0", "1"):
            raise ValueError("TPR_EXEC_PROFILE must be 0 or 1")
        runs = []
        for iteration in range(warmups + repeats):
            model.zero_grad(set_to_none=True)
            # Explicitly sync before and after the measured F/B as in the
            # original 20-step replay. Timer EXCLUDES gradnorm/optimizer.
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            if mode == "full":
                out_full = run_native_full_step(
                    model, batch, runtime, partial(ppo_loss, config=_PPO()),
                    capture_logprobs=capture_logprobs,
                )
                loss = float(out_full.loss)
            else:
                out = run_tpr_forward_backward_batch(
                    engine, batch, partial(ppo_loss, config=_PPO()),
                    forward_only=False,
                )
                loss = float(out["loss"][0])
            torch.npu.synchronize()
            own_seconds = time.perf_counter() - start
            critical = torch.tensor([own_seconds], dtype=torch.float64, device=runtime.device)
            dist.all_reduce(critical, op=dist.ReduceOp.MAX)
            peak = torch.tensor(
                [float(torch.npu.max_memory_allocated())], dtype=torch.float64,
                device=runtime.device,
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)

            # AFTER timing: profiler materialization and optional per-rank
            # snapshots, never included in the F/B critical path.
            if mode == "tpr" and os.getenv("TPR_EXEC_PROFILE", "0") == "1":
                profiler = engine._tpr_segment_event_profiler
                events = profiler.materialize()
                stage = dict(engine._tpr_stage_profile)
            else:
                events, stage = [], {}
            profile = {
                "rank": dist.get_rank(),
                "stage_ms": stage,
                "events": [{
                    "kind": e["kind"], "segment": e["segment_id"],
                    "length": e["length"], "prefix": e["prefix_length"],
                    "npu_event_ms": e["npu_event_ms"],
                    "cpu_enqueue_ms": e["cpu_enqueue_ms"],
                } for e in events],
            }
            profiles = [None] * dist.get_world_size()
            if os.getenv("TPR_EXEC_PROFILE", "0") == "1" and mode == "tpr":
                dist.all_gather_object(profiles, profile)
            else:
                profiles = []
            record = {
                "source_step": step,
                "mode": mode,
                "iteration": iteration,
                "warmup": iteration < warmups,
                "critical_fwd_bwd_s": float(critical.item()),
                "peak_allocated_gib": float(peak.item()) / 2**30,
                "loss": loss,
                "input_sha256": data_sha,
                "ppo_tokens": tokens,
                "recompute": False,
                "updated_weights": False,
                "profiler_enabled": os.getenv("TPR_EXEC_PROFILE", "0") == "1",
                "capture_logprobs": capture_logprobs,
                "profiles": profiles,
            }
            if dist.get_rank() == 0:
                print(
                    f"TPR_FIXED_SHAPE step={step} mode={mode} iter={iteration} "
                    f"warmup={int(iteration < warmups)} "
                    f"critical_fwd_bwd_s={record['critical_fwd_bwd_s']:.6f} "
                    f"peak_gib={record['peak_allocated_gib']:.3f} "
                    f"loss={loss:.9g} sha256={data_sha[:16]}",
                    flush=True,
                )
                runs.append(record)
            dist.barrier()
        gradient_witness = None
        if os.getenv("TPR_BENCH_GRAD_WITNESS", "0") == "1":
            from .gradient_witness import collect_tp2_gradient_witness
            gradient_witness = collect_tp2_gradient_witness(
                model, runtime, mode=mode,
                require_exact_replicas=(mode == "tpr"),
            )
        if dist.get_rank() == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"step{step}_{mode}_fixed_shape.json"
            if target.exists():
                raise FileExistsError(f"Refuse to overwrite benchmark: {target}")
            measured = [r["critical_fwd_bwd_s"] for r in runs if not r["warmup"]]
            target.write_text(json.dumps({
                "source_step": step, "mode": mode,
                "effective_sequence_cap": int(item.get("effective_sequence_cap", 40960)),
                "input_sha256": data_sha, "source_capture": str(capture),
                "checkpoint": str(checkpoint), "warmups": warmups,
                "repeats": repeats, "updated_weights": False,
                "capture_logprobs": capture_logprobs,
                "profiler_enabled": os.getenv("TPR_EXEC_PROFILE", "0") == "1",
                "median_fwd_bwd_s": statistics.median(measured),
                "gradient_witness": gradient_witness,
                "runs": runs,
            }, indent=2))
            print(
                f"TPR_FIXED_SHAPE_SUMMARY step={step} mode={mode} "
                f"median_fwd_bwd_s={statistics.median(measured):.6f} "
                f"n={len(measured)} output={target}", flush=True,
            )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
