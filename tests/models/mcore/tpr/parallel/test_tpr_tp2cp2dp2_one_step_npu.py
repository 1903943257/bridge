"""One recorded PPO step: Qwen3-4B TPR TP2 x RingCP2 x DP2 on 8 NPUs.

Compare against the EXISTING TP2 x CP4 x DP1 Native Full fixed-shape JSON.
Same 8 original Sessions and HF checkpoint; no rollout and no AdamW.
One *source* step is replayed for warmup/repeated F/B timing, without updates.
The DP placement is the DFS minimax planner imported from PR #5; this is
an isolated research experiment, NOT a production VERL DP2 enablement.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.models.mcore.tpr.dp_placement import plan_tpr_dta_dp
from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch
from verl.utils import tensordict_utils as tu

from ._qwen3_tp_checkpoint import make_real_qwen3_tp_model
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO, _load_real_pair

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_TP2CP2DP2_ONE_STEP") != "1",
    reason="opt in to the offline TP2 CP2 DP2 8-NPU one-step benchmark",
)


def _runtime():
    if int(os.getenv("WORLD_SIZE", "1")) != 8:
        raise AssertionError("TP2 x RingCP2 x DP2 requires exactly 8 ranks")
    import torch_npu  # noqa: F401
    from mindspeed.args_utils import get_full_args

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = argv
    repatch({
        "tensor_model_parallel_size": 2,
        "context_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_algo": "megatron_cp_algo",
        "sequence_parallel": False,
        "use_flash_attn": True,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    assert get_full_args().context_parallel_algo == "megatron_cp_algo"
    tp = parallel_state.get_tensor_model_parallel_group()
    cp = parallel_state.get_context_parallel_group()
    dp = parallel_state.get_data_parallel_group()
    # Reuse the native Megatron data+context group for weight-gradient
    # synchronization. CP ranks hold replicated weights, just like DP ranks.
    dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    assert (dist.get_world_size(tp), dist.get_world_size(cp),
            dist.get_world_size(dp), dist.get_world_size(dp_cp)) == (2, 2, 2, 4)
    device = torch.device("npu", local_rank)

    # Validate actual collective membership, not merely group size. An
    # all-reduce over the wrong two-rank group still sums two ones in the
    # prewarm and can silently combine CP peers instead of DP replicas.
    coords = torch.tensor(
        [dist.get_rank(), parallel_state.get_tensor_model_parallel_rank(),
         parallel_state.get_context_parallel_rank(),
         parallel_state.get_data_parallel_rank()],
        dtype=torch.int64, device=device,
    )
    all_coords = [torch.empty_like(coords) for _ in range(dist.get_world_size())]
    dist.all_gather(all_coords, coords)
    layout = [tuple(map(int, row.cpu().tolist())) for row in all_coords]
    if sorted(rank for rank, *_ in layout) != list(range(8)):
        raise AssertionError(f"Invalid global TP/CP/DP rank layout: {layout}")
    my_rank, my_tp, my_cp, my_dp = map(int, coords.cpu().tolist())
    expected = {
        "tp": {rank for rank, t, c, d in layout if (c, d) == (my_cp, my_dp)},
        "cp": {rank for rank, t, c, d in layout if (t, d) == (my_tp, my_dp)},
        "dp": {rank for rank, t, c, d in layout if (t, c) == (my_tp, my_cp)},
        "dp_cp": {rank for rank, t, c, d in layout if t == my_tp},
    }
    groups = {"tp": tp, "cp": cp, "dp": dp, "dp_cp": dp_cp}
    for label, group in groups.items():
        actual = set(dist.get_process_group_ranks(group))
        if actual != expected[label]:
            raise AssertionError(
                f"TPR_DP2_PROCESS_GROUP_MISMATCH rank={my_rank} label={label} "
                f"actual={sorted(actual)} expected={sorted(expected[label])} "
                f"layout={layout}"
            )
    # A separate FP32 WORLD probe is essential on Ascend 910B.
    # HCCL AllReduce does not support FP64 on this hardware, even though
    # AllGather is able to carry FP64 tensors.
    world_probe = torch.tensor(
        float(my_rank + 1), device=device, dtype=torch.float32,
    )
    dist.all_reduce(
        world_probe, op=dist.ReduceOp.SUM, group=dist.group.WORLD,
    )
    torch.npu.synchronize()
    if float(world_probe.item()) != 36.0:
        raise AssertionError(
            "TPR_DP2_WORLD_FP32_REDUCE_FAILED "
            f"rank={my_rank} received={world_probe.item()} expected=36"
        )
    if dist.get_rank() == 0:
        print(
            "TPR_DP2_WORLD_FP32_REDUCE_PASS rank_sum=36 world_size=8",
            flush=True,
        )
        print(
            f"TPR_DP2_PROCESS_GROUP_AUDIT layout={layout} "
            f"rank0_groups={{"
            + ", ".join(
                f"{key}:{sorted(dist.get_process_group_ranks(value))}"
                for key, value in groups.items()
            ) + "}",
            flush=True,
        )

    # Initialize the HCCL communicators BEFORE loading the Qwen3-4B model
    # and entering 40K-token forward/backward. Otherwise the first DP
    # all_reduce during finalize may lazily request ~1 GiB HCCL memory
    # after the activation/gradient peak has already been reached.
    for label, group, expected_size in (
        ("cp", cp, 2), ("dp", dp, 2), ("dp_cp", dp_cp, 4),
    ):
        # Rank-coded probe catches a wrong 2-rank group whose SUM(ones)
        # would still appear to have the correct world size.
        probe = torch.tensor(
            float(dist.get_rank() + 1), device=device, dtype=torch.float32,
        )
        dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=group)
        torch.npu.synchronize()
        expected_sum = float(sum(rank + 1 for rank in expected[label]))
        if len(expected[label]) != expected_size or float(probe.item()) != expected_sum:
            raise AssertionError(
                f"Wrong {label} communicator rank={dist.get_rank()} "
                f"group_ranks={dist.get_process_group_ranks(group)} "
                f"received_rank_sum={probe.item()} expected_rank_sum={expected_sum}"
            )
    if dist.get_rank() == 0:
        print(
            "TPR_DP2_HCCL_PREWARM cp=2 dp=2 dp_cp=4 "
            "grad_sync=native_dp_with_cp before_model_load=true",
            flush=True,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(261013)
    return SimpleNamespace(
        tp_group=tp, cp_group=cp, dp_group=dp, dp_cp_group=dp_cp,
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        cp_rank=parallel_state.get_context_parallel_rank(),
        dp_rank=parallel_state.get_data_parallel_rank(),
        rank=parallel_state.get_tensor_model_parallel_rank(),
        device=torch.device("npu", local_rank),
    )


def _engine_with_dp_finalize(model, runtime):
    """Native DP-with-CP weight-gradient average; never sync Prefix KV."""
    from .grad_sync_precision import offline_grad_sync_dtype, reduce_average_weight_grad
    grad_dtype = offline_grad_sync_dtype()
    model.config.no_sync_func = None
    model.config.grad_scale_func = None
    model.config.calculate_per_token_loss = False
    model.config.recompute_granularity = None

    def finalize(chunks, num_tokens, *, force_all_reduce=False, **kwargs):
        assert chunks == [model] and num_tokens is None and force_all_reduce
        # Each of the two CP ranks contributes a shard's gradients, and
        # VERL's token-mean PPO scales each DP replica by dp_size=2.
        # (CP SUM / 2) followed by (DP SUM / 2) equals the native
        # DP-with-CP SUM / 4, apart from BF16 reduction-order rounding.
        # A single four-rank collective avoids two HCCL communicator
        # switches per parameter.
        for name, param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Missing gradient before DP finalize: {name}")
            try:
                reduce_average_weight_grad(
                    param, runtime.dp_cp_group, name=name,
                    reduce_dtype=grad_dtype,
                )
            except RuntimeError:
                print(
                    f"TPR_DP2_GRAD_SYNC_FAILED rank={dist.get_rank()} "
                    f"param={name} shape={tuple(param.grad.shape)} "
                    f"dtype={param.grad.dtype} "
                    f"alloc_gib={torch.npu.memory_allocated()/2**30:.3f} "
                    f"reserved_gib={torch.npu.memory_reserved()/2**30:.3f}",
                    flush=True,
                )
                raise

    model.config.finalize_model_grads_func = finalize
    return SimpleNamespace(
        module=[model],
        engine_config=SimpleNamespace(
            tpr_enabled=True, tensor_model_parallel_size=2,
            context_parallel_size=2, pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            dynamic_context_parallel=False, tpr_cp_backend="ring",
            use_fused_kernels=False, override_transformer_config={},
        ),
        model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False)),
        enable_routing_replay=False,
        get_data_parallel_size=lambda: 2,
        get_data_parallel_group=lambda: runtime.dp_group,
    )


def _source_step():
    step = int(os.environ["TPR_BENCH_SOURCE_STEP"])
    manifest_path = Path(os.environ["TPR_OFFLINE_CAPTURE_MANIFEST"])
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("protocol") != "real_actor_session_first_8merged_v3"
            or manifest.get("steps") != 20
            or manifest.get("expected_sessions") != 8
            or manifest.get("max_seq_len") != 40960
            or not 1 <= step <= 20):
        raise ValueError("Requires original 20-step, session-first 40K capture manifest")
    if (manifest.get("only_step_derived") not in (None, step)
            or manifest.get("experimental_sequence_cap", 40960) !=
                manifest["items"][step - 1].get("effective_sequence_cap", 40960)):
        raise ValueError("Refuse to replay unrelated or mismatched 30K derived step")
    item = manifest["items"][step - 1]
    if (item["step"] != step or item["source_global_step"] != step
            or item["rows"] != list(range(8))
            or item.get("session_count") != 8
            or item["max_sequence_length"] > 40960):
        raise AssertionError("Refuse changed or untruncated original Session selection")
    capture = Path(item["capture"])
    if not capture.is_file():
        raise FileNotFoundError(capture)
    return step, capture, item


def _placement(capture, runtime, expected_tokens):
    # All eight ranks independently see the same global PPO input and plan.
    # No cross-DP prefix sharing: each DP replica builds an ordinary local tree.
    global_batch, global_keys, global_tokens, global_sha = _load_real_pair(
        capture, tuple(range(8)), "tpr",
    )
    if global_tokens != expected_tokens:
        raise AssertionError("Manifest/global PPO-token count disagrees")
    sequences = [list(map(int, ids.tolist())) for ids in global_batch["input_ids"].unbind(0)]
    placement = plan_tpr_dta_dp(
        sequences, global_keys, dp_size=2, enforce_equal_rows=False,
    )
    if (sorted(sum((list(p) for p in placement.partitions), [])) != list(range(8))
            or not all(placement.partitions)):
        raise AssertionError("DP placement lost/duplicated a Session")
    local_rows = placement.partitions[runtime.dp_rank]
    # Exactly the original Session rows, including captured advantages,
    # response_mask, old_log_probs and temperatures.
    batch, keys, local_tokens, _ = _load_real_pair(capture, local_rows, "tpr")
    if keys != tuple(global_keys[i] for i in local_rows):
        raise AssertionError("Physical row-to-PPO identity changed")
    tu.assign_non_tensor(
        batch, dp_size=2, batch_num_tokens=global_tokens,
        global_batch_size=8,
        tpr_capture_log_probs=os.getenv("TPR_BENCH_CAPTURE_LOGPROBS", "1") == "1",
    )
    # HCCL object collectives can decode a corrupted pickle-size header as a
    # gigantic allocation (all_gather_object -> resize_(max_object_size)).
    # Exchange a fixed-size device tensor instead. Eight rows fit one bitmask.
    local_mask = sum(1 << int(row) for row in local_rows)
    fingerprint = int(global_sha[:15], 16)  # 60 bits, fits int64
    payload = torch.tensor(
        [runtime.dp_rank, local_mask, local_tokens, fingerprint],
        dtype=torch.int64, device=runtime.device,
    )
    received = [torch.empty_like(payload) for _ in range(dist.get_world_size())]
    dist.all_gather(received, payload)
    assignments = [tuple(map(int, x.cpu().tolist())) for x in received]
    for dp_rank in (0, 1):
        members = [x for x in assignments if x[0] == dp_rank]
        expected_mask = sum(1 << int(row) for row in placement.partitions[dp_rank])
        if (len(members) != 4 or any(x != members[0] for x in members)
                or members[0][1] != expected_mask):
            raise AssertionError(f"TP/CP peers disagree on DP{dp_rank} row placement")
    if any(x[3] != fingerprint for x in assignments):
        raise AssertionError("Input fingerprint differs between ranks")
    if sum(next(x[2] for x in assignments if x[0] == d) for d in (0, 1)) != global_tokens:
        raise AssertionError("DP replicas do not cover exactly the global PPO tokens")
    return batch, placement, global_tokens, global_sha


def test_tpr_tp2cp2dp2_one_recorded_step():
    if os.getenv("TPR_EXPERIMENTAL_OFFLINE_DP2") != "1":
        raise ValueError("Offline DP2 adapter gate was not explicitly enabled")
    if os.getenv("TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS", "0") != "0":
        raise ValueError("Full baseline and this TPR run must both disable recompute")
    if os.getenv("TPR_TREE_POLICY") != "qwen_turn":
        raise ValueError("Use the same qwen_turn policy as the TP2CP4 Full comparison")
    if os.getenv("TPR_OFFLINE_MODE") != "tpr":
        raise ValueError("This experiment must run only the TPR path")
    warmups = int(os.getenv("TPR_BENCH_WARMUPS", "1"))
    repeats = int(os.getenv("TPR_BENCH_REPEATS", "3"))
    if not 0 <= warmups <= 10 or not 1 <= repeats <= 20:
        raise ValueError("warmups=0..10 and repeats=1..20 required")
    capture_lp = os.getenv("TPR_BENCH_CAPTURE_LOGPROBS", "1") == "1"
    lp_chunk_tokens = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", "512"))
    if lp_chunk_tokens < 0:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be nonnegative")
    profile = os.getenv("TPR_EXEC_PROFILE", "0") == "1"
    output = Path(os.environ["TPR_BENCH_OUTPUT_DIR"])
    checkpoint = Path(os.getenv("TPR_QWEN_4B_PATH", "/workspace/hf_models/Qwen3-4B-128K"))
    step, capture, source = _source_step()
    effective_sequence_cap = int(source.get("effective_sequence_cap", 40960))
    runtime = _runtime()
    try:
        from verl.workers.utils.losses import ppo_loss

        batch, plan, global_tokens, sha = _placement(
            capture, runtime, source["ppo_tokens"],
        )
        if max(int(x.numel()) for x in batch["input_ids"].unbind(0)) > effective_sequence_cap:
            raise AssertionError("DP2 input exceeds experimental sequence cap")
        model, hf = make_real_qwen3_tp_model(
            runtime, model_path=checkpoint, tp_size=2, cp_size=2,
            sequence_parallel=False, profile="4b", native_attention=False,
        )
        assert hf.num_hidden_layers == 36
        from .grad_sync_precision import offline_grad_sync_dtype
        grad_sync_dtype = offline_grad_sync_dtype()
        engine = _engine_with_dp_finalize(model, runtime)
        if dist.get_rank() == 0:
            print(
                f"TPR_DP2_GRAD_REDUCE_MODE dtype={grad_sync_dtype} "
                f"weighted_group_size={dist.get_world_size(runtime.dp_cp_group)}",
                flush=True,
            )
        runs = []
        for iteration in range(warmups + repeats):
            model.zero_grad(set_to_none=True)
            dist.barrier()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            result = run_tpr_forward_backward_batch(
                engine, batch, partial(ppo_loss, config=_PPO()), forward_only=False,
            )
            torch.npu.synchronize()
            own_seconds = time.perf_counter() - start

            # Ascend 910B HCCL Reduce does NOT support float64. These
            # world MAX reductions must use FP32 or their outputs may stay
            # rank-local, invalidating speed and peak-memory comparisons.
            critical = torch.tensor(
                own_seconds, dtype=torch.float32, device=runtime.device,
            )
            dist.all_reduce(
                critical, op=dist.ReduceOp.MAX, group=dist.group.WORLD,
            )
            peak = torch.tensor(
                float(torch.npu.max_memory_allocated()),
                dtype=torch.float32, device=runtime.device,
            )
            dist.all_reduce(
                peak, op=dist.ReduceOp.MAX, group=dist.group.WORLD,
            )
            # Engine returns the CP-SUMmed DP-local PPO contribution.
            # DP replicas own different samples, so their local losses
            # SHOULD differ. The global reporting objective is the mean
            # of the eight CP×TP replicated local contributions.
            # Use HCCL-supported FP32 for all_reduce; also all_gather
            # the original FP64-packaged local scalars below and
            # independently check the reduction result on every rank.
            # Neither reporting path contributes to autograd/grad sync.
            local_ppo_tokens = sum(
                int(x.bool().sum()) for x in batch["response_mask"].unbind(0)
            )
            if int(result["metrics"]["tpr/logical_loss_tokens"][0]) != local_ppo_tokens:
                raise AssertionError("TPR forest dropped or duplicated supervised PPO tokens")
            local_loss = float(result["loss"][0])
            global_loss = torch.tensor(
                local_loss, dtype=torch.float32, device=runtime.device,
            )
            dist.all_reduce(
                global_loss, op=dist.ReduceOp.SUM, group=dist.group.WORLD,
            )
            global_loss.div_(dist.get_world_size())

            # All fields have fixed width and are transferred as numeric
            # tensors; do not use all_gather_object with HCCL.
            stats = torch.tensor(
                [float(runtime.tp_rank), float(runtime.cp_rank),
                 float(runtime.dp_rank), local_loss, float(global_loss.item()),
                 float(result["metrics"]["tpr/unique_tree_tokens"][0]),
                 float(local_ppo_tokens)],
                dtype=torch.float64, device=runtime.device,
            )
            received_stats = [torch.empty_like(stats) for _ in range(dist.get_world_size())]
            dist.all_gather(received_stats, stats)
            gathered = []
            for rank_id, values in enumerate(received_stats):
                (tp_rank, cp_rank, dp_rank, rank_loss,
                 rank_global_loss, unique, tokens) = values.cpu().tolist()
                dp_rank = int(dp_rank)
                gathered.append({
                    "rank": rank_id, "tp_rank": int(tp_rank),
                    "cp_rank": int(cp_rank), "dp_rank": dp_rank,
                    "rows": list(plan.partitions[dp_rank]),
                    "local_loss_dp_scaled": rank_loss,
                    "global_loss": rank_global_loss,
                    "local_unique_tree_tokens": int(unique),
                    "local_ppo_tokens": int(tokens),
                    # Rank-zero stage profile is retained locally below;
                    # other ranks' variable-length dictionaries are not sent.
                    "stage_ms": {},
                })
            if profile and dist.get_rank() == 0:
                gathered[0]["stage_ms"] = dict(engine._tpr_stage_profile)
            from .tpr_dp2_loss_consistency import analyze_dp2_loss_report
            loss_report = analyze_dp2_loss_report(gathered)
            if dist.get_rank() == 0:
                print(
                    f"TPR_DP2_LOSS_REPORT step={step} iteration={iteration} "
                    f"dp0_local={loss_report['dp0_local_loss']:.9g} "
                    f"dp1_local={loss_report['dp1_local_loss']:.9g} "
                    f"tp_local_max_gap={loss_report['tp_local_max_gap']:.9g} "
                    f"cp_replica_max_gap={loss_report['cp_replica_max_gap']:.9g} "
                    f"global_reporting_max_gap={loss_report['global_reporting_max_gap']:.9g}",
                    flush=True,
                )
                item = {
                    "iteration": iteration, "warmup": iteration < warmups,
                    "critical_fwd_bwd_s": float(critical.item()),
                    "peak_allocated_gib": float(peak.item()) / 2**30,
                    "loss": loss_report["global_reporting_loss"],
                    "loss_report": loss_report,
                    "ranks": gathered,
                }
                runs.append(item)
                print(
                    f"TPR_TP2CP2DP2 step={step} iter={iteration} "
                    f"warmup={int(iteration < warmups)} "
                    f"critical_fwd_bwd_s={item['critical_fwd_bwd_s']:.6f} "
                    f"peak_gib={item['peak_allocated_gib']:.3f} "
                    f"global_loss={item['loss']:.9g} "
                    f"ppo_logprob_chunk_tokens={lp_chunk_tokens} "
                    f"rows={plan.partitions} costs={plan.tree_tokens_by_rank}",
                    flush=True,
                )
            dist.barrier()

        gradient_witness = None
        replica_policy = os.getenv("TPR_BENCH_GRAD_REPLICA_POLICY", "strict")
        if replica_policy not in ("strict", "diagnostic"):
            raise ValueError(
                "TPR_BENCH_GRAD_REPLICA_POLICY must be 'strict' or 'diagnostic'"
            )
        if os.getenv("TPR_BENCH_GRAD_WITNESS", "0") == "1":
            from .gradient_witness import collect_tp2_gradient_witness
            gradient_witness = collect_tp2_gradient_witness(
                model, runtime, mode="tpr_tp2cp2dp2",
                require_exact_replicas=(replica_policy == "strict"),
            )
        if dist.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=True)
            target = output / f"step{step}_tpr_tp2cp2dp2.json"
            if target.exists():
                raise FileExistsError(target)
            times = [r["critical_fwd_bwd_s"] for r in runs if not r["warmup"]]
            data = {
                "source_step": step, "mode": "tpr",
                "effective_sequence_cap": effective_sequence_cap,
                "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
                "source_capture": str(capture), "checkpoint": str(checkpoint),
                "input_sha256": sha, "global_ppo_tokens": global_tokens,
                "placement_policy": plan.policy,
                "dp_partitions": [list(x) for x in plan.partitions],
                "dp_tree_token_costs": list(plan.tree_tokens_by_rank),
                "global_tree_tokens": plan.global_tree_tokens,
                "duplicated_tree_tokens": plan.duplicated_tree_tokens,
                "equal_rows_per_rank": plan.equal_rows_per_rank,
                "warmups": warmups, "repeats": repeats,
                "capture_logprobs": capture_lp, "profiler_enabled": profile,
                "ppo_logprob_chunk_tokens": lp_chunk_tokens,
                "gradient_sync_reduce_dtype": grad_sync_dtype,
                "hccl_stats_reduce_dtype": "float32",
                "recompute": False, "updated_weights": False,
                "median_fwd_bwd_s": statistics.median(times),
                "gradient_witness": gradient_witness,
                "gradient_replica_policy": replica_policy,
                "gradient_replica_validation_passed": (
                    gradient_witness["replica_validation_passed"]
                    if gradient_witness is not None else None
                ),
                "runs": runs,
            }
            target.write_text(json.dumps(data, indent=2))
            print(
                f"TPR_TP2CP2DP2_SUMMARY step={step} "
                f"median_fwd_bwd_s={data['median_fwd_bwd_s']:.6f} "
                f"dp_rows={data['dp_partitions']} duplicated_tree_tokens="
                f"{plan.duplicated_tree_tokens} output={target}",
                flush=True,
            )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
