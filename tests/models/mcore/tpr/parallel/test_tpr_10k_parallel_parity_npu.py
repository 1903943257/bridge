"""10K controlled real-label PPO correctness: CP/DP/TP ablation on NPU.

Each torchrun executes exactly one topology. The same 10K capture, input
fingerprint, fixed PPO labels, Qwen3-4B initial weights and TPR backend are
used. 'unshared' uses one complete trajectory per tree via the existing
loader's synthetic nonsharing keys, NOT the native Megatron Full schedule.

No optimizer, no performance claim, no repaired PPO labels.
"""
from __future__ import annotations

import json
import os
import sys
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
from .grad_sync_precision import offline_grad_sync_dtype,reduce_average_weight_grad
from .test_real_actor_tp2cp4_offline_ppo_npu import _PPO,_load_real_pair
from .tpr_10k_canonical_witness import canonical_grad_witness,gather_logprobs

pytestmark=pytest.mark.skipif(
    os.getenv("TPR_RUN_10K_PARALLEL_PARITY")!="1",
    reason="Set TPR_RUN_10K_PARALLEL_PARITY=1 in isolated NPU torchrun",
)


def _runtime(tp: int,cp: int,dp: int):
    world=tp*cp*dp
    if int(os.getenv("WORLD_SIZE","1"))!=world:
        raise AssertionError(f"Expected {world} distributed ranks for TP{tp} CP{cp} DP{dp}")
    import torch_npu  # noqa: F401
    from mindspeed.args_utils import get_full_args
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    argv=sys.argv[:]
    try:
        sys.argv[:]=[sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:]=argv
    repatch({
        "tensor_model_parallel_size":tp,
        "context_parallel_size":cp,
        "pipeline_model_parallel_size":1,
        "context_parallel_algo":"megatron_cp_algo",
        "sequence_parallel":False,
        "use_flash_attn":True,
    })
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    t=parallel_state.get_tensor_model_parallel_group()
    c=parallel_state.get_context_parallel_group()
    d=parallel_state.get_data_parallel_group()
    dc=parallel_state.get_data_parallel_group(with_context_parallel=True)
    ranks=(parallel_state.get_tensor_model_parallel_rank(),
           parallel_state.get_context_parallel_rank(),
           parallel_state.get_data_parallel_rank())
    group_specs=(("tp",t,tp),("cp",c,cp),("dp",d,dp),("dp_cp",dc,cp*dp))
    def _size(group, expected_size):
        if group is None:
            if expected_size != 1:
                raise AssertionError("Missing process group for nontrivial parallel dimension")
            return 1
        return dist.get_world_size(group)
    actual=tuple(_size(group,n) for _,group,n in group_specs)
    if actual!=(tp,cp,dp,cp*dp):
        raise AssertionError(f"Wrong Megatron group sizes {actual} for {(tp,cp,dp)}")
    dev=torch.device("npu",local_rank)
    coords=torch.tensor([*ranks],dtype=torch.int64,device=dev)
    gathered=[torch.empty_like(coords) for _ in range(world)]
    dist.all_gather(gathered,coords)
    layout=[tuple(int(v) for v in x.cpu().tolist()) for x in gathered]
    if len(set(layout))!=world:
        raise AssertionError(f"Duplicate group coordinate layout: {layout}")
    expected={
        "tp": {i for i,(tt,cc,dd) in enumerate(layout) if (cc,dd)==ranks[1:]},
        "cp": {i for i,(tt,cc,dd) in enumerate(layout) if (tt,dd)==(ranks[0],ranks[2])},
        "dp": {i for i,(tt,cc,dd) in enumerate(layout) if (tt,cc)==ranks[:2]},
        "dp_cp": {i for i,(tt,cc,dd) in enumerate(layout) if tt==ranks[0]},
    }
    for name,group,n in group_specs:
        got=({dist.get_rank()} if group is None else
             set(dist.get_process_group_ranks(group)))
        if got!=expected[name]:
            raise AssertionError(f"{name} group wrong {got} != {expected[name]}")
        if len(got)>1:
            value=torch.tensor(float(dist.get_rank()+1),dtype=torch.float32,device=dev)
            dist.all_reduce(value,op=dist.ReduceOp.SUM,group=group)
            torch.npu.synchronize()
            if float(value.item())!=sum(j+1 for j in got):
                raise AssertionError(f"{name} HCCL group reduction failed")
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(261013)
    if dist.get_rank()==0:
        print(
            f"TPR_10K_PROCESS_GROUP_AUDIT tp={tp} cp={cp} dp={dp} "
            f"world={world} layout={layout} HCCL_DETERMINISTIC="
            f"{os.getenv('HCCL_DETERMINISTIC')}",
            flush=True,
        )
    return SimpleNamespace(
        tp_group=t,cp_group=c,dp_group=d,dp_cp_group=dc,
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        tp_rank=ranks[0],cp_rank=ranks[1],dp_rank=ranks[2],
        rank=ranks[0],device=dev,
    )


def _engine(model,runtime,tp,cp,dp):
    model.config.no_sync_func=None
    model.config.grad_scale_func=None
    model.config.calculate_per_token_loss=False
    model.config.recompute_granularity=None
    reduce_dtype=offline_grad_sync_dtype()

    def finalize(chunks,num_tokens,*,force_all_reduce=False,**kwargs):
        if chunks!=[model] or num_tokens is not None or not force_all_reduce:
            raise AssertionError("Unexpected TPR Forest gradient finalization")
        for name,param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"No gradient for {name}")
            if cp*dp>1:
                reduce_average_weight_grad(
                    param,runtime.dp_cp_group,
                    name=name,reduce_dtype=reduce_dtype,
                )
        # No collective required for CP=DP=1: each local gradient is
        # already on a unique model replica after its TP path.
    model.config.finalize_model_grads_func=finalize
    return SimpleNamespace(
        module=[model],
        engine_config=SimpleNamespace(
            tpr_enabled=True,tensor_model_parallel_size=tp,
            context_parallel_size=cp,pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,virtual_pipeline_model_parallel_size=None,
            dynamic_context_parallel=False,
            tpr_cp_backend="ring" if cp>1 else "auto",
            use_fused_kernels=False,override_transformer_config={},
        ),
        model_config=SimpleNamespace(mtp=SimpleNamespace(enable=False)),
        enable_routing_replay=False,
        get_data_parallel_size=lambda:dp,
        get_data_parallel_group=lambda:runtime.dp_group,
    )


def _load(capture: Path,runtime,mode: str,dp:int):
    all_batch,all_keys,global_tokens,global_sha=_load_real_pair(
        capture,tuple(range(8)), "tpr"
    )
    # Retaining TPR prefix sharing in the cropped 10K source matters:
    # otherwise an 'isolated TP/CP/DP TPR test' degenerates to eight Full rows.
    from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
    full_forest=build_tree_execution_plans(
        all_keys,all_batch,
        tree_policy=os.getenv("TPR_TREE_POLICY","qwen_turn"),
        turn_min_segment_tokens=int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS","128")),
    )
    logical_full=sum(int(x.numel()) for x in all_batch["input_ids"].unbind())
    unique_full=sum(
        segment.length for tree in full_forest.trees
        for segment in tree.segment_plan.segments.values()
    )
    if unique_full>=logical_full:
        raise AssertionError(
            "The controlled 10K source has no actual TPR Prefix reuse: "
            f"logical={logical_full} unique={unique_full}. "
            "Do not treat a no-sharing dataset as a TPR correctness gate."
        )
    if dp==1:
        partitions=[list(range(8))]
    else:
        seqs=[list(map(int,x.tolist())) for x in all_batch["input_ids"].unbind()]
        plan=plan_tpr_dta_dp(
            seqs,all_keys,dp_size=dp,enforce_equal_rows=False,
        )
        partitions=[list(p) for p in plan.partitions]
    if sorted(row for part in partitions for row in part)!=list(range(8)):
        raise AssertionError("DP forest placement lost/duplicated tokens")
    local_rows=partitions[runtime.dp_rank]
    # The loader produces synthetic independent roots in unshared mode.
    batch,selected_keys,local_tokens,sha=_load_real_pair(
        capture,tuple(local_rows),"full" if mode=="unshared" else "tpr",
    )
    if tuple(all_keys[i] for i in local_rows)!=selected_keys:
        raise AssertionError("Batch source row selection changed")
    tu.assign_non_tensor(
        batch,dp_size=dp,batch_num_tokens=global_tokens,
        global_batch_size=8,tpr_capture_log_probs=True,
    )
    if dp==1 and local_tokens!=global_tokens:
        raise AssertionError("DP1 token denominator mismatch")
    # DP group sample partitions are globally auditable: one representative
    # per DP rank, no HCCL Python-object collective.
    report=torch.tensor(
        [runtime.dp_rank, sum(1<<int(x) for x in local_rows),
         local_tokens,int(global_sha[:15],16)],
        dtype=torch.int64,device=runtime.device,
    )
    peers=[torch.empty_like(report) for _ in range(dist.get_world_size())]
    dist.all_gather(peers,report)
    decoded=[tuple(int(v) for v in x.cpu().tolist()) for x in peers]
    for d in range(dp):
        entries=[x for x in decoded if x[0]==d]
        expected=(d,sum(1<<int(x) for x in partitions[d]),
                  next(x[2] for x in entries),int(global_sha[:15],16))
        if len(entries)!=runtime.world_replicas or any(x!=expected for x in entries):
            raise AssertionError(f"Invalid DP{d} membership/sample ownership: {entries}")
    if sum(next(x[2] for x in decoded if x[0]==d) for d in range(dp))!=global_tokens:
        raise AssertionError("DP shards do not cover exact same global tokens")
    return batch,local_rows,partitions,global_tokens,global_sha,{
        "global_logical_tokens":logical_full,
        "global_unique_tree_tokens":unique_full,
        "global_reuse_ratio":logical_full/unique_full,
    }


def test_10k_parallel_factorial_one_topology():
    tp=int(os.getenv("TPR_10K_TP","2"))
    cp=int(os.getenv("TPR_10K_CP","1"))
    dp=int(os.getenv("TPR_10K_DP","1"))
    mode=os.getenv("TPR_10K_MODE","tpr")
    if (tp,cp,dp) not in (
        (2,1,1),(2,2,1),(2,1,2),(2,2,2)
    ):
        raise ValueError("Unsupported isolated 10K parity topology")
    if mode not in ("tpr","unshared"):
        raise ValueError("TPR_10K_MODE must be tpr or unshared")
    if os.getenv("TPR_10K_PARITY_ONLY")!="1":
        raise RuntimeError("10K parity-only gate not enabled")
    if os.getenv("HCCL_DETERMINISTIC")!="true":
        raise RuntimeError("HCCL_DETERMINISTIC=true required for BF16 replica validity")
    cap=Path(os.environ["TPR_10K_CAPTURE"]).resolve()
    output_dir=Path(os.environ["TPR_10K_OUTPUT"]).resolve()
    checkpoint=Path(os.getenv("TPR_QWEN_4B_PATH","/workspace/hf_models/Qwen3-4B-128K"))
    runtime=_runtime(tp,cp,dp)
    runtime.world_replicas=tp*cp
    try:
        from verl.workers.utils.losses import ppo_loss
        batch,local_rows,partitions,global_tokens,sha,reuse=_load(
            cap,runtime,mode,dp,
        )
        if max(int(x.numel()) for x in batch["input_ids"].unbind())>10240:
            raise AssertionError("10K window was not applied")
        model,hf=make_real_qwen3_tp_model(
            runtime,model_path=checkpoint,tp_size=tp,cp_size=cp,
            sequence_parallel=False,profile="4b",
            native_attention=False,
        )
        if hf.num_hidden_layers!=36:
            raise AssertionError("Must use Qwen3-4B 36-layer checkpoint")
        engine=_engine(model,runtime,tp,cp,dp)
        model.zero_grad(set_to_none=True)
        dist.barrier()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        result=run_tpr_forward_backward_batch(
            engine,batch,partial(ppo_loss,config=_PPO()),
            forward_only=False,
        )
        torch.npu.synchronize()
        loss_local=float(result["loss"][0])
        loss=torch.tensor(loss_local,device=runtime.device,dtype=torch.float32)
        dist.all_reduce(loss,op=dist.ReduceOp.SUM)
        loss.div_(dist.get_world_size())
        # All ranks participate; audits gradients before selecting one TP
        # shard and one CP/DP representative for canonical comparison.
        grads=canonical_grad_witness(
            model,tp_size=tp,tp_rank=runtime.tp_rank,
            cp_rank=runtime.cp_rank,dp_rank=runtime.dp_rank,
            device=runtime.device,
        )
        tokens=gather_logprobs(
            engine,local_rows=local_rows,total_tokens=global_tokens,
            device=runtime.device,
        )
        # Every case reads the same exact controlled 10K input bytes.
        if dist.get_rank()==0:
            if grads is None or tokens is None:
                raise AssertionError("Rank0 missing canonical witness")
            if output_dir.exists():
                raise FileExistsError(f"Refusing overwrite {output_dir}")
            output_dir.mkdir(parents=True)
            data={
                "mode":mode,"source_capture":str(cap),
                "input_sha256":sha,"checkpoint":str(checkpoint),
                "topology":{"tp":tp,"cp":cp,"dp":dp,"world":tp*cp*dp},
                "dp_partitions":partitions,
                "global_tpr_reuse_audit":reuse,
                "global_ppo_tokens":global_tokens,
                "global_loss":float(loss.item()),
                "gradient_sync_reduce_dtype":offline_grad_sync_dtype(),
                "hccl_deterministic":"true",
                "gradient_witness":grads,
                "token_logprobs":tokens,
                "memory_peak_allocated_gib_rank0": (
                    torch.npu.max_memory_allocated()/2**30
                ),
                "timed_scope":"not_benchmarked",
                "updated_weights":False,
                "capture_logprobs":True,
                "note":(
                    "Controlled fixed-label 10K parity; may truncate original "
                    "left prompt, not an online PPO equivalence claim"
                ),
            }
            dest=output_dir / "result.json"
            dest.write_text(json.dumps(data,indent=2))
            print(
                f"TPR_10K_PARITY_RESULT mode={mode} tp={tp} cp={cp} dp={dp} "
                f"ppo_tokens={global_tokens} global_loss={loss.item():.9g} "
                f"replica_grad_max_abs={grads['replica_max_abs']} "
                f"gradient_samples={len(grads['values'])} "
                f"logprob_tokens={len(tokens)} output={dest}",
                flush=True,
            )
        dist.barrier()
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
