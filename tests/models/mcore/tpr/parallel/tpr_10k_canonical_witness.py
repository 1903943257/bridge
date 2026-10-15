"""Topology-independent Qwen3 parameter-gradient and PPO token witnesses.

Canonical gradient coordinates are logical full parameter coordinates, not
8 per *local TP shard*. Exposes TP1/TP2 comparisons while de-duplicating
TP-replicated params and CP/DP replicas. Still sparse: not full-model rel-L2.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.distributed as dist

from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices

SAMPLES = 64


def shard_kind(name: str) -> str:
    if name.endswith(("self_attention.q_layernorm.weight",
                      "self_attention.k_layernorm.weight")):
        # Q/K RMSNorm weight is shared among heads but each TP rank owns
        # a subset of those heads. Its gradient requires a TP SUM.
        return "tp_sum"
    if name.endswith("embedding.word_embeddings.weight"):
        return "rows"
    if name.endswith("self_attention.linear_qkv.weight"):
        return "rows"
    if name.endswith("self_attention.linear_proj.weight"):
        return "cols"
    if name.endswith("mlp.linear_fc1.weight"):
        return "gate_up_rows"
    if name.endswith("mlp.linear_fc2.weight"):
        return "cols"
    return "replicated"


def canonical_shape(name: str, shape: tuple[int, ...], tp: int) -> tuple[int, ...]:
    kind=shard_kind(name)
    if kind in ("rows","gate_up_rows"):
        return (shape[0]*tp,*shape[1:])
    if kind=="cols":
        if len(shape)!=2:
            raise ValueError("Column-sharded weight must be 2D")
        return (shape[0],shape[1]*tp)
    return shape


def owner_and_local_flat(
    name: str, full_index: int, shape: tuple[int,...], tp: int
) -> tuple[int, int]:
    """Map one global logical flattened index to owning TP and local index."""
    kind=shard_kind(name)
    if kind in ("replicated","tp_sum"):
        return 0,full_index
    if kind=="rows":
        block=math.prod(shape)
        return full_index//block,full_index%block
    if kind=="cols":
        rows, local_cols=shape
        full_cols=local_cols*tp
        row,col=divmod(full_index,full_cols)
        owner,local_col=divmod(col,local_cols)
        return owner,row*local_cols+local_col
    if kind=="gate_up_rows":
        local_rows, hidden=shape
        if local_rows%2:
            raise ValueError("SwiGLU gate_up rows must be even")
        row,col=divmod(full_index,hidden)
        gate_chunk=local_rows//2
        gate_global=gate_chunk*tp
        if row < gate_global:
            owner,local_row=divmod(row,gate_chunk)
        else:
            owner,local_up=divmod(row-gate_global,gate_chunk)
            local_row=gate_chunk+local_up
        return owner,local_row*hidden+col
    raise AssertionError(kind)


@torch.no_grad()
def canonical_grad_witness(model, *, tp_size: int, tp_rank: int,
                           cp_rank: int, dp_rank: int, device) -> dict | None:
    """All TP shards contribute once, all other replicas must match exactly."""
    if tp_size not in (1,2):
        raise ValueError("10K parity supports TP=1/2 only")
    names=[]; shapes=[]; offsets=[]; counts=[]
    pieces=[]
    norm_pieces=[]
    norm_names=[]
    norm_shapes=[]
    tp_sum_slices=[]
    offset=0
    for name,param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        local_shape=tuple(param.grad.shape)
        logical_shape=canonical_shape(name,local_shape,tp_size)
        indices=exact_sample_indices(math.prod(logical_shape),SAMPLES)
        local_pos=[]; output_pos=[]
        for i,index in enumerate(indices):
            owner,loc=owner_and_local_flat(name,index,local_shape,tp_size)
            if owner==tp_rank or shard_kind(name)=="tp_sum":
                local_pos.append(loc)
                output_pos.append(i)
        chunk=torch.zeros((len(indices),),dtype=torch.float32,device=device)
        if local_pos:
            picks=param.grad.detach().reshape(-1).index_select(
                0,torch.tensor(local_pos,dtype=torch.int64,device=device)
            ).float()
            chunk[torch.tensor(output_pos,dtype=torch.int64,device=device)]=picks
        if "layernorm" in name or name.endswith(".norm.weight"):
            # Small complete Norm tensors, unlike huge GEMM weights, can
            # be captured in full for an actual Norm-only L2 comparison.
            whole=param.grad.detach().reshape(-1).float()
            # Q/K TP replicas have per-head partial gradients that must be
            # summed; other Norm params are TP replicas counted once.
            if cp_rank!=0 or dp_rank!=0 or (
                tp_rank!=0 and shard_kind(name)!="tp_sum"
            ):
                whole=torch.zeros_like(whole)
            norm_pieces.append(whole)
            norm_names.append(name)
            norm_shapes.append(list(logical_shape))
        names.append(name)
        shapes.append(list(logical_shape))
        offsets.append(offset)
        counts.append(len(indices))
        if shard_kind(name)=="tp_sum":
            tp_sum_slices.append((name,offset,offset+len(indices)))
        offset+=len(indices)
        pieces.append(chunk)
    local=torch.cat(pieces)
    # Compare same TP shard's local contributions among ALL CP/DP peers;
    # an existing discrepancy must not disappear after choosing a representative.
    states=torch.tensor(
        [float(tp_rank),float(cp_rank),float(dp_rank)],
        device=device,dtype=torch.float32,
    )
    all_states=[torch.empty_like(states) for _ in range(dist.get_world_size())]
    all_local=[torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(all_states,states)
    dist.all_gather(all_local,local)
    all_cpu=[x.cpu() for x in all_local]
    topology=[tuple(map(int,x.cpu().tolist())) for x in all_states]
    max_replica_abs=0.0
    for i,(t,c,d) in enumerate(topology):
        if c==0 and d==0:
            for j,(tj,cj,dj) in enumerate(topology):
                if tj==t:
                    max_replica_abs=max(
                        max_replica_abs,
                        float((all_cpu[i]-all_cpu[j]).abs().max().item()),
                    )
    if max_replica_abs:
        raise AssertionError(
            f"10K gradient replicas mismatch after finalize: {max_replica_abs}"
        )
    # A TP-shared Q/K Norm parameter can have different local gradients
    # because each rank processes different heads. Report raw TP difference
    # separately; the canonical *mathematical* gradient sums these
    # contributions, even though this offline finalize does not add a TP
    # synchronization to the model's actual .grad state.
    tp_sum_sample_max_abs=0.0
    for _,start,end in tp_sum_slices:
        refs={
            t:all_cpu[i][start:end]
            for i,(t,c,d) in enumerate(topology) if c==0 and d==0
        }
        if len(refs)==2:
            tp_sum_sample_max_abs=max(
                tp_sum_sample_max_abs,
                float((refs[0]-refs[1]).abs().max().item()),
            )
    if cp_rank!=0 or dp_rank!=0:
        local.zero_()
    dist.all_reduce(local,op=dist.ReduceOp.SUM)
    if not norm_pieces:
        raise AssertionError("No complete Norm gradients captured")
    norm_vector=torch.cat(norm_pieces)
    dist.all_reduce(norm_vector,op=dist.ReduceOp.SUM)
    if dist.get_rank()!=0:
        return None
    arr=local.cpu().tolist()
    norm_values=norm_vector.cpu().tolist()
    if not all(math.isfinite(x) for x in arr):
        raise ValueError("Nonfinite sampled parameter gradients")
    return {
        "names":names,"canonical_shapes":shapes,
        "samples_per_name":counts,
        "samples_per_parameter":SAMPLES,
        "values":arr,"replica_max_abs":max_replica_abs,
        "complete_norm_gradients":{
            "names":norm_names,
            "shapes":norm_shapes,
            "values":norm_values,
            "elements":len(norm_values),
            "note":(
                "EXACT complete Norm parameter gradients only; "
                "TP shared Q/K partial head grads summed across TP in witness. "
                "No optimizer update or physical TP sync is applied."
            ),
        },
        "qk_norm_tp_local_gradient_max_abs":tp_sum_sample_max_abs,
        "qk_norm_tp_sum_applied_to_witness":True,
        "qk_norm_tp_sum_applied_to_model_grads":False,
        "sampling":(
            "64 integer-strided logical full-parameter positions per parameter; "
            "TP-sharded weights are mapped to canonical TP1 coordinates, "
            "TP-replicated parameters counted once except Q/K RMSNorm, "
            "whose per-head TP contributions are summed in the witness only. "
            "Actual model .grad is unmodified. NOT a full-gradient metric."
        ),
    }


@torch.no_grad()
def gather_logprobs(engine, *, local_rows, total_tokens: int, device) -> list | None:
    """Gather (original_row_id, response_offset, new_logprob) once per token."""
    # Gather only TP=0; across CP ranks exactly one owns each query token.
    # DP replicas own different source trajectories.
    from megatron.core import parallel_state
    if parallel_state.get_tensor_model_parallel_rank()==0:
        source=engine._tpr_captured_log_probs
        local=[
            (int(local_rows[local_row]),int(offset),float(lp))
            for (local_row,offset),lp in source.items()
        ]
    else:
        local=[]
    n=torch.tensor([len(local)],dtype=torch.int64,device=device)
    sizes=[torch.empty_like(n) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes,n)
    max_size=max(int(x.item()) for x in sizes)
    padded=torch.zeros((max_size,3),dtype=torch.float64,device=device)
    if local:
        padded[:len(local)]=torch.tensor(local,dtype=torch.float64,device=device)
    buffers=[torch.empty_like(padded) for _ in sizes]
    dist.all_gather(buffers,padded)
    if dist.get_rank()!=0:
        return None
    found={}
    for buf,size in zip(buffers,sizes,strict=True):
        for row,offset,value in buf[:int(size.item())].cpu().tolist():
            key=(int(row),int(offset))
            if key in found:
                raise AssertionError(f"Duplicate CP/DP token owner: {key}")
            found[key]=float(value)
    if len(found)!=total_tokens:
        raise AssertionError(
            f"Missing PPO logprob refs: found={len(found)}, expected={total_tokens}"
        )
    if not all(math.isfinite(v) for v in found.values()):
        raise ValueError("Nonfinite PPO logprob")
    return [[*key,found[key]] for key in sorted(found)]
