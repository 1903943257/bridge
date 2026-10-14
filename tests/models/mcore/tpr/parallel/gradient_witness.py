"""Small deterministic post-timing gradient witness for TP2/CP/DP benchmarks.

Do not confuse sampled gradient parity with a full-parameter correctness gate.
All communications and CPU materialization happen AFTER F/B timing.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from megatron.core import parallel_state

from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices


@torch.no_grad()
def collect_tp2_gradient_witness(
    model, runtime, *, samples_per_parameter: int = 8,
    mode: str = "unknown", require_exact_replicas: bool = True,
):
    if not dist.is_initialized() or dist.get_world_size() != 8:
        raise RuntimeError("TP2 gradient witness requires eight ranks")
    if dist.get_world_size(runtime.tp_group) != 2:
        raise AssertionError("TP2 gradient witness requires TP=2")
    names = []
    chunks = []
    shapes = []
    sample_sizes = []
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient available for witness: {name}")
        names.append(name)
        shapes.append(list(param.shape))
        indices = exact_sample_indices(param.numel(), samples_per_parameter)
        sample_sizes.append(len(indices))
        idx = torch.tensor(indices, device=param.grad.device, dtype=torch.long)
        chunks.append(param.grad.detach().reshape(-1).index_select(0, idx).float())
    if not chunks:
        raise AssertionError("Empty sampled gradient witness")
    samples = torch.cat(chunks)
    cpdp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    # MIN/MAX on FP32 samples is a *diagnostic* of finalized replica
    # consistency, not part of training or performance timing.
    low, high = samples.clone(), samples.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=cpdp)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=cpdp)
    spread = (high - low).abs()
    local_max, worst_idx = spread.max(dim=0)
    local_relative = float(
        (torch.linalg.vector_norm(spread.float()) /
         torch.linalg.vector_norm(high.float()).clamp_min(1e-30)).item()
    )
    offset = 0
    worst_param = "<unknown>"
    worst_param_index = -1
    worst_sample = int(worst_idx.item())
    for param_index, (name, count) in enumerate(zip(names, sample_sizes, strict=True)):
        if offset <= worst_sample < offset + count:
            worst_param = name
            worst_param_index = param_index
            break
        offset += count
    group_members = tuple(
        dist.get_global_rank(cpdp, i) for i in range(dist.get_world_size(cpdp))
    )
    # Gather a fixed-size diagnostics vector across all eight ranks so the
    # earliest failing rank and TP shard are visible in the error.
    state = torch.tensor(
        [dist.get_rank(), runtime.tp_rank, runtime.cp_rank,
         getattr(runtime, "dp_rank", 0), float(local_max.item()),
         local_relative, float(worst_sample), float(worst_param_index)],
        device=samples.device, dtype=torch.float64,
    )
    all_states = [torch.empty_like(state) for _ in range(dist.get_world_size())]
    dist.all_gather(all_states, state)
    rank_states = [x.cpu().tolist() for x in all_states]
    error = max(x[4] for x in rank_states)
    if error != 0.0 and dist.get_rank() == 0:
        affected = [
            f"rank={int(x[0])}/tp={int(x[1])}/cp={int(x[2])}/dp={int(x[3])}:"
            f"max_abs={x[4]:.9g},rel_l2={x[5]:.9g},"
            f"param={names[int(x[7])] if 0 <= int(x[7]) < len(names) else '<unknown>'},"
            f"sample_idx={int(x[6])}"
            for x in rank_states if x[4] > 0
        ]
        print(
            f"TPR_GRAD_REPLICA_DIAG mode={mode} "
            f"require_exact={int(require_exact_replicas)} "
            f"world_max_abs={error:.9g} "
            f"rank0_cpdp_group={group_members} "
            f"affected={'; '.join(affected)}",
            flush=True,
        )
    # Full CP4 is the *reference*, not evidence that DP2 failed.
    # Preserve its tiny replica drift in the output rather than silently
    # passing an exactness check or aborting before the DP2 job can execute.
    # TPR DP2 retains its original strict sync correctness gate.
    if error != 0.0 and require_exact_replicas:
        raise AssertionError(
            f"TPR replica gradient mismatch in mode={mode}: "
            f"world_sample_max_abs={error:.9g}, rank={dist.get_rank()}, "
            f"cpdp_group={group_members}, local_worst_param={worst_param}, "
            f"local_worst_sample={int(worst_idx.item())}, "
            f"local_rel_l2={local_relative:.9g}"
        )

    # Exactly one CP0/DP0 representative from each TP shard contributes.
    witness = samples.new_zeros((2, samples.numel()))
    if runtime.cp_rank == 0 and getattr(runtime, "dp_rank", 0) == 0:
        witness[runtime.tp_rank].copy_(samples)
    dist.all_reduce(witness, op=dist.ReduceOp.SUM)
    if dist.get_rank() != 0:
        return None
    values = witness.cpu().tolist()
    return {
        "samples_per_parameter": samples_per_parameter,
        "names": names,
        "shapes": shapes,
        "samples_per_name": sample_sizes,
        "tp_shards": values,
        "replica_sample_max_abs": error,
        "replica_exact": error == 0,
        "replica_consistency_enforced": require_exact_replicas,
        "replica_diagnostics": {
            "world_rank_states": rank_states,
            "cpdp_group_rank0": list(group_members),
        },
    }
