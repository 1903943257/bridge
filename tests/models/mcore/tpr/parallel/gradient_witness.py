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
    # Read-only witness: gather the ACTUAL post-finalize gradients rather
    # than relying on additional HCCL MIN/MAX reductions to determine their
    # equality. This also reveals the exact values held by every CP/DP rank.
    # FP32 only converts the original BF16 sample losslessly.
    group_members = tuple(dist.get_process_group_ranks(cpdp))
    peers = [torch.empty_like(samples) for _ in group_members]
    dist.all_gather(peers, samples, group=cpdp)
    peer_values = torch.stack(peers, dim=0).cpu()
    low = peer_values.min(dim=0).values
    high = peer_values.max(dim=0).values
    spread = (high - low).abs()
    local_max, worst_idx_tensor = spread.max(dim=0)
    worst_sample = int(worst_idx_tensor.item())
    local_relative = float(
        (torch.linalg.vector_norm(spread.float()) /
         torch.linalg.vector_norm(high.float()).clamp_min(1e-30)).item()
    )
    differing_samples = int((spread != 0).sum().item())
    exact_local = all(torch.equal(peer_values[0], x) for x in peer_values[1:])
    worst_peer_values = [
        [int(rank), float(peer_values[i, worst_sample].item())]
        for i, rank in enumerate(group_members)
    ]
    offset = 0
    worst_param = "<unknown>"
    worst_param_index = -1
    for param_index, (name, count) in enumerate(zip(names, sample_sizes, strict=True)):
        if offset <= worst_sample < offset + count:
            worst_param = name
            worst_param_index = param_index
            break
        offset += count

    # For a small offending tensor (e.g. Qwen RMSNorm, 2560 elements),
    # also gather the ENTIRE original BF16 gradient. This distinguishes
    # sampled-gather artifacts from genuine one-ULP parameter drift.
    focus_grad = None
    for name, param in model.named_parameters():
        if name == worst_param and param.numel() <= 8192:
            focus_grad = param.grad.detach().reshape(-1).contiguous()
            break
    focus_summary = None
    if focus_grad is not None:
        focus_peers = [torch.empty_like(focus_grad) for _ in group_members]
        dist.all_gather(focus_peers, focus_grad, group=cpdp)
        focus_cpu = torch.stack(focus_peers, dim=0).float().cpu()
        focus_span = focus_cpu.max(dim=0).values - focus_cpu.min(dim=0).values
        focus_worst_value, focus_worst_pos = focus_span.abs().max(dim=0)
        focus_pos = int(focus_worst_pos.item())
        focus_summary = {
            "parameter": worst_param,
            "dtype": str(focus_grad.dtype),
            "elements": int(focus_grad.numel()),
            "differing_elements": int((focus_span != 0).sum().item()),
            "max_abs": float(focus_worst_value.item()),
            "worst_param_offset": focus_pos,
            "worst_peer_values": [
                [int(rank), float(focus_cpu[i, focus_pos].item())]
                for i, rank in enumerate(group_members)
            ],
        }
        if dist.get_rank() == 0 and focus_summary["differing_elements"]:
            print(
                f"TPR_GRAD_FULL_PARAM_GATHER_DIAG mode={mode} "
                f"cpdp_group={group_members} detail={focus_summary}",
                flush=True,
            )

    # Fixed-width WORLD all-gather only; no unsupported FP64 AllReduce.
    state = torch.tensor(
        [dist.get_rank(), runtime.tp_rank, runtime.cp_rank,
         getattr(runtime, "dp_rank", 0), float(local_max.item()),
         local_relative, float(worst_sample), float(worst_param_index),
         float(differing_samples), float(int(exact_local))],
        device=samples.device, dtype=torch.float32,
    )
    all_states = [torch.empty_like(state) for _ in range(dist.get_world_size())]
    dist.all_gather(all_states, state)
    rank_states = [x.cpu().tolist() for x in all_states]
    error = max(x[4] for x in rank_states)
    if error != 0.0 and dist.get_rank() == 0:
        affected = [
            f"rank={int(x[0])}/tp={int(x[1])}/cp={int(x[2])}/dp={int(x[3])}:"
            f"max_abs={x[4]:.9g},rel_l2={x[5]:.9g},"
            f"changed_samples={int(x[8])},"
            f"param={names[int(x[7])] if 0 <= int(x[7]) < len(names) else '<unknown>'},"
            f"sample_idx={int(x[6])}"
            for x in rank_states if x[4] > 0
        ]
        print(
            f"TPR_GRAD_REPLICA_GATHER_DIAG mode={mode} "
            f"require_exact={int(require_exact_replicas)} "
            f"world_max_abs={error:.9g} "
            f"rank0_cpdp_group={group_members} "
            f"rank0_worst_peer_values={worst_peer_values} "
            f"affected={'; '.join(affected)}",
            flush=True,
        )
    if error != 0.0 and require_exact_replicas:
        raise AssertionError(
            f"TPR replica gradient mismatch confirmed by raw AllGather in mode={mode}: "
            f"world_sample_max_abs={error:.9g}, rank={dist.get_rank()}, "
            f"cpdp_group={group_members}, local_worst_param={worst_param}, "
            f"local_worst_sample={worst_sample}, "
            f"local_worst_peer_values={worst_peer_values}, "
            f"local_differing_samples={differing_samples}, "
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
        "replica_validation_passed": error == 0,
        "replica_diagnostics": {
            "method": "FP32 AllGather of original post-finalize BF16 gradient samples",
            "world_rank_states": rank_states,
            "cpdp_group_rank0": list(group_members),
            "rank0_worst_peer_values": worst_peer_values,
            "rank0_worst_param": worst_param,
            "rank0_differing_samples": differing_samples,
            "rank0_focused_parameter": focus_summary,
        },
    }
