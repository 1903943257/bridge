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
def collect_tp2_gradient_witness(model, runtime, *, samples_per_parameter: int = 8):
    if not dist.is_initialized() or dist.get_world_size() != 8:
        raise RuntimeError("TP2 gradient witness requires eight ranks")
    if dist.get_world_size(runtime.tp_group) != 2:
        raise AssertionError("TP2 gradient witness requires TP=2")
    names = []
    chunks = []
    shapes = []
    for name, param in model.named_parameters():
        if param.grad is None:
            raise AssertionError(f"No gradient available for witness: {name}")
        names.append(name)
        shapes.append(list(param.shape))
        indices = exact_sample_indices(param.numel(), samples_per_parameter)
        idx = torch.tensor(indices, device=param.grad.device, dtype=torch.long)
        chunks.append(param.grad.detach().reshape(-1).index_select(0, idx).float())
    if not chunks:
        raise AssertionError("Empty sampled gradient witness")
    samples = torch.cat(chunks)
    cpdp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    # CP/DP parameter copies must agree exactly after weight-gradient finalization.
    low, high = samples.clone(), samples.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=cpdp)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=cpdp)
    local_error = (high - low).abs().max()
    dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
    error = float(local_error.item())
    if error > 0:
        raise AssertionError(
            f"Replicated parameter gradients differ after CP/DP finalize: "
            f"sample_max_abs={error:.9g}"
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
        "samples_per_name": [min(int(torch.tensor(s).prod().item()), samples_per_parameter) for s in shapes],
        "tp_shards": values,
        "replica_sample_max_abs": error,
    }
