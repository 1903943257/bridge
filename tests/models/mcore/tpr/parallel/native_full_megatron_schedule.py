"""Native Megatron schedule control for the fixed-trajectory TP2/CP4 PPO replay.

This is *not* the TPR unshared-forest path. It uses an unmodified Megatron
GPT attention module, VERL's THD/zigzag-CP preprocessing and PPO/log-prob
functions, and Megatron's get_forward_backward_func() schedule. The existing
offline CPU FP32-master AdamW is deliberately shared with the TPR experiment.

One original trajectory is one native Megatron microbatch. The eight losses
use the same 8-trajectory global PPO-token denominator, and the schedule's
1 / num_microbatches scaling is explicitly canceled. We average finalized
grads over CP replicas once, as in the existing offline experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


@dataclass
class NativeFullResult:
    loss: float
    metrics: dict
    logprobs: dict
    microbatches: int
    finalize_seconds: float


def native_cp_query_indices(length: int, cp_rank: int, cp_size: int = 4,
                            tp_size: int = 2) -> tuple[int, ...]:
    """Exact zigzag physical→logical position order used by preprocess_thd_engine.

    This mapping must NOT be replaced by contiguous CP slicing: the native
    MindSpeed causal Ring layout assigns two symmetric chunks per CP rank.
    """
    if not (0 <= cp_rank < cp_size) or length <= 0:
        raise ValueError("invalid length/CP rank")
    align = tp_size * cp_size * 2
    padded = ((length + align - 1) // align) * align
    half = padded // (2 * cp_size)
    return tuple(range(cp_rank * half, (cp_rank + 1) * half)) + tuple(
        range(padded - (cp_rank + 1) * half, padded - cp_rank * half)
    )


def _compact_native_ppo(loss_function, logprobs, response_offsets, batch, row,
                        global_tokens: int, device, dp_group):
    """Call the *same* VERL PPO loss as the TPR objective adapter.

    A one-token bookkeeping prompt is only for VERL's packed logprob unpadding;
    it is never fed into the model. All original PPO labels are retained.
    """
    n = len(response_offsets)
    if n == 0:
        raise ValueError("native PPO compact batch requires nonempty refs")
    offsets = torch.tensor(response_offsets, dtype=torch.long,
                           device=batch["old_log_probs"][row].device)
    old = batch["old_log_probs"][row].index_select(0, offsets).to(device=device, dtype=logprobs.dtype)
    adv = batch["advantages"][row].index_select(0, offsets).to(device=device, dtype=logprobs.dtype)
    fields = {
        "prompts": torch.zeros((1, 1), dtype=torch.long, device=device),
        "responses": batch["responses"][row].index_select(0, offsets).to(device).unsqueeze(0),
        "attention_mask": torch.ones((1, n + 1), dtype=torch.long, device=device),
        "response_mask": torch.ones((1, n), dtype=torch.bool, device=device),
        "old_log_probs": old.unsqueeze(0),
        "advantages": adv.unsqueeze(0),
    }
    for key in ("ref_log_prob", "rollout_is_weights"):
        if key in batch:
            fields[key] = batch[key][row].index_select(0, offsets).to(device=device, dtype=logprobs.dtype).unsqueeze(0)
    compact = TensorDict(fields, batch_size=[1], device=device)
    tu.assign_non_tensor(compact, dp_size=1, batch_num_tokens=global_tokens,
                         global_batch_size=len(batch["input_ids"]))
    # TPR also appends one unused terminal dummy to match no_padding_2_padding.
    logits_for_loss = torch.cat((logprobs, logprobs.new_zeros(1)))
    return loss_function(
        model_output={"log_probs": logits_for_loss},
        data=compact, dp_group=dp_group,
    )


def run_native_full_step(model, batch, runtime, loss_function, *,
                         capture_logprobs: bool) -> NativeFullResult:
    """Execute real no-PP Megatron F/B schedule over eight CP-sharded rows.

    Deliberately fails rather than silently falling back to SegmentExecutor.
    Native model and schedule versions need an 8-NPU integration gate.
    """
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from verl.models.mcore.util import preprocess_thd_engine
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
    from verl.models.mcore.tpr import TPRSelfAttention

    if any(isinstance(m, TPRSelfAttention) for m in model.modules()):
        raise AssertionError("Native Full must not contain TPRSelfAttention")
    if getattr(model.config, "context_parallel_size", None) != 4:
        raise AssertionError("Native Full requires CP4")
    if getattr(model.config, "tensor_model_parallel_size", None) != 2:
        raise AssertionError("Native Full requires TP2")
    if dist.get_world_size(runtime.cp_group) != 4:
        raise AssertionError("native CP group mismatch")

    input_rows = batch["input_ids"].unbind(0)
    masks = batch["response_mask"].unbind(0)
    prompts = batch["prompts"].unbind(0)
    responses = batch["responses"].unbind(0)
    temps = tu.get_non_tensor_data(batch, key="temperature", default=1.0)
    global_tokens = tu.get_non_tensor_data(batch, key="batch_num_tokens", default=None)
    if global_tokens is None:
        raise ValueError("Native Full requires original global batch_num_tokens")
    global_tokens = int(global_tokens)
    count = len(input_rows)
    if count != 8 or global_tokens <= 0:
        raise AssertionError("Native Full expects complete 8-Session PPO mini-batch")

    # Keep native configuration responsible for the native model's CP setup.
    model.config.no_sync_func = None
    model.config.grad_scale_func = None
    model.config.finalize_model_grads_func = None
    model.config.calculate_per_token_loss = False

    loss_parts = []
    metric_lists = {}
    logprob_parts = {}
    calls = [0]
    prepared = []
    for row in range(count):
        length = int(input_rows[row].numel())
        global_positions = native_cp_query_indices(length, runtime.cp_rank)
        local_lookup = {position: i for i, position in enumerate(global_positions)
                        if position < length}
        mask_indices = torch.nonzero(masks[row].bool(), as_tuple=False).flatten().tolist()
        p = int(prompts[row].numel())
        offsets = [j for j in mask_indices if (p + j - 1) in local_lookup]
        idxs = [local_lookup[p + j - 1] for j in offsets]
        if isinstance(temps, torch.Tensor):
            temperature = float(temps.reshape(-1)[0 if temps.numel() == 1 else row])
        elif isinstance(temps, (list, tuple)):
            temperature = float(temps[row])
        else:
            temperature = float(temps)
        if not 0 < temperature < float("inf"):
            raise ValueError("invalid original PPO temperature")
        prepared.append((row, length, offsets, idxs, temperature))

    # A silent loss of query ownership invalidates every subsequent parity
    # number, even if the native training job otherwise finishes successfully.
    local_ref_count = torch.tensor(
        sum(len(item[2]) for item in prepared),
        dtype=torch.int64, device=runtime.device,
    )
    dist.all_reduce(local_ref_count, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    if int(local_ref_count.item()) != global_tokens:
        raise AssertionError(
            f"Native CP PPO ownership lost tokens: {int(local_ref_count.item())} "
            f"vs original {global_tokens}"
        )

    def forward_step(data_iterator, module):
        row, length, response_offsets, local_indices, temperature = next(data_iterator)
        calls[0] += 1
        print(f"NATIVE_FULL_MB_BEGIN rank={dist.get_rank()} row={row} length={length} "
              f"ppo_local={len(response_offsets)} "
              f"allocated_gib={torch.npu.memory_allocated()/2**30:.3f}", flush=True)
        nested = tu.nested_tensor_from_tensor_list(
            [input_rows[row].to(runtime.device)], ragged_idx=1
        )
        packed_ids, packed_params, packed_positions = preprocess_thd_engine(
            nested, pre_process=True, need_roll=False, cp_layout="zigzag"
        )
        expected_indices = native_cp_query_indices(length, runtime.cp_rank)
        if packed_positions.shape[-1] != len(expected_indices):
            raise AssertionError("Native THD preprocessor changed CP partition length")
        if local_indices:
            actual_positions = packed_positions.reshape(-1).index_select(
                0, torch.tensor(local_indices, device=runtime.device)
            )
            expected_positions = torch.tensor(
                [int(prompts[row].numel()) + k - 1 for k in response_offsets],
                device=runtime.device, dtype=actual_positions.dtype
            )
            if not torch.equal(actual_positions, expected_positions):
                raise AssertionError("Native THD CP zigzag ownership mismatch")
        model_logits = module(
            input_ids=packed_ids, position_ids=packed_positions,
            attention_mask=None, packed_seq_params=packed_params,
        )
        if model_logits.ndim != 3:
            raise AssertionError(f"Unexpected native GPT logits shape: {model_logits.shape}")
        # GPTModel output is normally [B,S,V]; support [S,B,V] MCore layout.
        if model_logits.shape[0] == 1:
            physical_logits = model_logits[0]
        elif model_logits.shape[1] == 1:
            physical_logits = model_logits[:, 0]
        else:
            raise AssertionError(f"Unexpected native logits batch layout: {model_logits.shape}")
        if physical_logits.shape[0] != len(expected_indices):
            raise AssertionError("Native logit sequence does not match CP-sharded input")

        def loss_func(_output, non_loss_data=False):
            if non_loss_data:
                return {}
            if local_indices:
                all_lp = []
                # Avoid the original 804-MiB (or larger) one-shot vocab tensor.
                for start in range(0, len(local_indices), 256):
                    stop = min(start + 256, len(local_indices))
                    selected = physical_logits.index_select(
                        0, torch.tensor(local_indices[start:stop], device=runtime.device)
                    )
                    targets = responses[row].index_select(
                        0, torch.tensor(response_offsets[start:stop],
                                        device=responses[row].device)
                    ).to(runtime.device)
                    if temperature != 1.0:
                        selected = selected / temperature
                    all_lp.append(vocab_parallel_log_probs_from_logits(selected, targets))
                log_probs = torch.cat(all_lp)
                local_loss, metrics = _compact_native_ppo(
                    loss_function, log_probs, response_offsets,
                    batch, row, global_tokens, runtime.device, runtime.dp_group,
                )
                from verl.utils.metric import AggregationType, Metric
                for key, metric in metrics.items():
                    if isinstance(metric, Metric) and metric.aggregation is AggregationType.MEAN:
                        metric = Metric(
                            aggregation=AggregationType.SUM,
                            value=metric.aggregate() * len(response_offsets) / global_tokens,
                        )
                    metric_lists.setdefault(key, []).append(metric)
                if capture_logprobs:
                    values = log_probs.detach().float().cpu().tolist()
                    logprob_parts.update({
                        (row, offset): val for offset, val
                        in zip(response_offsets, values, strict=True)
                    })
            else:
                local_loss = physical_logits.sum() * 0.0
                metrics = {}
            loss_parts.append(local_loss.detach())
            print(f"NATIVE_FULL_MB_LOSS_DONE rank={dist.get_rank()} row={row} "
                  f"allocated_gib={torch.npu.memory_allocated()/2**30:.3f} "
                  f"peak_gib={torch.npu.max_memory_allocated()/2**30:.3f}",
                  flush=True)
            # Megatron's 2-tuple loss branch already multiplies by CP size
            # and divides by num_microbatches. Because each local_loss uses
            # the ORIGINAL global PPO-token denominator, cancel only /M:
            #   (local_loss * M) * CP / M = local_loss * CP.
            # The explicit CP grad all-reduce + /CP below then gives the
            # correct global PPO gradient. Multiplying CP here a second
            # time produced the earlier ~4x native gradient bug.
            return local_loss * count, {"metrics": metrics}

        return model_logits, loss_func

    schedule = get_forward_backward_func()
    # True Megatron no-pipeline schedule, not a hand-written F/B loop.
    schedule(
        forward_step_func=forward_step,
        data_iterator=iter(prepared),
        model=[model],
        num_microbatches=count,
        seq_length=1,
        micro_batch_size=1,
        forward_only=False,
    )
    if calls[0] != count or len(loss_parts) != count:
        raise AssertionError(f"Native Megatron schedule executed {calls[0]}/{count} rows")

    from .grad_sync_precision import (
        offline_grad_sync_dtype, reduce_average_weight_grad,
    )
    grad_dtype = offline_grad_sync_dtype()
    t0 = time.perf_counter()
    for name, param in model.named_parameters():
        reduce_average_weight_grad(
            param, runtime.cp_group, name=name, reduce_dtype=grad_dtype,
        )
    torch.npu.synchronize()
    finalize_seconds = time.perf_counter() - t0

    loss_value = torch.stack(loss_parts).sum()
    dist.all_reduce(loss_value, op=dist.ReduceOp.SUM, group=runtime.cp_group)
    return NativeFullResult(
        loss=float(loss_value.item()),
        metrics=metric_lists,
        logprobs=logprob_parts,
        microbatches=count,
        finalize_seconds=finalize_seconds,
    )
