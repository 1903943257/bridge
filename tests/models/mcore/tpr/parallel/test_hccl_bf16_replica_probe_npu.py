"""Isolated Ascend BF16 vs FP32 CP×DP HCCL reduction investigation.

The original BF16 SUM/4 probe returned different BF16 values across
rank replicas for 197/2560 elements. Disambiguate:
  (1) BF16 AllReduce SUM output, before division,
  (2) BF16 div_ on those outputs,
  (3) FP32-promoted SUM and div_ on EXACTLY the same BF16 inputs,
  (4) cast the FP32 result back to BF16 on each rank.

Each stage is synchronized and checked using raw AllGather; AllGather
never mutates the values being diagnosed. No model weights are loaded.
This is an isolated communication correctness test, not a performance test.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from .test_tpr_tp2cp2dp2_one_step_npu import _runtime

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_HCCL_BF16_GRAD_PROBE") != "1",
    reason="explicitly opt in to isolated 8-NPU BF16 gradient collective test",
)


def _gather_stage(tensor, group, ranks, label):
    """AllGather and analyze untouched values, reporting actual peer values."""
    torch.npu.synchronize()
    gathered = [torch.empty_like(tensor) for _ in ranks]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    torch.npu.synchronize()
    cpu = torch.stack(gathered).float().cpu()
    spread = cpu.amax(dim=0) - cpu.amin(dim=0)
    diff = spread.abs()
    max_diff, worst_index = diff.max(dim=0)
    offset = int(worst_index.item())
    values = [float(cpu[k, offset].item()) for k in range(len(ranks))]
    return {
        "stage": label,
        "dtype": str(tensor.dtype),
        "element_count": int(tensor.numel()),
        "max_abs": float(max_diff.item()),
        "differing_elements": int((diff != 0).sum().item()),
        "worst_offset": offset,
        "worst_rank_values": dict(zip(map(str, ranks), values)),
        "all_replicas_identical": all(torch.equal(cpu[0], x) for x in cpu[1:]),
    }


def test_bf16_gradient_finalize_collectives_match_across_cpdp_replicas():
    runtime = _runtime()
    try:
        cpdp = runtime.dp_cp_group
        group_members = tuple(dist.get_process_group_ranks(cpdp))
        rank = dist.get_rank()
        group_leader = group_members[0]
        for n in (2560, 1048576):
            # Values differ across the CP×DP members, matching the actual
            # weight-gradient reduce use case.
            base = torch.linspace(
                -0.025, 0.025, steps=n, dtype=torch.float32,
                device=runtime.device,
            )
            original = (
                base + (rank - 3.5) * 0.000244140625
            ).to(torch.bfloat16).contiguous()
            torch.npu.synchronize()

            # Keep a CPU host-side reference from raw original BF16 inputs.
            pre = [torch.empty_like(original) for _ in group_members]
            dist.all_gather(pre, original, group=cpdp)
            torch.npu.synchronize()
            input_cpu = torch.stack(pre).float().cpu()
            reference_sum = input_cpu.sum(dim=0)

            # The original experimental path: BF16 SUM, then in-place /4.
            reduced_bf16 = original.clone()
            dist.all_reduce(reduced_bf16, op=dist.ReduceOp.SUM, group=cpdp)
            bf16_sum = _gather_stage(
                reduced_bf16, cpdp, group_members, "bf16_sum_before_div",
            )
            reduced_bf16.div_(len(group_members))
            bf16_div = _gather_stage(
                reduced_bf16, cpdp, group_members, "bf16_after_div",
            )

            # Exact same pre-reduction BF16 inputs, promoted to FP32.
            reduced_fp32 = original.float().contiguous()
            dist.all_reduce(reduced_fp32, op=dist.ReduceOp.SUM, group=cpdp)
            fp32_sum = _gather_stage(
                reduced_fp32, cpdp, group_members, "fp32_sum_before_div",
            )
            reduced_fp32.div_(len(group_members))
            fp32_div = _gather_stage(
                reduced_fp32, cpdp, group_members, "fp32_after_div",
            )
            cast_bf16 = _gather_stage(
                reduced_fp32.to(torch.bfloat16),
                cpdp, group_members, "fp32_mean_cast_to_bf16",
            )

            # Cross-group checksum: no new FP64 HCCL reductions.
            host_sum_max_error = float(
                (reduced_fp32.float().cpu() * len(group_members)
                 - reference_sum).abs().max().item()
            )
            stages = (
                bf16_sum, bf16_div, fp32_sum, fp32_div, cast_bf16,
            )
            if rank == group_leader:
                for stage in stages:
                    print(
                        f"TPR_HCCL_DTYPE_STAGE group={group_members} "
                        f"elements={n} stage={stage['stage']} "
                        f"max_abs={stage['max_abs']:.9g} "
                        f"differing_elements={stage['differing_elements']} "
                        f"worst_offset={stage['worst_offset']} "
                        f"worst_rank_values={stage['worst_rank_values']}",
                        flush=True,
                    )
                print(
                    f"TPR_HCCL_DTYPE_SUM_REFERENCE group={group_members} "
                    f"elements={n} fp32_max_error={host_sum_max_error:.9g}",
                    flush=True,
                )
            for stage in (fp32_sum, fp32_div, cast_bf16):
                if not stage["all_replicas_identical"]:
                    raise AssertionError(
                        f"Promoted FP32 HCCL gradient sync is NOT replica-identical: "
                        f"elements={n} group={group_members} stage={stage}"
                    )
            if host_sum_max_error > 1e-5:
                raise AssertionError(
                    "FP32 HCCL reduction does not match independently gathered "
                    f"input sum: group={group_members} elements={n} "
                    f"max_abs={host_sum_max_error}"
                )
            if os.getenv("TPR_PROBE_REQUIRE_BF16_EXACT", "0") == "1":
                for stage in (bf16_sum, bf16_div):
                    if not stage["all_replicas_identical"]:
                        raise AssertionError(
                            f"BF16 gradient replica mismatch: {stage}"
                        )
            dist.barrier()
        if rank == 0:
            print(
                "TPR_HCCL_DTYPE_PROBE_PASS "
                "fp32_reduce_replica_identical=true "
                "bf16_reduce_reported_separately=true",
                flush=True,
            )
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
