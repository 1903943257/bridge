# Pure TP2 (Sequence Parallel=False) — implementation and NPU gates

Updated 2026-10-10. **Code prepared; no distributed NPU correctness run
has yet been reported.** This is not an E2E PPO or multi-dimensional result.

## Upstream TP/SP evidence

- [Schedule-Level](https://arxiv.org/abs/2606.01143): reports optimizer
  agreement for TP/CP/PP/EP mixtures. The public abstract does **not**
  specify whether its TP experiments had Megatron SP on; do not label
  them TP-only.
- [HARTS](https://arxiv.org/pdf/2608.28158): Table 4 explicitly lists
  `DP4/TP2/SP/PP1/EP8` and `DP2/TP2/SP/PP2/EP4`. Its demonstrated
  TP2 setups **enable SP**.
- [psRL prefix-sharing](https://arxiv.org/abs/2608.25683): TP/SP
  implementation details are not sufficiently evidenced publicly.
  Do not confuse with the independent PSRL asynchronous RL framework.
- [Megatron native TP and SP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html):
  TP shards QKV projection heads and MLP weights; SP additionally shards
  portions of the activation sequence. Phase 1 uses TP only, SP=False.

## New TPR changes, strictly TP2 CP1 DP1 PP1 EP1

1. `megatron_adapter.run_tpr_forward_backward` (formal CE) and
   `run_tpr_forward_backward_batch` (VERL PPO Forest) accept TP2 with
   `sequence_parallel=False`, retaining CP/DP/PP/EP guards. TP=1
   existing CP paths remain unchanged.
2. `tp_validation.py` hashes SegmentIds, hierarchy, original source
   token bytes, DFS events, loss labels, and PPO objective refs; verifies
   all ranks in Megatron's native TP group have identical digests before
   model fwd or vocab-parallel CE. HCCL AllGather moves only 32 digest bytes
   as eight int64 words per rank, **not KV**.
3. `segment_executor._compute_loss` fixes a real TP correctness issue:
   next-token labels are global vocabulary IDs while TP2 logits are
   vocabulary-sharded. Do not compare global label against local_vocab;
   use native CE / global model vocab instead.
4. Existing `TPRSelfAttention` retains Megatron native ColumnParallel
   QKV and RowParallel projection. `PrefixState/KVStack` retain rank-local
   head shards. No custom TP AllReduce/AllGather and no KV cross-shard sync.
5. Native `no_sync_func` and `finalize_model_grads_func` are unchanged.

## Test matrix

- CPU `tests/models/mcore/tpr/unit/test_tp_validation.py` verifies
  stable/different plan hash, different PPO labels and TP mismatch errors.
- Existing `test_tpr_self_attention.py` already covers stub local-head
  TP2 gradient and explicit SP rejection.
- New **two-NPU** test
  `tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py`:
  `TPR_RUN_TP2=1 torchrun --standalone --nproc_per_node=2 -m pytest -vv -s ...`
    - Native TP2 independent complete trajectories vs TPR TP2 Tree
      `MegatronEngine.forward_backward_batch`: BF16 scalar CE,
      rank-local parameter gradients, and one SGD step.
    - Native TP2 vocab-sharded NLL vs TPR PPO Forest objective adapter:
      the same response-token loss, branch-point query ownership,
      TP2 vocab-parallel logprob gradients.
    - These are isolated F/B tests (NLL specialization for PPO adapter),
      not actual multi-step GRPO/PPO end-to-end training.

## Command on the already-synced Docker runtime

```bash
cd /workspace/uni-agent/verl
python -m pytest -vv -s \
    tests/models/mcore/tpr/unit/test_tp_validation.py \
    tests/models/mcore/tpr/unit/test_tpr_self_attention.py

TPR_RUN_TP2=1 torchrun --standalone --nproc_per_node=2 \
    -m pytest -vv -s \
    tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py
```

The two-rank NPU tests are the acceptance gate; **do not claim TP2
correctness, throughput or numerical parity before they pass**.
Do not enable TP+SP or TP+CP implicitly. Preserve ongoing BF16 GEMM
precision debugging by avoiding modifications to native kernels.
