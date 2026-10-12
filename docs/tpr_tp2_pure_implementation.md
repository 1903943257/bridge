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
- New **two-NPU Qwen3-1.7B** test
  `tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py`:
  `TPR_RUN_TP2=1 torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29531 -m pytest -vv -s ...`
    - Uses **real** `/workspace/hf_models/Qwen3-1.7B` checkpoint,
      28 layers, 2048 hidden, 16 Q / 8 KV heads, 151936 vocab,
      and **native TP2** sharding. Source checkpoint paths can be set
      using `TPR_QWEN_1_7B_PATH`.
    - New `_qwen3_tp_checkpoint.py` loads HF weights into native
      Megatron GQA-group QKV shards, per-partition SwiGLU gate/up,
      row-parallel O/MLP down projections and vocabulary-sharded tied
      embeddings; checks all parameter coverage. This is necessary
      because the existing real-Qwen CP weight loader assumes TP=1.
    - Megatron TP2 independent complete trajectories vs TPR TP2 Tree
      `MegatronEngine.forward_backward_batch`: BF16 scalar CE,
      rank-local parameter gradients, and an SGD update. Vocab labels
      target **the second TP vocab shard** (global IDs above 75968).
    - Native TP2 vocab-sharded NLL vs TPR PPO Forest objective adapter:
      same response-token loss, branch-point query ownership, TP2
      vocab-parallel logprob gradients.
    - Trajectory token IDs are deterministic test inputs; **weights and
      architecture are genuine pretrained Qwen3-1.7B**. Tests are
      isolated F/B gates (PPO NLL specialization), not full GRPO E2E.

## Command on the already-synced Docker runtime

```bash
cd /workspace/uni-agent/verl
python -m pytest -vv -s \
    tests/models/mcore/tpr/unit/test_tp_validation.py \
    tests/models/mcore/tpr/unit/test_tpr_self_attention.py

TPR_RUN_TP2=1 torchrun --nproc_per_node=2 \
    --master_addr=127.0.0.1 --master_port=29531 \
    -m pytest -vv -s \
    tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py
```

The two-rank NPU tests are the acceptance gate; **do not claim TP2
correctness, throughput or numerical parity before they pass**.
Do not enable TP+SP or TP+CP implicitly. Preserve ongoing BF16 GEMM
precision debugging by avoiding modifications to native kernels.

## 2026-10-10 — two independent test failures after switching to real Qwen3-1.7B

1. **Device kernel failure during the first full-trajectory reference**:
   `ScaledMaskedSoftmax` expected a 4D mask but received a 2D mask.
   Merely setting `config.use_flash_attn=True` selected MindSpeed's
   two-dimensional flash causal mask but did **not** switch the Megatron
   `DotProductAttention` module to a flash kernel. The genuine checkpoint
   and TP2 weight placement were loaded; failure preceded the TPR path.
   Fix: install the **existing real-Qwen controlled CANN square-causal
   reference** (`_ProfileFusedCausalAttention`) into the model spec's
   `core_attention` on both reference and TPR models; assert correct
   class after construction. This is **not** a validated unmodified
   MindSpeed native-FlashAttention comparison. It avoids changing TPR
   runtime or upstream MindSpeed operators.
2. **Forest raised 'response is not an input_ids suffix'** in a following
   test. The test builds both tensors via the *same suffix* in
   `torch.cat(prefix, suffix)` and `torch.stack(suffixes)`; no code
   evidence so far supports a real Radix/Forest mapping failure.
   Given the preceding unrecoverable asynchronous NPU operator error,
   a poisoned device context is possible. Added explicit, CPU-side
   preflight of both input/response rows to distinguish malformed test
   data from a stale NPU runtime. If this still reproduces in a **fresh
   separate process**, inspect the preflight and report the first
   mismatching row; do not weaken the production Forest invariants.
3. Run in a fresh torchrun process; do not treat results from a worker
   that previously hit E89999 as an independent clean GPU/NPU validation.

TP2 accuracy remains UNVERIFIED until real two-rank Qwen3-1.7B tests pass.


## 2026-10-10 — TP2 thin-entry test routing and PPO metadata fixes

The last two errors were in the **test harness**, prior to completed
numerical comparison, not TP2/TPR gradient mismatches:

- `KeyError: loss_mask`: `_tpr_engine_run` previously passed a
  request-only TensorDict into `MegatronEngine.forward_backward_batch`.
  The installed upstream native method accesses `data["loss_mask"]`
  in its global-token-count preamble. Because the bridge branch does not
  contain the deployed VERL Engine override, injecting a dummy
  `loss_mask` would only push the error further into native batch
  preparation, and would **not prove TPR dispatch**. The controlled TP2
  correctness gate now calls the existing
  `megatron_adapter.run_tpr_forward_backward(engine, request,
  forward_only=False)` directly. It still executes the real Megatron
  Qwen3-1.7B TP2 model/TP collectives/TPR F+B, but **does not certify
  `MegatronEngine.forward_backward_batch` automatic interception**.
  The real production VERL routing test needs to be completed separately
  using the exact installed Engine version.
- `TypeError: get_non_tensor_data() missing default`: fixed the
  `_tpr_ppo_nll_run` nested NLL callback by passing
  `default=None` and verifying positive `batch_num_tokens`.
  No changes to the real VERL loss function or PPO semantics.

After these changes, the two-NPU numerical gate still needs to run.
