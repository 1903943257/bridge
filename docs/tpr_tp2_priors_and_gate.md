# TPR TP2 roadmap: reuse Megatron tensor parallelism, not a second TP stack

Date: 2026-10-09. Design only; no multi-NPU correctness claim.
The new direct-compressed `build_trajectory_trees` is the default CPU tree
builder after strict legacy equivalence and 21.58–40.47x CPU benchmarks.
DP placement is still offline/planning-only; runtime Trainer dispatch is
not enabled. TP can be developed independently.

## Prior art and strength of evidence

1. [Schedule-Level Shared-Prefix Reuse for LLM RL Training](https://arxiv.org/abs/2606.01143)
   explicitly reports compatibility and optimizer-update alignment under
   multiple TP/CP/PP/EP combinations. It motivates keeping prefix KV and
   gradient interfaces **local to the rank's attention-head shard** and
   reusing the existing TP projection collectives.
2. [HARTS: hybrid attention over arbitrary rollout trees](https://arxiv.org/abs/2608.28158)
   §4.3 explicitly composes with existing TP/SP; Table 4 reports real
   TP=2 with SP (DP4/TP2/SP/PP1/EP8 and DP2/TP2/SP/PP2/EP4).
   This is **not** a proof that our no-SP first phase is validated by HARTS;
   it demonstrates compatibility via native layer partitioning.
3. [psRL: Efficient Training for Agentic AI via Training-Time Prefix Sharing](https://arxiv.org/abs/2608.25683)
   emphasizes global prefix reuse, workload distribution and KV placement.
   We have not identified a publicly inspectable psRL TP-specific
   implementation or a dedicated TP correctness matrix: do not claim TP
   support on title/abstract alone.
4. [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
   provides the real production TP semantics; use its QKV/output projection
   and vocab-parallel CE collective paths unchanged.

## Proposed minimal TP2 validation: dense Qwen3 / GPT, TP2 PP1 CP1 DP1 EP1

Assumptions: consistent model weights and RNG across two TP ranks, B=1,
`sequence_parallel=False`, dropout=0, no PP, EP, GDN, FP8, or recompute.

- Every TP peer receives the **same original trajectory keys, SegmentPlan,
  branch ordering, and per-token PPO objective metadata**; assert a stable
  per-plan digest across TP ranks BEFORE collective-bearing forward/backward.
  The radix builder emits deterministic SegmentIDs for both ranks.
- `TPRSelfAttention.get_query_key_value_tensors` uses **native Megatron
  ColumnParallel QKV**: local Q heads and local K/V groups per rank.
  Its RoPE, causal rectangular FA and prefix K/V state all stay local-head
  shards, with no TP-specific KV all-gather. Let the existing Megatron
  RowParallel projection perform output TP all-reduce.
- `PrefixState`, KVStack and gradient relay hold local `K, V, dK, dV`
  only. Sum contributions from descendant segments on each TP rank, then
  native TP projection autograd and native Megatron gradient finalize handle
  their existing collectives. **Do not all-reduce dKV across TP peers.**
- PPO vocab-sharded logprob and entropy should use native VERL/Megatron
  TP-aware functions; native vocab-parallel loss path is mandatory when TP>1.
  Inspect native `sequence_parallel=False` behavior and replicated vs
  partitioned gradient reductions; no custom TP communications in TPR.
- `TPRBatchRunner` must produce one identical **collective order** on
  both TP peers. Prefix graph or KV operations must not diverge based on a
  rank-local condition. `no_sync` and gradient finalization happen once
  per logical TPR minibatch via the native Megatron hooks.

### Gates (the outer Engine TP1 restriction remains until the last gate)

A. **Native TP2 baseline**: same model, identical fixed trajectories,
   vocab-parallel CE/logprob/entropy plus 1 optimizer update.
B. **Rank-local attention**: head-sharded QKV shapes and RoPE; compare
   rectangular vs reference causal full sequence at TP2; verify prefix
   dK/dV nonzero and per-rank shards match.
C. **TP-plan agreement**: identical forest digest/events/objective sample
   rows on TP ranks, including duplicate, prefix-terminal, multilevel trees.
D. **End-to-end TP2 with isolated TPR**: same seed, batch, and parameter
   initialization as native TP2; PPO scalar/logprob, per-parameter *sharded*
   gradient and optimizer delta alignment; two successive updates, not only
   a single fwd pass. Use tolerance calibrated to native repeat-run noise.
E. Only then lift **TP restriction** in both
   `run_tpr_forward_backward` and `run_tpr_forward_backward_batch`.
   Keep DP/PP/EP gates and context-specific CP guards independent.
F. Later expand to `sequence_parallel=True` and CP×TP. SP implies
   native AG/RS transformations of sequence-sharded non-attention tensors;
   never silently enable it through a TP2 success with SP=False.

Do not copy HARTS compact-row execution or Schedule-Level's entire
scheduler; follow their principle of **orthogonal TP** by preserving
existing Megatron TP layers, collectives and gradient synchronization.

## Current branch implementation boundary

`attention.py::_validate_tree_forward` already permits rank-local TP2
heads with `sequence_parallel=False`, and explicitly rejects TP+SP.
`megatron_adapter.py` still rejects TP != 1 in both formal TPR and PPO
entries. No new TP kernels or collectives are planned; next work is a
two-NPU *comparison test* before changing these gates.
