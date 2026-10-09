# TPR TP2 audit and prefix-aware DP placement (2026-10-09)

Status: **audit + isolated planning implementation**; **NOT a claim of multi-NPU TP2/DP2 correctness**.
Branch: `feat/tpr-tp2-dp-placement-audit`. Main is untouched until review.
Based on `bridge` main `af5607ad79f30a30abc6b334abc02f6182e69626`.

## Upstream sources (follow existing stacks, do not implement distributed primitives)

- DTA: [AReaL `feat/dta/areal/experimental/dta/dp.py`](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/dp.py), [`tree_time_model.py`](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/tree_time_model.py), [DP tests](https://github.com/areal-project/AReaL/blob/feat/dta/tests/experimental/dta/test_dp.py).
- VERL: [PPO trainer `_balance_batch`](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/ray_trainer.py), [`get_group_balanced_partitions`](https://github.com/verl-project/verl/blob/main/verl/utils/seqlen_balancing.py), [`TrainingWorker.train_mini_batch`](https://github.com/verl-project/verl/blob/main/verl/workers/engine_workers.py), [MegatronEngine `forward_backward_batch`](https://github.com/verl-project/verl/blob/main/verl/workers/engine/megatron/transformer_impl.py).
- Schedule-Level: preserve Megatron TP projection/output collectives; Prefix KV is rank-local **head shard**. Do not all-gather Prefix KV across TP.
- HARTS: microbatch/DP/slot co-scheduling is an *optimization phase*, not an initial DP2 dependency.

## 1. TP2 source audit

| Current location | Existing contract | What to change | Status |
| --- | --- | --- | --- |
| `attention.py::_tree_forward` | Megatron native `get_query_key_value_tensors`, RoPE, local rectangular attention, `linear_proj`; prefix KV saved as local tensors | Do not implement TP collectives; verify local Q/KV head layout; reject SP at first | TP gate loosened **only in local attention**; CPU smoke added |
| `attention.py::_validate_tree_forward` | Explicit TP=1 restriction | Permit TP local-head shapes while PP/EP still unsupported | Changed in branch; TP+SP rejected |
| `megatron_adapter.py::run_tpr_forward_backward` | Explicit TP=DP=PP=EP=1; CP already supported | After NPU TP2 gate passes, relax TP alone; retain PP/EP gates; ensure identical plan on TP peers | **Not changed** |
| `megatron_adapter.py::run_tpr_forward_backward_batch` | **All sizes=1**, `dp_size==1`, builds a local forest; native finalize once per mini-batch | Separate TP and DP enablement gates; TP group must see same plan / row metadata; DP objective normalization must remain native | **Not changed** |
| `segment_executor.py::_compute_loss` | Normal GPTModel uses `compute_language_model_loss` (Megatron vocabulary-parallel CE), test-double fallback uses `F.cross_entropy` | For TP2 CE, **require** native compute loss; forbid fallback on local vocab; verify output layout | NPU gate required |
| `objective_adapter.py` | Uses VERL's vocab-parallel logprob/entropy | Verify TP-sharded vocab and scalar / gradient symmetry; no duplicate TP reduction | Reuse native, no rewrite |
| `prefix_state.py` / `kv_stack.py` | Store tensor-local Post-RoPE KV, collect leaf gradients, relay at Pop | No TP cross-rank KV gather/reduce; shape check on local heads only | No changes needed |
| `tpr_batch_runner.py` | One `no_sync` around forest, finalize once | TP collective order must match across model-parallel ranks; DP requires same optimizer-step count | Existing mechanism reused |
| `parallel/*` CP backends | CP group shards sequence | TP×CP phase: verify local head count, Ulysses head divisibility, hybrid subgroup global ranks | Defer |

### TP2 prerequisites and correctness gates

1. Start Dense Qwen3 / GPT, `TP=2, CP=DP=PP=EP=1`, `sequence_parallel=False`, `dropout=0`. Check Q heads and GQA groups **per TP rank**; do not assume that every GQA configuration supports every TP degree.
2. Check reference native Megatron TP2 first. Both TP ranks receive identical token rows and `FixedTopologyScheduler` events. Assert a deterministic digest of segment `(parent, start, length, token ids)` is equal across the TP group **before** collective-bearing forward/backward.
3. Compare one-layer local Q/KV shapes, rectangular forward, dK/dV including ancestors, local (sharded) and replicated parameter gradients to **native TP2**, then final optimizer update. Compare parameter shards correctly and avoid accidental extra loss scaling.
4. CE: confirm vocab-parallel path works with local vocab logits. PPO: verify `vocab_parallel_log_probs_from_logits` and entropy match native, including GQA and temperature handling.
5. Only after actual **NPU multi-rank** tests pass, relax the outer Engine TP2 gate. Do **not** interpret the CPU local-head unit test as distributed TP support.
6. CP×TP, GDN, Sequence Parallel, activation offload, recompute, PP, EP are separate phase gates.

## 2. DP location: *before* sharding, not inside SegmentExecutor

The actual upstream order is:

1. `RayPPOTrainer._balance_batch(batch: DataProto)` runs on the **global controller** and calls `batch.reorder(global_idx)`, before actor worker DP dispatch.
2. The dispatcher maps the globally reordered batch into equal-sized per-DP actor-worker inputs (possibly replicated across TP/CP peers).
3. `TrainingWorker.train_mini_batch` **re-slices and possibly shuffles** the per-DP batch via `tu.make_iterator`. This can destroy shared-prefix locality again if same-UID trajectories land in different mini-batches.
4. `MegatronEngine.forward_backward_batch` computes `batch_num_tokens` via the **native DP group** and then the TPR patch routes before `prepare_micro_batches`.
5. `run_tpr_forward_backward_batch` builds a **rank-local** `ForestExecutionPlan` and finalizes gradients once.

Therefore **do not** implement cross-replica placement inside `megatron_adapter.py`: it sees data after the global split.

### Existing VERL balancing is the baseline

VERL already exposes `get_group_balanced_partitions(seqlen_list, uid_list, k_partitions)` and uses it in `RayPPOTrainer._balance_batch` when PrefixGrouper is enabled. It uses equal numbers of UID groups, assuming each group has identical rollout count, and requires contiguous same-UID rows. The entire sample count per replica must still be equal. **Do not turn on `use_prefix_grouper=True` for TPR** just to get the balancing branch: PrefixGrouper also patches model attention, is intended for FSDP, and is not this TPR implementation.

For the first real DP2 implementation, reuse this native balancing **directly** in a new TPR-specific opt-in path at `_balance_batch` (not a new trainer, optimizer, or distributed backend). Validate UID contiguity and actual equal row counts. Assert mini-batch iteration keeps related UID rows together; otherwise install a group-aware sampler using VERL's existing iterator interface or fail fast with an actionable config error.

### DTA as an isolated second placement policy

`verl/models/mcore/tpr/dp_placement.py` now provides:

- `plan_verl_uid(...)`: delegates to upstream VERL `get_group_balanced_partitions`; validates contiguous UIDs and equal number of rows in each DP replica. **Only this policy is initially suitable for VERL's existing equal-shard dispatcher.**
- `plan_dta_dfs(...)`: existing lightweight DFS-contiguous minimax baseline, with equal-cardinality tie handling for repeated trajectories. Kept for VERL contract regression until DTA direct path is validated in the container.
- `plan_areal_dta(...)`: **directly calls vendored upstream AReaL `LB_by_DFS_and_TM`**, with original `TokenTrie`, `CompressedTrie`, `pred_time`, `try_divide` and optional `TreeTimeModel`. Only package imports were relocated; algorithm bodies are preserved in `_vendor/areal_dta/`. The adapter validates empty ranks, duplicate/prefix-contained leaf collapse, and VERL equal row cardinality.
- `DPPlacementPlan`: rows-per-rank, tree-token costs, global cost, duplicated-prefix cost, equal-cardinality flag.
- Default `enforce_equal_rows=True`: throw if the DTA plan violates VERL's current equal-row DP dispatch contract. `False` is **offline analysis only**; do not feed variable-sized partitions straight into `batch.reorder`.

Sample usage, on global-controller **un-padded** token sequences, not on padded `input_ids`:

```python
from verl.models.mcore.tpr.dp_placement import plan_verl_uid, plan_dta_dfs

# Safe baseline (when same-UID groups have fixed rollout.n):
native = plan_verl_uid(seqs, uid_list, dp_size=2)

# Compare how much reuse DTA would retain after variable-size partitioning:
offline = plan_dta_dfs(seqs, dp_size=2, enforce_equal_rows=False)

# Reuse AReaL-DTA original DP algorithm, with the baseline tree-token model:
from verl.models.mcore.tpr.dp_placement import plan_areal_dta
areal = plan_areal_dta(seqs, dp_size=2, enforce_equal_rows=False)
print(areal.partitions, areal.tree_tokens_by_rank)
```

AReaL's `TreeTimeModel` (learned coefficients, NNLS fit) is vendored unchanged at `_vendor/areal_dta/tree_time_model.py` and can be passed as `time_model=` once `numpy/scipy` are available and it has sufficient calibration data; the default adapter uses its original interface with a dependency-free tree-token predictor.

**Upstream limitation:** AReaL `TokenTrie` merges duplicate or prefix-contained rows into one trie leaf. The upstream solver can therefore yield fewer than `dp_size` non-empty bins even if total original rows are enough. The VERL wrapper rejects these inputs instead of introducing new AReaL-internal algorithm patches. The existing local `plan_dta_dfs` handles the duplicate/equal tie case and remains a fallback until actual distributed data dispatch can be changed.

**License/provenance:** vendored sources come from AReaL commit `a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6` and retain Apache-2.0 notices, as well as MIT attribution to the original DynamicTreeAttn upstream. See `_vendor/areal_dta/THIRD_PARTY_NOTICES.md`. Nothing requires installing AReaL as a runtime package.

### Necessary DP2 gates before wiring the planner into Trainer

- Samples partitioned once globally, no loss of examples, and actor's TP/CP peers get **identical** local Forest/metadata.
- Per-DP sample counts and `num_mini_batch` iteration counts are equal under current native dispatch. If the data contains only one UID group and DP2, a no-split baseline that preserves all siblings on one rank is **not possible** under normal equal batch placement without duplicating/nonstandard work; reject or explicitly choose a policy that splits and independently recomputes Prefix.
- No group is split across *mini-batches on the same DP rank* unless intended, because prefix KV does not survive `TPRBatchRunner` calls.
- Native `batch_num_tokens` and `dp_size` continue to drive VERL's global objective normalization. Before allowing DP2, remove its current `dp_size==1` assumption carefully, and check any token-mean loss / metrics and custom objective for DP collectives inside per-Segment calls.
- Use `config.no_sync_func`, native `finalize_model_grads_func`, and native optimizer. Do **not** add TPR DP gradient synchronization or cross-DP KV transfer.
- End-to-end numerical gate: native DP2 vs TPR DP2 on same global trajectory multiset, compare PPO scalar, logprobs, local/global param gradients, and **multiple optimizer steps**, then measure tree-token savings / replica stragglers.

## 3. Order of work

1. **This branch:** TP local-head support gated at outer Engine + CPU smoke; independent, reviewed DP placement planner + CPU contract tests + upstream insertion analysis.
2. **Native TP2 NPU gate:** add real TP2 CE+PPO two-rank numerical tests and relax *only* TP Engine guards once tests pass.
3. **Native DP2 functional gate:** add TPR-aware route at controller `_balance_batch` using `plan_verl_uid`, group-aware local mini-batch boundaries, and native loss/grad semantics.
4. **DTA DP performance:** benchmark offline `plan_dta_dfs`; make variable-cardinality dispatch opt-in only after native dispatch/worker step counts can handle it.
5. TP×DP, then TP×CP; HARTS joint microbatch/slot packing and DTA time-model only when straggler profiling justifies them.

Nothing in this document claims that TP2/DP2 is enabled today. The existing outer Engine restrictions intentionally remain until a real multi-rank correctness gate.
