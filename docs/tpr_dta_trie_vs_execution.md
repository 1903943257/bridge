# TPR data-parallel design — DTA-inspired placement only

**Scope**: This change modifies only *which complete trajectory rows belong to a
logical DP replica*. It is NOT a DTA runtime or a second TPR tree builder.

Reference: [AReaL-DTA original DP partition](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/dp.py)
and [DynamicTreeAttn](https://github.com/Whisper-6/DynamicTreeAttn/blob/main/data_parallel.py).

## Production path (unchanged downstream components)

```
global real TQ actor trajectories and trajectory_keys
   |
   +-- DP placement: plan_tpr_dta_dp (DTA DFS-contiguous minimax idea)
   |       outputs original row-index partitions only
   |
VERL controller global batch.reorder + native DP dispatch  [NOT WIRED YET]
   |
TrainingWorker local minibatch iterator                 [NOT WIRED YET]
   |
run_tpr_forward_backward_batch (existing)
   |
build_tree_execution_plans (existing)
   +-- build_trajectory_trees (existing, UID-scoped, prefix-exact)
   +-- SegmentPlan + SegmentObjectiveRef (existing)
   |
FixedTopologyScheduler + SegmentExecutor (existing)
   |
TPR KV / gradient relay, PPO, native Megatron optimizer (existing)
```

`plan_tpr_dta_dp(token_sequences, trajectory_keys, dp_size, enforce_equal_rows=True)`
uses the same `trajectory_keys` accepted by the TPR tree builder. It extracts
UIDs through the original TPR key parser and sorts DFS candidates by UID then
token sequence. Its cost estimator computes exact unique token counts **separately
per UID**, mirroring the existing `TrajectoryTree` scope. It never merges
two UIDs simply because tokens happen to match. The min-max, greedy threshold,
binary search, and equal-row optimal-tie logic are DP scheduling only.

`plan_dta_dfs` remains a historical flat/global-token cost baseline for
independent unit tests; **it is not used by the TPR-aware DP path unless
called with a UID list**. `plan_verl_uid` remains a wrapper around the native
VERL UID group balancing strategy. No AReaL library or TokenTrie package
is copied or imported. The original AReaL optimizer's fitted TreeTimeModel is
not used; using such a model later would require measuring actual TPR forest
time rather than DTA execution time.

## Why the previous vendor approach was removed

AReaL's `dp.py` imports `TokenTrie` and `CompressedTrie` to estimate time
and partition leaf groups. Copying these into TPR inadvertently made a
parallel data-model stack and cost estimates that considered prefix reuse
across UIDs. It also introduced duplicate/contained sequence leafization
semantics not needed by our execution path. All vendored source files and
tests exercising only the duplicate DTA trie were removed.

## Verification — what is and is not checked

1. Existing `test_dp_placement.py`: DFS minimax, exact sample coverage,
   duplicated trajectories, equal-cardinality safety, UID-scoped cost.
2. **New direct integration**
   `tests/models/mcore/tpr/integration/test_tpr_dp_placement_to_forest_cpu.py`:
   uses TPR's actual `build_tree_execution_plans` on each DP partition,
   validates physical Segment costs, exact logical PPO token references, and
   original row identity after reindexing.
3. Medium/large scale `test_dp_placement_scale_cpu.py`: 128×2K, 256×4K,
   272 duplicated/prefix-terminal rows, opt-in 512×16K. Measures DTA-inspired
   planner and VERL UID balancer; also measures TPR's *actual* executable
   `TrajectoryTree` build on medium cases.
4. **Not verified**: real DP2 VERL dispatch, equal minibatch update counts,
   PPO gradient scaling, optimizer synchronization, actual multi-NPU speedup.
   These require a separate Trainer entry change and end-to-end gate.

```bash
python -m pytest -vv -s \
  tests/models/mcore/tpr/unit/test_dp_placement.py \
  tests/models/mcore/tpr/integration/test_tpr_dp_placement_to_forest_cpu.py \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py

TPR_DTA_SCALE_STRESS=1 python -m pytest -vv -s \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py \
  -k long_prefix_16k
```

Current worker adapter and Engine DP=1/TP=1 gates remain intact until
parallel NPU correctness is proven. This change does not touch the other
ongoing BF16/GEMM numerical experiments.
