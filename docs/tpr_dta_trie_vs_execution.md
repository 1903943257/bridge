# AReaL-DTA trie vs TPR trajectory tree: design and scale gates

Sources:
- DTA paper: https://arxiv.org/pdf/2602.00482 (sections 3–4; Algorithms 1–4).
- Exact implementation: https://github.com/areal-project/AReaL/tree/feat/dta/areal/experimental/dta
- Local execution: `verl/models/mcore/tpr/{trajectory_tree.py,tree_plan_builder.py,segment_plan.py,fixed_topology_scheduler.py}`.

## The two tries have overlapping LCP logic but different consumers

| Layer | AReaL-DTA | Our TPR | Decision |
| --- | --- | --- | --- |
| Global planner (pre-DP) | `TokenTrie`: lexical sort, adjacent LCP, merge duplicates and contained terminal rows into `attach_lists` | Previous `plan_dta_dfs` repeats lexical/LCP logic | Reuse AReaL `TokenTrie` via vendored `plan_areal_dta`; keep previous algorithm as duplicate/equal-row fallback |
| Workload prediction | `CompressedTrie`: lightweight topology, optimized forward/backward DFS order, per-tree metrics | `TrajectoryTree` stores explicit executable node spans | Use AReaL for DP time prediction; do not create `SegmentPlan` globally |
| DP partition | `LB_by_DFS_and_TM`: binary-search minimax threshold with contiguous DFS leaf ranges; optional fitted time model | No live DP dispatcher yet | Prefix-aware placement must run at trainer/controller `_balance_batch`, *before* per-rank data dispatch |
| Local executable tree | `TokenTrie` + DTAEngine take advantage of attachment indices and LCP, running one active sequence path | `TrajectoryTree` has `SegmentRef`, `member_rows`, `terminal_rows`, parents/children | Keep local `TrajectoryTree` + `SegmentPlan` + `FixedTopologyScheduler` |
| PPO token ownership | DTA attachment captures per-original-row loss | `SegmentObjectiveRef` captures original `sample_row`, `response_offset`, and query/target across parent-child boundary | Preserve our `tree_plan_builder.py` |
| Execution | DTA `push` saves detached KV + forward-only cache; `pop` redoes differentiable forward (chunked backwards) and injects KV gradient | TPR Push/Visit/Pop is integrated with Megatron/CP and optional offload/recompute | Do NOT replace runtime with DTAEngine |

### The UID scope matters

Our `build_trajectory_trees` groups by `uid` first; within each UID it computes exact token-prefix sharing. AReaL-DTA's global token sort can cluster *different UIDs* that happen to have identical tokens, whereas the local TPR executor will **not** share across those UID boundaries. Hence the current placement's `DPPlacementPlan.tree_tokens_by_rank` is an optimistic upper bound on saved work when different UIDs share long prefixes. Before production dispatch, compute predicted cost from the same UID-scoped execution forest, or explicitly opt in to cross-UID reuse after verifying semantics.

Our builder creates one executable tree per real first-token root (dummy root is not an executable segment), so it does not assume that the root equals a prompt boundary. DTA `attach_lists` and our `terminal_rows` both preserve duplicate and prefix-contained **logical trajectories**; neither implies losses should be de-duplicated.

### DP vs microbatch boundary

AReaL DP partition cost is an estimate of compute, not a guarantee of current VERL dispatchability. Native VERL requires equal row counts and synchronizable local mini-batch iterations. After global placement, shuffling or re-slicing same-UID trajectories in `TrainingWorker.train_mini_batch` can erase planned prefix reuse. Validate mini-batch grouping before claiming real speedup.

### CPU scale gates

```bash
pytest -vv -s \
  tests/models/mcore/tpr/unit/test_dta_trie_semantics.py \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py

# Explicit long-prefix stress: 512 trajectories x >=16K tokens, DP=8
TPR_DTA_SCALE_STRESS=1 pytest -vv -s \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py
```

Workloads include 128 and 256 unique trajectories with 2K/4K group prefix, four-level branching and variable suffix; 272 logical rows with duplicates/prefix-terminal attachments; 512 x 16K explicit stress. The scale gate runs native VERL `plan_verl_uid`, directly vendored AReaL `plan_areal_dta`, and old `plan_dta_dfs` side-by-side, measuring wall time, maximum rank tree tokens, extra duplicated tree tokens and per-rank sample counts. Timings are diagnostic, not stable CI thresholds. These workloads are synthetic; real Agent trajectories should be measured separately before accepting scheduling efficiency.

This is **not** a TP2/DP2 numerical or throughput test: Engine's existing parallelism gate is unchanged.
