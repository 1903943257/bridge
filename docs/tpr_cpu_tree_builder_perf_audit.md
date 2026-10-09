# TPR CPU tree builder performance audit: optional direct-radix alternative

## Finding: distinguish DP planning from executable tree building

The reported **256 rows, 4K prefix, DP8** workload had:

- Native VERL UID placement: **0.152 s**
- TPR-aware DTA-inspired DP placement: **0.216 s**
- Existing TPR `build_trajectory_trees`: **2.659 s**
- Existing TPR execution tree: **276,480 unique tree tokens -> 480 compact segment nodes**

These timings measure different operations. The TPR DP planner itself is the
same order of magnitude as native VERL; it is **the separate local CPU tree
construction** that is roughly 12x slower than that planner. The 2.659 seconds
are not DP planner latency, and the first phase of local TPR construction is
measured on a *global batch*, not yet on individual DP-rank shards.

## Root cause in the source

`trajectory_tree._insert_sequence` loops through every token in every row,
allocating `_TrieNode` (position, member_rows, terminal_rows, children) on the
first visit to each distinct token. Every insertion appends row references to
all ancestors. Only **after all rows are inserted** does
`_compress_real_root` construct the small executable Segment tree.

On the example above, the temporary Python trie can have approximately
276,480 ordinary token nodes, while only 480 actual compressed segment nodes
are retained. The cost is millions of Python list/dict/object operations and
much larger peak temporary metadata than the final plan requires. This is
a direct-compression opportunity, **not an Attention, KV or DP algorithm flaw**.

## Experimental design (strictly opt-in, no default behavior change)

`verl/models/mcore/tpr/trajectory_tree_radix.py` exports:

```python
from verl.models.mcore.tpr.trajectory_tree_radix import build_trajectory_trees_radix
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans

trees = build_trajectory_trees_radix(keys, batch)
forest = build_tree_execution_plans(
    keys, batch, trees=trees, require_loss_mask_alignment=True
)
```

It uses a **direct-compressed radix/LCP insertion**, inspired by DTA's
compressed-prefix approach, not the AReaL `TokenTrie` or DTA runtime.
It allocates one temporary span node per actual branch, terminal, or duplicate
endpoint rather than one node per token. The same TPR `TrajectoryTree`,
`TrajectoryNode`, `SegmentRef`, and `SegmentPlan` are returned.

Semantics deliberately preserved:
- UID is *only* the candidate sharing boundary. Identical prefixes in
  different UIDs are **not** shared.
- No fake prompt-root assumption; dummy first-token root is not executable.
- Tree order, branch order, `tree_index`, `node_id`, `source_row`,
  segment offsets and original sample-row order are preserved.
- Strict-prefix/duplicate trajectories keep their logical terminal ownership;
  PPO query/target rows and gradient contributions are not collapsed.
- Forward/backward, recompute/offload, CP, TP, DP, optimizer and scheduler
  remain **completely unchanged**.

## Impact assessment / side-effect gates

| Contract | Expected impact | Validation |
| --- | --- | --- |
| UID split, first-token root | none | complete `TrajectoryTree` object equality |
| Branch/terminal/duplicate rows | none | fixed edge cases, deterministic random differential tests |
| `SegmentRef(row,start,end)` | none | full node equality |
| `node_id`, tree order, DFS Push/Pop | none | exact plan event equality |
| PPO query/target/sample ownership | none | `SegmentObjectiveRef` equality |
| Model/KV/CP/TP/offload/grad | none by default | only if explicit opt-in later; before enabling production run NPU correctness |
| CPU memory/latency | potentially large improvement | print old vs radix times on same input, never assume before measurement |

Existing TPR `build_trajectory_trees` is not modified. The fast builder
is **NOT enabled by default**. The regular PPO `megatron_adapter.py` entry
now selects the constructor via `TPR_TREE_BUILDER` and forwards it into
the original `build_tree_execution_plans` and original TPR execution:
```bash
# Existing single-rank TPR PPO path remains unchanged by default.
unset TPR_TREE_BUILDER

# Explicit training-side opt-in; do not enable during BF16/GEMM diagnostics
# unless running a separate paired native-vs-TPR numerical check.
export TPR_TREE_BUILDER=radix
```
An invalid selector fails fast instead of silently changing semantics.
The switch only changes CPU topology construction, not attention, backward,
optimizer, or gradient finalization. Run NPU correctness before changing
the default globally.

## Tests to run

```bash
python -m pytest -vv -s \
  tests/models/mcore/tpr/unit/test_trajectory_tree_radix.py \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py

# 512 rows, 16K shared prefixes, DP8; fast tree builder only
TPR_DTA_SCALE_STRESS=1 python -m pytest -vv -s \
  tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py -k long_prefix_16k
```

Medium-scale test `tree_builder_compare` prints old and radix CPU times on
the whole global batch, and `rank_local_tree_compare` *separately* measures
old/radix builders on each DP partition (reporting max-rank latency as the
relevant local straggler proxy). Every comparison requires exact tree equality.
The 16K stress gate only constructs the new radix tree to avoid allocating
millions of one-token legacy nodes.

### Observed initial CPU benchmark (user's Docker, 2026-10-09)

128 trajectories (2K shared prefix): original 0.867s vs radix 0.021s,
**40.47x**, per-token insertion accounted for 0.844s (97.3%).
DP4 rank-local max (sequential timing proxy): old 0.350s vs new 0.004s.

256 trajectories (4K shared prefix): original 2.075s vs radix 0.096s,
**21.58x**, per-token insertion accounted for 1.979s (95.4%).
DP8 rank-local max (sequential timing proxy): old 0.386s vs new 0.008s.

Both benchmark cases passed `radix_trees == trees` equality checks;
the structural differential unit suite and model-weight/gradient parity
must be verified independently before claiming training correctness.
Per-rank times are serial CPU estimates (potential warm-up and GC effects),
**not** measured concurrent multi-DP worker times or train throughput.
