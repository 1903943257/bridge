# Stage 4: GDN TPR + CP

Sequence agreed with the user:

1. 4.1 single stateful GDN CP2 (this change).
2. 4.2 pure-GDN TPR + CP2: Push/Branch/Pop and multi-level state relay.
3. 4.3 small Hybrid CP2, real GDN/GDN/GDN/FA pattern.
4. Before full-Qwen CP integration: one CP1 20-50-step A/B to assess actual
   training impact of the tracked full/split numerical drift.
5. 4.4 full Qwen3.5 CP2.
6. 4.5 THD/remove-padding.

Stage 3.3 per-layer Linear investigation stops here. See its known-issue note;
controlled PASS is not silently promoted to unmodified-baseline PASS.

## 4.1 seam and scope

`verl/models/mcore/tpr/parallel/gdn_state.py` exposes
`forward_gdn_cp_with_state(layer, hidden_states, initial_state=None)` returning
`((output, bias), GDNLayerState)`. It reuses the actual MindSpeed helpers:

in_proj -> six-section CP->HP A2A + undo zigzag -> local Stateful Conv ->
local Stateful GDR -> gated norm -> redo zigzag + HP->CP A2A -> out_proj.

Only CP1/2, TP=SP=1, B=1, non-packed are accepted. Each call's input is the
segment's native zigzag `[S/CP,1,1024]`; state is head/channel-local after A2A.
Stateful primitives are the unchanged Stage 1 APIs. No tree/context CP dispatch
is enabled in `TPRGatedDeltaNet` or SegmentExecutor yet; that belongs to 4.2.
No changes to Megatron, MindSpeed, Ops, precision or projection chunk controls.

Qwen0.8B state placement from native A2A and parameter slicing:

| State | CP1 | CP2 per rank | Shard axis |
|---|---|---|---|
| conv | `[1,6144,4]` | `[1,3072,4]` | dim 1, separately inside Q/K/V sections |
| recurrent | `[1,16,128,128]` | `[1,8,128,128]` | dim 1, value heads |

Conv CP2 rank r holds `[Q_r(1024), K_r(1024), V_r(1024)]`, not one contiguous
3072-channel chunk of global QKV. W=4 is the primitive's boundary-state width;
recurrent final two dimensions are key/value feature dimensions, not shards.
Conv is BF16 and recurrent is expected FP32; the NPU test prints actual dtypes.
The caller must preserve rank/group/segment ownership. These bare tensor states
are not yet a validated serialized CP PrefixState or a cross-rank restore API.

## Tests and criteria

`test_stateful_gdn_cp_npu.py` uses the Stage 3.2 CP1 model as an independent
reference. CP2 uses the new seam on the native MindSpeed layer with identical
parameters. Two cases: zero initial state and nonzero conv/recurrent initial
state. S=128, BF16+SiLU. The objective contains output loss plus both final-state
loss terms, exercising final-state VJPs as well as dh0 with nonzero initial state.

- Gather zigzag output/input gradients to logical order.
- SUM CP2 replicated parameter gradients exactly once.
- Compare local final states and initial-state gradients against the appropriate
  CP1 Q/K/V and value-head slices, with global loss normalization.
- CP1 A2A calls=0; CP2 cp2hp=6, hp2cp=1 per forward. There is no FA in this test.
- Output atol/rtol=0.005; loss atol=1e-7, rtol=0.0005.
- Input/parameter/state comparisons: relative-L2 <=0.02 and cosine >=0.999;
  no missing/non-finite compared gradients. These are single-layer criteria,
  not the relaxed full-Hybrid envelope. State norms/dtypes/shapes are printed.

This does not yet certify tree branch accumulation, suffix continuation across
multiple calls, all lengths, BT=1 dw, FA, or packed sequences. BT=1 dw remains
an explicit unresolved kernel-coverage item; do not infer coverage from S=128.

Sync the production seam and test to the corresponding server verl paths.
Retain the previously validated server dirty patches including the conv dh0 fix.
The existing baseline helper provides exact SHA/import/dirty environment checks.

```bash
torchrun --master_addr=127.0.0.1 --master_port=29565 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_stateful_gdn_cp_npu.py
```

Optional CPU shape contract regression (on an environment with torch/pytest):

```bash
python -m pytest -v tests/models/mcore/tpr/unit/test_gdn_cp_state_shapes.py
```

Status: **Stage 4.1 PASS**, as reported by the user after server execution.
This covers the scoped single-layer cases above, not tree or packed execution.

## 4.2 pure-GDN TPR + CP2

`parallel/gdn_tree.py` adds a residual `GDNCPStack` and focused
`GDNCPBranchExecutor`. Each block is `hidden + GDN(hidden)`; no MLP/FA/norm
is added outside the GDN. It uses Stage 4.1's stateful CP seam unchanged.

- Push saves graph-free, compact per-layer states and a detached local input
  for recomputation, without evaluating its own loss.
- Visit creates independent direct-parent anchors, backward, then accumulates
  conv/recurrent state gradients on the parent. No global all-reduce of state
  gradients is appropriate: states remain in their rank-local HP placement.
- Pop consumes the sibling gradient sum once, recomputes its saved input,
  backward through its own loss plus state VJPs, then relays to its parent and
  releases its saved state.
- Parameters must remain unchanged while prefixes are active. Duplicate IDs,
  wrong parents/Pop order and post-failure reuse are rejected. All ranks must
  supply identical event order; automatic distributed schedule negotiation and
  rank-migration/serialized state restore are not implemented.

The NPU test uses **three real GDN layers** and a two-level tree
`R -> P -> (S1,S2)`. Every segment has 128 global tokens, BF16+SiLU, CP2
local zigzag length=64. R/P loss multiplicity=2 and S1/S2=1; global denominator
is `2 * 3 * 128 * 1024`. The executor does not multiply losses by CP or reduce
parameters; the test SUM-reduces parameter gradients exactly once after each
complete CP2 backward. Four independently rerun controls share identical weights:

1. CP1 independent full trajectories R+P+S1 and R+P+S2 (Stage 3.2 implementation).
2. CP1 connected segmented graph, including boundary clones for state VJPs.
3. CP2 connected segmented graph (Stage 4.1 seam).
4. CP2 graph-free Push/Visit/Pop, with intermediate P gradients relayed to R.

Gates print independently, then the test fails if any gate fails:

- CP1 materialized vs CP2 tree: outputs/input/parameter gradients rel-L2 <=0.08,
  cosine >=0.995 (Stage 3.2 envelope).
- CP1 connected vs CP2 connected and CP2 connected vs tree: <=0.02/>=0.999
  (Stage 4.1 envelope); every conv/recurrent boundary gradient at R and P also
  compared individually using correct per-section CP1 slices.
- Loss relative difference <=0.002, finite outputs and compared gradients.
- CP1 A2A=0; CP2 connected cp2hp/hp2cp=72/12; CP2 tree=108/18 per rank.
- Both prefix own losses evaluated once, direct-parent-only accumulation,
  both states released, executor empty. There is no FA/Ring in this test.

This is focused pure-GDN tree execution, **not a GPT/Engine/tree-request test**;
production Engine/Hybrid dispatch remains unchanged. No packed metadata,
projection chunk control, optimizer step, or third-party kernel edit is added.

Sync these files to corresponding server verl paths:

```text
verl/models/mcore/tpr/parallel/gdn_tree.py
tests/models/mcore/tpr/parallel/test_pure_gdn_tree_cp_npu.py
tests/models/mcore/tpr/parallel/test_stateful_gdn_cp_npu.py
tests/models/mcore/tpr/unit/test_gdn_branch_executor.py
```

The existing Stage 4.1 `gdn_state.py` and baseline helpers are prerequisites.
The updated 4.1 test helper only adds an optional layer_number argument; its
single-layer defaults and gates are unchanged.

```bash
python -m pytest -v tests/models/mcore/tpr/unit/test_gdn_branch_executor.py
torchrun --master_addr=127.0.0.1 --master_port=29566 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_pure_gdn_tree_cp_npu.py
```

Status: Stage 4.2 implemented; NPU results pending. The Stage 3.3 numerical
known issue and BT=1 coverage item remain tracked; neither is declared fixed.
