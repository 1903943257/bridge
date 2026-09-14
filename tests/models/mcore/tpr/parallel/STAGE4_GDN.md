# Stage 4: GDN TPR + CP

Sequence agreed with the user:

1. 4.1 single stateful GDN CP2 (this change).
2. 4.2 pure-GDN TPR + CP2: Push/Branch/Pop and multi-level state relay.
3. 4.3 small Hybrid CP2, real GDN/GDN/GDN/FA pattern.
4. 4.4 full Qwen3.5 CP2 (moved ahead of the short training A/B).
5. One CP1 20-50-step A/B to assess actual training impact of the tracked
   full/split numerical drift; this alone does not establish CP2 training stability.
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

Status: **Stage 4.2 all PASS, reported from the server (2026-09-14).**
The Stage 3.3 numerical known issue and BT=1 coverage item remain tracked;
neither is declared fixed. No new numerical metrics were supplied for this run.

### Stage 4.2 concise closeout

- Problem: a single CP2 GDN does not establish multilayer tree-state relay.
- Goal: three real GDN layers, R -> P -> (S1, S2), CP2/non-packed.
- Method: graph-free Push; independent sibling anchors; summed state gradients;
  reverse Pop with each prefix's owned loss once, then relay to its parent.
- New code: `parallel/gdn_tree.py`, `test_pure_gdn_tree_cp_npu.py`,
  `unit/test_gdn_branch_executor.py`; reused the Stage 4.1 state seam.
- Tests/results: all four controls and output/loss/input/parameter/state gates
  PASS; A2A contract and release checks PASS. Pure GDN only; no FA/Engine claim.

## Stage 4.3: small Hybrid CP2 (server CONTROLLED PASS; baseline gate still fails)

Random Qwen dimensions, **GDN -> GDN -> GDN -> FA**, including real norms/MLPs,
BF16, CP1/CP2, non-packed. Tree P128 -> (S1_128, S2_128). No projection controls,
optimizer or kernel changes. This is the production SegmentExecutor seam,
not yet Full-Qwen Engine integration or THD.

`TPRGatedDeltaNet` uses the passed Stage 4.1 A2A seam in a CP2 Ring context.
SegmentExecutor retains direct-parent GDN state and all-ancestor FA KV relay.
Other CP algorithms, CP4 and padded GDN segments are rejected. CP1 is unchanged.

FA uses the existing **TPR Ring extension with real MindSpeed RingP2P**:
external Prefix KV requires a rectangular block schedule, not the equal-length
native `dot_product_attention.ringattn_context_parallel` entry. Tests probe both
the Ring attention calls and actual RingP2P sends, without replacing their math.

Three runs with identical parameters:

1. CP1 connected segmented graph (no A2A/Ring).
2. CP2 connected segmented graph (A2A cp2hp/hp2cp=54/9, FA Ring=3).
3. CP2 Push/Visit/Pop (A2A=72/12, FA Ring=4, actual RingP2P sends >0).

Compare output probes, loss, embedding-output/input gradients, all parameter
gradients, and all **8 boundary gradients** (three conv/recurrent pairs, FA K/V).
CP1 states are sliced by Q/K/V section or recurrent heads; FA KV by native
zigzag token placement. Boundary clones in connected controls exclude the
prefix-owned loss from external dState. Graph-free saves, owned loss once,
sibling accumulation and cache release are checked.

Retain Stage 4.1/4.2 CP/relay gates: rel-L2 <=0.02, cosine >=0.999;
loss relative difference <=0.002, finite values. No numerical-drift waiver.
Standalone parameter gradients are SUM-reduced once; cancel the executor's
Engine-oriented CP loss multiplier in the test only. Boundary gradients are
rank-local and must not be all-reduced.

Sync modified/new files (in addition to existing Stage 4.1/4.2 prerequisites):

```text
verl/models/mcore/tpr/gated_delta_net.py
verl/models/mcore/tpr/segment_executor.py
tests/models/mcore/baseline/_qwen35_baseline_utils.py
tests/models/mcore/tpr/parallel/test_small_hybrid_tree_cp_npu.py
tests/models/mcore/tpr/unit/test_hybrid_segment_executor.py
```

```bash
python -m pytest -v tests/models/mcore/tpr/unit/test_hybrid_segment_executor.py
torchrun --master_addr=127.0.0.1 --master_port=29567 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_small_hybrid_tree_cp_npu.py
```

Initial local validation was syntax/static only. Server results are recorded
below. Updated order: 4.4 Full Qwen CP2, then CP1 short multi-step A/B; THD remains 4.5.

### Server feedback: CP state gate FAIL; localization pending

Reported loss CP1-connected/CP2-connected/CP2-tree:
13.204019547 / 13.203976631 / 13.203976631.
CP1 vs CP2 parameter rel-L2 ~0.0166; individual boundary gradients exceed 0.02:
GDN1 conv 0.02343, recurrent 0.02211; FA4 key ~0.0238--0.0267 in another
diagnostic. CP2 connected vs tree state rel-L2 ~1e-6--5e-6. Communication
matches: CP1 zero; CP2 connected A2A54/9, Ring3/P2P10; tree72/12, Ring4/P2P11.
This localizes the failing comparison, **not the responsible operator**.
No thresholds changed and Stage 4.3 is not declared PASS.

Sync the updated test plus `_hybrid_divergence_probe.py` in this directory.
Enable observational connected-path hooks (tree path unchanged):

```bash
STAGE43_TRACE=1 torchrun --master_addr=127.0.0.1 --master_port=29567 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_small_hybrid_tree_cp_npu.py
```

The trace covers embedding, each layer input/output, attention input/output,
GDN/FA projections, norm/MLP boundaries and final norm. Compare CP1's matching
zigzag slice against CP2 locally for both values and VJPs; no extra CP
collectives or division of rank-local activation gradients. Each rank/segment
prints first nonzero forward and reverse-boundary-order backward locations,
plus norms/absolute-L2/relative-L2/cosine/max-abs. All eight boundary gradient
metrics print before gating. This is module-boundary localization; it does not
trace inside conv/GDR/Ring kernels. A nonzero backward difference may be inherited
from a different upstream gradient and requires a same-upstream replay before
attributing it to a backward implementation. CPU snapshots can affect execution
timing; trace-off remains the original correctness run.

### First-layer out_proj canonical-input replay

Latest trace reports the first forward difference at
`layer1.attention.out_proj.output` for segment2 on both ranks. The first
reverse-boundary VJP difference at `final_norm.output` uses different upstream
gradients and is not evidence of a backward kernel defect.

With `STAGE43_TRACE=1`, after all three measured runs, the probe now replays
first-layer out_proj for P/S1/S2 (same command above). Sync both the probe and
test file. Parameters/buffers must match. Canonical input is the captured CP1
full input, restored to its original dtype in contiguous storage; its native
zigzag slice supplies the same-input local controls. Print:

- Captured CP1/CP2 input and output, own-input output reproduction.
- Same full input and same local input through both module instances.
- Full versus local shape within each instance on canonical inputs.
- CP2 canonical versus captured-own input, plus full/local repeatability.
- Shapes, original dtype/strides, exact equality, norms, rel-L2, cosine, max-abs.

Forward autograd remains enabled like connected mode, but no replay backward
is called. No full-model substitution, optimizer/kernel change, or new numerical
threshold. Replay is outside the measured communication contract. This is a
forward localization control, not a backward same-upstream replay. If own-input
reproduction fails, contiguous layout/dispatch or repeatability must be resolved
before attributing the observed original difference solely to sequence length.
Actual replay results remain pending on NPU.

### Optional causal control: CP1 layer1 out_proj 128 -> zigzag 2x64

Server replay found identical outputs across CP1/CP2 module instances on the
same shape/input, but full vs shard within one instance differed (segment2:
rank0 rel-L2 1.10e-5/max-abs 2.44e-4; rank1 2.78e-6/6.10e-5).
This identifies a shape-dependent local difference, not proof that it explains
all final gradient drift.

`STAGE43_OUT_PROJ_ZIGZAG64=1` intervenes ONLY during CP1 connected F/B:
first GDN out_proj processes token groups [0:32,96:128] and [32:96], then
inverts their permutation. This retains autograd and parameter-gradient
accumulation; it is not contiguous `split(64)`. Three full calls (P/S1/S2)
must be controlled. CP2, other layers, kernels and thresholds remain unchanged.
The module is restored on exit. Standalone native replay is skipped in this
mode because captured controlled outputs must not be labelled native replay.

Sync `_first_gdn_zigzag_control.py`, the updated main test, and optionally
`../unit/test_first_gdn_zigzag_control.py`; keep the existing trace helper.

```bash
STAGE43_TRACE=1 STAGE43_OUT_PROJ_ZIGZAG64=1 \
  torchrun --master_addr=127.0.0.1 --master_port=29567 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_small_hybrid_tree_cp_npu.py
python -m pytest -v tests/models/mcore/tpr/unit/test_first_gdn_zigzag_control.py
```

Compare to the trace-on/control-off results: does first-layer out_proj (and
then the whole first layer) become exact, and how much do output/input/parameter
and all individual state errors drop? Whole-layer exactness is an observation,
not guaranteed by controlling only out_proj. Passing existing gates is labelled
**CONTROLLED PASS (not unmodified baseline PASS)**. No threshold for 'significant
drop' is invented. Default control=0 preserves the baseline. Server results
and CPU torch regression results remain pending; local checks are static only.

### Additive layer1 MLP fc2 control

Server out_proj-only control removed that first difference; first nonzero moved
to layer1 MLP linear_fc2. Output/input/parameter rel-L2 dropped to approximately
0.00627/0.0150/0.01376, but individual state gates still failed (worst 0.02824).
It does not establish an unmodified baseline PASS.

`STAGE43_MLP_FC2_ZIGZAG64=1` independently applies the same rank-zigzag 2x64
intervention to CP1 layer1 MLP linear_fc2 only. Both switches default to zero.
Each enabled projection must count exactly three interventions (P/S1/S2).
The control remains bias-free and differentiable, restoring token order and
original methods on exit. No CP2/kernel/threshold change.

Sync `_first_gdn_zigzag_control.py`, `_hybrid_divergence_probe.py`,
`test_small_hybrid_tree_cp_npu.py`, and the optional unit regression file.

```bash
STAGE43_TRACE=1 STAGE43_OUT_PROJ_ZIGZAG64=1 STAGE43_MLP_FC2_ZIGZAG64=1 \
  torchrun --master_addr=127.0.0.1 --master_port=29567 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_small_hybrid_tree_cp_npu.py
```

`STAGE-4.3 LAYER1` explicitly reports whether all observed layer1 forward
boundaries (including layer output) are exact and whether the first difference
is in layer2, per rank/segment. This is not a claim about unobserved internal
ops. Compare existing output/input/parameter/all-eight-state metrics to both
baseline and out_proj-only runs. Results pending; CONTROLLED PASS/FAIL labelling
and original numerical thresholds are retained.

### Stage 4.3 closeout: double-control server PASS

Both ranks: all observed first-layer boundaries/output exact; first divergence
moved to layer2 attention.out_proj. Double-control output/input/parameter rel-L2
~0.00455/0.0134/0.01204 versus baseline ~0.00823/0.0180/0.0166.
State aggregate rank0/rank1=0.01788/0.01332; worst=0.01916/0.01926.
CP2 connected vs tree: output exact, input/parameter ~0.00350/0.00344,
state ~1e-6. **CONTROLLED PASS only**, not unmodified-baseline PASS.
Stop layerwise controls; retain this numerical-drift known issue.

## Stage 4.4: Full Qwen CP2 Engine tree (implemented; NPU pending)

New test: `test_full_qwen35_tree_cp_npu.py`. Random Qwen3.5-0.8B dimensions,
24 layers/18 GDN/6 FA, BF16, dropout=0, TP/PP/EP/DP=1, non-packed.
Tree P128 -> (S1_128, S2_128), with shared-prefix owned loss weighted for both
materialized trajectories, including each branch's distinct boundary label.
Three independent runs have identical initial weights and inputs:

1. CP1 connected segmented graph.
2. CP2 connected segmented graph.
3. CP2 Engine `forward_backward_batch` + explicit `TPRForwardBackwardRequest`.

The third run executes the real thin entry, CP runtime resolution, fixed tree
scheduler and SegmentExecutor. Only an observing executor subclass is injected;
no scheduling/forward/backward math is replaced. The Engine fixture supplies a
random GPT model, no distributed optimizer, and a real SUM parameter-gradient
finalizer called exactly once. `grad_scale_func=loss/2` cancels the executor's
CP multiplier to match the connected globally sum-normalized objective.
The Engine itself aggregates CP loss; the test does not reduce it twice.
This validates Engine tree routing, **not DDP/optimizer initialization or updates**.

Checks:

- Every forward dispatches all 24 layers; layer boundaries `[128/CP,1,1024]`.
- GDN states: CP-local conv `[1,3072,4]`, recurrent `[1,8,128,128]`.
- All 48 boundary gradients: 36 GDN conv/recurrent and 12 FA K/V, each separately.
- Logical output probes, target logprobs, loss, embedding-output/input gradients,
  all parameter gradients; finite values. State gradients remain rank-local.
- Graph-free Push; P owned loss once at Pop, each leaf once; caches released.
- CP1 no A2A/Ring. CP2 connected A2A324/54, FA Ring18; Engine tree A2A432/72,
  FA Ring24. Real MindSpeed RingP2P must execute (TPR rectangular Ring extension).

Two independent numerical reports: `cross-CP-connected` and
`CP2-connected-vs-Engine-tree`. Retain 4.3 rel-L2<=0.02/cosine>=0.999 gates
for outputs/input/parameters/state, loss rtol<=0.002. Target-logprob maps use
the same rel-L2/cosine envelope. Every state gate is evaluated even after an
earlier one fails. Both ranks' failure flags are combined; any failure keeps
the overall test FAIL. Cross-CP failure cannot silently invalidate or promote
the independently reported Engine-relay result. No shape controls are installed;
nonzero Stage 3.3/4.3 control environment variables are rejected.

Only one model is kept on NPU at a time; initial weights and gradient snapshots
remain on CPU. Host memory must accommodate the initial state plus three full
gradient maps. No automatic architecture/sequence reduction on OOM.

Sync the new test and this document, with existing Stage 4.3 prerequisites
already synchronized (`test_small_hybrid_tree_cp_npu.py` supplies its plan,
communication probe and state-sharding helper). No production file or third-party
source change is required in this stage. Existing Engine TPR patch/entry from
Stage 3.3 and CP2 Ring/GDN dispatch from Stage 4.3 must be installed.

```bash
torchrun --master_addr=127.0.0.1 --master_port=29568 --nproc_per_node=2 \
  -m pytest -s -v tests/models/mcore/tpr/parallel/test_full_qwen35_tree_cp_npu.py
```

Local syntax/static checks only (no local torch/pytest/NPU). Actual loss/gradient
metrics and PASS/FAIL await the server. Next: CP1 20-50-step A/B, then 4.5 THD.

### Stage 4.4 first server run and focused relay diagnostic

Initial server report: cross-CP output/input/parameter/state rel-L2 approximately
0.05892/0.13832/0.14342/0.13010. Engine relay also has a distinct failing
`gdn.5.conv` gate: rel-L2=0.02209246, cosine=0.999766760. Overall **FAIL**;
neither failure is waived or automatically attributed to BF16 rounding.

The test now appends one CP2 connected repeat after the original three runs,
with a freshly built identical model loaded from the same initial parameters,
same seed/plan, and no optimizer update. It uses the original connected path,
including its existing communication and lifecycle checks. Peak host storage
now includes four gradient snapshots; still only one model is kept on NPU.

Both ranks print `STAGE-4.4 DIAGNOSTIC` for connected-vs-repeat,
connected-vs-Engine-tree, and repeat-vs-Engine-tree:

- Loss values/relative difference; aggregate parameter and state metrics.
- `gdn.5.conv` full tensor and slots 0..3 on its final state axis: exactness,
  reference/actual norm, norm ratio, absolute-L2, relative-L2, cosine, max-abs.
- Slot numbers mean state storage order; no unsupported temporal interpretation.

Diagnostics execute before the original gates. No repeatability error is
subtracted and no new numerical threshold is introduced. Sync the updated
`test_full_qwen35_tree_cp_npu.py` and run the same port-29568 command above.
Local syntax checks only; repeat NPU results pending.
