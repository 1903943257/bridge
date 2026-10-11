# P0-4 — Real Qwen3-1.7B 50-step Full vs TPR P512/S512

Status: **implementation committed, Ascend NPU execution pending**.

## Why P0-4 follows P0-3

The actual P0-3 experiment on one real TQ row measured:

| Pair | Response mean absolute logprob difference | Sampled gradient rel L2 |
|---|---:|---:|
| Controlled Full vs TPR One-Node M1024 | **0** | ~0.01111 |
| TPR One-Node M1024 vs TPR P512/S512 | **0.03291573** | ~0.26541 |
| Native CP1 vs native CP2 | **0.03568786** | ~0.31301 |

A one-node TPR call has precisely equal response logprobs to
full-sequence native attention at the same M1024. Only changing to
physical P512/S512 introduces the much larger forward difference;
the cause remains unresolved among physical GEMM M, TPR rectangular
attention/KV and cross-layer propagation. This comparison alone
does not prove that gradient relays are correct or that 50-step
training is equally effective.

Previous **P0-B 50-step CP1/2/4** was NOT a TPR optimizer training test.
P0-4 adds the missing Full-vs-TPR control.

## Experiment design

- Model: **real pretrained Qwen3-1.7B** using the existing
  `test_tpr_qwen3_compatibility_npu._make_qwen_model` fixture,
  BF16 forward/backward, 28 layers, no dropout, no FP32 GEMM.
- Both modes construct the **same TPR-capable model** from the original
  Qwen checkpoint, with bitwise identical sampled initial FP32-master
  and BF16 weights verified by the comparator.
  - `full`: no active TPRAttentionContext; regular complete
    1024-token full GPT forward/backward. This is the same
    *controlled Full* path validated by P0-3, which matched
    official native CP1 response logprobs exactly.
  - `tpr`: actual production `SegmentExecutor`
    `push(P512) -> visit_leaf(S512) -> pop(P512)`,
    rectangular causal attention, detached prefix KV reuse,
    backward dKV relay into 28 ancestor layers.
- Data: real `tq_batch.pt`; identical deterministic real-TQ
  1024-token window schedule and same source-data SHA,
  sampled via P0-B's `_load_real_tq_rows`.
  Each window uses 512 previous real tokens and 512 future
  real response tokens. Neighbor windows may overlap in context,
  but supervised response spans do not overlap.
  Hold out **entire original trajectories**, never windows of a
  training trajectory, for validation.
- Objective: **real response_masked NLL**. Every run uses the
  exact same actual supervised response tokens.
  **The root's last query at position 511 predicts response[0]**
  if that mask bit is true; the leaf's local queries 0..510
  predict response[1..511]. The denominator is the actual
  count of true response_mask bits per 1024-token window,
  not unconditionally 512. This is more faithful than
  P0-3's all-512-target NLL.
- Optimizer: FP32-master AdamW, LR `2e-5`, betas `(0.9,0.95)`,
  eps `1e-8`, weight decay=0, grad clip norm 1.0.
  BF16 model gradients are transferred to FP32 master grads
  before clipping and stepping. Both modes have identical
  optimizer initialization and LR schedule.
- Duration: `50` actual optimizer steps, same window IDs at
  corresponding steps. No distributed DP/CP, just CP1;
  no synthetic data and no artificial padding.
- **Shared evaluation oracle:** after steps
  `0,1,2,5,10,20,30,40,50`, both models are evaluated on
  held-out real trajectories using the **same no-context
  Full-M1024 model forward**. The reported validation NLL
  gap and masked per-token logprob gap therefore compare
  *trained parameters*, not different evaluation backends.
  A fixed real training window is also measured through the
  shared full oracle as a train-probe.
- Parameter accuracy: sampled FP32 master
  `delta_theta(step) = theta(step)-theta(initial)`
  relative L2 and cosine, sampled weight L2, and **BF16
  preclip gradient samples at steps 1,2,5,10,20,30,40,50**,
  with relative L2/cosine. These are
  **strided sampled values from all 226 parameter names**, not
  all-parameter global L2 or complete optimizer state.
  Each step prints NLL, pre-clip FP32 grad norm and clipping state.

Limitations:
- Only a true *two-node chain*, no siblings and no shared-prefix
  benefit across separate trajectory windows. This is a
  numerical training-parity experiment, not a speedup claim.
- This is a controlled masked-NLL replay, **not PPO/GRPO**.
  The real actor requires advantages, behavior-policy old
  logprobs, clipping, policy KL and more complex tree topology.
- A limited pool of real SWE trajectories is reused, and
  validation has few independent trajectories. 50 steps
  cannot demonstrate generalization or RL reward convergence.
- The test uses explicit FP32-master AdamW instead of native
  Megatron distributed optimizer; never claim that it
  reproduces VERL's distributed actor optimizer or replay.

## Running P0-4

**On the host** (do NOT `git pull` in the Docker VERL checkout):

```bash
cd /workspace/bridge
git pull --ff-only
```

**Inside the Ascend NPU Docker**:

```bash
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
cd /workspace/uni-agent/verl

# Uses same default real dump as earlier P0 stages; override only if needed:
# export TPR_REAL_TQ_BATCH=/path/to/the/actual/tq_batch.pt
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B

# Preflight with *two actual optimizer steps in both modes*:
TPR_P0_TPR50_STEPS=2 bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_full_multistep.sh

# Full 50-step Full-vs-TPR run (no overlap of output filenames):
TPR_P0_TPR50_STEPS=50 bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_full_multistep.sh
```

Launches two fresh **single-NPU** torchrun processes sequentially,
with localhost ports 29821 and 29822; change port-base via
`TPR_P0_TPR50_MASTER_PORT_BASE`. Never `--standalone`.
Both runs write to `/tmp/tpr_p0_tpr50/` by default:

- `full_train50.pt` — checkpoints, training events, 50 optimizer steps.
- `tpr_train50.pt` — same schema/schedule, TPR 50 optimizer steps.

If one mode has succeeded and the other fails, rerun **only** the
unsuccessful mode rather than repeating the completed 50 optimizer steps:

```bash
# TPR-only retry; Full artifact remains untouched:
TPR_P0_TPR50_STEPS=50 TPR_P0_TPR50_RUN_MODES=tpr \
  bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_full_multistep.sh
# Or run Full alone with TPR_P0_TPR50_RUN_MODES=full.
```

At milestones, each saved artifact is an **atomic replacement**.
It includes `steps_completed`; the comparator refuses incomplete
runs and checks that data SHA, schedule, checkpoints, sampled initial
weights, mask, optimizer hyperparameters and shared evaluation backend
match across modes. It does NOT support optimizer-state resume after
interruption; rerun the failed mode from step0.

After the artifacts exist, comparison-only is safe and does not
launch the NPU:

```bash
TPR_P0_TPR50_STEPS=50 TPR_P0_TPR50_COMPARE_ONLY=1 \
  bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_full_multistep.sh

# Or, after rsync, in Docker VERL:
python tests/models/mcore/tpr/parallel/compare_real_tq_tpr_full_multistep.py \
  /tmp/tpr_p0_tpr50 --steps 50
```

Key comparison lines:
`P0_4 STEP=50 COMMON_FULL_VALIDATION`,
`P0_4 STEP=50 HELDOUT_RESPONSE_LOGPROB`,
`P0_4 STEP=50 MASTER_UPDATE`,
and `P0_4 STEP=50 PRECLIP_GRAD`.

### How to assess the 50-step result

A small loss gap alone is not enough; compare:

1. Shared Full-oracle heldout NLL trajectories and **improvement
   from step0**, measured after identical actual optimizer counts.
2. Shared Full-oracle token-level logprob MAE/max/p95 at steps
   0,1,2,5,10,20,30,40,50 (only actual mask=true tokens).
3. The FP32-master **cumulative update vector** rel L2/cosine;
   update-path separation can grow while loss remains similar.
4. Training NLL (different forward paths, same logical targets)
   and preclip grad norms; clipping events can differ.
5. Milestone preclip BF16 gradient sampled relative L2/cosine (step1 reflects the initial numerical difference; later steps also include drift from changed parameters).

This 50-step exercise is **a first experimental numerical stability
gate**. If both paths learn similarly it does not prove TPR is
semantically correct for multi-sibling trees, PPO/GRPO or later
long-horizon optimizer training; if they diverge, follow up with
matched-gradient, layer-token and optimizer-state diagnostics.

## Observed NPU results: complete 50-step Full vs P512/S512 TPR (2026-10-11)

**Measured, not simulated.** User-provided output:
`P0_4 CONTROL status=PASS identical_initial_weights=True`.
There were **26 training windows from original trajectories [0..5]**
and heldout original trajectories **[6,7]**, with **4068 supervised
validation token logprobs** per milestone. CP=1, original Qwen3-1.7B,
BF16 model math, FP32-master AdamW, `clip_norm=1`,
real response_mask, common Full-M1024 evaluation for *both* learned
models. All 50 optimizer steps completed for each branch.

### Same-backend heldout-validation NLL

| Step | Full NLL | TPR NLL | TPR - Full |
|---:|---:|---:|---:|
| 0 | 0.895805299 | 0.895805299 | 0 |
| 1 | 0.749957465 | 0.750150315 | +0.000192851 |
| 2 | 0.703529477 | 0.704159990 | +0.000630513 |
| 5 | 0.667764090 | 0.666939482 | -0.000824608 |
| 10 | 0.743025258 | 0.742520280 | -0.000504978 |
| 20 | 0.637367889 | 0.635400288 | -0.001967601 |
| 30 | 0.576584235 | 0.574091919 | -0.002492316 |
| 40 | 0.642815270 | 0.644875944 | +0.002060674 |
| 50 | **0.613095485** | **0.617147543** | **+0.004052058** |

Full heldout NLL improvement `val0-val50=0.282709815`
(31.56% of its initial NLL), TPR improvement `0.278657757`
(31.11%). Both improved substantially. On these heldout windows TPR's
final NLL is 0.004052 higher, approximately **0.66% relative to Full
final NLL**, and achieved ~1.43% less *absolute NLL improvement*.
Do not call this a statistical significance estimate: only two
independent heldout original trajectories, correlated windows, one
checkpoint and one deterministic training order.

### Parameter, gradient and prediction drift

All `MASTER_UPDATE` and `PRECLIP_GRAD` entries refer to **fixed strided
samples of up to 512 elements per parameter across 226 parameter names**,
**not** full-parameter errors. Master update rel L2 compares
`theta_mode(step)-theta_initial` with Full's update as the denominator.
Later-step gradient comparisons also include different parameter and
optimizer histories; **only step1 is a same-initial-weights gradient
control**.

| Step | Sampled grad rel L2 | Sampled grad cosine | Sampled master update rel L2 | Master update cosine | Shared Full eval response logprob MAE | Logprob max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | n/a | n/a | n/a | n/a | 0 | 0 |
| 1 | 0.146133691 | 0.991007621 | 0.248846856 | 0.969032967 | 0.021116894 | 0.778535 |
| 2 | 0.113112301 | 0.994308336 | 0.189557474 | 0.982029440 | 0.019914421 | 0.496387 |
| 5 | 0.178606692 | 0.985311731 | 0.142001972 | 0.989917492 | 0.019759780 | 0.436587 |
| 10 | 0.112014186 | 0.993916297 | 0.129703733 | 0.991584621 | 0.029003737 | 1.094727 |
| 20 | 0.413394915 | 0.935090088 | 0.157737747 | 0.987553644 | 0.030930072 | 0.783214 |
| 30 | 0.251800642 | 0.970540259 | 0.190190046 | 0.981894177 | 0.033486567 | 0.997735 |
| 40 | 0.223563157 | 0.975948215 | 0.219935542 | 0.975789019 | 0.049198478 | 2.048016 |
| 50 | **0.421599437** | **0.924154019** | **0.249367389** | **0.968881573** | **0.053906588** | **2.159992** |

At Step50 the shared Full eval response logprob p95 absolute
difference = `0.266277403`, rel L2 = `0.094000271`,
cosine = `0.995649409`. Sampling master weights gives an
absolute vector diff norm of `0.011095521`, Full update sampled norm
`0.044494674`, and TPR update sampled norm `0.044456427`.
The update norm magnitudes are similar while the sampled vector
direction differs; a near-identical norm is not parameter parity.

At step1, sampled preclip gradient rel L2 already differs by 14.61%
under *identical* initial weights and first example. At step50
the sampled preclip gradient difference is 42.16%, but it now includes
different **trained weights**, unlike a fresh same-weight comparison.
The measured step50 prediction differences are from two different
learned policies evaluated on the same backend. Do not interpret
these as step50 same-weight TPR-vs-Full kernel drift.

Training NLL also remained finite with both modes completing.
Representative step50 current-window training NLL:
Full `0.155159459`, TPR `0.147678509`. That single training
window does not replace heldout evaluation and must not be read as
a general improvement of TPR.

### What this establishes, and what it does not

**Observed:** 50 deterministic matched optimizer steps under
response-masked NLL do not show a catastrophic or monotonic relative
degradation in heldout NLL. The two paths do produce meaningfully
different sampled optimizer update vectors and token-level
probabilities, with logprob MAE rising from ~0.021 at step1 to
~0.054 at step50 despite loss similarity.

**Unproven:** TPR has no missing/shared KV gradients, production
PPO/GRPO reward convergence equivalence, persistent stability for
hundreds/thousands of steps, or correctness with several children
sharing a prefix. Likewise this is not a statement that a 25% sampled
update-direction difference is a tolerable acceptance threshold.
Native CP2's earlier 50-step sampled parameter update rel L2
`0.2950` was from a *different CP attention/validation setup*;
do not use it to declare P0-4 TPR better or worse than official CP.

### Suggested acceptance progression

1. **P0-4 status:** 50-step matched-NLL numerical stability
   *diagnostic completed*, not strict gradient parity PASS.
2. **P0-5 next:** fixed true multi-sibling prefix tree:
   identical training trajectories, response masks, ancestor
   dK/dV sums, saved roots; compare against independent Full
   trajectories with same weighted objective and FP32 master updates.
3. **Then actual PPO replay:** fixed rollout/logprobs/advantages
   from an identical actor snapshot and identical clip/KL/entropy
   configuration. Measure PPO policy-loss, clip fraction,
   importance-ratio outlier rates, KL and common-oracle
   response logprobs at successive optimizer steps.
4. For any PPO training divergence, compare a *same-weight*
   Full vs TPR PPO backward at matching milestones to distinguish
   numerical/scheduling error from accumulated policy-parameter drift.
5. Performance and memory benefit analysis should happen only
   after real shared-prefix workloads are covered; this two-node
   chain does not quantify actual KV reuse speedup.

**Implementation and 50-step user Ascend NPU measurement complete; broader TPR/PPO training parity remains open.**
