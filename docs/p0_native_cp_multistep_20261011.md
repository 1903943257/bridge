# P0-B: 50-step real-TQ native MindSpeed CP training control (2026-10-11)

Purpose: test whether CP=1/2/4 forward/backward numerical drift accumulates
or changes learning under **actual optimizer steps** on identical real tokens.

## Scope and interpretation

- Checkpoint: **Qwen3-1.7B pretrained**, BF16 model forward/backward.
- Attention: **unaltered** Megatron + MindSpeed `megatron_cp_algo`.
  The CP1/2/4 sharding and model builders reuse
  `test_native_cp1_cp2_cp4_drift_npu.py`.
- Training: 50 sequential optimizer updates, no invented synthetic tokens,
  fixed deterministic real-window order and disjoint held-out trajectory IDs.
- Dataset: the existing `tq_batch.pt` contains **8 SWE trajectories**.
  By default each eligible trajectory is decomposed into up to **16
  1024-token REAL windows** with non-overlapping 512-token supervised
  response spans: 512 immediately preceding context tokens (prompt or prior
  response) + 512 actual response tokens. Neighboring windows intentionally
  overlap in the CONTEXT half, not in the supervised response half,
  with the corresponding recorded response_mask. This can yield up to
  128 different real-token windows, but **still only 8 independent
  trajectories**. No padding or synthetic filler is used. Trajectories
  with insufficient prompt/response length or all-zero masks are skipped.
  The final quarter of *original trajectories* is held out entirely,
  with up to 8 held-out windows selected for efficient validation.
  If fewer than 2 trajectories are eligible, the test refuses to run.
  Window order is deterministically shuffled for training and identical
  across CP=1/2/4. Configure
  `TPR_P0_B_MAX_WINDOWS_PER_TRAJECTORY` (default 16, maximum 64).
- More than 64 genuinely **independent trajectories** would be preferred
  for strong training conclusions. The 50-step run reports
  `low_independent_trajectory_count=True` with only 8 originals.
  Distinct 1024-token windows from the same rollout are real training
  examples but are *not statistically independent* tasks. This is a
  controlled optimizer-drift experiment, **not** a generalization claim.
- Loss: masked next-token NLL, *not* PPO/GRPO. The prompt-boundary
  query at global index 511 predicts the first response token; the
  supervision denominator is the global count of true response_mask
  bits, not batch_size * CP.
- Every rank uses the **same pretrained weights and same optimizer step
  schedule**. Parameters are BF16 for model compute, but optimizer
  has explicit FP32 master parameters and FP32 AdamW moments
  (`betas=(.9,.95)`, `eps=1e-8`, weight decay=0).
  Per-rank `param.grad` originates from BF16 backward, and its
  CP reduction is **FP32 SUM by default** before clipping to 1.0.
  `TPR_P0_B_GRAD_REDUCE=bf16` optionally repeats the earlier P0-1
  BF16 gradient SUM for attribution.
- This is a manually synchronized bare GPTModel + FP32 master AdamW
  training probe, **not** a verified production Megatron Distributed
  Optimizer / VERL actor PPO training path.
- The primary validation is performed in each run's **own native CP**
  backend (same real held-out row indices). It is not a shared CP1
  evaluator: absolute CP2/4-vs-CP1 validation gaps also include
  backend numerical differences. Compare *validation improvement from
  step0*, and sampled parameter update vectors
  `master_param(step)-master_param(0)`.
- All 226 parameter names are covered, with up to 512 deterministic
  strided sampled master values each. Reported parameter relative L2 and
  cosine are **sampled-only**, not full-gradient / full-model metrics.
  No full optimizer checkpoints are written by default.

## Run (host pull, Docker rsync and torchrun)

**Outside Docker, on the host**:

```bash
cd /workspace/bridge
git pull --ff-only
```

**Inside the NPU Docker container**, after the host updates the mounted
bridge checkout:

```bash
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
cd /workspace/uni-agent/verl

# Use the same real dump path as P0-1. The launcher has a repo-known default.
export TPR_REAL_TQ_BATCH=/path/to/real/tq_batch.pt
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_P0_B_OUTPUT_DIR=/tmp/tpr_p0_native_cp_b

# Optional preflight smoke: writes native_cp{1,2,4}_train2.pt, separate from 50.
TPR_P0_B_STEPS=2 bash tests/models/mcore/tpr/parallel/run_native_cp_multistep.sh

# Full run: native CP 1/2/4, exactly 50 sequential optimizer steps each.
TPR_P0_B_STEPS=50 bash tests/models/mcore/tpr/parallel/run_native_cp_multistep.sh
```

This uses three fresh `torchrun` groups with
`--master_addr=127.0.0.1`,
`--master_port=29661/29662/29664`, and
`--nproc_per_node=1/2/4`. Set
`TPR_P0_B_MASTER_PORT_BASE` to a different integer for port conflicts.
It does **not** use `--standalone`.

For a comparison-only rerun after all three output files exist:

```bash
TPR_P0_B_STEPS=50 TPR_P0_B_COMPARE_ONLY=1 bash tests/models/mcore/tpr/parallel/run_native_cp_multistep.sh

# Or call the synced comparator directly inside Docker:
python tests/models/mcore/tpr/parallel/compare_native_cp_multistep.py \
  /tmp/tpr_p0_native_cp_b --steps 50
```

Artifacts: `native_cp1_train50.pt`, `native_cp2_train50.pt`,
`native_cp4_train50.pt` under the output directory. Each is
atomically rewritten at milestones 1, 2, 5, 10, 20, 30, 40, 50.
If interrupted, **do not mistake** partial `steps_completed` for
completed training; there is no optimizer-state resume in this P0.

At each step: global train NLL, number of supervised tokens, FP32
master grad norm before clip, estimated FP32 master update norm,
step number and real trajectory:window ID. At milestone: held-out mean NLL
and per-row NLL, all model parameter-name-strided sample witnesses
for FP32 masters and BF16 model copies; rank consensus of model weights.

## How to interpret results

1. Verify identical `data_sha`, `stream_sha`, schedule, initial
   sampled BF16 weight hashes, checkpoint and optimizer hyperparameters.
   Comparator refuses mismatches and incomplete runs.
2. The meaningful loss metric is held-out `val_delta = val50 - val0`;
   also compare `val_improvement_gap` between CP configurations,
   not just their own already-different step0 losses.
3. Inspect the **sampled FP32 master update vector** relative L2/cosine
   to identify update-direction drift, rather than raw pretrained
   parameter cosine (which can hide tiny step differences).
4. This can detect severe trajectory divergence but alone cannot
   establish whether CP or TPR meets production PPO parity.
   Next run fixed official CP Q/K/V attention and single Linear X/W/dY
   isolation controls, and then replay PPO with real old_logprobs and
   advantages.

## P0-2 isolation plan (2A measured, 2B implemented, NPU pending)

- **P0-2A GEMM replay**: real-X/W/dY single-NPU replay using
  official Megatron Linear and Zigzag M=1024/512/256; **completed on user
  Ascend** and reported below.
- **P0-2B Attention replay**: implemented with real pretrained Qwen
  layer-core Q/K/V and the *same upstream dO*, captured from CP1.
  Native CP1 core attention is replayed once for a self-consistency gate,
  then MindSpeed CP2/CP4 official Ring runs with the frozen operands.
  Outputs and dQ/dK/dV are restored to original global positions.
  An LSE comparison requires a separate kernel-statistics capture and
  is not part of the current probe.
- If the Stage-B 50-step gaps are large, perform both isolation tests
  *before* attributing the drift to any one kernel family.

**Stage B and P0-2A have been run on the user's Ascend host. P0-2B is implemented but pending NPU execution.**


## Stage B measured results (user Ascend, 2026-10-11)

The full 50-step CP1/CP2/CP4 runs completed and the comparator printed
`P0_B status=DIAGNOSTIC_ONLY`. This is **real measured data**, not
a simulated curve. Validation uses **each configuration's own CP backend**.

| Metric vs CP1 | CP2 | CP4 |
|---|---:|---:|
| Step0 validation NLL gap | -0.00173233449 | -0.000970900059 |
| Step1 sampled FP32-master update rel L2 | 0.238965762 | 0.228927748 |
| Step5 sampled FP32-master update rel L2 | 0.143791291 | 0.135750437 |
| Step20 sampled FP32-master update rel L2 | 0.180051274 | 0.175499841 |
| Step50 sampled FP32-master update rel L2 | **0.295016591** | **0.323183914** |
| Step50 sampled FP32-master update cosine | 0.956440711 | 0.947757730 |
| Step50 validation NLL gap | -0.00103177875 | +0.00134871155 |
| Step50 validation improvement gap | -0.000700555742 | -0.00231961161 |

`val_improvement_gap` = CP candidate `(val0 - val50)` minus CP1
`(val0 - val50)`: negative values mean slightly *less improvement*.
It does not demonstrate CP training correctness, convergence parity,
PPO clipping parity or TPR correctness. The sampled parameter update
vectors **do** drift apart over steps 10-50, while the own-backend
validation NLL gap oscillates near zero, rather than growing monotonically.
Do not infer a numerical tolerance for TPR from this experiment.

### Immediately next: P0-2A real GEMM decomposition (implemented)

`tests/models/mcore/tpr/parallel/test_native_gemm_m_replay_npu.py`
captures **actual pretrained Qwen CP1 X and upstream dY** during a true
masked real-TQ loss/backward. For each target native Megatron QKV/Proj/FC1/FC2
module it independently replays the SAME module and weight with:

- Full physical `M=1024`;
- Rank-wise CP2 Zigzag `M=512` and restored global row order;
- Rank-wise CP4 Zigzag `M=256` and restored global row order.

It logs native forward replay vs originally captured Y, restored Y,
dX, and full-vs-sharded dW using both diagnostic FP32 and BF16 SUM.
This **holds upstream dY fixed** and does not change attention,
training, weight dtype, model weights, or patch native kernels.
This is a single-GEMM attribution test, **not** proof that all CP drift
comes from GEMM. Configure `TPR_P0_2A_LAYERS=1,4` for additional layers
(the default is zero-based layer 1). The result is a JSON artifact, not
a pass/fail float-tolerance gate.

**HOST:**

```bash
cd /workspace/bridge && git pull --ff-only
```

**DOCKER:**

```bash
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
cd /workspace/uni-agent/verl
# Reuse the exact REAL TQ file from stage B if the default location differs:
# export TPR_REAL_TQ_BATCH=/path/to/tq_batch.pt
bash tests/models/mcore/tpr/parallel/run_native_gemm_m_replay.sh
```

P0-2B native Ring attention Q/K/V and backward isolation is now
**implemented**; see the next section. It has not yet been run on NPU.


## P0-2A actual single-Linear GEMM results (user Ascend, 2026-10-11)

**Same real pretrained Qwen X/W/upstream dY** for all shapes; native
Megatron GEMMs, no TPR, no parameter update, no forced FP32 GEMM.

| Real layer-1 module | M | Y relative L2 | dX relative L2 | dW FP32 SUM rel L2 | dW BF16 SUM rel L2 |
|---|---:|---:|---:|---:|---:|
| QKV | 512 | 1.73645e-5 | 3.15012e-5 | 0.00226216 | 0.00280380 |
| QKV | 256 | 2.72333e-5 | 3.15012e-5 | 0.00216493 | 0.00338855 |
| Proj | 512 | 1.71770e-5 | 2.31285e-5 | 0.00229084 | 0.00281560 |
| Proj | 256 | 2.07720e-5 | 2.31285e-5 | 0.00226512 | 0.00357058 |
| FC1 | 512 | 2.42566e-5 | 4.24973e-5 | 0.00227529 | 0.00285062 |
| FC1 | 256 | 2.42566e-5 | 4.24973e-5 | 0.00223980 | 0.00354264 |
| FC2 | 512 | 3.39641e-5 | 2.69217e-5 | 0.00230942 | 0.00281991 |
| FC2 | 256 | 2.51617e-5 | 2.83232e-5 | 0.00229348 | 0.00358830 |

**Interpretation:** physical M changes produce measurable BF16-native
GEMM differences at layer1; per-module Y/dX relative L2 is ~1.7e-5
to 4.3e-5, while **dW differs by ~0.216%-0.231% using FP32 SUM**
or **~0.280%-0.359% using BF16 SUM**. BF16 aggregation is an
additional drift contributor on this real operand control.

This is **not a whole-model accumulation bound**: the observed layer1
single-Linear errors can propagate nonlinearly across 28 layers and
through attention, and the fixed upstream dY removes feedback coupling.
The P0-1 CP logprob drift and P0-B 50-step sampled weight divergence
remain unexplained by a single isolated GEMM.

The P0-2A original output was
`/tmp/tpr_p0_gemm_replay/p0_2a_gemm_replay.json`
with 8 measured module/CP reports. No user server environment changes
are required to inspect this JSON.

## P0-2B native Attention replay: implemented, pending Ascend measurement

The new test
`tests/models/mcore/tpr/parallel/test_native_attention_qkv_replay_npu.py`
records CP1's **real Qwen layer core-attention Q/K/V**, actual O and
upstream dO from a real TQ NLL backward pass. It saves real dQ/dK/dV
and independently replays the CP1 native Attention with the frozen
Q/K/V/dO to make sure its reference is self-consistent.

Fresh CP2/4 groups load those exact Q/K/V/dO, apply the validated
official Zigzag layout and call **the original Megatron/MindSpeed
`core_attention` module**. The native Ring dispatch must occur
exactly once per chosen layer on each rank. After backward the test
restores O/dQ/dK/dV to original global-token indices and reports
mean/max/p95, relative L2 and cosine. All rank-to-rank gathers are
disjoint-token restores; there is no GEMM or parameter gradient
reduction in this replay.

It is a **one-layer Attention causality diagnostic**, not a production
TPR correctness test. The original native CP1 and CP2/4 backends
remain unchanged. LSE capture is excluded at this stage.

**HOST:**
```bash
cd /workspace/bridge
git pull --ff-only
```

**DOCKER:**
```bash
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
cd /workspace/uni-agent/verl
# export TPR_REAL_TQ_BATCH=/path/to/your/real/tq_batch.pt
bash tests/models/mcore/tpr/parallel/run_native_attention_qkv_replay.sh
```

Optional: `TPR_P0_2B_LAYERS=1,4` changes the checked zero-based
Qwen layers (default 1); `TPR_P0_2B_REUSE_CP1=1` reuses an already
captured compatible CP1 oracle. Default localhost ports are
29771, 29772, 29774. No `--standalone`.

Outputs: `/tmp/tpr_p0_attn_replay/native_cp1_attention_oracle.pt`
plus `native_cp2_attention_replay.json` and
`native_cp4_attention_replay.json`.

## Deferred: direct TPR vs Full 50-step training comparison

**Not started and not implemented here.** After P0-2B is measured,
run a CP-matched, data-matched, optimizer-matched 50-step **real TPR vs
complete-sequence native baseline**. Important controls:

- Same original Qwen checkpoint; same exact physical real trajectories,
  logical supervised tokens and train/validation split; fixed tree.
- Identical NLL first (then separately authentic PPO with real old
  logprobs, advantages and clipping). No reinterpretation of synthetic
  coefficients as real advantages.
- Compare (a) CP1 native Full vs TPR CP1 and (b) CP2/CP4 native Full
  vs TPR at the **same CP**, after official backend/schedule compatibility
  checks. Do not use cross-CP results as TPR correctness thresholds.
- Capture train NLL, per-step gradients, FP32 master weights/optimizer
  state and a **common native CP1 validation oracle**, not solely
  separate CP-specific evaluations.
- Do not silently replace reference operators with padding/FP32
  shape normalization or force tree topology to prompt-root.

The current P0-B replay of independent 1024-token windows is **not yet
an apples-to-apples TPR tree benefit/equivalence workload**. A later
tree-aware trajectory scheduler must use the same logical losses while
preserving genuine shared prefixes and repeated trajectories.
