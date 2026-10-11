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

## P0-2 isolation plan (2A implemented, 2B pending)

- **P0-2A GEMM replay**: implemented as real-X/W/dY single-NPU replay
  using official Megatron Linear and Zigzag shape M=1024/512/256.
  Runtime measurements are pending.
- **P0-2B Attention replay**: supply identical global Q/K/V and
  corresponding causal positions to native full attention and official
  MindSpeed CP2/4 Ring, compare restored O/dQ/dK/dV and LSE.
- If the Stage-B 50-step gaps are large, perform both isolation tests
  *before* attributing the drift to any one kernel family.

**Implementation is committed; execution on the user's Ascend host is pending.**


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

P0-2B (native Ring attention Q/K/V and backward isolation) is still
**not implemented**; it is the next attribution control after P0-2A.
