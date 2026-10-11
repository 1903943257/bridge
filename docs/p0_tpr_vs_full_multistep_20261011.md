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
  relative L2 and cosine, sampled weight L2, and **first
  step BF16 gradient** relative L2/cosine. These are
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
and `P0_4 STEP1_GRAD`.

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
5. Step1 preclip BF16 gradient sampled relative L2/cosine.

This 50-step exercise is **a first experimental numerical stability
gate**. If both paths learn similarly it does not prove TPR is
semantically correct for multi-sibling trees, PPO/GRPO or later
long-horizon optimizer training; if they diverge, follow up with
matched-gradient, layer-token and optimizer-state diagnostics.

**Implementation committed, user NPU measurements are pending.**
