# P0-B: 50-step real-TQ native MindSpeed CP training control (2026-10-11)

Purpose: test whether CP=1/2/4 forward/backward numerical drift accumulates
or changes learning under **actual optimizer steps** on identical real tokens.

## Scope and interpretation

- Checkpoint: **Qwen3-1.7B pretrained**, BF16 model forward/backward.
- Attention: **unaltered** Megatron + MindSpeed `megatron_cp_algo`.
  The CP1/2/4 sharding and model builders reuse
  `test_native_cp1_cp2_cp4_drift_npu.py`.
- Training: 50 sequential optimizer updates, no invented synthetic tokens,
  fixed deterministic train row schedule and held-out validation row IDs.
- Dataset: the existing `tq_batch.pt` contains **8 SWE trajectories**.
  Cropping keeps prompt tail 512 + response head 512, and consumes the
  **recorded response_mask**. Rows missing the required token span or
  with an empty supervised mask are skipped and recorded.
  The final quarter of eligible rows are held out for validation.
  If <2 eligible rows remain, the experiment refuses to run.
- More than 64 eligible, independent real rows are preferred for strong
  training conclusions. When only 8 real trajectories exist the 50-step
  run is intentionally marked `LOW_DIVERSITY`: repeated deterministic
  replay is a stress test, **not** a demonstration of generalization.
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
step number and real trajectory ID. At milestone: held-out mean NLL
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

## Pending P0-2 isolation tests (not yet implemented)

- **P0-2A GEMM replay**: capture identical real X/W/dY for QKV and
  FC1/FC2 and compare M=1024 vs 512/256 Zigzag without modifying
  training compute or padding it back to full M.
- **P0-2B Attention replay**: supply identical global Q/K/V and
  corresponding causal positions to native full attention and official
  MindSpeed CP2/4 Ring, compare restored O/dQ/dK/dV and LSE.
- If the Stage-B 50-step gaps are large, perform both isolation tests
  *before* attributing the drift to any one kernel family.

**Implementation is committed; execution on the user's Ascend host is pending.**
