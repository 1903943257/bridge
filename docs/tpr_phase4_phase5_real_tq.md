# TPR Phase 4/5: VERL native PPO and real Qwen3-1.7B

### Fixed: invalid FP32 linspace indices in weak-E2E optimizer sampling (2026-10-09)

The first Native/TPR weak-E2E attempt failed during bounded gradient
sampling with `ACL stream synchronize failed, error code:507035`, at
`flat.index_select(0, indices).float().cpu()`. The **test's diagnostic
sampler**, not a known TPR GEMM bug, was generating indices using:

```python
torch.linspace(0, flat.numel() - 1, steps=min(512, flat.numel()),
               device=flat.device).long()
```

For large embedding parameters, the last legal index may not be exactly
representable in FP32: e.g., 151936*2048 elements has legal maximum
311164927, but `float32(311164927)` is **311164928**, already one
past the end. NPU gather with such an index can produce the
Vector-Core address fault reported later by an asynchronous `.cpu()`
copy. This is a **strong code-grounded root cause**, although a single
traceback without the device error log cannot prove which NPU op
faulted first.

**Fix on `bridge/main`:**
`correctness/_qwen17_weak_e2e_sampling.py::exact_sample_indices`
computes all sampled indices using Python integer division (CPU) and
copies the tiny int64 index array to NPU. New CPU-only regression:
`unit/test_qwen17_weak_e2e_sampling.py`. No model kernel, gradient
math, precision or optimizer settings were changed.

```bash
# From the bridge/VERL training tree with the latest code:
python -m pytest -q \
  tests/models/mcore/tpr/unit/test_qwen17_weak_e2e_sampling.py

# Rerun original full Native+TPR weak optimizer-step smoke:
bash tests/models/mcore/tpr/correctness/run_qwen17_real_tq_weak_e2e.sh

# Bypass the optional *sampling* entirely, if diagnosing remaining errors:
TPR_QWEN17_WEAK_E2E_SAMPLE=0 \
  bash tests/models/mcore/tpr/correctness/run_qwen17_real_tq_weak_e2e.sh
```

If a new process **still** reports 507035 after this fix and sampler
bypass, set `ASCEND_LAUNCH_BLOCKING=1` *before* starting pytest,
and inspect the FIRST `AICORE`/`AIVEC` `fault kernel_name` or
`origin_op_name` in the Ascend plog, not only the Python `.cpu()`
trace. Synchronous launch is **debug only**, as it changes performance.

## First runnable weak E2E: real TQ -> default BF16 PPO -> actual AdamW step

`tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_weak_e2e_npu.py`
is a **standalone training-side entry**, reading the already-generated
`tq_batch.pt`; it does NOT launch UniAgent, invoke ClaudeCode, or require
the older `uni-agent` E2E rollout scripts. Run it from an environment where
the bridge TPR code is installed and the real Qwen3-1.7B checkpoint and TQ
dump are accessible. This is intentionally insulated from the
`uniagent-cc` runner/version difference; do **not** modify or downgrade
the working `uniagent-cc` rollout checkout to run this test.

Scope is the **default BF16 Linear** and controlled CANN attention in the
existing Phase-5 Qwen fixture. It calls the existing Engine's
`forward_backward_batch` TPR Forest adapter, consumes the real VERL PPO
loss and backpropagates its real autograd graph, and then actually executes
`torch.optim.AdamW.step()` on **FP32 optimizer master parameters**,
copies updated weights back to the BF16 Qwen model and runs a
post-update Forward. FP32 here describes **optimizer state/master
storage**, not any FP32 QKV/MLP GEMM. No fixed-M GEMM, grouped GEMM,
or custom FP32-dW autograd is installed.

**Recommended single command:** A checked-in shell runner now executes
Native and TPR as two **separate** NPU pytest processes, checks real
AdamW-step PASS for both, and prints loss, Forward/Backward time,
optimizer total time and peak allocated HBM comparisons. It also
writes bounded strided samples across **every trainable parameter**
(maximum 512 entries each) and compares the actual clipped gradients
and FP32 AdamW updates by sampled relative L2, cosine and sign flips.
These samples are diagnostic, not full-parameter equality gates:

```bash
export TPR_REAL_TQ_BATCH=/absolute/path/from/uniagent-cc/tq_batch.pt
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
# Optional: CPU master for low-HBM correctness smoke only
# export TPR_QWEN17_WEAK_E2E_MASTER_DEVICE=cpu
bash tests/models/mcore/tpr/correctness/run_qwen17_real_tq_weak_e2e.sh
```

Run from the installed training-side bridge/VERL source directory,
not from an unrelated old `uni-agent` checkout. The current server
has to use bridge's compatible `verl.models.mcore.tpr` modules and
correctness tests. The script does not consume credentials or start
the rollout system.

To get a comparable, **controlled** Native baseline, run the exact
same command with `TPR_QWEN17_WEAK_E2E_EXECUTION=native`, in its own
pytest process. The default TPR run and the Native control use the
same checkpoint, TQ rows, loss inputs, AdamW settings, gradient
norm clipping, FP32 master semantics and CANN Attention Oracle.
The performance numbers distinguish Forward/Backward seconds and
whole optimizer seconds (including FP32 master setup, grad transfer,
optimizer work and BF16 weight copy), and report NPU
peak-allocated/reserved. First-run CANN/torch_npu compilation and
CPU IO may affect comparisons; compare repeats with identical
runtime conditions rather than treating the very first invocation
as a production throughput benchmark.

```bash
# From bridge/verl's source tree with working Megatron/MindSpeed/NPU runtime:
export TPR_RUN_QWEN17_WEAK_E2E=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
export TPR_QWEN17_WEAK_E2E_PROMPT=128
export TPR_QWEN17_WEAK_E2E_RESPONSE=64
export TPR_QWEN17_WEAK_E2E_LR=0.0001
export TPR_QWEN17_WEAK_E2E_MASTER_DEVICE=npu
unset TPR_QWEN17_PPO_TILE_GEMM TPR_QWEN17_GPT_FP32_GEMM
unset TPR_QWEN17_PPO_FC2_FIXED_M TPR_QWEN17_SPLIT_FP32_DW_BACKWARD

for MODE in native tpr; do
  TPR_QWEN17_WEAK_E2E_EXECUTION="$MODE" \
    python -m pytest -x -s -q --tb=long \
      tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_weak_e2e_npu.py \
      > "/tmp/tpr_weak_e2e_${MODE}.log" 2>&1 || {
        tail -100 "/tmp/tpr_weak_e2e_${MODE}.log"
        exit 1
      }
  grep -E 'P0 WEAK_TQ|PASSED|FAILED|ERROR' \
    "/tmp/tpr_weak_e2e_${MODE}.log"
done
```

**NPU capacity escape hatch**: change
`TPR_QWEN17_WEAK_E2E_MASTER_DEVICE=cpu` on **both** control and
TPR, keeping the model and GEMMs on NPU. This validates optimizer
math / model weight updates without the NPU memory cost of
FP32 AdamW states; however its optimizer time becomes dominated by
host transfers and is NOT an optimizer performance reference.

**First acceptance**: require each run to print
`P0 WEAK_TQ BACKWARD status=PASS`,
`P0 WEAK_TQ OPTIMIZER_STEP status=PASS`, and
`P0 WEAK_TQ RESULT status=PASS`, with nonzero BF16
parameter updates, finite AdamW moments, a finite post-update Forward,
matching supervised token totals, and recorded memory/time. Record
Native/TPR loss discrepancy; this smoke does **not** enforce
full-parameter update parity yet. It is a **real optimizer update**,
unlike the previous sampled AdamW math-only diagnostic.

**Strict boundary**: this is **not** a production VERL
`BaseEngine.train_batch` / Megatron fused-distributed optimizer test:
the fixture constructs an isolated Engine for backward and uses
`torch.optim.AdamW` on FP32 masters to update actual model weights,
not the production distributed optimizer/GradBuffer.
Its TQ inputs are real, but old logprobs are checkpoint-recomputed
and advantages may be diagnostic; by default it explicitly crops
the recorded 8 trajectories to P=128/S=64. Thus the label
is `WEAK_TQ`, not `REAL_ACTOR_E2E` or `FULL_TQ_PASS`.
After this passes, use a real actor-update capture and the correct
newer `uniagent-cc` integration to accept true production Actor
optimizer semantics and full sequences.

## Default BF16 vs AReaL-DTA: do not make numerical probes mandatory (2026-10-09)

This comparison was inspected against AReaL's **actual `feat/dta`
branch**, at commit `a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6`,
not against the separate newer *sparse tree training* implementation.

**AReaL-DTA facts and source files:**

- `examples/tau2/dta/config_1.7b_airline_dta.yaml` selects
  `backend: archon:d2`, `dtype: bfloat16`,
  `gradient_checkpointing: false`, `tree_training_mode: dta`,
  AdamW/ZeRO-1, `gradient_clipping: 1.0`, and `eps_clip: 0.4`.
- `areal/experimental/engine/archon_engine.py` expressly **rejects**
  activation checkpointing with DTA; its DTA-specific `train_batch`
  executes `DTAWrapper.run_backward_with_scaled_loss`, then
  `all_reduce_zero1_gradients`, then a real `optimizer_step`.
  The branch's FSDP and Megatron engines reject DTA mode entirely.
- `areal/experimental/models/archon/qwen3/model/model.py` uses
  ordinary **separate** Q/K/V and MLP `torch.nn.Linear` modules.
  There is **no forced GEMM M=128 tile and no FP32 Linear-forward
  substitution** in DTA's train path.
- `areal/experimental/dta/dta_engine.py` performs no-grad prefix cache
  Push and with-grad segment recompute/Pop, with detached
  `requires_grad_(True)` prefix KV and explicit gradient injection.
  KV/grad KV buffers use model dtype (BF16 in this config);
  logprob/entropy caches are FP32. The Archon training path sets
  `reduce_dtype=torch.float32`, which is gradient communication
  precision, **not FP32 QKV/MLP GEMM**.
- `tests/experimental/archon/test_dta.py` actually calls
  `train_batch` and checks grad_norm / model update. It accepts
  grad_norm relative gap below 0.25; its `compare_tensors` for
  individual parameter updates uses `rtol=0.3`, then merely **prints**
  mismatches without failing the test. Do **not** call DTA's
  full-parameter updates bitwise/strictly equivalent to Dense/FSDP.

**What is non-default in our bridge?** Three independent layers must
not be confused:

1. **TPR architecture itself:** compressed DFS, no-grad Push,
   graph-building Pop, connected gradient injection, rectangular
   Attention instead of ordinary square Causal Attention. The
   variable physical Linear M dimension is an inevitable consequence
   of per-segment execution; DTA has the same logical phenomenon.
2. **Phase-5 test fixture baseline (always active):** Qwen3-1.7B
   pretrained weights are loaded into a single-rank BF16 Megatron
   model. For BOTH `tpr=False` and `tpr=True`, the fixture
   **replaces native core-attention with a controlled CANN fused
   attention adapter**, unless `core_attention_module="native"` is
   explicitly requested. It additionally uses dropout=0,
   TE=false, no fused LM head, no native Megatron recompute, and
   CPU checkpoint initialization. These are **not** sufficient to
   identify it as a production VERL/MindSpeed Native configuration.
3. **Optional debugging interventions (not production defaults):**
   `TPR_QWEN17_PPO_TILE_GEMM=128` wraps QKV/proj/FC1/FC2 on BOTH
   Native and Forest using physical-M=128 BF16 GEMMs; partial tiles
   are padded with zero rows and trimmed. Other separate probes use
   `TPR_QWEN17_GPT_FP32_GEMM` (FP32 Linear forward then BF16 cast),
   `TPR_QWEN17_SPLIT_FP32_DW_BACKWARD` (test-only alternative dW),
   or independent grouped-dX/`npu_matmul_add_fp32` autograd.
   **None of these optional probes is installed in the ordinary
   `run_tpr_forward_backward_batch` PPO implementation**.

**Why do Native and Forest differ without these controls?** BF16 GEMMs
with physical M=full-sequence vs M=segment may dispatch through
different reduction/tiling paths on Ascend CANN, producing slightly
different rounded outputs even though both compute valid BF16
approximations. Differences compound over 28 layers. DTA runs on a
different GPU+Archon stack and does not require strict bitwise parity;
its training ability does not show that a shape-dependent BF16
difference is a correctness bug in our Ascend kernels.

The short recorded-data `P=128,S=64,N=8` Phase-5 experiment passed
its PPO numerical gate **only after symmetric test-only BF16 tiling**
while sampled full-precision AdamW-update disagreement remained.
These results establish a useful diagnostic oracle, **not** a
production requirement to force all Linear GEMMs to M=128. Do not
promote FP32 Linear or grouped FP32 dW into the normal TPR path on this
evidence.

### P0 default-first acceptance matrix

Keep the **same initial weights, true captured actor-update PPO
fields and actual optimizer** throughout. Run these configurations
in order and record **loss, logical logprobs, all gradients,
grad_norm, all parameter deltas, AdamW moments, step success,
NPU peak and wall time**:

| Run | Backend / shape | Numerical interventions | Purpose |
| --- | --- | --- | --- |
| A | Native production VERL + MindSpeed | None | Real Actor baseline |
| B | TPR production path + native BF16 Linear | None | **P0 primary acceptance** |
| C | Both controlled CANN Native + TPR, same BF16 | None | Attribute attention/backend differences |
| D | Both controlled CANN Native + TPR, BF16 M=128 | Test-only **cropped** oracle | Attribute physical-M precision only |
| E | Both paths with alternate FP32 wgrad/GMM | Explicit opt-in only | Investigate any *proven* backward blocker |

**Do not require B to be bitwise A.** First require a finite,
non-skipped update, reasonable PPO/new-logprob comparison, bounded
normalized gradient and parameter-update errors, a second optimizer
step retaining moments, and no unaccounted sample/token loss.
Numerical tolerance should be recorded as an explicit predeclared
gate rather than silently relaxed to make a failing result pass.
Only after B is accepted should end-to-end performance be compared.

**Important input precondition**: the original `tq_batch.pt` is a
rollout/TQ capture, not a post-PPO-preprocessing actor-update
mini-batch. The Phase-5 fixture recomputes old logprobs and may
synthesize advantages; consequently it **cannot serve as true
production PPO Actor Optimizer-Step acceptance**. The strict
`test_real_actor_update_minibatch_contract.py` requires genuine
`old_log_probs` and `advantages`; after it passes, invoke the
**actual** `BaseEngine.train_batch` / `EngineWorker.train_batch`
(which calls optimizer zero_grad, `forward_backward_batch`, and
`optimizer_step`), not a hand-constructed Phase-5 engine and not
the sampled CPU AdamW counterfactual.

To capture the unmodified **default numerical control** while waiting
for the genuine actor snapshot:

```bash
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_PPO_ACCEPTANCE=1
unset TPR_QWEN17_PPO_TILE_GEMM
unset TPR_QWEN17_PPO_FC2_FIXED_M
unset TPR_QWEN17_GPT_FP32_GEMM
unset TPR_QWEN17_SPLIT_FP32_DW_BACKWARD
unset TPR_QWEN17_DENSE_BACKWARD_AUTOGRAD
# Explicit cropped-data numerical diagnosis only, NOT Actor E2E:
export TPR_QWEN17_PPO_PROMPT=128
export TPR_QWEN17_PPO_RESPONSE=64
python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

A failure in the test's strict comparison should be investigated, but
must not be reinterpreted as evidence that a default BF16 PPO
optimizer step cannot run or train.

## P0: real-TQ actor-update E2E closure (2026-10-09)

**This is the current P0 deliverable.** The isolated GEMM/GMM
performance work is a supporting P1 optimization unless a numerical
defect is conclusively blocking a real actor optimizer step. Do not
wait for a fast production grouped-GEMM path before executing an
initial slow, correctness-first offline E2E.

Definition of the **offline actor-update E2E P0**:

1. Capture an actual Uni-Agent rollout actor `mini_batch_td` at the
   update boundary. Preserve **original** `old_log_probs` from the
   behavior policy, **original** advantages, responses, masks, sampling
   temperature, trajectory IDs, grouping IDs, and weights/config used
   during that update. Do not recompute old policy values from the
   checkpoint or silently synthesize advantages.
2. Make sure the eight trajectories meant to share a prefix appear
   together in **the same actor update mini-batch**; merely naming a
   directory `GBS1_N8` is not evidence they share a training batch.
   Verify `tpr_trajectory_keys` / grouping, the resulting forest
   topology, logical loss-token count, and total PPO denominator.
3. With **the same initialization, real PPO inputs and optimizer
   configuration**, execute Native and Forest on the same NPU setup:
   forward -> loss -> backward -> gradient synchronization/finalization
   -> optimizer step. Compare logical response logprobs, loss,
   all trainable parameters' pre/post gradients, parameter updates,
   optimizer states, and one subsequent forward. Do not use the
   Phase-5 test's hand-constructed engine (which only calls
   `forward_backward_batch`) as evidence of optimizer E2E.
4. Verify at least one more consecutive update with fresh gradient
   reset and preserved optimizer moments. Report exceptions/OOM as
   blockers, not numerical PASS; don't loosen the existing gates.
5. Measure NPU peak allocated/reserved and the end-to-end wall time
   against a **matched**, actual production Native reference. The
   controlled CANN Native oracle and cropped symmetric BF16 tile oracle
   are debugging controls, not automatically the production baseline.

**First unblock the E2E without waiting for GMM:** Start with the
smallest genuine actor mini-batch that fits in memory; retain the
recorded token IDs/advantages/old logprobs/masks unchanged. The
existing Python fixed-tile GEMM is a *short-sequence diagnosis only*,
not an unconditional long-context production intervention. If the
full recorded trajectories fail capacity, report the memory limit and
enable a supported checkpoint/offload/length strategy with explicit
correctness revalidation rather than mislabeling a cropped smoke
as a complete E2E.

**Follow-on P0:** after offline actor-update succeeds, run one
Uni-Agent rollout -> trajectory grouping -> VERL actor update ->
next rollout using the updated checkpoint. Verify checkpoint change,
finite optimizer state, stable trajectory identities, logical loss
token count, and actual end-to-end runtime. Grouped GEMM performance,
alternative CP/TP/DP setups and full-length stress sweeps can be
optimized independently after an initially correct E2E.

### One-time opt-in capture hook in the real VERL Actor worker

This code is now checked into `bridge/main`:

- `verl/models/mcore/tpr/actor_capture.py`: save a copy of the **actual**
  actor-update mini-batch immediately before its real train step. It
  never overwrites existing captures, invents advantages/old logprobs
  or changes the original training tensors.
- `patches/apply_tpr_actor_capture.py`: idempotently inserts the
  capture call in the working VERL `TrainingWorker.train_mini_batch`
  immediately before `actor_output = self.train_batch(mini_batch_td)`.
  Default behavior remains unchanged when the env variable is unset.
- `tests/models/mcore/tpr/unit/test_actor_capture_patcher.py`:
  CPU-only patcher validation.

Only run this if the currently running VERL checkout has the latest
TPR helper module installed, and **review the target path first**.
Do not run `git pull` inside the VERL working checkout just to apply it.

```bash
# From bridge (the source repository), inspect compatibility first:
python patches/apply_tpr_actor_capture.py --check \
  /workspace/uni-agent/verl/verl/workers/engine_workers.py
python -m pytest -q \
  tests/models/mcore/tpr/unit/test_actor_capture_patcher.py

# Verify actor_capture.py exists in the *running VERL Python package*:
test -f /workspace/uni-agent/verl/verl/models/mcore/tpr/actor_capture.py

# Then apply the reviewed targeted patch (does not touch optimizer):
python patches/apply_tpr_actor_capture.py \
  /workspace/uni-agent/verl/verl/workers/engine_workers.py

# Set in the ACTOR WORKER process environment for a real training run:
export TPR_CAPTURE_ACTOR_MINIBATCH_DIR=/workspace/tpr_actor_capture
```

After the actual actor job runs through `train_mini_batch`, look for
`actor_update_rank0_batch0.pt`; run the strict P0 contract below
with `TPR_REAL_ACTOR_MINIBATCH` pointed to that file. Other DP ranks
write their own rank-labelled captures. If the real TQ trajectory
identity was not propagated through the actor data loader, the capture
will contain an empty `keys` tuple and the contract must FAIL:
fix identity propagation rather than constructing artificial row keys.

This hook **captures before the optimizer update**. The next P0
validation must inspect the *actual* `train_batch` results, the
non-skipped optimizer step, parameter deltas and optimizer state. A
successful capture on its own does not establish any of these.

### First P0 gate: capture the actual actor-update mini-batch

A strict, opt-in CPU test has been added at
`tests/models/mcore/tpr/integration/test_real_actor_update_minibatch_contract.py`.
It requires the **post-PPO-preprocessing** `mini_batch_td`, not the
original rollout-only `tq_batch.pt`. At the real VERL actor update
boundary, preserve the exact mini-batch plus its original trajectory
keys in this snapshot format:

```python
torch.save({
    "capture_stage": "actor_update_mini_batch",
    "tensordict": mini_batch_td.cpu(),
    "keys": tuple(exact_trajectory_keys),
}, "/tmp/tpr_real_actor_update.pt")
```

The exact keys must correspond to the same mini-batch rows (and must be
derived from actual rollout identity; do not fabricate them).
Do not store credentials or sensitive trajectory contents in a public
repo; keep this snapshot on the private NPU server.

```bash
export TPR_RUN_REAL_ACTOR_MINIBATCH_CONTRACT=1
export TPR_REAL_ACTOR_MINIBATCH=/tmp/tpr_real_actor_update.pt
python -m pytest -s -q --tb=short \\
  tests/models/mcore/tpr/integration/test_real_actor_update_minibatch_contract.py
```

This fails if the capture lacks any original
`old_log_probs`, `advantages`, `temperature`, masks or trajectory IDs,
or if token alignment/Forest PPO denominator disagrees. **It does not
run PPO backward or optimizer step.** Next execute the captured real
mini-batch through actual Native and TPR `train_batch` with a matched
optimizer; this remains the acceptance-critical part of P0.

**Observed current status (not yet E2E PASS):** real checkpoint,
recorded TQ, compressed forest, native VERL PPO loss and TPR backward
already run in the Phase-5 fixture; a cropped symmetric BF16
fixed-tile comparison passed the numerical gate, with selected
gradient relative-L2 around 0.0121. The actor-mini-batch capture,
full optimizer integration and rollout-to-next-rollout loop remain
unverified.

## Supported / deliberately unsupported

* Training target: **real Qwen3-1.7B checkpoint**, no synthetic model.
* Trajectory source: the recorded 8-row django11163 `tq_batch.pt`, not fabricated token sequences.
* Phase-4 PPO: vanilla, token-mean, TP=PP=CP=EP=DP=1.
* Fused LM head, Megatron native recompute, router replay, MoE, and MTP are not enabled for this first integration.
* The old CE request path is retained as a regression oracle; it is not the production PPO entry.
* This is a correctness gate, **not yet a full Uni-Agent RL rollout-to-optimizer loop**.

## Engine integration contract

`MegatronEngine.forward_backward_batch` first attaches global `batch_num_tokens`
and `dp_size`. Before `prepare_micro_batches`, training batches with
`tpr_enabled` dispatch to:

```python
run_tpr_forward_backward_batch(engine, data, loss_function, forward_only=False)
```

The adapter resolves `tpr_trajectory_keys` (preferred) or a per-row `uid`,
builds `ForestExecutionPlan`, binds each tree's PPO objective, and runs one
`FixedTopologyScheduler` per tree. The executor performs Push/Visit/Pop
and forwards prefix gradients. Native `finalize_model_grads_func` runs **once**
per complete mini-batch; `BaseEngine.train_batch` retains zero_grad and the
optimizer step. **There is no artificial factor of physical segment count**.

Important: to obtain prefix reuse across the recorded 8 trajectories, the actor
update mini-batch must contain them together. The `GBS1_N8` directory label
alone does not prove this; inspect actor `mini_batch_td` and the actual keys.

## Apply integration to an existing VERL checkout

`bridge` is the source repository for the added TPR files and
`patches/3_5_megatron_engine.patch`. Do not run `git pull` inside
`/workspace/uni-agent/verl` based on these instructions.

Ensure the target VERL checkout already includes the **latest** TPR modules.
For an unpatched checkout, inspect:

```bash
cd /workspace/uni-agent/verl
git apply --check /path/to/bridge/patches/3_5_megatron_engine.patch
```

If the old CE patch was already applied, **do not reapply the whole patch**.
Use the focused incremental hunk instead, after checking compatibility:

```bash
git apply --check /path/to/bridge/patches/3_5_megatron_engine_phase4_incremental.patch
git apply /path/to/bridge/patches/3_5_megatron_engine_phase4_incremental.patch
```

If either check fails because your native file differs in the area around
`routed_num_tokens`, patch that location manually; don't discard local
modifications or force the patch. Verify:

```bash
grep -n 'run_tpr_forward_backward_batch' \
  verl/workers/engine/megatron/transformer_impl.py
```

## Real TQ CPU Phase-4 tests

```bash
python -m pytest -s -q \
  tests/models/mcore/tpr/integration/test_real_tq_trajectory_tree.py \
  tests/models/mcore/tpr/integration/test_real_tq_tree_plan_builder.py \
  tests/models/mcore/tpr/integration/test_real_tq_phase4_training_keys.py
```

If the dump lives elsewhere, set `TPR_REAL_TQ_BATCH` to its absolute path.
These tests require the real dump and skip if absent; no synthetic substitution.

## Real Qwen3-1.7B Phase-5 numerical gate

Use the **complete 8 recorded variable-length trajectories by default**.
No synthetic GPT proxy. Do **not** set `TPR_QWEN17_PPO_PROMPT` or
`TPR_QWEN17_PPO_RESPONSE` for the whole-TQ correctness test.

```bash
cd /workspace/uni-agent/verl
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
unset TPR_QWEN17_PPO_PROMPT TPR_QWEN17_PPO_RESPONSE

python -m pytest -x -vv -s --tb=long \\
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

This may exceed single-NPU memory because a native row can span more
than 40k tokens and the checkpoint vocabulary is 151,936. Memory failure
is a real capacity finding, **not** Phase-5 numerical correctness PASS.

For a *separate*, explicitly labeled cropped-real-data diagnostic:

```bash
export TPR_QWEN17_PPO_PROMPT=64
export TPR_QWEN17_PPO_RESPONSE=64
python -m pytest -x -vv -s --tb=long \\
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

Both windows are copied from recorded TQ samples, but cropping changes
causal context and tree topology and does **not** verify the full TQ.

When the cropped real-TQ numerical check fails, do **not** loosen
`rtol=0.02, atol=0.2` automatically. The Phase5 test now prints:

- `NATIVE REPEATABILITY`: same Native weights/tokens, no-grad versus backward-enabled
  forward; estimates the numerical baseline noise floor.
- `TPR LOGPROB DIAG`: per-real-row and root/nonroot mismatch counts, followed
  by the worst 20 logical tokens with absolute query coordinates, segment
  span, Native logprob, TPR logprob, and exact tolerance.
- `TPR MODEL NATIVE-FORWARD`: *same TPR checkpoint* in non-tree
  SelfAttention mode versus Native reference. This isolates checkpoint,
  model-spec, or ordinary-forward differences.
- `WHOLE-SEGMENT TPR`: same TPR model and attention kernel with an
  unsplit single physical segment for the worst mismatched real row; compare
  its error with the Forest error to diagnose decomposition/shape effects.
- PPO scalar loss and sampled parameter gradient relative-L2 and cosine
  are reported even if the logprob gate fails.

### Interpretation warning: this 'Native' baseline is a controlled CANN reference

The Phase-5 correctness fixture uses **real Qwen3-1.7B checkpoint weights**
and a Megatron `GPTModel` with `tpr=False`, but it is **not** a production
VERL/MindSpeed reference as currently configured. The helper
`test_tpr_qwen3_compatibility_npu._make_qwen_model` calls
`_replace_core_attention(spec)`, which replaces the model's native
`core_attention` with `_ProfileFusedCausalAttention`. The latter wraps the
TPR `rectangular_causal_attention` helper (including for square causal
attention). Thus 'Native full/cutoff' in these diagnostics means:

```text
real Qwen3 checkpoint + Megatron GPTModel + controlled CANN attention
    full: input_ids[0:128], position_ids[0:128], S=128
    cutoff: input_ids[0:segment_end], position_ids[0:segment_end]
```

No TPR tree, prefix cache, or rectangular external-KV merge is used in
that full/cutoff comparison. Its observed causal shape sensitivity is
**real for the controlled CANN fixture**, but it is not proof that the
unmodified Megatron/MindSpeed attention backend has the same discrepancy.
Similarly, CORE SHAPE ORACLE compares the **same** CANN helper at
square and rectangular query lengths using post-RoPE Q/K/V captured
from one forward. Zero difference demonstrates agreement for the measured
tensor inputs and lengths; it does **not** independently validate the CANN
kernel's outputs against mathematical attention.

A properly independent follow-up requires:
1. A reference with unmodified production attention module spec, matched
   checkpoint/config, and explicit confirmation of the active backend; and
2. FP32 attention math on captured identical post-RoPE Q/K/V and a
   correctly aligned causal mask as an independent small-shape kernel oracle.
Only then can we distinguish backend-specific numerical drift from
issues elsewhere in TPR.

### Identical-QKV square versus rectangular Attention experiment

Enable `TPR_QWEN17_PPO_CORE_ORACLE=1` (the short-real-TQ
diagnostic default). It captures **post-RoPE Q, K and V** at selected
layers (1, 2, 3, 4, 14, 28) from the **same real Qwen3-1.7B Native
128-token forward**, then calls the production
`rectangular_causal_attention` without replacing any inputs:

* Square: the Native model's `128×128` causal core output.
* Rectangular: the last `58×128`, `34×128` and `14×128`
  queries against **the same 128 K/V tokens** (segment starts 70, 94, 114).

These are full-sequence-ending segments, so CANN's bottom-right causal
alignment is valid without an explicit mask. Compare `CORE SHAPE ORACLE`
per-layer metrics and `CORE TOKEN` metrics. If they differ, there is
direct evidence that the square-vs-rectangular kernel geometry changes
the result even with identical projected Q/K/V. If they agree, the
observed full-model Forest drift must arise from a difference **before**
the isolated core call (e.g. prefix activations or QKV GEMM shape) or
from multi-step effects; this does not by itself prove all TPR kernels
correct. This is a short real-token diagnostic, not a synthetic test
or a change to the numerical acceptance threshold.

### Native physical-segment cutoff oracle (Phase-5 accuracy)

The 64+64 real-TQ experiment confirmed two positions at which the Native
**truncated-prefix** result exactly equals TPR, while Native full 128
differs:

| Absolute query | Native full 128 | Native cutoff | TPR forest | Cutoff length |
| --- | ---: | ---: | ---: | ---: |
| 69 | -4.79739285 | -4.78902054 | -4.78902054 | 70 |
| 76 | -1.70170808 | -2.12716579 | -2.12716579 | 94 |

These positions prove that TPR-specific KV/gradient mistakes are **not
required** to produce the observed divergence. They do not prove every
logical token and every branch is correct.

To extend the causal control to **every real TQ objective reference**:

```bash
export TPR_QWEN17_PPO_SEGMENT_ORACLE=1
```

For each compressed-tree physical Segment the Phase-5 test now performs
a Native forward on the original real tokens up to the Segment end and
scores all of that Segment's original logical targets, including
different child targets for a shared parent query. It prints
`NATIVE FULL vs NATIVE PER-SEGMENT CUTOFF` and
`TPR FOREST vs NATIVE PER-SEGMENT CUTOFF`, and per-Segment max errors.

If `TPR FOREST vs NATIVE PER-SEGMENT CUTOFF` is small for all tokens,
the large original Native-Full/Forest discrepancy is predominantly the
precision-dependent sequence-shape effect. If not, pursue the residual
TPR-specific discrepancy in attention/KV. This cutoff oracle is only
for short **real-token debug runs** (<=256 total tokens) and will skip
full-length real TQ to avoid reproducing its observed OOM. It does not
relax the strict original Native-full gate.

### Root/branch BF16 numerical diagnosis

For the Qwen3-1.7B cropped-real-TQ case, current evidence includes:

- Same checkpoint Native / TPR module in **context-free** mode: logprob max diff zero.
- One **unsplit** TPR segment of length 128: logprob max diff zero.
- 70-token root and smaller rectangular queries: layer 1 matches at
  selected queries, layer 2 begins to differ by BF16-size units, later
  layers amplify the difference.
- A 0.425 logprob drop can move the PPO ratio to approximately 0.65,
  changing clipping. Therefore matching PPO gradients cannot be achieved
  merely by making the loss aggregation or gradient relay correct.

**Causal truncation control (opt-in diagnostic enabled by default with
layer tracing):**

```bash
export TPR_QWEN17_PPO_TRACE_LAYERS=1
export TPR_QWEN17_PPO_NATIVE_CUTOFF=1
```

The NPU test compares at the exact *real* row-0 token positions:

1. Native full 128-token forward (the canonical reference);
2. Native forward truncated to the real segment's `position_end`
   (e.g. 70 for query 69 and 94 for query 76);
3. TPR single, **unsplit** segment with exactly the same cutoff length;
4. TPR Forest's segmented Prefix-KV execution.

Look for `NATIVE TRUNCATION LOGPROB` and
`TPR SINGLE-CUTOFF LOGPROB` alongside the already reported Forest
per-token error. If Native itself changes substantially with causal
sequence length, the underlying BF16 kernel shape sensitivity must be
quantified before attributing the drift to a tree-specific bug. If Native
is invariant at that cutoff but Forest differs, investigate rectangular
FA, post-RoPE K/V and ancestor-prefix state. This probe changes **no**
model arithmetic, inference defaults, or pass/fail tolerance.

When the 64+64 TQ crop gives repeated mismatches in nonroot Segment 1
(e.g. real absolute query 76) while whole-segment TPR is bit-identical,
the next gate is **attention-layer-local**. Set
`TPR_QWEN17_PPO_TRACE_LAYERS=1` (default) to compare the actual real row-0
queries at response offsets 6 and 13 across all Qwen layers. For each layer
the test reports the relative L2 and maximum absolute differences of
`self_attention` **input and output**. It maps physical Segment ownership
from `ForestExecutionPlan`, not from hard-coded node IDs; sibling branches
at the same absolute position cannot contaminate the trace.

- First-layer input mismatch: investigate embeddings, sequence positions,
  Qwen model/row mismatch.
- First-layer input matches but output differs: investigate TPR attention
  projection, RoPE, rectangular FA and NPU shape sensitivity.
- First-layer output close but drift grows through later layers: quantify
  BF16 shape-dependent accumulation before considering tolerances.

Also, this diagnostic's PPO advantages are intentionally signed (+1.0/-0.5)
rather than an exact +1/-1 cancellation: previously native PPO loss was 0
by construction and hid loss-relative deviation. This is a mathematical
probe, not real rollout advantage data.

The numerical gate remains strict; none of these diagnostics changes
acceptance thresholds or marks mismatches as passing.

Qwen3-1.7B normally has 40,960 maximum positions; the recorded TQ
contains a 41,029-token trajectory. The shared real-Qwen test fixture now
honors requested test length (RoPE extrapolation with unchanged theta),
and Phase5 reports actual maximum length.

`old_log_probs` in the diagnostic are recomputed with the frozen real
checkpoint. If the TQ dump lacks `advantages` (before actor update),
deterministic signed coefficients are used **solely** for probing the
mathematical gradient equivalence. They are *not* true rollout advantages.
To validate production PPO, capture `mini_batch_td` at actor update with
actual `old_log_probs`, `advantages`, `response_mask`, `loss_mask`,
temperature and stable trajectory identity.

The Phase5 test also sanitizes an empty-named field from MindSpeed's
`get_full_args()`, as is already done in the real-Qwen Ring profile.
If a dataclass error remains, save the **full** `--tb=long` traceback.

## Not yet validated

* No server NPU run has been performed by the GitHub edits themselves.
* The actual Qwen3-1.7B and native/TPR numerical gate may need NPU/runtime
  compatibility fixes once real logs are available.
* Full ~17k prompt trajectories may need LM-head chunking and activation
  memory work before they can fit; the crop is intentionally much smaller.
* CP>1, TP>1, DP>1 and full actor-update integration remain separate work.
