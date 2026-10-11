# P0-1: Official MindSpeed native CP=1/2/4 numerical control

This is an **independent native CP control** for the TPR M-shape drift study.
It does **not** use `replace_self_attention_with_tpr`, the TPR executor,
a custom rectangular Attention backend, fixed M padding, FP32 GEMM, or PPO.
It uses **unmodified Megatron GPTModel native core-attention spec**,
MindSpeed `repatch`, `megatron_cp_algo` (Ring), and MindSpeed's own
`get_batch_on_this_cp_rank` (2*CP Zigzag). The driver validates official
shard indices and captures qkv/proj/fc1/fc2 GEMM M at all 28 layers.
For CP>1 it counts calls through MindSpeed native Ring dispatch and
**fails closed** if it cannot verify at least one dispatch per layer.

Canonical MindSpeed method:
- https://github.com/Ascend/MindSpeed/blob/master/mindspeed/core/context_parallel/get_batch_utils.py
- https://github.com/Ascend/MindSpeed/blob/master/mindspeed/core/context_parallel/dot_product_attention.py
- https://www.hiascend.com/document/detail/en/MindSpeed/2610/Core/MindSpeed_Core/docs/en/user-guide/megatron_performance_optimization.md

## Run workflow: host git pull -> Docker rsync and train

The **host** owns the `bridge` Git working tree. Do not run `git pull`
inside Docker. Once the host updates it, the mounted
`/workspace/bridge` path is the source of truth inside Docker.
The runnable project is **`/workspace/uni-agent/verl` inside Docker**.
This is the older UniAgent checkout, and must not be git-pulled.

**Step 1 — HOST shell (outside Docker):**

```bash
cd /workspace/bridge
git pull --ff-only
```

**Step 2 — Docker shell (inside the existing NPU container):**

```bash
# Sync only tests, using the established Docker-mounted paths.
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/

cd /workspace/uni-agent/verl
export TPR_P0_NATIVE_CP=1
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_P0_CP_OUTPUT_DIR=/tmp/tpr_p0_native_cp
bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh
```

The launcher runs each CP size in a fresh process, using
`--master_addr=127.0.0.1` and `--master_port` of
`29561 / 29562 / 29564` (for CP1/2/4). For port conflicts, set
`TPR_P0_CP_MASTER_PORT_BASE` before launching.

The actual CP=4 launch inside Docker is:

```bash
torchrun --master_addr=127.0.0.1 --master_port=29564 --nproc_per_node=4 \
  -m pytest -x -q -s \
  tests/models/mcore/tpr/parallel/test_native_cp1_cp2_cp4_drift_npu.py
```

The synced launch script also executes
`/workspace/bridge/tools/compare_native_cp_drift.py`. This script stays
in bridge, since your `rsync` only copies `tests/models/mcore/tpr/`.
If the bridge `tools/` path is not visible inside Docker, copy the
comparator into Docker and set `TPR_P0_CP_COMPARE_SCRIPT` to that path.

To use real TQ tokens rather than the default seeded input, set
`TPR_REAL_TQ_BATCH=/path/to/tq_batch.pt` and optionally
`TPR_P0_CP_TQ_ROW=0` **inside Docker** before launching.
The chosen trajectory needs at least 512 prompt and 512 response tokens.

**Native batch configuration:** P0 explicitly sets `reset_attention_mask=False`
for the normal non-EOD/non-packed `megatron_cp_algo` path. Lightweight
MindSpeed args may omit this field without `megatron.training`. The test
still calls the official MindSpeed `get_batch_on_this_cp_rank`; there is
no local data-sharding implementation or masking override.

**Dependency note:** Existing bridge CP test fixtures use
`mindspeed.args_utils.get_full_args()` because Docker may have
Megatron-Core without `megatron.training`. This P0-1 test now uses
the same MindSpeed parameter interface and does not import
`megatron.training` directly.

### Recovering existing P0-1 evidence without rerunning the NPU jobs

If CP=1/2/4 each printed `P0_NATIVE_CP ...` but comparison failed on
`torch.torch_version.TorchVersion`, **do not rerun the distributed tests**:
the three `.pt` artifacts are already valid and saved. The comparator now
keeps `weights_only=True` and only allowlists the PyTorch `TorchVersion`
str subclass needed to read earlier artifacts. New captures save the version
as plain `str`.

After pulling bridge on the **host**, execute inside **Docker**:

```bash
cd /workspace/uni-agent/verl
python /workspace/bridge/tools/compare_native_cp_drift.py /tmp/tpr_p0_native_cp
```

Or, after syncing the updated test launcher into Docker with the same rsync
command as above, set `TPR_P0_CP_COMPARE_ONLY=1` to skip CP=1/2/4 and
preserve prior artifacts:

```bash
TPR_P0_CP_COMPARE_ONLY=1 bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh
```

All jobs run BF16 Qwen3-1.7B, exactly 1024 tokens, one trajectory,
TP=PP=DP=1, dropout=0, 512-token suffix next-token NLL objective.
The first response token is predicted at the last prompt query position (511),
and the last response token at query 1022. The common denominator is 512. Logical query indices/targets are **global**;
never roll labels within a Zigzag shard. CP local M is 1024/512/256.

The test saves rank-0 artifacts for CP1/2/4 with all 1023 per-token
logprobs, objective, Linear M witnesses, native Ring call count, full
parameter names, sampled BF16 weight signature and strided gradient samples. Gradient synchronization
is a **bare-GPTModel HCCL SUM diagnostic** in the native BF16 grad dtype,
with global token denominator already applied (no /CP): it is not a
claim that Megatron DistributedDataParallel or optimizer was run.

Default comparison uses exact integer-derived <=512 indices per
parameter, NOT all-parameter gradient metrics. For full gradient
comparisons, `export TPR_P0_CP_SAVE_FULL_GRADS=1` before **all three**
runs. This writes a large BF16 gradient tensor for every parameter and
requires substantial CPU RAM and disk (about 3.4 GB per CP run).

The comparator **rejects** mismatched token SHA, sampled BF16 model-weight signature, model, source, checkpoint
path, objective, sequence length or software versions. It prints
response-only max/mean/p95 logprob drift, sampled/full gradient relL2,
cosine and worst parameters; no arbitrary PASS tolerance is applied.

### Interpretation

- This controls *native CP* end-to-end numerics (both Ring/FA and GEMM/
  dW accumulation). It does **not** by itself isolate GEMM as the cause.
- Native CP=1 is unmodified Full with its *own* native attention, not the
  controlled CANN Full oracle used in some TPR tests. Do not mix those
  baselines to attribute GEMM error.
- If CP4 deviates from CP1, the next P0-2 experiment must isolate
  single-Linear GEMM M with identical X/W; CP also changes Attention
  and gradient communication.
- Reproducibility is limited to the recorded software/weights/input:
  this script was authored without access to the user's Ascend host.
  **NPU execution and numerical results are pending.**



## Observed native CP1/CP2/CP4 results (2026-10-11, user NPU run)

All three **native** runs completed their forward/backward paths:
Qwen3-1.7B BF16, actual recorded TQ row0 (`prompt_tail512+response_head512`),
1024 logical tokens, 512 response targets, 226 trainable gradient entries.
No TPR or custom attention was installed in this control.

| Metric | CP=1 | CP=2 | CP=4 |
|---|---:|---:|---:|
| Local physical Linear M (all 28 layers) | 1024 | 512 | 256 |
| MindSpeed native Ring entry calls on rank0 | 0 | 28 | 28 |
| NLL loss | 0.956898391 | 0.950917602 | 0.947620749 |
| Loss delta vs CP1 | — | -0.0059807897 | -0.0092776418 |

| Numeric difference vs CP1 | CP2 | CP4 |
|---|---:|---:|
| All logprobs (1023) mean abs | 0.037954772 | 0.036326010 |
| All logprobs max abs | 0.49981308 | 0.99390936 |
| Response logprobs (512) mean abs | 0.035687864 | 0.032613402 |
| Response logprobs p95 abs | 0.245665634 | 0.228623867 |
| Response logprobs max abs | 0.49981308 | 0.99390936 |
| Response logprobs rel L2 | 0.03961506 | 0.04343556 |
| Response logprobs changed (nonzero difference) | 335 / 512 | 345 / 512 |
| **Strided sampled-only** parameter grad relative L2 | **0.31300916** | **0.11868224** |
| **Strided sampled-only** parameter grad cosine | 0.95312057 | 0.99295477 |

The output's gradient statistics are **NOT** complete-parameter
global metrics: each parameter contributes at most 512 deterministic
strided elements (and some tiny parameters fewer). The 226 group
counts indicate compared parameter names, not full-gradient coverage.
Notably CP2's sampled grad rel L2 is larger than CP4's: the drift is
**non-monotonic** with physical Linear M.

**What is established:** On this model/checkpoint/input, the official
non-TPR distributed configuration also has material logprob differences
between CP1 and CP2/4. Neither the CP loss nor the forward/backward
metrics imply that the TPR tree scheduler or its KV-gradient relay is
correct. These observations do **not** isolate GEMM as the root cause.

**Confounders verified in the P0-1 source:**
- CP1 attention uses its normal whole-sequence path; CP2/4 dispatch
  `mindspeed.core.context_parallel.dot_product_attention.ringattn_context_parallel`
  28 times (one per layer). The probe checks entry dispatch, not a
  per-kernel numerical-equivalence proof. Attention decomposition and
  collective communication change along with Linear M.
- The bare GPTModel gradients are synchronized after local backward
  via `dist.all_reduce(parameter.grad, SUM)` **in BF16** for CP>1,
  while CP1 has no collective. This is a test control, not a
  verified production Megatron-DDP/main-grad optimizer finalizer.
  BF16 accumulation/communication may contribute to gradient drift.
- The `sampled_weight_sha256_rank0` and `token_sha256` equality gates
  verify bounded BF16 model-weight witnesses and identical tokens,
  but the former is not a full-checkpoint cryptographic hash.
- Older TPR results used cropped 128+64 tokens across 8 trajectories,
  often PPO and a controlled CANN attention path. Similar mean
  logprob errors (~0.03) are **not apples-to-apples TPR/CP parity**.

**Follow-up control priority (diagnostic only; production kernels unchanged):**
1. **P0-2 real-operands single-Linear GEMM replay:** capture identical
   X, W and upstream dY from CP1 real Qwen at the layer1 qkv and MLP
   projections; compare full M=1024 to sharded M=512/256 (including
   Zigzag reordering), report Y/dX/dW separately. Do not pad full or
   install fixed-M GEMM in TPR or native training.
2. **Attention-only CP isolation:** hold identical Q/K/V, compare CP1
   whole attention with official CP2/4 Ring, mapping outputs/dQ/dK/dV
   back to their global token positions. This controls attention
   transport separately from Linear M.
3. **Gradient reduction control:** for the same CP2/4 local gradients,
   compare current BF16 SUM vs an explicitly **diagnostic-only** FP32
   SUM (without altering model forward/backward). The existing
   saved strided gradient artifacts do not contain per-rank pre-SUM
   tensors, so this requires a future test run.
4. Consider same-shape/same-backend CP-matched TPR vs native control
   only after attention backend equivalence is established; previous
   TPR Ring is a separately implemented backend.

This stage is a valid native CP numerical diagnostic, **not** a
strict correctness PASS and not proof that the non-monotonic sampled
gradient deltas are expected training behavior.
