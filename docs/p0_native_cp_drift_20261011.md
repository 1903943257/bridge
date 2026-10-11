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

## Running on the NPU host (bridge -> rsync -> old UniAgent)

**Source checkout:** `/workspace/bridge`. Pull **only here**. The actual
Megatron/VERL test runtime is the older `/workspace/uni-agent/verl`, not
the bridge checkout. Sync the `tpr/` test subtree exactly as used on the
server. The native P0 test and its launch script live under this subtree,
so both arrive at the old runtime via rsync.

```bash
# 1. Update source-of-truth bridge repo (never git pull in /workspace/uni-agent/verl).
cd /workspace/bridge
git pull --ff-only

# 2. Sync the P0 test AND its launch script into the runnable VERL checkout.
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/

# 3. Run against the old UniAgent/VERL environment, with 1/2/4 visible NPUs.
cd /workspace/uni-agent/verl
export TPR_P0_NATIVE_CP=1
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_P0_CP_OUTPUT_DIR=/tmp/tpr_p0_native_cp
bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh
```

The synced launcher invokes `torchrun` separately for CP=1/2/4 using
`--master_addr=127.0.0.1` and **explicit** `--master_port` values
`29561/29562/29564`. There is no standalone rendezvous mode.
To avoid port collisions, set `TPR_P0_CP_MASTER_PORT_BASE` (default
`29560`; each CP uses base+CP).

For clarity, the actual distributed command run inside the script is:

```bash
# From /workspace/uni-agent/verl, shown for CP=4:
torchrun --master_addr=127.0.0.1 --master_port=29564 --nproc_per_node=4 \
  -m pytest -x -q -s \
  tests/models/mcore/tpr/parallel/test_native_cp1_cp2_cp4_drift_npu.py
```

The launcher also executes the comparator from
`/workspace/bridge/tools/compare_native_cp_drift.py`, since your rsync
only copies `tests/models/mcore/tpr/` (not `tools/`). That bridge path
must remain visible in the test environment; if it is mounted elsewhere,
set `TPR_P0_CP_COMPARE_SCRIPT=/path/to/compare_native_cp_drift.py`.
For reproducible real TQ instead of the fixed seeded synthetic tokens,
set `TPR_REAL_TQ_BATCH` to the dump's full path and optionally
`TPR_P0_CP_TQ_ROW=0` before launching; the selected row needs at least
512 prompt and 512 response tokens.


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
