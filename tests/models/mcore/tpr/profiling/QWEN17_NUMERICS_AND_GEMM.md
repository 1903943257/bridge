# Qwen3-1.7B real-TQ numerical acceptance and GEMM physical-M study

Experimental test-only controls in the **bridge** repository. Do not patch
Megatron/MindSpeed production kernels until measurements support a narrow fix.
All tests are CP=TP=PP=1 on Ascend NPU with actual 1.7B checkpoint weights.

## A. Real PPO numerical acceptance (Native vs Forest TPR)

The existing real TQ PPO test already builds eight real-token trajectories,
uses VERL's actual PPO loss, compares selected parameter gradients, and applies
its existing strict numerical gates. Its optional `ACCEPTANCE` report adds:

- Native no_grad(A) vs Native no_grad(B), same checkpoint and shape.
- Native no_grad vs Native grad-enabled (existing reference noise measure).
- Advantage-aware PPO clipping branch disagreements, holding old-policy
  logprobs fixed for Native and TPR. A ratio merely outside [0.8,1.2] is
  **not** necessarily the clipped PPO branch for the token's advantage sign.
- PPO per-token clipped objective, plus the existing real VERL PPO loss.
- Sampled FIRST AdamW update, with identical initial weights and zero optimizer
  moments, computed on CPU FP32 for at most 1024 elements per chosen parameter.
  This is a diagnostic **not** a full model or sharded optimizer step.
- Existing full selected-gradient cosine and relative-L2 gates remain active.

Run using the existing real TQ file and real Qwen checkpoint:

```bash
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_PPO_ACCEPTANCE=1
# Clear experimental knobs retained in shell from earlier work:
unset TPR_QWEN17_PPO_FC2_FIXED_M
unset TPR_QWEN17_GPT_FP32_GEMM
unset TPR_QWEN17_GPT_TILE_GEMM
unset TPR_QWEN17_GPT_REPLAY_FULL_LAYER_INPUTS

# OPTIONAL smoke with *real* cropped rows (not a training acceptance test):
# export TPR_QWEN17_PPO_PROMPT=128
# export TPR_QWEN17_PPO_RESPONSE=64

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py \
  -k real_qwen3_1_7b_tq_ppo_loss_and_gradients \
  > /tmp/qwen17_real_ppo_acceptance.log 2>&1

grep -E 'ACCEPTANCE|NATIVE REPEATABILITY|PPO native_loss|PPO grad:|NUMERICAL GATE|FAILED|ERROR' \
  /tmp/qwen17_real_ppo_acceptance.log
```

An exit code of 1 from the **existing** strict gradient/logprob gate is not
equivalent to the diagnostic itself failing. Inspect all acceptance reports
before interpreting a gate failure. A previously exported
`TPR_QWEN17_PPO_FC2_FIXED_M` can make this test abort **before** any TPR
forward; always clear it unless deliberately testing FC2 physical padding.

IMPORTANT: this fixture currently **recomputes** `old_log_probs` using the
same native checkpoint instead of consuming the saved rollout behavior-policy
logprobs. Its PPO branch comparison is therefore a checkpoint-matched
counterfactual, NOT a verified historical rollout-old-policy test. Advantages
are similarly synthetic signed coefficients if missing from the TQ dump.
Never claim a real policy-lag experiment based solely on this result.

Do **not** claim 20-100-step real PPO replay or full AdamW optimizer equivalence
from the sampled first-step probe. A multi-step experiment needs repeatedly
recomputed gradients and optimizer moments on both actual models, with identical
fixed trajectories and no intervening rollout changes.

CPU-only unit contract:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_qwen17_ppo_acceptance.py
```

## B. BF16 GEMM physical-M / GroupedMatmul feasibility

Uses real pretrained Qwen3-1.7B weights and controlled (synthetic) BF16 hidden
activations, **without** running the 28-layer model. Measures each of the four
Linear families at M=128,1024,1152. Contrasts the original Megatron Linear
forward against the exact same module called in token tiles. Each result
includes latency and elementwise/relative error against the native GEMM.
These latencies include Python dispatch, repeated kernel launches and
synchronization; they are not end-to-end throughput measurements.

```bash
export TPR_RUN_QWEN17_GEMM_BENCH=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_GEMM_TILE=128
export TPR_QWEN17_GEMM_M=128,1024,1152
export TPR_QWEN17_GEMM_GROUPS=qkv,proj,fc1,fc2
# Either 'random' or real hidden activations captured from the unmodified
# Full Qwen forward using actual recorded TQ token IDs:
export TPR_QWEN17_GEMM_INPUT=real
export TPR_QWEN17_GEMM_LAYER=1
export TPR_QWEN17_GEMM_WARMUP=2
export TPR_QWEN17_GEMM_REPEATS=8

# Separate opt-in API probe. ONLY forward support is tested here.
export TPR_QWEN17_GEMM_TRY_GROUPED=1

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/profiling/test_qwen3_1_7b_gemm_tile_bench_npu.py \
  > /tmp/qwen17_gemm_tile_bench.log 2>&1

grep -E 'QWEN17 GEMM BENCH|FAILED|ERROR' /tmp/qwen17_gemm_tile_bench.log
```

For the grouped kernels, the diagnostic now explicitly compares **three**
results: grouped vs **fixed-tile**, grouped vs **native full-M**, and native
vs fixed-tile, including per-128-token mismatch counts. This is necessary
because FC2 grouped forward can differ from fixed-tiling even when its speed
is comparable to native (and relative-L2 alone does not prove it matches native).

`npu_grouped_matmul` receives one physical weight per group *by reference*;
this does not guarantee any on-device weight reuse, supported backward, or
the identical numeric path as repeated Megatron Linear. An UNSUPPORTED result
is valid evidence about this installed NPU stack, not a TPR mathematical
failure. Some older Ascend API versions explicitly target inference only.

Reference: https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md

## Decision criteria before touching production

1. Compare Native self-noise floor vs TPR logprob differences and actual
   advantage-aware PPO clip branches, real VERL loss, and full gradients.
2. Only if numerical mismatch matters for optimization, use fixed-tile BF16 as
   **numerical oracle**, not necessarily the production execution strategy.
3. Profile full Megatron native vs 128-token Python tile and optional
   GroupedMatmul; require bitwise comparison against the fixed-tile oracle
   before treating a grouped kernel as numerically equivalent.
4. Grouped GEMM training backward support, selective FP32 dW accumulation,
   unaligned segment lengths and CP>1 remain separate required gates.

## C. Measured results 2026-10-09, and selective fast-tile validation

Real 8-row TQ smoke (P=128/S=64), checkpoint-matched old-policy logprobs:
Native no_grad vs no_grad repeated bitwise, Native no_grad vs grad-enabled
bitwise; TPR changes advantage-aware PPO clipped branch for **31 / 512**
tokens. PPO loss -0.250000 vs -0.247312, sampled selected-gradient relative
L2=0.523553, cosine=0.89627931. This should be treated as a current
numerical acceptance failure, not dismissed using tiny relative-to-weight
first-step AdamW differences. Updated AdamW diagnostic also reports
`update_rel_l2`, `update_cosine`, and `gradient_sign_flips`.

Real layer-1 activations, M=1024/1152, torch_npu 2.9 GroupedMatmul:
- FC2 both GMM layouts are bitwise **native full-M**, not fixed tile.
  Thus fast but NOT a drop-in numerical oracle for FC2.
- Projection both GMM layouts are bitwise **fixed-tile**, not native full-M,
  while Single-X timing was around native GEMM in this microbenchmark.
- QKV and FC1 require the same *real-activation* triad before drawing
  conclusions.

Test-only full-model intervention, after fixing tile=128 for four GEMM
families, selectively substitute GroupedMatmul for Projection while all
other families stay on fixed token tiles:

```bash
export TPR_RUN_QWEN17_SPLIT=1
export TPR_QWEN17_SPLIT_P=1024
export TPR_QWEN17_SPLIT_S=128
export TPR_QWEN17_GPT_TILE_GEMM=128
export TPR_QWEN17_GPT_TILE_GEMM_GROUPS=qkv,proj,fc1,fc2
export TPR_QWEN17_GPT_GROUPED_GEMM_GROUPS=proj
export TPR_QWEN17_GPT_GROUPED_GEMM_MODE=single_x_multi_w
unset TPR_QWEN17_GPT_FP32_GEMM
unset TPR_QWEN17_GPT_REPLAY_FULL_LAYER_INPUTS
unset TPR_QWEN17_GPT_PREFIX_QKV_FIXED_M_LAYERS

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_split_equivalence_npu.py \
  -k full_gpt_vs_single_split \
  > /tmp/qwen17_full28_grouped_proj.log 2>&1

grep -E 'GROUPED_GEMM|BF16_TILED_GEMM|GPT_SUFFIX_LOGPROBS|PPO_RATIO_MAX_DEVIATION|FULL-GPT EQUIVALENCE|FAILED|ERROR' \
  /tmp/qwen17_full28_grouped_proj.log
```

If the final logprobs are bitwise, replace the Python tiled Projection
with the fused GMM **only inside the diagnostic**. A full-model bitwise result
does not prove that NPU GroupedMatmul has usable training autograd, avoids
weight transpose buffers, or supports TP/CP.

Optionally change
`TPR_QWEN17_GPT_GROUPED_GEMM_GROUPS=fc2` (same four
`TPR_QWEN17_GPT_TILE_GEMM_GROUPS`) as an expected negative control.
If bitwise fails, it validates that a numerically-native-like GMM cannot
replace the fixed-tile FC2 oracle.

Before choosing QKV/FC1 fast routes, run with
`TPR_QWEN17_GEMM_GROUPS=qkv,fc1`, `TPR_QWEN17_GEMM_INPUT=real`, and
`TPR_QWEN17_GEMM_TRY_GROUPED=1` to compare Grouped-vs-Tile-vs-Native.

## D. 2026-10-09 follow-up: GroupedMatmul single-tile probe and PPO cutoff branches

Additional real activations / layer-1 microbench:
- QKV grouped single-X matches fixed tile at both M=1024 and M=1152;
  at M=1152 original native also happens to match the tile oracle.
  Relative-to-native latency M=1152 was ~0.882.
- FC1 grouped single-X matches fixed tile but costs ~1.526x native at
  M=1152, so immediate production speed benefit is unproven.
- Previous Projection matches tile near native speed; FC2 matches native
  and differs from tile. No conclusion about other layers yet.

Attempting full-28-layer selective Projection GroupedMatmul initially
raised `ERR00100 PTA call acl api failed`, without a provided full traceback.
The previous microbench only covered M=1024 and M=1152; the split suffix
uses M=128 = ONE group. The full-model diagnostic now routes one group to
ordinary native M=128 GEMM (which is exactly the fixed-tile calculation),
and offers `TPR_QWEN17_GPT_GROUPED_GEMM_TRACE=1` to synchronously attribute
any remaining NPU kernel error to its layer and M. **Single-group root
cause remains a hypothesis until rerun**; it has not been NPU verified.

```bash
export TPR_QWEN17_GPT_GROUPED_GEMM_TRACE=1
# Keep the prior four-family tile=128, grouped_groups=proj environment.
python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_split_equivalence_npu.py \
  -k full_gpt_vs_single_split \
  > /tmp/qwen17_grouped_proj_trace.log 2>&1
grep -E 'GROUPED_GEMM|GPT_SUFFIX_LOGPROBS|PPO_RATIO_MAX_DEVIATION|FAILED|ERROR' \
  /tmp/qwen17_grouped_proj_trace.log
```

Real TQ PPO acceptance now explicitly reports 3 clip-branch comparisons
when `TPR_QWEN17_PPO_SEGMENT_ORACLE=1`:
- `native_full_vs_tpr_forest`,
- `native_full_vs_native_cutoff`,
- `native_cutoff_vs_tpr_forest`.

All three comparisons use exactly the same checkpoint-recomputed old
logprobs and the same signed advantages. They help attribute clipping
differences to ordinary shape/truncation versus Forest-specific execution;
they do NOT by themselves prove KV-reuse or scheduling bug isolation.

Previously observed: 31/512 Native-vs-Forest clipping branches disagree;
selected parameter gradient relative L2=0.523553 / cosine=0.89627931;
sampled fresh AdamW update relative L2=0.783774842 / cosine=0.693042952.
The native Full vs Native Cutoff logprob max_abs=0.37499237, while
TPR Forest vs Native Cutoff max_abs=0.73060608. Both differences matter.

## E. 2026-10-09: full-28 Projection success, PPO cutoff attribution, symmetric M oracle

**Verified on user's Ascend stack:**
- Full 28-layer Qwen3-1.7B with fixed token tiles in QKV/FC1/FC2
  and Grouped Projection (single-X / multiple-W) **passes bitwise logprob
  equality**. The 1-group suffix path falls back to ordinary M=128 GEMM.
- Native Full repeat is bitwise (512 tokens).
- PPO native Full clipping = 0, cutoff clipping = 13, Forest clipping = 31;
  cutoff-vs-Forest branch XOR = 30. Therefore cutoff/Forest clipped-set
  intersection = 7, cutoff-only = 6, Forest-only = 24. This proves extra
  Forest-vs-Cutoff numerical behavior, **not** necessarily a semantic KV bug.
- Cropped real-TQ selected grads native-vs-Forest relative L2 ~0.524,
  cosine ~0.896; sampled AdamW step direction cosine ~0.693.
  The native Cutoff and Forest GEMM execution shapes differ!

### Next NPU step A: symmetric real-TQ fixed token tile

Native Full, per-segment Native Cutoff, and TPR Forest must ALL use the
same BF16 physical token tile size, **including ragged segments**. This
test-only oracle pads the LAST GEMM tile to M=128 and slices off dummy rows.
It does not change sequence/attention positions, tree topology, or KV length.
Activation/training time will be slower; gradient accumulation paths may
still differ even if Forward becomes bitwise.

```bash
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN17_PPO_ACCEPTANCE=1
export TPR_QWEN17_PPO_PROMPT=128
export TPR_QWEN17_PPO_RESPONSE=64
export TPR_QWEN17_PPO_SEGMENT_ORACLE=1
export TPR_QWEN17_PPO_TILE_GEMM=128
export TPR_QWEN17_PPO_TILE_GEMM_GROUPS=qkv,proj,fc1,fc2
export TPR_QWEN17_PPO_TRACE_LAYERS=0
export TPR_QWEN17_PPO_CORE_ORACLE=0
unset TPR_QWEN17_PPO_FC2_FIXED_M

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py \
  -k real_qwen3_1_7b_tq_ppo_loss_and_gradients \
  > /tmp/qwen17_ppo_symmetric_tile.log 2>&1

grep -E 'SYMMETRIC|ACCEPTANCE CLIP|ADAMW_SAMPLED|NATIVE FULL vs NATIVE PER-SEGMENT CUTOFF|TPR FOREST vs NATIVE PER-SEGMENT CUTOFF|PPO grad:|NUMERICAL GATE|FAILED|ERROR' \
  /tmp/qwen17_ppo_symmetric_tile.log
```

A TPR failure with full/cutoff/Forest symmetric shapes suggests
additional attention-state, tree/segment, or backward numerical pathways.
A successful full forward but failing parameter gradients indicates an
additional backward accumulation problem, not necessarily Forward error.

CPU-only test for arbitrary ragged M and chain-rule preserved by the probe:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_qwen17_fixed_tile_probe.py
```

### Next NPU step B: combine fast QKV + fast Projection in full 28 layers

```bash
export TPR_RUN_QWEN17_SPLIT=1
export TPR_QWEN17_SPLIT_P=1024
export TPR_QWEN17_SPLIT_S=128
export TPR_QWEN17_GPT_TILE_GEMM=128
export TPR_QWEN17_GPT_TILE_GEMM_GROUPS=qkv,proj,fc1,fc2
export TPR_QWEN17_GPT_GROUPED_GEMM_GROUPS=qkv,proj
export TPR_QWEN17_GPT_GROUPED_GEMM_MODE=single_x_multi_w
unset TPR_QWEN17_GPT_FP32_GEMM
unset TPR_QWEN17_GPT_REPLAY_FULL_LAYER_INPUTS
unset TPR_QWEN17_GPT_PREFIX_QKV_FIXED_M_LAYERS

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_split_equivalence_npu.py \
  -k full_gpt_vs_single_split \
  > /tmp/qwen17_grouped_qkv_proj_full28.log 2>&1

grep -E 'GROUPED_GEMM|BF16_TILED_GEMM|GPT_SUFFIX_LOGPROBS|PPO_RATIO_MAX_DEVIATION|FULL-GPT EQUIVALENCE|FAILED|ERROR' \
  /tmp/qwen17_grouped_qkv_proj_full28.log
```

Keep `TPR_QWEN17_GPT_GROUPED_GEMM_TRACE=0` for non-trace timing tests;
the traced version synchronizes every GroupedMatmul and is far slower.
Even full-forward bitwise success does NOT prove autograd/backward support.

## F. 2026-10-09: symmetric Tile PPO PASS, combined Grouped QKV+Projection PASS

**Verified on user's Ascend 910B2C environment** (real recorded TQ cropped
128 prompt / 64 response, 8 trajectories, 512 valid policy tokens):
- With symmetric BF16 tile=128 in Native Full, Native Cutoff and Forest,
  all three PPO effective clip-branch comparisons are **0/512**.
  Native Full-vs-Forest ratio maximal absolute deviation ~9.54e-7;
  native Full-vs-Cutoff logprob max_abs ~9.54e-7;
  Forest-vs-Cutoff max_abs ~2.38e-7.
- The REAL VERL PPO numerical gate prints PASS; selected-parameter grad
  relative-L2=0.012100, cosine=0.99992683.
- Sampled fresh AdamW **update** relative-L2 still ~0.111952, cosine
  ~0.993734, with 59/21224 nonzero gradient signs flipped.
  The update metrics depend on the sampled parameters and first-step
  zero-moment/no-clip assumptions. They are not full-optimizer equivalence.
- Complete 28-layer Qwen3-1.7B BF16 forward with **Grouped QKV +
  Grouped Projection**, while FC1/FC2 use fixed token tiles:
  logprob **bitwise=True** and ratio max deviation zero. 128-token
  one-group suffix uses native M=128 GEMM.

These outcomes strongly implicate physical BF16 Linear M-shape numerical
paths for this SHORT real-TQ forward discrepancy, but do not establish
any universal model/length/TP/CP guarantee or exact source of the remaining
backward differences. The "global gradient" figure is for the **selected
parameter subset** (decoder layers 0, 13, 27, plus embeddings/norm/head),
not all 1.7B model parameters.

### Next step: isolate remaining selected-gradient disagreement

```bash
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN17_PPO_ACCEPTANCE=1
export TPR_QWEN17_PPO_PROMPT=128
export TPR_QWEN17_PPO_RESPONSE=64
export TPR_QWEN17_PPO_TILE_GEMM=128
export TPR_QWEN17_PPO_TILE_GEMM_GROUPS=qkv,proj,fc1,fc2
export TPR_QWEN17_PPO_SEGMENT_ORACLE=0
export TPR_QWEN17_PPO_TRACE_LAYERS=0
export TPR_QWEN17_PPO_CORE_ORACLE=0
export TPR_QWEN17_PPO_GRAD_BREAKDOWN=1
unset TPR_QWEN17_PPO_FC2_FIXED_M

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py \
  -k real_qwen3_1_7b_tq_ppo_loss_and_gradients \
  > /tmp/qwen17_ppo_fixed_gradient_breakdown.log 2>&1

grep -E 'GRAD BREAKDOWN|PPO grad:|ADAMW_SAMPLED|NUMERICAL GATE|FAILED|ERROR' \
  /tmp/qwen17_ppo_fixed_gradient_breakdown.log
```

This extra diagnostic ranks selected parameters by share of squared
gradient discrepancy to identify which layer and family contributes
most. This does NOT establish BF16 wgrad accumulation as the cause.

### Next step: check GroupedMatmul training autograd WITHOUT patching TPR

The 28-layer Grouped experiment is FORWARD ONLY; its packed per-layer
weights use `detach()`, so the wrapper itself does not propagate any
learnable weight gradient. Before production, test whether torch_npu
GroupedMatmul has native autograd on a SHARED trainable weight:

```bash
export TPR_RUN_QWEN17_GROUPED_BACKWARD=1
export TPR_QWEN17_GROUPED_BACKWARD_TILE=128
export TPR_QWEN17_GROUPED_BACKWARD_N=2

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/profiling/test_qwen3_1_7b_grouped_autograd_npu.py \
  > /tmp/qwen17_grouped_backward.log 2>&1

grep -E 'GROUPED BACKWARD|FAILED|ERROR' /tmp/qwen17_grouped_backward.log
```

Possible statuses: `SUPPORTED_AUTOGRAD`, `NO_AUTOGRAD`, `UNSUPPORTED`.
If torch_npu's direct grouped operation lacks autograd, prefer searching
the matching *installed* MindSpeed GMM training wrapper/backward kernel
before considering new custom operations.

## G. P0: native MindSpeed shared-weight GMM autograd and FP32 main_grad

**Code inspection (Ascend/MindSpeed master, 2026-10-09):**
- `mindspeed/ops/grouped_matmul.py` implements a native
  `torch.autograd.Function` which calls `torch_npu.npu_grouped_matmul`
  in forward, and GMM group_type=0 for dX / group_type=2 for dW.
  Its weight shape is **[num_groups,K,N]**; it is designed for
  per-group/MoE weights. The probe tests whether
  `W.T.unsqueeze(0).expand(G,-1,-1)` can share one actual [N,K]
  parameter, with PyTorch summing expanded gradients.
- `mindspeed/ops/gmm.py` provides separate `npu_gmm`,
  `GMMFunction.backward`, optional `gemm_fusion=True`,
  `original_weight.main_grad`, and calls
  `npu_groupmatmul_add_fp32`. This ALSO expects grouped 3D weights.
  In the current `mindspeed/ops/npu_groupmatmul_add.py` path, the
  A5 branch explicitly views `main_grad` as
  `[num_groups,K,N]`; for 910B2C/A2 the extension receives the same
  grouped inputs. A regular shared dense Linear has
  `main_grad` shape **[N,K]**, so the grouped fusion should NOT be
  assumed to accumulate directly into this single 2D buffer.
- `mindspeed/ops/npu_matmul_add.py` ALREADY offers
  `npu_matmul_add_fp32(total_input, grad_output, main_grad)`.
  It accumulates regular dense dW to the SAME 2D FP32 main_grad,
  segment by segment. This is the appropriate existing primitive to
  probe before designing any new shared-weight grouped backward.
- Multiplying all weights by G via `.contiguous()` could make the
  runtime group layout valid but may negate TPR memory/performance wins.
  A zero-stride `.expand()` view only avoids the forward weight copies
  if the installed GMM accepts that layout; even then grouped 3D dW
  can still consume G times weight-size temporary memory.

**P0.1 — shared-weight MindSpeed GMM autograd:** separate processes
per mode, because a device-kernel error can poison an NPU context.

```bash
export TPR_RUN_QWEN17_MINDSPEED_GMM=1
export TPR_QWEN17_P0_TILE=128
export TPR_QWEN17_P0_GROUPS=2
export TPR_QWEN17_P0_K=256
export TPR_QWEN17_P0_N=384

for MODE in grouped_view grouped_contiguous gmm_view gmm_contiguous gmm_fp32_fusion; do
  export TPR_QWEN17_P0_MODE="$MODE"
  python -m pytest -s -q --tb=short \
    tests/models/mcore/tpr/profiling/test_qwen17_mindspeed_shared_gmm_npu.py \
    > "/tmp/tpr_p0_gmm_${MODE}.log" 2>&1
  echo "===== $MODE ====="
  grep -E 'P0 SHARED_GMM|FAILED|ERROR' "/tmp/tpr_p0_gmm_${MODE}.log" | tail -30
done
```

A `SUPPORTED_SHARED_AUTOGRAD` outcome requires BOTH dX and
shared 2D dW non-None. Forward error and gradient error are printed
separately. The `gmm_fp32_fusion` route measures a GROUPED 3D FP32
buffer with a **separate FP32 sum back to shared W**; it is NOT the
production 2D `main_grad` interface. Statuses are capability reports,
not unconditional correctness PASS.

**P0.2 — the existing dense FP32 main_grad primitive:**

```bash
export TPR_RUN_QWEN17_FP32_MAIN_GRAD=1
python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/profiling/test_qwen17_fp32_main_grad_npu.py \
  > /tmp/tpr_p0_main_grad.log 2>&1

grep -E 'P0 MAIN_GRAD|FAILED|ERROR' /tmp/tpr_p0_main_grad.log
```

This compares 128 once, 64 twice and 32 four times on EXACTLY the
same BF16 X/dY and writes to a shared 2D FP32 main_grad using the
MindSpeed native `npu_matmul_add_fp32`. It also reports reverse
order, a deliberately naive BF16 accumulation, and a FP32 GEMM
reference. It **does not yet replace** Megatron's production Linear
backward or integrate with Megatron's DDP main-grad allocation.

**Performance/memory warning:** 8 groups with K=2048, N=4096 would
require 256 MiB of FP32 [8,K,N] temporary group weight gradients
per QKV layer if fully materialized. This is an illustrative
minimum-sized group-gradient tensor, NOT a measured runtime peak.
Converting per-group gradients back to the single original dW in
FP32 remains necessary.

## H. P0 NPU findings and P1 real Qwen backward/temporary-memory gate (2026-10-09)

**P0 verified on Ascend 910B2C (torch_npu 2.9.0):**

- `mindspeed.ops.gmm.npu_gmm` with `W.T.unsqueeze(0).expand(G,-1,-1)`
  at G=2, M=256, K=256,N=384: **bitwise equivalent** to token-tiled
  BF16 GEMM forward, dX and *shared-parameter dW* via PyTorch autograd.
  This is the non-contiguous, zero-stride **expanded weight view**;
  it does not deliberately make G explicit forward weight copies.
- `gmm_contiguous` also passed forward/dX/shared dW, but duplicates the
  logical weight G times and is not a preferred production memory path.
- `mindspeed.ops.grouped_matmul` (grouped_view and
  grouped_contiguous) printed only CONFIG/REFERENCE in the supplied
  grepped log. Those do **not** establish support or failure; import
  skips may have been hidden. The test now prints explicit
  WRAPPER_IMPORT_UNAVAILABLE reasons.
- `gmm_fp32_fusion` forward succeeded but its backward raised
  `RuntimeError: setup failed!`. The probe now prints a synchronous
  BEFORE_BACKWARD/AFTER_BACKWARD marker to localize this in the
  installed extension. Do not treat this as proof that all FP32 wgrad
  accumulation is unsupported.
- `mindspeed.ops.npu_matmul_add.npu_matmul_add_fp32` with identical
  BF16 X and dY, writing into a single [N,K] FP32 buffer:
  M=128 one chunk / M=64 times 2 / M=32 times 4:
  relative L2 to FP32 reference approximately 6.65e-8 /
  7.24e-8 / 7.35e-8. The corresponding naive BF16 accumulation
  errors were ~0.00165 / 0.00246 / 0.00314.
  Partitioned FP32 accumulation is numerically stable in this
  standalone experiment, but it **does not** eliminate shape-sensitive
  BF16 **forward** drift when M changes.

**Next validation: pretrained Qwen real weight + real activation GMM G=8/9.**
This compares the exact Megatron M=128 tiled forward against a
F.linear(M=128) reference before checking MindSpeed GMM, and then
compares dX/shared-dW as well. It reports NPU memory usage and the
logical/storage difference of an expanded weight view. It is still a
standalone autograd experiment, not complete 28-layer training.

```bash
export TPR_RUN_QWEN17_MINDSPEED_REAL_GMM=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_REAL_GMM_TILE=128
export TPR_QWEN17_REAL_GMM_LAYER=1

for LINEAR in qkv proj; do
  for G in 8 9; do
    export TPR_QWEN17_REAL_GMM_LINEAR="$LINEAR"
    export TPR_QWEN17_REAL_GMM_GROUPS="$G"
    log="/tmp/tpr_p1_real_gmm_${LINEAR}_g${G}.log"
    python -m pytest -s -q --tb=short \
      tests/models/mcore/tpr/profiling/test_qwen3_1_7b_mindspeed_gmm_real_npu.py \
      > "$log" 2>&1
    echo "===== $LINEAR G=$G ====="
    grep -E 'P1 REAL_SHARED_GMM|FAILED|ERROR' "$log" | tail -35
  done
done
```

A critical distinction: BF16 GMM backward already returned a shared
BF16 dW via PyTorch expanded-weight autograd, but this **does not
mean** it writes the FP32 Megatron `weight.main_grad` using a shared
2D buffer. For production, prefer reusing existing Megatron/MindSpeed
FP32 grad accumulation rather than per-group temporary gradients
and an extra BF16 reduction. This requires separate end-to-end
parameter-grad and optimizer-state validation.

**No production changes or new low-level kernel have been made.**

## I. P1 real Qwen G=8/9 findings: BF16 dW drift and memory pressure

**Actual NPU log supplied on 2026-10-09:**

| Linear | G | grouped fwd vs tile | grouped dX rel L2 | grouped shared dW rel L2 | reported reference peak MiB | reported grouped peak MiB |
|---|---:|---|---:|---:|---:|---:|
| qkv | 8 | bitwise | 4.1863e-5 | 0.0040484 | 151.520 | 386.524 |
| qkv | 9 | bitwise | 4.0971e-5 | 0.0042349 | 155.583 | 424.083 |
| proj | 8 | bitwise | 2.7966e-5 | 0.0040107 | 91.017 | 205.021 |
| proj | 9 | bitwise | 2.6966e-5 | 0.0042138 | 95.830 | 225.084 |

The original memory comparison was **confounded**: the reference
forward/dX/dW tensors were still resident when GroupedMatmul was measured.
Do not subtract the printed raw peaks and claim isolated per-op overhead.
The new test keeps reference outputs/grads on CPU, frees the NPU copies,
and reports `baseline_mib` plus `incremental_peak_mib` separately.

The shared expanded weight still uses only one [N,K] physical storage
(16 MiB QKV / 8 MiB Projection), but the GMM backward may allocate
per-group [G,K,N] temporary gradients (128/144 MiB logical BF16 QKV,
64/72 MiB Projection). The 0.4% discrepancy in shared dW is **BF16
autograd vs BF16 tiled-autograd**, not yet proof of a wrong kernel.
Compare both against an independent FP32 dW and MindSpeed's FP32
`npu_matmul_add_fp32` with identical X/dY before choosing an implementation.

The updated P1 real-Qwen probe prints:
- `native_tile_reference incremental_peak_mib`
- `mindspeed_gmm_autograd incremental_peak_mib` with GPU reference removed
- `FP32_main_grad_vs_FP32_full`
- `BF16_tile_dW_vs_FP32_full`
- `BF16_grouped_dW_vs_FP32_full`
- `BF16_grouped_dW_vs_FP32_main_grad`

Recommended quick two-case rerun (same parameters as earlier):

```bash
export TPR_RUN_QWEN17_MINDSPEED_REAL_GMM=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_REAL_GMM_TILE=128
export TPR_QWEN17_REAL_GMM_LAYER=1
export TPR_QWEN17_REAL_GMM_GROUPS=8

for LINEAR in qkv proj; do
  export TPR_QWEN17_REAL_GMM_LINEAR="$LINEAR"
  python -m pytest -s -q --tb=short \
    tests/models/mcore/tpr/profiling/test_qwen3_1_7b_mindspeed_gmm_real_npu.py \
    > "/tmp/tpr_real_gmm_fp32_${LINEAR}.log" 2>&1
  grep -E 'P1 REAL_SHARED_GMM|FAILED|ERROR' \
    "/tmp/tpr_real_gmm_fp32_${LINEAR}.log"
done
```

Interpretation:
- If both BF16 paths differ from FP32 similarly, **rounding and
  accumulation** can account for much of the discrepancy; not necessarily
  a GMM implementation correctness issue.
- If MindSpeed GMM differs markedly more from FP32 than native-tiled
  BF16, investigate its backward kernel/group gradient sum.
- If grouped incremental peak remains large after memory isolation,
  do not ship it as the default TPR training backward; benchmark a
  native dense dX path plus MindSpeed FP32 shared `main_grad` route.
- A PASS from this numerical/peak probe still does not verify training
  optimizer integration, end-to-end performance or the PPO gradient gate.

## J. 2026-10-09: independent real-Qwen FP32 dW comparison and dx-only probe

User NPU measurements (real pretrained Qwen layer-1, recorded real tokens,
G=8, M=1024):

| Family | GMM BF16 dW vs FP32 | native BF16 tiled dW vs FP32 | MindSpeed FP32 main_grad vs FP32 |
| --- | ---: | ---: | ---: |
| qkv | 0.00234818 | 0.00408423 | 2.19676e-7 |
| proj | 0.00234757 | 0.00404395 | 2.32716e-7 |

MindSpeed GMM's BF16 shared weight gradient is actually **closer** to
the FP32 reference than the native tiled BF16 result. The previously
observed 0.004 GMM-vs-tile dW difference is a difference BETWEEN TWO
BF16 backward algorithms, not direct evidence the GMM kernel is worse.
The results do not validate full-model backward; optimizer semantics
are still unverified.

Memory measured with separate reference GPU gradient cleanup and equal
input baselines:

| Family | Native tile incremental peak | MindSpeed ordinary GMM incremental peak |
| --- | ---: | ---: |
| qkv | 109.008 MiB | 304.007 MiB |
| proj | 57.007 MiB | 156.006 MiB |

The GMM dW path consumes substantial peak temporary memory despite the
zero-stride expanded **forward** weight view being genuinely shared.
These are independent operator-probe peaks, not measured full-model
added HBM or latency.

**New isolated P1 test:** probe already-existing MindSpeed GMM dx-only
backward fusion separately from the known-failing grouped FP32 wgrad
fusion, OR use torch_npu grouped GEMM directly for dX. Both modes then
feed real Qwen BF16 X/dY tile partitions into the existing
`npu_matmul_add_fp32` with one shared [N,K] FP32 buffer. Both are
existing vendor kernels; the test is NOT a production custom autograd
Function. In particular, the native MindSpeed `GMMFunction` path
`gemm_fusion=True` currently errors with 'setup failed', potentially
in either dx-fusion or group-gradient add. The isolated test distinguishes
these stages.

```bash
export TPR_RUN_QWEN17_DENSE_FP32_BACKWARD=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_DENSE_BACKWARD_GROUPS=8
export TPR_QWEN17_DENSE_BACKWARD_LAYER=1

for LINEAR in qkv proj; do
  for MODE in grouped_dx mindspeed_fused; do
    export TPR_QWEN17_DENSE_BACKWARD_LINEAR="$LINEAR"
    export TPR_QWEN17_DENSE_BACKWARD_MODE="$MODE"
    LOG="/tmp/tpr_p1_dense_${LINEAR}_${MODE}.log"
    python -m pytest -s -q --tb=short \
      tests/models/mcore/tpr/profiling/test_qwen17_dense_fp32_shared_gmm_backward_npu.py \
      > "$LOG" 2>&1
    echo "===== $LINEAR $MODE ====="
    grep -E 'P1 DENSE_FP32|FAILED|ERROR' "$LOG" | tail -35
  done
done
```

An `DX_AND_SHARED_FP32_WGRAD_AVAILABLE` result means the EXISTING
operator combination works in isolation, NOT that its manual calls are
wired into autograd, Megatron FP32 main_grad, TPR Forest, optimizer
fusion or end-to-end PPO. Do not merge into production before autograd
glue, gradient scaling/microbatch reduction, and throughput profiling.

## K. 2026-10-09: grouped dX + shared FP32 main_grad validation

User's real-Qwen layer-1 P1 run on Ascend 910B2C (recorded TQ
activations, BF16 weight and synthetic BF16 upstream dY, G=8/M=1024)
verified **both** independent backward primitive combinations:

| Family | Route | dX vs BF16 tiled rel-L2 | shared FP32 dW vs FP32 rel-L2 | Incremental peak |
| --- | --- | ---: | ---: | ---: |
| qkv | grouped_dx | 3.50516e-5 | 2.19258e-7 | 36.002 MiB |
| qkv | mindspeed_fused | 3.50516e-5 | 2.19258e-7 | 148.003 MiB |
| proj | grouped_dx | 3.71101e-5 | 2.33174e-7 | 20.001 MiB |
| proj | mindspeed_fused | 3.71101e-5 | 2.33174e-7 | 84.002 MiB |

The corresponding BF16 dX vs the mathematically full FP32 matrix
product is ~0.00166; native BF16 tiled dX has the **same** FP32
discrepancy. This is NOT evidence that grouped dX introduced a 0.16%
regression relative to the native BF16 execution. Unlike an earlier
failed `GMMFunction(gemm_fusion=True)` probe, the **isolated** MindSpeed
dX-fusion primitive also completed successfully in this run.
The fused dX route is more memory-intensive for these exact shapes;
its speed is still unmeasured. Both routes write dW using the same
**separate** existing MindSpeed `npu_matmul_add_fp32` primitive.

### Test-only P2: hook the combination into real autograd

The P1 backward test now has a strictly opt-in P2 experimental path.
It adds a small `torch.autograd.Function` around *existing* GMM forward,
`torch_npu.npu_grouped_matmul` dX, and `npu_matmul_add_fp32` dW, without
creating a custom NPU kernel or editing production Linear implementations.
Two successive backward calls use the same shared BF16 weight and
one persistent FP32 main_grad. Gates: bitwise forward vs tiled BF16;
dX relative L2 <=2e-4 vs tiled BF16; FP32 main_grad relative L2
<=1e-5 vs a full-M FP32 oracle after each backward; no BF16
`weight.grad` materialization; FP32 buffer identity unchanged.

```bash
export TPR_RUN_QWEN17_DENSE_FP32_BACKWARD=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_DENSE_BACKWARD_GROUPS=8
export TPR_QWEN17_DENSE_BACKWARD_LAYER=1
export TPR_QWEN17_DENSE_BACKWARD_MODE=grouped_dx
export TPR_QWEN17_DENSE_BACKWARD_AUTOGRAD=1

for LINEAR in qkv proj; do
  export TPR_QWEN17_DENSE_BACKWARD_LINEAR="$LINEAR"
  log="/tmp/tpr_p2_shared_fp32_autograd_${LINEAR}.log"
  python -m pytest -s -q --tb=short \
    tests/models/mcore/tpr/profiling/test_qwen17_dense_fp32_shared_gmm_backward_npu.py \
    > "$log" 2>&1
  echo "===== $LINEAR ====="
  grep -E 'P1 DENSE_FP32|P2 DENSE_FP32|FAILED|ERROR' "$log" | tail -40
done
```

This new P2 path is **not yet NPU-executed**: the four table rows above
are the P1 measurements supplied by the user. A P2 PASS does **not**
validate `weight.main_grad` ownership, gradient scale, DDP reduction,
Megatron zero_grad/reset, optimizer updates, or full PPO.

### Explicit path to end-to-end acceptance

1. **P2 standalone autograd**: run the new optional test above with
   actual recorded-token activations, repeated backward, and memory report.
2. **P3 model autograd adapter (test-only first)**: integrate one shared
   Dense Linear in a real 28-layer Qwen execution, including M<128
   / ragged tail, G=1 fallback, BF16 outputs, FP32 main_grad,
   loss scaling, and consistent parameter/gradient ownership. Extend
   from qkv/proj to fc1/fc2 only after family-specific bitwise triads.
3. **P4 offline full actor update**: feed *captured* actor
   `mini_batch_td` including genuine rollout `old_log_probs`,
   `advantages`, masks, temperature, stable trajectory identity;
   compare Native vs Forest new logprobs, real PPO objective,
   all trainable parameter gradients, optimizer step, and optimizer
   state. Run at least one repeated step; a cropped/selected-parameter
   smoke is not a full acceptance result.
4. **P5 full recorded-TQ lengths and capacity**: remove cropped prompt/
   response settings, manage large LM-head and activations, test
   variable segments and memory/offload under the actual trajectory
   distribution; do not confuse 128/64-token numerical PASS with this.
5. **P6 Uni-Agent E2E**: rollout -> TQ grouping/tree building ->
   actual VERL actor minibatch -> TPR backward -> optimizer step ->
   updated rollout. Benchmark time-to-train and peak memory against a
   matched native baseline; TP/CP/DP>1 require separate correctness
   validation if the deployment needs them.

Neither the P1 logs nor the P2 isolated autograd test establish an
end-to-end training throughput gain. Keep the baseline physically
shape-matched when attributing numerical differences to reuse.
