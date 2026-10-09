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
