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
before interpreting a gate failure.

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
export TPR_QWEN17_GEMM_WARMUP=2
export TPR_QWEN17_GEMM_REPEATS=8

# Separate opt-in API probe. ONLY forward support is tested here.
export TPR_QWEN17_GEMM_TRY_GROUPED=1

python -m pytest -s -q --tb=short \
  tests/models/mcore/tpr/profiling/test_qwen3_1_7b_gemm_tile_bench_npu.py \
  > /tmp/qwen17_gemm_tile_bench.log 2>&1

grep -E 'QWEN17 GEMM BENCH|FAILED|ERROR' /tmp/qwen17_gemm_tile_bench.log
```

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
