#!/usr/bin/env bash
# Fixed recorded SWE-TQ parity: Native Full vs TPR CP1.
# No rollout, no task execution, no synthetic token stream.
set -euo pipefail

TQ_DIR="${TPR_SWE_TQ_DIR:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke}"
TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-${TQ_DIR}/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B-128K}"
MODE="${TPR_SWE_PARITY_MODE:-cropped}"
PROMPT="${TPR_SWE_PARITY_PROMPT:-128}"
RESPONSE="${TPR_SWE_PARITY_RESPONSE:-64}"
TEST=tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py

[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Missing TQ: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }
[[ -f "$TEST" ]] || { echo "Run in the bridge/VERL root containing $TEST" >&2; exit 2; }

export TPR_REAL_TQ_BATCH TPR_QWEN_1_7B_PATH
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_PPO_TRACE_LAYERS="${TPR_QWEN17_PPO_TRACE_LAYERS:-0}"
export TPR_QWEN17_PPO_CORE_ORACLE="${TPR_QWEN17_PPO_CORE_ORACLE:-0}"
export TPR_QWEN17_PPO_NATIVE_CUTOFF="${TPR_QWEN17_PPO_NATIVE_CUTOFF:-0}"
export TPR_QWEN17_PPO_ACCEPTANCE="${TPR_QWEN17_PPO_ACCEPTANCE:-1}"
export TPR_QWEN17_PPO_LOGIT_PARTS="${TPR_QWEN17_PPO_LOGIT_PARTS:-1}"
export TPR_QWEN17_PPO_LOGIT_PARTS_TOPK="${TPR_QWEN17_PPO_LOGIT_PARTS_TOPK:-12}"
# Never run the deprecated fixed-tile numerical oracle for parity.
if [[ -n "${TPR_QWEN17_PPO_TILE_GEMM:-}" ]]; then
  echo "ERROR: unset TPR_QWEN17_PPO_TILE_GEMM; fixed tile changes Native math" >&2
  exit 2
fi

case "$MODE" in
  cropped)
    [[ "$PROMPT" =~ ^[1-9][0-9]*$ && "$RESPONSE" =~ ^[1-9][0-9]*$ ]] || {
      echo "Positive integer TPR_SWE_PARITY_PROMPT and _RESPONSE required" >&2; exit 2;
    }
    export TPR_QWEN17_PPO_PROMPT="$PROMPT"
    export TPR_QWEN17_PPO_RESPONSE="$RESPONSE"
    echo "CROPPED REAL TOKENS: diagnostic only, does NOT establish 40K parity"
    ;;
  full)
    unset TPR_QWEN17_PPO_PROMPT TPR_QWEN17_PPO_RESPONSE
    echo "FULL 8-row real TQ: CP1 Native/TPR training can OOM on long SWE trajectories"
    ;;
  *)
    echo "Invalid TPR_SWE_PARITY_MODE=$MODE (cropped|full)" >&2; exit 2 ;;
esac

echo "TQ=$TPR_REAL_TQ_BATCH"
echo "CHECKPOINT=$TPR_QWEN_1_7B_PATH"
echo "MODE=$MODE"
python -m pytest -x -s -q --tb=short "$TEST::test_real_qwen3_1_7b_tq_ppo_loss_and_gradients"
