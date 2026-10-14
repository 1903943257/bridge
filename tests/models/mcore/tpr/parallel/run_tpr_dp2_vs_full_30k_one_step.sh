#!/usr/bin/env bash
# Real Qwen3-4B actor PPO Step 6 (or selected source step):
# prepare ONE immutable 30-Ki derived capture, then run matched
# Full TP2CP4DP1 and TPR TP2CP2DP2 benchmarks on the same eight NPUs.
# No rollout, reward relabeling, or optimizer update.
# Usage: bash this_script.sh ORIGINAL_40K_MANIFEST NEW_ROOT [STEP=6]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 ORIGINAL_40K_MANIFEST NEW_OUTPUT_ROOT [STEP=6]" >&2
  exit 2
fi
SOURCE_MANIFEST="$(realpath "$1")"
ROOT="$2"
STEP="${3:-6}"
CAP="${TPR_BENCH_SEQUENCE_CAP:-30720}"
[[ -f "$SOURCE_MANIFEST" ]] || { echo "Missing source manifest: $SOURCE_MANIFEST" >&2; exit 2; }
[[ ! -e "$ROOT" ]] || { echo "Refusing to overwrite output: $ROOT" >&2; exit 2; }
[[ "$CAP" == "30720" ]] || { echo "30K fairness gate requires TPR_BENCH_SEQUENCE_CAP=30720" >&2; exit 2; }
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Must use the same physical NPUs 8..15" >&2; exit 2;
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
export TPR_BENCH_CAPTURE_LOGPROBS="${TPR_BENCH_CAPTURE_LOGPROBS:-1}"
# Isolated benchmark knob, NOT a VERL production flag. Use fp32 only
# after the standalone HCCL FP32 control has passed on this machine.
export TPR_OFFLINE_GRAD_SYNC_DTYPE="${TPR_OFFLINE_GRAD_SYNC_DTYPE:-bf16}"
[[ "$TPR_OFFLINE_GRAD_SYNC_DTYPE" == "bf16" || "$TPR_OFFLINE_GRAD_SYNC_DTYPE" == "fp32" ]] || {
  echo "TPR_OFFLINE_GRAD_SYNC_DTYPE must be bf16 or fp32" >&2
  exit 2
}
# Sampled parameter gradients checked AFTER timing in both topologies.
export TPR_BENCH_GRAD_WITNESS=1
export TPR_EXEC_PROFILE=0
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_BENCH_SOURCE_STEP="$STEP"
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

python tests/models/mcore/tpr/parallel/prepare_tpr_30k_one_step.py \
  "$SOURCE_MANIFEST" "$ROOT" --step "$STEP" --cap "$CAP"
ROOT="$(realpath "$ROOT")"
MANIFEST="$ROOT/capture_manifest.json"
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_BENCH_OUTPUT_DIR="$ROOT/full"
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_OFFLINE_MODE=full
mkdir -p "$TPR_BENCH_OUTPUT_DIR"
echo "TPR_30K_FULL_START source_step=$STEP cap=$CAP topology=TP2CP4DP1"
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
  --master_port="${TPR_BENCH_MASTER_PORT:-29817}" \
  -m pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee "$ROOT/full/step${STEP}_full.log"

unset TPR_RUN_FIXED_SHAPE_40K_BENCH
echo "TPR_30K_DP2_START source_step=$STEP cap=$CAP topology=TP2CP2DP2 chunk=$TPR_PPO_LOGPROB_CHUNK_TOKENS"
bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh \
  "$MANIFEST" "$ROOT/tpr" "$STEP" \
  "$ROOT/full/step${STEP}_full_fixed_shape.json"
echo "TPR_30K_COMPARISON=$ROOT/tpr/comparison.json"
