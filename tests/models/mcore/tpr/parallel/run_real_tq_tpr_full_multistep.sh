#!/usr/bin/env bash
# P0-4: 50 real optimizer steps, Full vs SegmentExecutor TPR P512/S512.
# HOST ONLY: cd /workspace/bridge && git pull --ff-only
# DOCKER:
#   rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
#   cd /workspace/uni-agent/verl
#   bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_full_multistep.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test_file="tests/models/mcore/tpr/parallel/test_real_tq_tpr_full_multistep_npu.py"
compare_file="tests/models/mcore/tpr/parallel/compare_real_tq_tpr_full_multistep.py"
[[ -f "$test_file" && -f "$compare_file" ]] || {
  echo "P0-4 files not found. Sync /workspace/bridge/tests/... into Docker VERL tests first." >&2
  exit 2
}
export TPR_P0_TPR50=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_TPR50_OUTPUT_DIR="${TPR_P0_TPR50_OUTPUT_DIR:-/tmp/tpr_p0_tpr50}"
export TPR_P0_TPR50_STEPS="${TPR_P0_TPR50_STEPS:-50}"
export TPR_P0_TPR50_LR="${TPR_P0_TPR50_LR:-2e-5}"
base="${TPR_P0_TPR50_MASTER_PORT_BASE:-29820}"

if [[ ! "$base" =~ ^[0-9]+$ ]]; then
  echo "Port base must be a decimal number" >&2
  exit 2
fi
base=$((10#$base))
if ((base < 1024 || base + 2 > 65535)); then
  echo "Port base out of range [1024,65533]" >&2
  exit 2
fi
[[ -f "$TPR_REAL_TQ_BATCH" ]] || {
  echo "Real TQ dump unavailable: $TPR_REAL_TQ_BATCH" >&2
  exit 2
}
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || {
  echo "Real Qwen checkpoint missing: $TPR_QWEN_1_7B_PATH" >&2
  exit 2
}
mkdir -p "$TPR_P0_TPR50_OUTPUT_DIR"
if [[ "${TPR_P0_TPR50_COMPARE_ONLY:-0}" == "1" ]]; then
  python "$compare_file" "$TPR_P0_TPR50_OUTPUT_DIR" --steps "$TPR_P0_TPR50_STEPS"
  exit 0
fi
case "${TPR_P0_TPR50_RUN_MODES:-both}" in
  both) modes=(full tpr) ;;
  full) modes=(full) ;;
  tpr) modes=(tpr) ;;
  *) echo "TPR_P0_TPR50_RUN_MODES must be both, full or tpr" >&2; exit 2 ;;
esac
for mode in "${modes[@]}"; do
  export TPR_P0_TPR50_MODE="$mode"
  if [[ "$mode" == "full" ]]; then
    port=$((base + 1))
  else
    port=$((base + 2))
  fi
  echo "P0-4 MODE=$mode steps=$TPR_P0_TPR50_STEPS localhost:127.0.0.1:$port" >&2
  torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
    -m pytest -x -q -s "$test_file"
done
if [[ -f "$TPR_P0_TPR50_OUTPUT_DIR/full_train${TPR_P0_TPR50_STEPS}.pt" &&
      -f "$TPR_P0_TPR50_OUTPUT_DIR/tpr_train${TPR_P0_TPR50_STEPS}.pt" ]]; then
  python "$compare_file" "$TPR_P0_TPR50_OUTPUT_DIR" --steps "$TPR_P0_TPR50_STEPS"
else
  echo "P0-4 one-sided training complete. Run the missing mode and compare." >&2
fi
