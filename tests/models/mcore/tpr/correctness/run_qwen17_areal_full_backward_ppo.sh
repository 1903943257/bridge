#!/usr/bin/env bash
# Run only in the container's /workspace/uni-agent/verl checkout.
# Sync from /workspace/bridge separately; git pull belongs on host.
set -euo pipefail
: "${TPR_REAL_TQ_BATCH:?Set real TQ batch path}"
: "${TPR_QWEN_1_7B_PATH:?Set HF checkpoint}"
[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Missing TQ dump: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Missing model dir: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }
export TPR_RUN_QWEN17_DTA_BACKWARD=1
export TPR_QWEN17_DTA_HF_ATTN="${TPR_QWEN17_DTA_HF_ATTN:-sdpa}"
export TPR_DTA_BWD_MODES="${TPR_DTA_BWD_MODES:-native,m_split,fp32_linear}"
export TPR_DTA_BWD_ROWS="${TPR_DTA_BWD_ROWS:-8}"
export TPR_DTA_BWD_GEMM_M_TILE="${TPR_DTA_BWD_GEMM_M_TILE:-32}"
export TPR_DTA_BWD_BLOCK="${TPR_DTA_BWD_BLOCK:-64}"
log_dir="${TPR_DTA_BWD_LOG_DIR:-$(mktemp -d /tmp/tpr_dta_backward.XXXXXX)}"
mkdir -p "$log_dir"
unit_file="tests/models/mcore/tpr/unit/test_qwen17_areal_native_backward.py"
npu_file="tests/models/mcore/tpr/correctness/test_qwen3_1_7b_areal_backward_ppo_npu.py"
echo "P1 DTA_BACKWARD START log_dir=$log_dir rows=$TPR_DTA_BWD_ROWS modes=$TPR_DTA_BWD_MODES"
python -m pytest -x -q --tb=short "$unit_file" > "$log_dir/unit.log" 2>&1 || {
  echo "P1 DTA_BACKWARD UNIT_FAILED log=$log_dir/unit.log" >&2
  tail -100 "$log_dir/unit.log" >&2
  exit 1
}
cat "$log_dir/unit.log"
python -m pytest -x -q -s --tb=short "$npu_file" > "$log_dir/npu.log" 2>&1 || {
  echo "P1 DTA_BACKWARD NPU_FAILED log=$log_dir/npu.log" >&2
  tail -150 "$log_dir/npu.log" >&2
  exit 1
}
grep -E '^P1 DTA_BACKWARD (CONFIG|FULL|SUMMARY|WORST_GRAD|ROW|RESULT)' "$log_dir/npu.log"
grep -q '^P1 DTA_BACKWARD RESULT status=PASS backward=EXECUTED' "$log_dir/npu.log" || {
  echo "P1 DTA_BACKWARD missing completion marker" >&2
  exit 1
}
echo "P1 DTA_BACKWARD FULL_LOG=$log_dir/npu.log"
