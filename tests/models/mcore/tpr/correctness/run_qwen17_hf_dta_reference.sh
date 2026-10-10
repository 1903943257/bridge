#!/usr/bin/env bash
# Standalone HF Qwen3-1.7B Dense vs independent DynamicCache DTA-style control.
# Run from the bridge-backed VERL root. Do not sync via /workspace/uni-agent/verl.
# Does NOT touch the Megatron TP/DP/CP init or production TPR code.
set -euo pipefail

: "${TPR_REAL_TQ_BATCH:?Set path to real uniagent tq_batch.pt}"
: "${TPR_QWEN_1_7B_PATH:?Set path to Qwen3-1.7B HF checkpoint}"
[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Missing TQ: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Missing Qwen: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }

export TPR_RUN_QWEN17_DTA_REF=1
export TPR_QWEN17_DTA_PROMPT=128
export TPR_QWEN17_DTA_RESPONSE=64
export TPR_QWEN17_DTA_HF_ATTN="${TPR_QWEN17_DTA_HF_ATTN:-sdpa}"
log_dir="${TPR_QWEN17_DTA_LOG_DIR:-$(mktemp -d /tmp/tpr_qwen17_hf_dta.XXXXXX)}"
mkdir -p "$log_dir"
test_file="tests/models/mcore/tpr/correctness/test_qwen3_1_7b_hf_dta_reference_npu.py"
unit_file="tests/models/mcore/tpr/unit/test_qwen17_dta_style_reference.py"

echo "P0 DTA_HF START log_dir=$log_dir checkpoint=$TPR_QWEN_1_7B_PATH"
echo "P0 DTA_HF START tq=$TPR_REAL_TQ_BATCH attn=$TPR_QWEN17_DTA_HF_ATTN"
python -m pytest -x -q --tb=short "$unit_file" > "$log_dir/unit.log" 2>&1 || {
  echo "DTA HF CPU unit tests failed: $log_dir/unit.log" >&2
  tail -100 "$log_dir/unit.log" >&2
  exit 1
}
cat "$log_dir/unit.log"
python -m pytest -x -q -s --tb=short "$test_file" > "$log_dir/hf_dta.log" 2>&1 || {
  echo "DTA HF NPU experiment failed: $log_dir/hf_dta.log" >&2
  tail -100 "$log_dir/hf_dta.log" >&2
  exit 1
}
grep -E '^P0 DTA_HF (CONFIG|DFS_ROW|DFS_SUMMARY|FIXED_PATH|ROOT_V_SHAPE|ROOT_CUTOFF_TO_DTA|RESULT)' "$log_dir/hf_dta.log"
if ! grep -q '^P0 DTA_HF RESULT status=PASS execution=FORWARD_CONTROL' "$log_dir/hf_dta.log"; then
  echo "DTA HF reference did not complete: $log_dir/hf_dta.log" >&2
  exit 1
fi
echo "P0 DTA_HF FULL_LOG=$log_dir/hf_dta.log"
