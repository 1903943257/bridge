#!/usr/bin/env bash
# AReaL DTA training loss Forward oracle. Run from bridge's VERL root.
# Do not git pull in /workspace/uni-agent/verl.
set -euo pipefail
: "${TPR_REAL_TQ_BATCH:?Set real TQ batch path}"
: "${TPR_QWEN_1_7B_PATH:?Set HF Qwen3-1.7B model path}"
[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Missing TQ: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Missing HF model: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }
export TPR_RUN_QWEN17_DTA_TRAIN_FWD=1
export TPR_QWEN17_DTA_PROMPT=128
export TPR_QWEN17_DTA_RESPONSE=64
export TPR_QWEN17_DTA_HF_ATTN="${TPR_QWEN17_DTA_HF_ATTN:-sdpa}"
export TPR_QWEN17_DTA_TRAIN_BLOCK_SIZES="${TPR_QWEN17_DTA_TRAIN_BLOCK_SIZES:-64,-1}"
export TPR_QWEN17_DTA_TRAIN_CUT_F1_TAIL="${TPR_QWEN17_DTA_TRAIN_CUT_F1_TAIL:-1}"
log_dir="${TPR_QWEN17_DTA_TRAIN_LOG_DIR:-$(mktemp -d /tmp/tpr_qwen17_dta_train.XXXXXX)}"
mkdir -p "$log_dir"
unit_file="tests/models/mcore/tpr/unit/test_qwen17_areal_training_forward.py"
forward_unit_file="tests/models/mcore/tpr/unit/test_qwen17_areal_exact_reference.py"
npu_file="tests/models/mcore/tpr/correctness/test_qwen3_1_7b_hf_dta_training_forward_npu.py"
echo "P0 DTA_TRAIN START log_dir=$log_dir attn=$TPR_QWEN17_DTA_HF_ATTN block_sizes=$TPR_QWEN17_DTA_TRAIN_BLOCK_SIZES"
python -m pytest -x -q --tb=short "$unit_file" "$forward_unit_file" > "$log_dir/unit.log" 2>&1 || {
  echo "P0 DTA_TRAIN UNIT_FAILED log=$log_dir/unit.log" >&2
  tail -100 "$log_dir/unit.log" >&2
  exit 1
}
cat "$log_dir/unit.log"
python -m pytest -x -q -s --tb=short "$npu_file" > "$log_dir/train_forward.log" 2>&1 || {
  echo "P0 DTA_TRAIN NPU_FAILED log=$log_dir/train_forward.log" >&2
  tail -100 "$log_dir/train_forward.log" >&2
  exit 1
}
grep -E '^P0 DTA_TRAIN (CONFIG|FWD_ONLY_RESPONSE|SUMMARY|RESPONSE_SUMMARY|VS_FWD_ONLY|EVENT|ROW|OUTLIER|TARGET|RESULT)' "$log_dir/train_forward.log"
grep -q '^P0 DTA_TRAIN RESULT status=PASS execution=TRAIN_LOSS_FORWARD_CONTROL' "$log_dir/train_forward.log" || {
  echo "P0 DTA_TRAIN missing success marker" >&2
  exit 1
}
echo "P0 DTA_TRAIN FULL_LOG=$log_dir/train_forward.log"
