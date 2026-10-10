#!/usr/bin/env bash
# Compare HF Full/AReaL-DTA and Megatron Native/real TPR Forest.
# Git pulls happen on the host; this script runs in Docker's VERL tree.
set -euo pipefail
: "${TPR_REAL_TQ_BATCH:?Real TQ dump required}"
: "${TPR_QWEN_1_7B_PATH:?HF Qwen3-1.7B path required}"
[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Missing TQ: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Missing Qwen: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }

export TPR_RUN_QWEN17_DTA_BACKWARD=1
export TPR_RUN_QWEN17_TRIPLET_TPR=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_DTA_HF_ATTN="${TPR_QWEN17_DTA_HF_ATTN:-sdpa}"
export TPR_DTA_BWD_MODES=native
export TPR_DTA_BWD_ROWS=8
export TPR_DTA_BWD_BLOCK="${TPR_DTA_BWD_BLOCK:-64}"
export TPR_DTA_BWD_OBJECTIVE="${TPR_DTA_BWD_OBJECTIVE:-ppo_unclipped}"

log_dir="${TPR_DTA_TRIPLET_DIR:-$(mktemp -d /tmp/tpr_dta_tpr_triplet.XXXXXX)}"
mkdir -p "$log_dir"
export TPR_DTA_TRIPLET_DIR="$log_dir"
base="tests/models/mcore/tpr"
hf_test="$base/correctness/test_qwen3_1_7b_areal_backward_ppo_npu.py"
tpr_test="$base/correctness/test_qwen3_1_7b_tpr_dta_triplet_npu.py"
cpu_test="$base/unit/test_qwen17_dta_tpr_triplet.py"
summary="$base/correctness/compare_qwen17_dta_tpr_triplet.py"
echo "P1 TRIPLET START objective=$TPR_DTA_BWD_OBJECTIVE log_dir=$log_dir"
python -m pytest -x -q --tb=short "$cpu_test" > "$log_dir/unit.log" 2>&1 || {
  echo "P1 TRIPLET UNIT_FAILED log=$log_dir/unit.log" >&2
  tail -120 "$log_dir/unit.log" >&2
  exit 1
}
cat "$log_dir/unit.log"

# Separate pytest subprocesses: Megatron/MindSpeed initialization must
# not leak into the HF checkpoint numerical control.
python -m pytest -x -q -s --tb=short "$hf_test" > "$log_dir/hf.log" 2>&1 || {
  echo "P1 TRIPLET HF_FAILED log=$log_dir/hf.log" >&2
  tail -120 "$log_dir/hf.log" >&2
  exit 1
}
grep -E '^P1 (DTA_BACKWARD (CONFIG|FULL|SUMMARY|RESULT)|DTA_TRIPLET HF_ARTIFACT)' "$log_dir/hf.log"
python -m pytest -x -q -s --tb=short "$tpr_test" > "$log_dir/tpr.log" 2>&1 || {
  echo "P1 TRIPLET TPR_FAILED log=$log_dir/tpr.log" >&2
  tail -160 "$log_dir/tpr.log" >&2
  exit 1
}
grep -E '^P1 TPR_TRIPLET (NATIVE|SUMMARY|MEGATRON_ARTIFACT|RESULT)' "$log_dir/tpr.log"
if [[ ! -f "$log_dir/hf_full_dta.pt" || ! -f "$log_dir/megatron_full_tpr.pt" ]]; then
  echo "P1 TRIPLET missing aligned HF/Megatron artifacts in $log_dir" >&2
  exit 1
fi
python "$summary" "$log_dir" | tee "$log_dir/comparison.log"
echo "P1 TRIPLET FULL_LOG_DIR=$log_dir"
