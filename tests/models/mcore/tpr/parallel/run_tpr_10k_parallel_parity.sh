#!/usr/bin/env bash
# Controlled Step6 10,240-token Qwen3-4B TPR CP/DP correctness matrix (TP fixed at 2).
# One-time 10K capture from previous matched 30K pair. Repeatable/resumable;
# no model updates, no performance comparison, strict BF16 replica agreement.
# Usage: bash tests/models/mcore/tpr/parallel/run_tpr_10k_parallel_parity.sh \
#   /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
#   /workspace/tpr_10k_axes_correctness_v1
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
if [[ "$#" != 2 ]]; then
  echo "Usage: $0 EXISTING_30K_PAIR_ROOT NEW_10K_OUTPUT_ROOT" >&2
  exit 2
fi
SOURCE="$(realpath "$1")"
ROOT="$2"
[[ -f "$SOURCE/step6_cap30720.pt" ]] || {
  echo "Missing original derived 30K Step6 capture" >&2; exit 2
}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Requires physical NPUs 8..15 to preserve experiment locality" >&2; exit 2
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || {
  echo "Missing Qwen3-4B-128K checkpoint" >&2; exit 2
}
export HCCL_DETERMINISTIC=true
export TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_10K_PARITY_ONLY=1
export TPR_RUN_10K_PARALLEL_PARITY=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_PPO_LOGPROB_CHUNK_TOKENS=512
export TPR_EXEC_PROFILE=0
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

if [[ ! -e "$ROOT" ]]; then
  python -m tests.models.mcore.tpr.parallel.prepare_tpr_10k_parity_capture \
    "$SOURCE" "$ROOT" --step 6
fi
ROOT="$(realpath "$ROOT")"
[[ -f "$ROOT/parity_capture_audit.json" ]] || {
  echo "Existing ROOT is not a prepared 10K correctness workspace" >&2; exit 2
}
export TPR_10K_CAPTURE="$ROOT/step6_cap10240_parity.pt"
[[ -f "$TPR_10K_CAPTURE" ]] || {
  echo "10K capture missing" >&2; exit 2
}

# One-factor-at-a-time comparisons:
#   TPR algorithm (with CP=1): unshared TP2 vs shared TP2
#   TP1 is intentionally NOT part of this matrix (single-NPU OOM).
#   CP only:   TP2CP1DP1 vs TP2CP2DP1
#   DP only:   TP2CP1DP1 vs TP2CP1DP2
#   DP @ CP2:  TP2CP2DP1 vs TP2CP2DP2
# No layer recompute, dropout, changes to historical PPO labels, optimizer.
while read -r NAME TP CP DP MODE; do
  [[ -n "$NAME" ]] || continue
  if [[ -f "$ROOT/$NAME/result.json" ]]; then
    echo "TPR_10K_CASE_REUSE name=$NAME path=$ROOT/$NAME/result.json"
    continue
  fi
  if [[ -e "$ROOT/$NAME" ]]; then
    echo "Refusing partial or unrelated case directory: $ROOT/$NAME" >&2
    exit 2
  fi
  export TPR_10K_TP="$TP"
  export TPR_10K_CP="$CP"
  export TPR_10K_DP="$DP"
  export TPR_10K_MODE="$MODE"
  export TPR_10K_OUTPUT="$ROOT/$NAME"
  WORLD=$((TP * CP * DP))
  echo "TPR_10K_CASE_START name=$NAME tp=$TP cp=$CP dp=$DP world=$WORLD"
  torchrun --nproc_per_node="$WORLD" --master_addr=127.0.0.1 \
    --master_port="${TPR_BENCH_MASTER_PORT:-29861}" \
    -m pytest -q -s -x \
    tests/models/mcore/tpr/parallel/test_tpr_10k_parallel_parity_npu.py \
    2>&1 | tee "$ROOT/$NAME.log"
done <<'CASES'
unshared_tp2cp1dp1 2 1 1 unshared
tpr_tp2cp1dp1 2 1 1 tpr
tpr_tp2cp2dp1 2 2 1 tpr
tpr_tp2cp1dp2 2 1 2 tpr
tpr_tp2cp2dp2 2 2 2 tpr
CASES

if [[ ! -e "$ROOT/comparison_10k.json" ]]; then
  python -m tests.models.mcore.tpr.parallel.compare_tpr_10k_parallel_parity \
    "$ROOT" | tee "$ROOT/comparison_10k.log"
else
  echo "TPR_10K_PARITY_COMPARISON_REUSE=$ROOT/comparison_10k.json"
fi
