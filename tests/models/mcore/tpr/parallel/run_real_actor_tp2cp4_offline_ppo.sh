#!/usr/bin/env bash
# Isolated 8-NPU TP2 x RingCP4 true-PPO recorded-trajectory one-step profile.
# Must run from the synced Docker VERL root, no rollout/Ray is launched.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
export TPR_RUN_REAL_ACTOR_TP2CP4=1
export TPR_REAL_ACTOR_MINIBATCH="${TPR_REAL_ACTOR_MINIBATCH:-/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_OFFLINE_ROWS="${TPR_OFFLINE_ROWS:-4,6}"
export TPR_OFFLINE_OUTPUT_DIR="${TPR_OFFLINE_OUTPUT_DIR:-/workspace/tpr_offline_qwen4b_tp2cp4_row4_row6}"
: "${ASCEND_RT_VISIBLE_DEVICES:=8,9,10,11,12,13,14,15}"
export ASCEND_RT_VISIBLE_DEVICES
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9,10,11,12,13,14,15" ]]; then
    echo "ERROR: expected physical NPU 8-15; got $ASCEND_RT_VISIBLE_DEVICES" >&2
    exit 2
fi
[[ -f "$TPR_REAL_ACTOR_MINIBATCH" ]] || { echo "Missing actor capture" >&2; exit 2; }
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
mkdir -p "$TPR_OFFLINE_OUTPUT_DIR"
case "${TPR_OFFLINE_RUN_MODES:-both}" in
   both) modes=(full tpr) ;;
   full) modes=(full) ;;
   tpr) modes=(tpr) ;;
   compare) modes=() ;;
   *) echo "Unknown TPR_OFFLINE_RUN_MODES" >&2; exit 2;;
esac
errors=()
for mode in "${modes[@]}"; do
    export TPR_OFFLINE_MODE="$mode"
    echo "TPR_OFFLINE_START mode=$mode rows=$TPR_OFFLINE_ROWS device=$ASCEND_RT_VISIBLE_DEVICES"
    # An OOM in Full must not prevent the independent TPR memory experiment.
    if torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29784 \
       -m pytest -vv -s -x \
       tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_offline_ppo_npu.py \
       2>&1 | tee "$TPR_OFFLINE_OUTPUT_DIR/${mode}.log"; then
       echo "TPR_OFFLINE mode=$mode completed"
    else
       errors+=("$mode")
       echo "TPR_OFFLINE mode=$mode FAILED; retaining the log and trying remaining modes" >&2
    fi
done
if [[ "${#errors[@]}" == 0 ]] && \
   [[ "${TPR_OFFLINE_RUN_MODES:-both}" == "both" || "${TPR_OFFLINE_RUN_MODES:-both}" == "compare" ]]; then
    python tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_offline_ppo.py \
      "$TPR_OFFLINE_OUTPUT_DIR" | tee "$TPR_OFFLINE_OUTPUT_DIR/comparison.log"
fi
if (("${#errors[@]}" > 0)); then
    echo "TPR_OFFLINE failed_modes=${errors[*]} (logs preserved)" >&2
    exit 1
fi
