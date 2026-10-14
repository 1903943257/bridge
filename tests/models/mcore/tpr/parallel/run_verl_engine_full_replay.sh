#!/usr/bin/env bash
# Genuine VERL/MindSpeed Engine Full on the existing 8-Session, 20-update Actor
# manifest; never runs Rollout, never touches the old Full/TPR snapshots.
# Usage: bash run_verl_engine_full_replay.sh RUN_ROOT [1..20]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

RUN_ROOT="${1:?Usage: bash $0 RUN_ROOT [steps=1]}"
STEPS="${2:-1}"
if ! [[ "$STEPS" =~ ^([1-9]|1[0-9]|20)$ ]]; then
  echo "steps must be 1..20" >&2
  exit 2
fi
[[ -f "$RUN_ROOT/capture_manifest.json" ]] || {
  echo "Existing capture_manifest.json not found: $RUN_ROOT" >&2
  exit 2
}
[[ -d /workspace/hf_models/Qwen3-4B-128K ]] || {
  echo "Missing original model checkpoint" >&2
  exit 2
}
export PYTHONUNBUFFERED=1
export TPR_RUN_VERL_ENGINE_FULL_REPLAY=1
export TPR_OFFLINE_CAPTURE_MANIFEST="$RUN_ROOT/capture_manifest.json"
export TPR_OFFLINE_OUTPUT_DIR="$RUN_ROOT/offline"
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_VERL_FULL_REPLAY_STEPS="$STEPS"
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-2e-5}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9,10,11,12,13,14,15" ]]; then
  echo "Requires exactly physical NPUs 8..15; do not silently change topology" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/offline"
echo "VERL_ENGINE_FULL_LAUNCH engine=MindspeedEngineWithLMHead steps=$STEPS recompute=OFF" 
echo "VERL_ENGINE_FULL_LAUNCH output_prefix=$RUN_ROOT/offline/verl_full_step"
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29787 \
  -m pytest -vv -s -x \
  tests/models/mcore/tpr/parallel/test_verl_engine_full_replay_npu.py \
  2>&1 | tee "$RUN_ROOT/logs/verl_engine_full_replay.log"
