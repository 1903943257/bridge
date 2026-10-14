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
# Match the already-successful UniAgent E2E environment, NOT the NVIDIA
# Megatron-Bridge PEFT stack (which unconditionally imports TransformerEngine).
# Online: use_mbridge=True + vanilla_mbridge=True -> legacy "mbridge".
# Do not inject /workspace/Megatron-Bridge/src or pip-install TE on Ascend.
if [[ -n "${TPR_MEGATRON_BRIDGE_SRC:-}" ]]; then
  echo "ERROR: unset TPR_MEGATRON_BRIDGE_SRC; VERL Full uses vanilla_mbridge=True" >&2
  exit 2
fi
# An inherited source path from the previous failed experiment can shadow the
# working legacy MBridge environment; strip only this known incompatible path.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$(python - <<'PY'
import os
print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":")
               if p and p != "/workspace/Megatron-Bridge/src"))
PY
)"
fi
export PYTHONPATH="/workspace/vllm-ascend:/workspace/uni-agent-cc/verl:/workspace/uni-agent-cc:${PYTHONPATH:-}"

# Check the FULL Engine source branch (not just the package name) before NPU
# launch. A different upstream VERL snapshot might lack vanilla_mbridge even
# though the working E2E script passes that setting.
python - <<'PY'
import inspect
import sys
import mbridge
from verl.workers.config import McoreEngineConfig
from verl.workers.engine.mindspeed.transformer_impl import MindspeedEngineWithLMHead
from verl.models.mcore.mbridge import AutoBridge
fields = inspect.signature(McoreEngineConfig).parameters
if "vanilla_mbridge" not in fields:
    raise SystemExit(
        "VERL_FULL_PREFLIGHT_FAIL: installed McoreEngineConfig lacks "
        "vanilla_mbridge. Need the SAME VERL code revision as the successful "
        "UniAgent E2E, not an unrelated upstream Megatron-Bridge-only version."
    )
source = inspect.getsource(MindspeedEngineWithLMHead._build_tf_config)
if "vanilla_mbridge" not in source and "vanilla_bridge" not in source:
    raise SystemExit(
        "VERL_FULL_PREFLIGHT_FAIL: installed Engine lacks the proven legacy "
        "mbridge branch. Inspect the actual successful E2E Engine revision."
    )
print("VERL_FULL_PREFLIGHT_PASS", "python=", sys.executable,
      "legacy_mbridge=", getattr(mbridge, "__file__", None),
      "AutoBridge=", AutoBridge,
      "config_vanilla_mbridge=True", flush=True)
PY

export PYTHONUNBUFFERED=1
export TPR_RUN_VERL_ENGINE_FULL_REPLAY=1
export TPR_OFFLINE_CAPTURE_MANIFEST="$RUN_ROOT/capture_manifest.json"
export TPR_OFFLINE_OUTPUT_DIR="$RUN_ROOT/offline"
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_VERL_FULL_REPLAY_STEPS="$STEPS"
# Smoke and full-20 jobs must start independently from the original checkpoint.
# Never resume a 20-step optimizer run from only a one-step diagnostic snapshot.
if [[ "$STEPS" == 20 ]]; then
  export TPR_VERL_FULL_REPLAY_PREFIX=verl_full
elif [[ "$STEPS" == 1 ]]; then
  export TPR_VERL_FULL_REPLAY_PREFIX=verl_full_smoke
else
  export TPR_VERL_FULL_REPLAY_PREFIX="verl_full_test${STEPS}"
fi
export TPR_OFFLINE_ADAMW_LR="${TPR_OFFLINE_ADAMW_LR:-2e-5}"
export TPR_OFFLINE_ADAMW_CLIP_NORM="${TPR_OFFLINE_ADAMW_CLIP_NORM:-1.0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
if [[ "$ASCEND_RT_VISIBLE_DEVICES" != "8,9,10,11,12,13,14,15" ]]; then
  echo "Requires exactly physical NPUs 8..15; do not silently change topology" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/offline"
echo "VERL_ENGINE_FULL_LAUNCH engine=MindspeedEngineWithLMHead steps=$STEPS recompute=OFF" 
echo "VERL_ENGINE_FULL_LAUNCH output_prefix=$RUN_ROOT/offline/${TPR_VERL_FULL_REPLAY_PREFIX}_step"
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29787 \
  -m pytest -vv -s -x \
  tests/models/mcore/tpr/parallel/test_verl_engine_full_replay_npu.py \
  2>&1 | tee "$RUN_ROOT/logs/${TPR_VERL_FULL_REPLAY_PREFIX}_replay.log"
