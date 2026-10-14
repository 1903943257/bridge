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
# The legacy online capture stack used the separate "mbridge" package.
# This genuine VERL Engine imports "megatron.bridge" from Megatron-Bridge.
# Do NOT install/upgrade arbitrary Megatron or MindSpeed packages here:
# first reuse the exact local Megatron-Bridge source when it exists.
if [[ -n "${TPR_MEGATRON_BRIDGE_SRC:-}" ]]; then
  [[ -f "${TPR_MEGATRON_BRIDGE_SRC}/megatron/bridge/__init__.py" ]] || {
    echo "Invalid TPR_MEGATRON_BRIDGE_SRC: ${TPR_MEGATRON_BRIDGE_SRC}" >&2
    exit 2
  }
  export PYTHONPATH="${TPR_MEGATRON_BRIDGE_SRC}:${PYTHONPATH:-}"
elif [[ -f /workspace/Megatron-Bridge/src/megatron/bridge/__init__.py ]]; then
  # Prefer a locally installed package, but allow the known local source tree
  # when the megatron.bridge submodule is missing.
  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
assert importlib.util.find_spec("megatron.bridge") is not None
PY
  then
    export PYTHONPATH="/workspace/Megatron-Bridge/src:${PYTHONPATH:-}"
  fi
fi

# Fail before launching eight NPU workers if the true Engine dependency is
# missing or incompatible.  mbridge != megatron.bridge.
python - <<'PY'
import importlib.util
import sys
try:
    import megatron.core as core
    from megatron.bridge import AutoBridge
except (ImportError, ModuleNotFoundError) as exc:
    raise SystemExit(
        "VERL_FULL_PREFLIGHT_FAIL: genuine VERL/MBridge Engine requires "
        "'megatron.bridge.AutoBridge', which is NOT the separate 'mbridge' "
        "package used by the older capture script. Set "
        "TPR_MEGATRON_BRIDGE_SRC=/path/to/Megatron-Bridge/src if that "
        "source tree is already available; otherwise install a "
        "Megatron-Bridge version compatible with this NPU/Megatron-Core "
        "environment. Do not upgrade Megatron-Core/MindSpeed blindly. "
        f"Original error: {exc!r}"
    ) from exc
print("VERL_FULL_PREFLIGHT_PASS", "python=", sys.executable,
      "megatron_core=", getattr(core, "__file__", None),
      "AutoBridge=", AutoBridge, flush=True)
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
