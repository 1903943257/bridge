#!/usr/bin/env bash
# E2 + E3 ONLY. Reuse an existing 20-step REAL Actor Capture. No Ray/rollouts.
# Usage: bash this_script.sh CAPTURE_DIR [RUN_ROOT] [both|full|tpr|compare]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "Usage: bash $0 /workspace/YOUR_20_STEP_ACTOR_CAPTURE_DIR [RUN_ROOT] [both|full|tpr|compare]" >&2
  exit 2
fi
CAPTURE_DIR="$(realpath "$1")"
[[ -d "$CAPTURE_DIR" ]] || { echo "Missing Capture Dir: $CAPTURE_DIR" >&2; exit 2; }
RUN_ROOT="${2:-/workspace/tpr_20step_offline_only_$(date +%Y%m%d_%H%M%S)}"
RUN_MODE="${3:-both}"
case "$RUN_MODE" in both|full|tpr|compare) ;; *) echo "Bad mode: $RUN_MODE" >&2; exit 2;; esac
if [[ "$RUN_MODE" == "compare" && ! -d "$RUN_ROOT/offline" ]]; then
  echo "Compare mode requires prior Full/TPR snapshot directory" >&2; exit 2
fi
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/offline"
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_OFFLINE_CAPTURE_MANIFEST="$RUN_ROOT/capture_manifest.json"
export TPR_OFFLINE_OUTPUT_DIR="$RUN_ROOT/offline"
export TPR_OFFLINE_STEPS=20
export TPR_OFFLINE_ADAMW_LR=2e-5
export TPR_OFFLINE_ADAMW_CLIP_NORM=1.0
# Strict parity run: disable Native Full recompute, matching the TPR route.
# Dedicated capacity experiments can invoke the native runner separately.
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_OFFLINE_SAVE_FULL_GRADS=0
unset TPR_OFFLINE_FULL_GRAD_STEPS || true
export TPR_EXEC_PROFILE=1
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_FULL_RING_BACKEND="${TPR_OFFLINE_FULL_RING_BACKEND:-native}"
unset TPR_MEM_TRACE || true
RUNNER=tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_multistep.sh
SELECTOR=tests/models/mcore/tpr/parallel/prepare_real_actor_20step_manifest.py
[[ -f "$RUNNER" && -f "$SELECTOR" ]] || {
  echo "Offline scripts missing: sync Bridge main tests into the Docker VERL checkout" >&2; exit 2;
}
if [[ "$RUN_MODE" != "compare" ]]; then
  if [[ ! -f "$TPR_OFFLINE_CAPTURE_MANIFEST" ]]; then
    python "$SELECTOR" "$CAPTURE_DIR" "$TPR_OFFLINE_CAPTURE_MANIFEST" \
      --steps 20 --max-seq-len 49152 --expected-sessions 8 \
      2>&1 | tee "$RUN_ROOT/logs/manifest.log"
  fi
fi
# One ONLINE step can contain several genuine Actor mini-batch captures.
# Replay ALL retained updates in manifest chronological order; never
# confuse capture_index with a Session trajectory index.
readarray -t manifest_info < <(python - "$TPR_OFFLINE_CAPTURE_MANIFEST" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d=json.load(f)
if (d.get("protocol")!="real_actor_session_first_8merged_v3"
        or d.get("online_steps")!=20 or d.get("steps")!=20
        or d.get("expected_sessions")!=8
        or len(d.get("items",[]))!=20
        or any(len(item["rows"])!=8 for item in d["items"])):
    raise SystemExit("Require full 8-Session per-step merged PPO manifest (v3)")
print(d["steps"])
print(d["items"][0]["capture"])
print(",".join(str(x) for x in d["items"][0]["rows"]))
PY
)
if [[ "${#manifest_info[@]}" -ne 3 ]]; then
    echo "Invalid session-first offline manifest" >&2; exit 2
fi
export TPR_OFFLINE_STEPS="${manifest_info[0]}"
export TPR_REAL_ACTOR_MINIBATCH="${manifest_info[1]}"
export TPR_OFFLINE_ROWS="${manifest_info[2]}"
echo "TPR_SESSION_FIRST_CAPTURE_SELECTION online_steps=20 replay_updates=$TPR_OFFLINE_STEPS sessions_per_step=8, NO total-token cap"
# Full uses the genuine Megatron schedule + unmodified MindSpeed attention.
# Do not require the old TPR ring monkeypatch's trailing-padding API here.
if [[ "$RUN_MODE" == both || "$RUN_MODE" == full ]]; then
  python - <<'PY'
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from verl.models.mcore.util import preprocess_thd_engine
from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits
from verl.models.mcore.tpr import TPRSelfAttention
print("NATIVE_FULL_PREFLIGHT=PASS megatron_schedule=available verl_thd=available")
PY
fi
echo "TPR_20STEP_OFFLINE_START capture_dir=$CAPTURE_DIR root=$RUN_ROOT mode=$RUN_MODE"
echo "E2: Native Megatron F/B schedule + MindSpeed Ring on FULL complete sequences"
echo "E3: Custom TPR Ring with Turn-based prefix reuse"
full_exit=0
tpr_exit=0
if [[ "$RUN_MODE" == "both" || "$RUN_MODE" == "full" ]]; then
  export TPR_OFFLINE_RUN_MODES=full
  bash "$RUNNER" 2>&1 | tee "$RUN_ROOT/logs/e2_full_native_ring.log" || full_exit=$?
  echo "E2_FINISHED status=$full_exit"
fi
if [[ "$RUN_MODE" == "both" || "$RUN_MODE" == "tpr" ]]; then
  export TPR_OFFLINE_RUN_MODES=tpr
  bash "$RUNNER" 2>&1 | tee "$RUN_ROOT/logs/e3_tpr_turn.log" || tpr_exit=$?
  echo "E3_FINISHED status=$tpr_exit"
fi
if [[ "$full_exit" == "0" && "$tpr_exit" == "0" ]] && \
  { [[ "$RUN_MODE" == "both" ]] || [[ "$RUN_MODE" == "compare" ]] ||
    { [[ "$RUN_MODE" == "full" ]] &&
      [[ -f "$RUN_ROOT/offline/tpr_step${TPR_OFFLINE_STEPS}_rank7.pt" ]]; }; }; then
  export TPR_OFFLINE_RUN_MODES=compare
  bash "$RUNNER" 2>&1 | tee "$RUN_ROOT/logs/final_comparison.log"
fi
echo "OFFLINE_20STEP_ROOT=$RUN_ROOT full_exit=$full_exit tpr_exit=$tpr_exit"
if [[ "$full_exit" != "0" || "$tpr_exit" != "0" ]]; then exit 1; fi
