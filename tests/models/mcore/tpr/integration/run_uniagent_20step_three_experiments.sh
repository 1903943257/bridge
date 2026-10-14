#!/usr/bin/env bash
# ONE launch: E1 real online Native recompute 20 steps -> E2 Full offline
# 20 DISTINCT captured PPO minibatches -> E3 Turn-TPR same 20 captures.
#
# WARNING: E2 is independent complete-sequence TPR-capable RingCP4 control,
# NOT production Megatron/MindSpeed Native FA. The replay optimizer is
# FP32-master CPU AdamW, not VERL online optimizer.
set -euo pipefail

VERL_ROOT=/workspace/uni-agent-cc/verl
MODEL=/workspace/hf_models/Qwen3-4B-128K
ONLINE_SCRIPT="$VERL_ROOT/tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh"
REPLAY_SCRIPT="$VERL_ROOT/tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_multistep.sh"
PREP_SCRIPT="$VERL_ROOT/tests/models/mcore/tpr/parallel/prepare_real_actor_20step_manifest.py"
CAPTURE_PATCH="$VERL_ROOT/verl/models/mcore/tpr/actor_capture.py"
if [[ ! -f "$ONLINE_SCRIPT" || ! -f "$REPLAY_SCRIPT" || ! -f "$PREP_SCRIPT" || ! -f "$CAPTURE_PATCH" ]]; then
  echo "ERROR: sync the current Bridge main tests and verl/models/mcore/tpr into $VERL_ROOT first" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "ERROR: missing real Qwen3-4B checkpoint: $MODEL" >&2
  exit 2
fi
# Real PPO capture must be wired into the actual VERL TrainingWorker;
# just exporting TPR_CAPTURE_ACTOR_MINIBATCH_DIR is not sufficient.
WORKER="$VERL_ROOT/verl/workers/engine_workers.py"
if [[ ! -f "$WORKER" ]] || ! grep -Fq 'TPR P0 real actor-update capture' "$WORKER"; then
  echo "ERROR: TrainingWorker capture hook is not installed: $WORKER" >&2
  echo "Apply/check Bridge patches/apply_tpr_actor_capture.py before the nightly run." >&2
  exit 2
fi
if [[ -f /workspace/.venvs/uniagent-cc/bin/activate ]]; then
  source /workspace/.venvs/uniagent-cc/bin/activate
fi
cd "$VERL_ROOT"

# Every invocation gets an immutable result directory. Avoid any reuse of an
# old actor capture, optimizer snapshot, or performance log.
RUN_ROOT="/workspace/tpr_nightly_20step_$(date +%Y%m%d_%H%M%S)"
if [[ -e "$RUN_ROOT" ]]; then
  echo "ERROR: output exists: $RUN_ROOT" >&2; exit 2
fi
mkdir -p "$RUN_ROOT"/{logs,captures,offline}
echo "$RUN_ROOT" > /workspace/tpr_nightly_20step_latest.txt
printf "TPR_20STEP_RUN_ROOT=%s\n" "$RUN_ROOT"

# Online runner installs the new opt-in 48K session hardcap in UniAgent
# before Ray launches. The patch script MUST be discoverable locally.
if [[ -f "$VERL_ROOT/patches/apply_uniagent_session_hardcap.py" ]]; then
  export BRIDGE_ROOT="$VERL_ROOT"
elif [[ -f /workspace/bridge/patches/apply_uniagent_session_hardcap.py ]]; then
  export BRIDGE_ROOT=/workspace/bridge
else
  echo "ERROR: 48K session patcher unavailable in either $VERL_ROOT/patches or /workspace/bridge/patches" >&2
  echo "Sync Bridge patches/ to one of those locations before the overnight run." >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

# ================================================================
# E1: FULL-SEQUENCE ONLINE REAL 20 STEPS, NATIVE RECOMPUTE ENABLED.
# Explicitly set every critical value here: NO caller env tuning.
# ================================================================
export N=8
export TOTAL_TRAINING_STEPS=20
export TOTAL_EPOCHS=20
export NATIVE_RECOMPUTE=True
export N_GPUS_PER_NODE=8
export TRAIN_PROMPT_BSZ=1
export GEN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export SESSION_HARD_CAP=True
export MODEL_PATH="$MODEL"
export RUN_NAME="$(basename "$RUN_ROOT")"
export TQ_DUMP_DIR="$RUN_ROOT/tq_dump"
export TPR_CAPTURE_ACTOR_MINIBATCH_DIR="$RUN_ROOT/captures"
echo "TPR_NIGHTLY_E1_START native_online_20step_recompute_true"
if bash "$ONLINE_SCRIPT" 2>&1 | tee "$RUN_ROOT/logs/e1_online_20step.log"; then
  echo "TPR_NIGHTLY_E1_TRAINER_EXIT=0"
else
  echo "TPR_NIGHTLY_E1_FAILED: online training failed; skipping offline jobs" >&2
  exit 1
fi

# VERL/V1 uses Ray + vLLM-Ascend for E1. Tear down only this container's
# local Ray runtime before torchrun E2/E3, so retained actor/rollout workers
# do not consume NPU memory and invalidate the offline peak comparison.
ray stop --force > "$RUN_ROOT/logs/e1_ray_cleanup.log" 2>&1 || true

# Preflight: one rank0 original real Actor minibatch per online step;
# align steps 1..20, preserve old logprobs/advantages, select bounded
# COMPLETE trajectories, and refuse duplicates/missing captures.
MANIFEST="$RUN_ROOT/capture_manifest.json"
echo "TPR_NIGHTLY_CAPTURE_PREFLIGHT_START"
if python "$PREP_SCRIPT" "$RUN_ROOT/captures" "$MANIFEST" \
    --steps 20 --max-seq-len 49152 --expected-sessions 8 \
    2>&1 | tee "$RUN_ROOT/logs/capture_preflight.log"; then
  echo "TPR_NIGHTLY_CAPTURE_PREFLIGHT_PASS"
else
  echo "TPR_NIGHTLY_CAPTURE_PREFLIGHT_FAILED: incomplete or unsafe batch selection" >&2
  exit 1
fi

# Two independent replay jobs initialize from exactly the same HF checkpoint.
# Each job consumes all 20 recorded minibatches in the same capture-step
# order, with identical PPO objectives and CPU FP32-master AdamW settings.
export TPR_QWEN_4B_PATH="$MODEL"
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_REAL_ACTOR_MINIBATCH="$(python - "$MANIFEST" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['items'][0]['capture'])
PY
)"
readarray -t _selected_capture_info < <(python - "$MANIFEST" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
if m.get("protocol")!="real_actor_session_first_8merged_v3":
    raise SystemExit("expected the full-eight merged PPO replay manifest")
if m["online_steps"]!=20 or m["steps"]!=20:
    raise SystemExit("expected exactly one merged batch per online step")
if any(len(item["rows"])!=8 for item in m["items"]):
    raise SystemExit("merged batch is not complete N=8")
print(m["steps"])
print(",".join(map(str,m["items"][0]["rows"])))
PY
)
if [[ "${#_selected_capture_info[@]}" -ne 2 ]]; then
  echo "ERROR: invalid session-first multi-capture replay manifest" >&2; exit 2
fi
export TPR_OFFLINE_ROWS="${_selected_capture_info[1]}"
export TPR_OFFLINE_STEPS="${_selected_capture_info[0]}"
export TPR_OFFLINE_ADAMW_LR=2e-5
export TPR_OFFLINE_ADAMW_CLIP_NORM=1.0
export TPR_OFFLINE_OUTPUT_DIR="$RUN_ROOT/offline"
export TPR_OFFLINE_SAVE_FULL_GRADS=0
unset TPR_OFFLINE_FULL_GRAD_STEPS || true
export TPR_EXEC_PROFILE=1
export TPR_TURN_MIN_SEGMENT_TOKENS=128
unset TPR_MEM_TRACE || true

# ================================================================
# E2: OFFLINE FULL 20-step correctness/reference and segment perf.
# ================================================================
export TPR_TREE_POLICY=qwen_turn  # Full's keys intentionally never share
export TPR_OFFLINE_RUN_MODES=full
echo "TPR_NIGHTLY_E2_START offline_full_ring_20_distinct_captures"
full_exit=0
bash "$REPLAY_SCRIPT" > >(tee "$RUN_ROOT/logs/e2_full.log") \
                      2> >(tee "$RUN_ROOT/logs/e2_full.err" >&2) || full_exit=$?
echo "TPR_NIGHTLY_E2_EXIT=$full_exit"

# ================================================================
# E3: OFFLINE SHARED TURN TPR 20-step correctness and segment perf.
# Independent of E2 result. Full OOM must NOT skip TPR.
# ================================================================
export TPR_TREE_POLICY=qwen_turn
export TPR_OFFLINE_RUN_MODES=tpr
echo "TPR_NIGHTLY_E3_START offline_turn_tpr_20_distinct_captures"
tpr_exit=0
bash "$REPLAY_SCRIPT" > >(tee "$RUN_ROOT/logs/e3_tpr.log") \
                      2> >(tee "$RUN_ROOT/logs/e3_tpr.err" >&2) || tpr_exit=$?
echo "TPR_NIGHTLY_E3_EXIT=$tpr_exit"

# Both complete => CPU comparison of every capture step's Full/TPR
# logprob, sampled gradient, exact global BF16 gradnorm, PPO loss,
# FP32-master cumulative updates, and each segment's timings/peaks.
if [[ "$full_exit" == "0" && "$tpr_exit" == "0" ]]; then
  export TPR_OFFLINE_RUN_MODES=compare
  echo "TPR_NIGHTLY_COMPARISON_START"
  bash "$REPLAY_SCRIPT" 2>&1 | tee "$RUN_ROOT/logs/final_comparison.log"
else
  echo "TPR_NIGHTLY_PARTIAL: one offline mode failed; logs and successful rank snapshots retained" >&2
fi
echo "TPR_NIGHTLY_SUMMARY online=PASS full_exit=$full_exit tpr_exit=$tpr_exit root=$RUN_ROOT"
if [[ "$full_exit" != "0" || "$tpr_exit" != "0" ]]; then
  exit 1
fi
