#!/usr/bin/env bash
# Replay ONE original 8-Session 40K PPO source step on eight Ascend NPUs.
# TP2 x RingCP2 x DP2 TPR vs existing TP2 x RingCP4 x DP1 Native Full JSON.
# No optimizer updates; measured F/B includes gradient finalization, not AdamW.
# Usage: bash this_script.sh 40K_MANIFEST NEW_OUTPUT_DIR [SOURCE_STEP=6] [FULL_STEP_JSON]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

if (( $# < 2 || $# > 4 )); then
  echo "Usage: $0 40K_MANIFEST NEW_OUTPUT_DIR [SOURCE_STEP=6] [FULL_STEP_JSON]" >&2
  exit 2
fi
MANIFEST="$(realpath "$1")"
OUTPUT="$2"
STEP="${3:-6}"
FULL_JSON="${4:-}"
[[ -f "$MANIFEST" ]] || { echo "Missing 40K manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Output exists (refusing overwrite): $OUTPUT" >&2; exit 2; }
if [[ -n "$FULL_JSON" ]]; then
  FULL_JSON="$(realpath "$FULL_JSON")"
  [[ -f "$FULL_JSON" ]] || { echo "Missing Full baseline JSON" >&2; exit 2; }
fi
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Requires same physical NPUs 8..15 as Full baseline" >&2; exit 2;
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B-128K checkpoint" >&2; exit 2; }
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_BENCH_OUTPUT_DIR="$OUTPUT"
export TPR_BENCH_SOURCE_STEP="$STEP"
export TPR_OFFLINE_MODE=tpr
export TPR_RUN_TP2CP2DP2_ONE_STEP=1
export TPR_EXPERIMENTAL_OFFLINE_DP2=1
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
export TPR_BENCH_CAPTURE_LOGPROBS="${TPR_BENCH_CAPTURE_LOGPROBS:-1}"
# Isolated benchmark knob, NOT a VERL production flag. Use fp32 only
# after the standalone HCCL FP32 control has passed on this machine.
export TPR_OFFLINE_GRAD_SYNC_DTYPE="${TPR_OFFLINE_GRAD_SYNC_DTYPE:-bf16}"
[[ "$TPR_OFFLINE_GRAD_SYNC_DTYPE" == "bf16" || "$TPR_OFFLINE_GRAD_SYNC_DTYPE" == "fp32" ]] || {
  echo "TPR_OFFLINE_GRAD_SYNC_DTYPE must be bf16 or fp32" >&2
  exit 2
}
# Recompute PPO logprob intermediates per chunk instead of retaining the
# whole CP2 local-vocabulary softmax workspace until backward. Set 0 to
# reproduce the original code path; chunking is not native Megatron recompute.
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_EXEC_PROFILE="${TPR_EXEC_PROFILE:-0}"
export PYTHONUNBUFFERED=1
unset TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true
if [[ "${TPR_BENCH_MEM_TRACE:-0}" == "1" ]]; then
  export TPR_MEM_TRACE=1
else
  unset TPR_MEM_TRACE || true
fi

python - "$MANIFEST" "$STEP" <<'PY'
import json, pathlib, sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
step=int(sys.argv[2])
assert m["protocol"]=="real_actor_session_first_8merged_v3"
assert m["steps"]==20 and m["expected_sessions"]==8 and m["max_seq_len"]==40960
assert 1<=step<=20
x=m["items"][step-1]
assert x["step"]==x["source_global_step"]==step
assert x["rows"]==list(range(8)) and x["session_count"]==8
assert x["max_sequence_length"]<=40960
assert pathlib.Path(x["capture"]).is_file()
assert m["historical_ppo_labels_modified"] is False
print("TPR_DP2_INPUT step=%s source=%s ppo_tokens=%s full_tokens=%s" %
      (step,x["capture"],x["ppo_tokens"],x["logical_full_tokens"]),flush=True)
PY

echo "TPR_DP2_START source_step=$STEP topology=TP2,CP2,DP2 source=$MANIFEST"
echo "TPR_DP2_TIMING forward_backward_plus_CP_DP_gradient_finalize; no_optimizer_update logprob_chunk_tokens=$TPR_PPO_LOGPROB_CHUNK_TOKENS"
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
  --master_port="${TPR_BENCH_MASTER_PORT:-29793}" \
  -m pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_tp2cp2dp2_one_step_npu.py \
  2>&1 | tee "$OUTPUT/tpr_tp2cp2dp2_step${STEP}.log"

if [[ -n "$FULL_JSON" ]]; then
  python tests/models/mcore/tpr/parallel/compare_tpr_tp2cp2dp2_one_step.py \
    "$FULL_JSON" "$OUTPUT/step${STEP}_tpr_tp2cp2dp2.json" \
    "$OUTPUT/comparison.json" | tee "$OUTPUT/comparison.log"
else
  echo "TPR_DP2_RESULT=$OUTPUT/step${STEP}_tpr_tp2cp2dp2.json"
  echo "To compare, pass the existing step${STEP}_full_fixed_shape.json as fourth argument"
fi
