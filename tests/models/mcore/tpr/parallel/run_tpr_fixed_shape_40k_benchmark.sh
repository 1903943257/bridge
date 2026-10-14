#!/usr/bin/env bash
# Isolated Step6/16/17 with existing session-first 40K captures.
# Same HF checkpoint, 1 warmup + 3 measured F/B for each mode. NO AdamW.
# Usage: bash this_script.sh 40K_MANIFEST NEW_OUTPUT_DIR
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
if [[ "$#" != 2 ]]; then
  echo "Usage: $0 40K_MANIFEST NEW_OUTPUT_DIR" >&2
  exit 2
fi
MANIFEST="$(realpath "$1")"
OUTPUT="$2"
[[ -f "$MANIFEST" ]] || { echo "Missing 40K manifest: $MANIFEST" >&2; exit 2; }
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists, refuse overwrite: $OUTPUT" >&2; exit 2; }
mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_BENCH_OUTPUT_DIR="$OUTPUT"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
# Original 20-step stored logprobs. Keep capture ON for comparable PPO timing.
export TPR_BENCH_CAPTURE_LOGPROBS="${TPR_BENCH_CAPTURE_LOGPROBS:-1}"
# Disable TPR-only segment profiler for primary FAIR performance runs.
export TPR_EXEC_PROFILE="${TPR_EXEC_PROFILE:-0}"
unset TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC TPR_MEM_TRACE || true
export PYTHONUNBUFFERED=1
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing checkpoint: $TPR_QWEN_4B_PATH" >&2; exit 2; }
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Requires physical NPUs 8..15 as in the original 40K replay" >&2
  exit 2
}

python - "$MANIFEST" <<'PY'
import json, pathlib, sys
m=json.load(open(sys.argv[1]))
assert m["protocol"]=="real_actor_session_first_8merged_v3"
assert m["steps"]==20 and m["expected_sessions"]==8 and m["max_seq_len"]==40960
for step in (6,16,17):
    x=m["items"][step-1]
    assert x["step"]==step and x["source_global_step"]==step
    assert x["rows"]==list(range(8)) and x["session_count"]==8
    assert x["max_sequence_length"]<=40960 and pathlib.Path(x["capture"]).is_file()
    print(f"TPR_FIXED_SHAPE_INPUT step={step} input={x['capture']} tokens={x['logical_full_tokens']} ppo={x['ppo_tokens']}",flush=True)
PY

for step in 6 16 17; do
  for mode in full tpr; do
    export TPR_BENCH_SOURCE_STEP="$step"
    export TPR_OFFLINE_MODE="$mode"
    echo "TPR_FIXED_SHAPE_START step=$step mode=$mode warmups=$TPR_BENCH_WARMUPS repeats=$TPR_BENCH_REPEATS profiler=$TPR_EXEC_PROFILE"
    torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
      --master_port="${TPR_BENCH_MASTER_PORT:-29791}" \
      -m pytest -q -s -x \
      tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
      2>&1 | tee "$OUTPUT/step${step}_${mode}.log"
  done
done

python - "$OUTPUT" <<'PY'
import json, pathlib, sys
d=pathlib.Path(sys.argv[1])
result=[]
for step in (6,16,17):
    full=json.loads((d/f"step{step}_full_fixed_shape.json").read_text())
    tpr=json.loads((d/f"step{step}_tpr_fixed_shape.json").read_text())
    for field in ("input_sha256","source_capture","checkpoint","warmups",
                  "repeats","capture_logprobs","profiler_enabled"):
        if full[field]!=tpr[field]:
            raise AssertionError(f"Step {step} Full/TPR mismatch field={field}")
    f=full["median_fwd_bwd_s"]
    t=tpr["median_fwd_bwd_s"]
    result.append({"step":step,"full_median_s":f,"tpr_median_s":t,
                   "speedup":f/t,"input_sha256":full["input_sha256"]})
    print(f"TPR_FIXED_SHAPE_COMPARE step={step} full_median_s={f:.6f} "
          f"tpr_median_s={t:.6f} speedup={f/t:.6f} "
          f"profile={int(full['profiler_enabled'])}",flush=True)
(d/"comparison.json").write_text(json.dumps(result,indent=2))
print("TPR_FIXED_SHAPE_COMPARISON="+str(d/"comparison.json"),flush=True)
PY
