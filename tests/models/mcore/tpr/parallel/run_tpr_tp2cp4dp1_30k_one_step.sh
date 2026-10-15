#!/usr/bin/env bash
# One NEW NPU experiment: TPR TP2CP4DP1 on the EXACT Step-6 30K source
# previously used by Full TP2CP4DP1 vs TPR TP2CP2DP2 paired benchmark.
# Existing Full and DP2 JSONs are reused; no re-truncation or extra training.
# Usage:
#   bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp4dp1_30k_one_step.sh \
#     /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
#     /workspace/tpr_30k_cp4_deterministic_v1 [STEP=6]
set -euo pipefail
cd "$(dirname "$0")/../../../../.."
if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 EXISTING_30K_PAIR_ROOT NEW_OUTPUT_DIR [STEP=6]" >&2
  exit 2
fi
ROOT="$(realpath "$1")"
OUTPUT="$2"
STEP="${3:-6}"
MANIFEST="$ROOT/capture_manifest.json"
FULL_JSON="$ROOT/full/step${STEP}_full_fixed_shape.json"
DP2_JSON="$ROOT/tpr/step${STEP}_tpr_tp2cp2dp2.json"
[[ -f "$MANIFEST" && -f "$FULL_JSON" && -f "$DP2_JSON" ]] || {
  echo "Missing original derived manifest or Full/TPR-DP2 JSONs in $ROOT" >&2
  exit 2
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing overwrite of output directory: $OUTPUT" >&2
  exit 2
}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"
[[ "$ASCEND_RT_VISIBLE_DEVICES" == "8,9,10,11,12,13,14,15" ]] || {
  echo "Must use the same physical NPUs 8..15 as the existing pair" >&2
  exit 2
}
export TPR_QWEN_4B_PATH="${TPR_QWEN_4B_PATH:-/workspace/hf_models/Qwen3-4B-128K}"
[[ -d "$TPR_QWEN_4B_PATH" ]] || { echo "Missing Qwen3-4B checkpoint" >&2; exit 2; }
export HCCL_DETERMINISTIC="${HCCL_DETERMINISTIC:-true}"
[[ "$HCCL_DETERMINISTIC" == "true" ]] || {
  echo "This experiment requires HCCL_DETERMINISTIC=true" >&2; exit 2
}
export TPR_OFFLINE_GRAD_SYNC_DTYPE="${TPR_OFFLINE_GRAD_SYNC_DTYPE:-bf16}"
export TPR_BENCH_WARMUPS="${TPR_BENCH_WARMUPS:-1}"
export TPR_BENCH_REPEATS="${TPR_BENCH_REPEATS:-3}"
export TPR_BENCH_CAPTURE_LOGPROBS="${TPR_BENCH_CAPTURE_LOGPROBS:-1}"
export TPR_PPO_LOGPROB_CHUNK_TOKENS="${TPR_PPO_LOGPROB_CHUNK_TOKENS:-512}"
export TPR_EXEC_PROFILE=0
export TPR_BENCH_GRAD_WITNESS=1
export TPR_OFFLINE_CAPTURE_MANIFEST="$MANIFEST"
export TPR_OFFLINE_MODE=tpr
export TPR_BENCH_SOURCE_STEP="$STEP"
export TPR_RUN_FIXED_SHAPE_40K_BENCH=1
export TPR_OFFLINE_NATIVE_RECOMPUTE_LAYERS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export PYTHONUNBUFFERED=1
unset TPR_MEM_TRACE TPR_FIRST_LEAF_DIAG TPR_FIRST_LEAF_DIAG_SYNC || true

# FAIL BEFORE loading the checkpoint if any comparison dimension differs.
python - "$MANIFEST" "$FULL_JSON" "$DP2_JSON" "$STEP" <<'PY'
import json, os, pathlib, sys
manifest_path, full_path, dp_path, step_arg = sys.argv[1:5]
step = int(step_arg)
m = json.loads(pathlib.Path(manifest_path).read_text())
full = json.loads(pathlib.Path(full_path).read_text())
dp2 = json.loads(pathlib.Path(dp_path).read_text())
if (m.get("only_step_derived") != step or
        m.get("experimental_sequence_cap") != 30720 or
        m.get("historical_ppo_labels_modified") is not False):
    raise AssertionError("Requires the existing immutable, label-preserving 30K derived manifest")
item = m["items"][step - 1]
if (item["step"] != step or item["rows"] != list(range(8)) or
        item["max_sequence_length"] > 30720):
    raise AssertionError("Not the exact original eight 30K trajectories")
if pathlib.Path(item["capture"]).resolve() != pathlib.Path(full["source_capture"]).resolve():
    raise AssertionError("Derived manifest no longer points to recorded Full capture")
if full["source_capture"] != dp2["source_capture"] or full["input_sha256"] != dp2["input_sha256"]:
    raise AssertionError("Existing Full/DP2 baselines do not refer to identical capture bytes")
if full["source_step"] != dp2["source_step"] or full["source_step"] != step:
    raise AssertionError("Baseline source steps differ")
for key in ("warmups", "repeats", "capture_logprobs", "profiler_enabled",
            "checkpoint", "gradient_sync_reduce_dtype", "hccl_stats_reduce_dtype"):
    if full[key] != dp2[key]:
        raise AssertionError(f"Existing Full vs DP2 mismatched {key}")
for key, expected in (
    ("warmups", int(os.environ["TPR_BENCH_WARMUPS"])),
    ("repeats", int(os.environ["TPR_BENCH_REPEATS"])),
    ("capture_logprobs", os.environ["TPR_BENCH_CAPTURE_LOGPROBS"] == "1"),
    ("profiler_enabled", False),
    ("gradient_sync_reduce_dtype", os.environ["TPR_OFFLINE_GRAD_SYNC_DTYPE"]),
    ("hccl_stats_reduce_dtype", "float32"),
):
    if full[key] != expected:
        raise AssertionError(f"30K CP4 experiment {key}: expected={full[key]} current={expected}")
if pathlib.Path(full["checkpoint"]).resolve() != pathlib.Path(os.environ["TPR_QWEN_4B_PATH"]).resolve():
    raise AssertionError("Checkpoint differs from original Full/DP2 experiment")
if dp2["topology"] != {"tp":2,"cp":2,"dp":2,"world":8}:
    raise AssertionError("Wrong original DP2 topology")
if (full.get("effective_sequence_cap") != 30720 or
        dp2.get("effective_sequence_cap") != 30720):
    raise AssertionError("Full/DP2 results are not the 30K pair")
if dp2["ppo_logprob_chunk_tokens"] != int(os.environ["TPR_PPO_LOGPROB_CHUNK_TOKENS"]):
    raise AssertionError("TPR PPO logprob chunking differs from original DP2")
if full.get("gradient_witness") is None or dp2.get("gradient_witness") is None:
    raise AssertionError("Original Full/DP2 comparison requires both gradient witnesses")
print(
    "TPR_30K_CP4_PREFLIGHT_PASS "
    f"step={step} sha={full['input_sha256']} "
    f"global_ppo_tokens={dp2['global_ppo_tokens']} "
    f"warmups={full['warmups']} repeats={full['repeats']} "
    f"grad_dtype={full['gradient_sync_reduce_dtype']}",
    flush=True,
)
PY

mkdir -p "$OUTPUT"
OUTPUT="$(realpath "$OUTPUT")"
export TPR_BENCH_OUTPUT_DIR="$OUTPUT"
echo "TPR_30K_CP4_START source_step=$STEP topology=TP2CP4DP1 HCCL_DETERMINISTIC=$HCCL_DETERMINISTIC mode=tpr"
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 \
  --master_port="${TPR_BENCH_MASTER_PORT:-29827}" \
  -m pytest -q -s -x \
  tests/models/mcore/tpr/parallel/test_tpr_fixed_shape_40k_benchmark_npu.py \
  2>&1 | tee "$OUTPUT/tpr_tp2cp4dp1_step${STEP}.log"

python -m tests.models.mcore.tpr.parallel.compare_tpr_30k_three_way \
  "$FULL_JSON" "$OUTPUT/step${STEP}_tpr_fixed_shape.json" "$DP2_JSON" \
  "$OUTPUT/comparison_three_way.json" | tee "$OUTPUT/comparison_three_way.log"

echo "TPR_30K_CP4_RESULT=$OUTPUT/step${STEP}_tpr_fixed_shape.json"
echo "TPR_30K_THREE_WAY_COMPARISON=$OUTPUT/comparison_three_way.json"
