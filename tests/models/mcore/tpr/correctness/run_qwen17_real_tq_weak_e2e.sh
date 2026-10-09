#!/usr/bin/env bash
# P0 *weak* offline Qwen3-1.7B actor optimizer smoke.
# Run from a working VERL/bridge training runtime. This script does NOT
# invoke UniAgent, ClaudeCode, or require old uni-agent rollout scripts.
# Reads a pre-existing real TQ dump; PPO history is diagnostic/recomputed.
set -euo pipefail

: "${TPR_REAL_TQ_BATCH:?Set TPR_REAL_TQ_BATCH to real uniagent-cc tq_batch.pt}"
: "${TPR_QWEN_1_7B_PATH:?Set TPR_QWEN_1_7B_PATH to real Qwen3-1.7B checkpoint}"
if [[ ! -f "$TPR_REAL_TQ_BATCH" ]]; then
  echo "Real TQ file missing: $TPR_REAL_TQ_BATCH" >&2
  exit 2
fi
if [[ ! -d "$TPR_QWEN_1_7B_PATH" ]]; then
  echo "Real Qwen3-1.7B checkpoint missing: $TPR_QWEN_1_7B_PATH" >&2
  exit 2
fi

export TPR_RUN_QWEN17_WEAK_E2E=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN17_WEAK_E2E_PROMPT="${TPR_QWEN17_WEAK_E2E_PROMPT:-128}"
export TPR_QWEN17_WEAK_E2E_RESPONSE="${TPR_QWEN17_WEAK_E2E_RESPONSE:-64}"
export TPR_QWEN17_WEAK_E2E_LR="${TPR_QWEN17_WEAK_E2E_LR:-0.0001}"
export TPR_QWEN17_WEAK_E2E_MASTER_DEVICE="${TPR_QWEN17_WEAK_E2E_MASTER_DEVICE:-npu}"

# Fail explicitly rather than hiding left-over numerical interventions.
for knob in \
  TPR_QWEN17_PPO_TILE_GEMM \
  TPR_QWEN17_GPT_FP32_GEMM \
  TPR_QWEN17_PPO_FC2_FIXED_M \
  TPR_QWEN17_SPLIT_FP32_DW_BACKWARD; do
  if [[ -n "${!knob:-}" ]]; then
    echo "Unset diagnostic knob $knob before the default-BF16 training test" >&2
    exit 2
  fi
done

log_dir="${TPR_QWEN17_WEAK_E2E_LOG_DIR:-/tmp/tpr_qwen17_weak_e2e}"
mkdir -p "$log_dir"
test_path="tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_weak_e2e_npu.py"
if [[ ! -f "$test_path" ]]; then
  echo "Run from the bridge/VERL root where $test_path exists" >&2
  exit 2
fi
echo "P0 WEAK_TQ START checkpoint=$TPR_QWEN_1_7B_PATH"
echo "P0 WEAK_TQ START tq=$TPR_REAL_TQ_BATCH"
echo "P0 WEAK_TQ START master_device=$TPR_QWEN17_WEAK_E2E_MASTER_DEVICE"
echo "P0 WEAK_TQ START log_dir=$log_dir"

for mode in native tpr; do
  logfile="$log_dir/${mode}.log"
  echo "===== $mode default BF16 training with real TQ tokens ====="
  TPR_QWEN17_WEAK_E2E_EXECUTION="$mode" \
    python -m pytest -x -s -q --tb=long "$test_path" > "$logfile" 2>&1 || {
      echo "$mode weak training step FAILED; log=$logfile" >&2
      tail -120 "$logfile" >&2
      exit 1
    }
  grep -E 'P0 WEAK_TQ (CONFIG|BACKWARD|OPTIMIZER_STEP|RESULT)' "$logfile"
  if ! grep -q "P0 WEAK_TQ RESULT status=PASS execution=$(echo "$mode" | tr '[:lower:]' '[:upper:]')" "$logfile"; then
    echo "$mode: missing explicit completed optimizer/updated-forward PASS" >&2
    exit 1
  fi
done

python - "$log_dir/native.log" "$log_dir/tpr.log" <<'PY'
import re, sys
data = {}
for name, path in zip(("native", "tpr"), sys.argv[1:]):
    content = open(path, encoding="utf-8").read()
    line = next((v for v in content.splitlines()
                 if "P0 WEAK_TQ RESULT status=PASS" in v), None)
    loss_line = next((v for v in content.splitlines()
                      if "P0 WEAK_TQ BACKWARD status=PASS" in v), None)
    if line is None or loss_line is None:
        raise SystemExit(f"{name}: missing final PASS")
    def metric(text, key):
        found = re.search(r"(?:^| )" + re.escape(key) + r"=([+-]?[0-9.eE+-]+)", text)
        if not found:
            raise SystemExit(f"{name}: missing {key}")
        return float(found.group(1))
    data[name] = dict(
        loss=metric(loss_line, "loss"),
        fb=metric(line, "forward_backward_seconds"),
        opt=metric(line, "optimizer_seconds"),
        peak=metric(line, "npu_peak_alloc_mib"),
    )
native, tpr = data["native"], data["tpr"]
print("P0 WEAK_TQ COMPARISON (CONTROLLED CANN, CROPPED REAL TQ, NOT PRODUCTION)")
print(f"  loss_native={native['loss']:.9g}, loss_tpr={tpr['loss']:.9g}, abs_delta={abs(native['loss']-tpr['loss']):.9g}")
print(f"  fwd_bwd_native={native['fb']:.5f}s, tpr={tpr['fb']:.5f}s, speedup={native['fb']/max(tpr['fb'],1e-12):.4f}x")
print(f"  optimizer_total_native={native['opt']:.5f}s, tpr={tpr['opt']:.5f}s")
print(f"  peak_alloc_native={native['peak']:.3f}MiB, tpr={tpr['peak']:.3f}MiB")
print("  all_parameter_update_parity=UNVERIFIED true_rollout_old_logprobs=UNVERIFIED")
PY
