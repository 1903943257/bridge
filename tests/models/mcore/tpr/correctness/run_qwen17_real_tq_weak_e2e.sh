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

log_dir="${TPR_QWEN17_WEAK_E2E_LOG_DIR:-$(mktemp -d /tmp/tpr_qwen17_weak_e2e.XXXXXX)}"
mkdir -p "$log_dir"
# Sampling is only a diagnostic, NEVER a prerequisite for the real
# optimizer-step correctness gate. Set SAMPLE=0 to isolate a sampling
# kernel error while retaining Native and TPR AdamW step validation.
if [[ "${TPR_QWEN17_WEAK_E2E_SAMPLE:-1}" == "1" ]]; then
  export TPR_QWEN17_WEAK_E2E_SIGNATURE_DIR="$log_dir"
else
  unset TPR_QWEN17_WEAK_E2E_SIGNATURE_DIR
fi
if [[ "${TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE:-0}" == "1" ]]; then
  export TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE_DIR="$log_dir"
  echo "P0 WEAK_TQ WARNING: token capture synchronizes Segment logprobs; timing is NOT a benchmark"
else
  unset TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE_DIR
fi
test_path="tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_weak_e2e_npu.py"
if [[ ! -f "$test_path" ]]; then
  echo "Run from the bridge/VERL root where $test_path exists" >&2
  exit 2
fi
echo "P0 WEAK_TQ START checkpoint=$TPR_QWEN_1_7B_PATH"
echo "P0 WEAK_TQ START tq=$TPR_REAL_TQ_BATCH"
echo "P0 WEAK_TQ START master_device=$TPR_QWEN17_WEAK_E2E_MASTER_DEVICE"
echo "P0 WEAK_TQ START log_dir=$log_dir"

# Select "tpr" alone after the Native gate has already passed.
# Default remains both modes for same-run numerical comparisons.
read -r -a modes <<< "${TPR_QWEN17_WEAK_E2E_MODES:-native tpr}"
if [[ "${#modes[@]}" -eq 0 ]]; then
  echo "Set TPR_QWEN17_WEAK_E2E_MODES to native, tpr or 'native tpr'" >&2
  exit 2
fi
run_native=0
run_tpr=0
for mode in "${modes[@]}"; do
  case "$mode" in
    native)
      if [[ "$run_native" == 1 ]]; then echo "Duplicate mode: native" >&2; exit 2; fi
      run_native=1 ;;
    tpr)
      if [[ "$run_tpr" == 1 ]]; then echo "Duplicate mode: tpr" >&2; exit 2; fi
      run_tpr=1 ;;
    *)
      echo "Unsupported mode: $mode (valid: native, tpr)" >&2
      exit 2 ;;
  esac
done

for mode in "${modes[@]}"; do
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

if [[ "$run_native" != 1 || "$run_tpr" != 1 ]]; then
  echo "P0 WEAK_TQ single-mode result: ${modes[*]}; skipping cross-run comparison"
  exit 0
fi

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
import os
if os.environ.get("TPR_QWEN17_WEAK_E2E_SAMPLE", "1") != "1":
    print("  sampled_grad_update=SKIPPED (optimizer step still executed)")
    sys.exit(0)
import torch
native_sample = torch.load(
    sys.argv[1].replace("native.log", "native_optimizer_sample.pt"),
    map_location="cpu", weights_only=True,
)
tpr_sample = torch.load(
    sys.argv[2].replace("tpr.log", "tpr_optimizer_sample.pt"),
    map_location="cpu", weights_only=True,
)
if set(native_sample) != set(tpr_sample):
    raise SystemExit("Native/TPR sampled trainable parameter names disagree")
for metric_name in ("grad", "update"):
    delta2 = ref2 = cand2 = dot = 0.0
    sign_flips = nonzero = entries = 0
    for name in native_sample:
        a = native_sample[name][metric_name].float()
        b = tpr_sample[name][metric_name].float()
        if not torch.equal(
            native_sample[name]["indices"], tpr_sample[name]["indices"]
        ):
            raise SystemExit(f"{name}: sampled indices do not align")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise SystemExit(f"{name}: sampled {metric_name} nonfinite")
        delta2 += float(((a - b) ** 2).sum())
        ref2 += float((a ** 2).sum())
        cand2 += float((b ** 2).sum())
        dot += float((a * b).sum())
        common = (a != 0) & (b != 0)
        sign_flips += int(((a * b < 0) & common).sum())
        nonzero += int(common.sum())
        entries += a.numel()
    print(
        f"  sampled_{metric_name}_rel_l2={(delta2/max(ref2,1e-24))**0.5:.9g}, "
        f"cosine={dot/max((ref2*cand2)**0.5,1e-24):.9g}, "
        f"sign_flips={sign_flips}/{nonzero}, "
        f"sampled_entries={entries} trainable_tensors={len(native_sample)}"
    )
print("  all_parameter_update_parity=UNVERIFIED true_rollout_old_logprobs=UNVERIFIED")
PY

if [[ "${TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE:-0}" == "1" ]]; then
  python tests/models/mcore/tpr/correctness/_qwen17_weak_e2e_token_capture.py \
    "$log_dir/native_ppo_tokens.pt" "$log_dir/tpr_ppo_tokens.pt"
fi
