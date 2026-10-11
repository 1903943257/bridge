#!/usr/bin/env bash
# P0-5 actual real SWE trajectory siblings with common 512-token prompt tail.
# Host: cd /workspace/bridge && git pull --ff-only
# Docker: rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
#         cd /workspace/uni-agent/verl
#         bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test_file="tests/models/mcore/tpr/parallel/test_real_tq_multisibling_prefix_npu.py"
compare="tests/models/mcore/tpr/parallel/compare_real_tq_multisibling_prefix.py"
if [[ ! -f "$test_file" || ! -f "$compare" ]]; then
  echo "Sync bridge tests to Docker VERL before running P0-5" >&2; exit 2
fi
export TPR_P0_REAL_SIBLINGS=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_BRANCH_OUTPUT_DIR="${TPR_P0_BRANCH_OUTPUT_DIR:-/tmp/tpr_p0_real_siblings}"
export TPR_P0_BRANCH_OBJECTIVE="${TPR_P0_BRANCH_OBJECTIVE:-nll}"
# Actual PPO fields are produced only at actor-update time, not by the
# original TQ rollout dump. Set TPR_REAL_ACTOR_MINIBATCH to use that capture.
if [[ "$TPR_P0_BRANCH_OBJECTIVE" == "ppo" &&
      -n "${TPR_REAL_ACTOR_MINIBATCH:-}" ]]; then
  export TPR_REAL_TQ_BATCH="$TPR_REAL_ACTOR_MINIBATCH"
fi
export TPR_P0_BRANCH_SIBLINGS="${TPR_P0_BRANCH_SIBLINGS:-2}"
export TPR_P0_BRANCH_STEPS="${TPR_P0_BRANCH_STEPS:-1}"
base="${TPR_P0_BRANCH_MASTER_PORT_BASE:-29840}"
if [[ ! "$base" =~ ^[0-9]+$ ]]; then
  echo "Port base must be decimal" >&2; exit 2
fi
base=$((10#$base))
if ((base < 1024 || base + 2 > 65535)); then
  echo "Port base must be in [1024,65533]" >&2; exit 2
fi
[[ -f "$TPR_REAL_TQ_BATCH" ]] || { echo "Real TQ missing: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
[[ -d "$TPR_QWEN_1_7B_PATH" ]] || { echo "Checkpoint missing: $TPR_QWEN_1_7B_PATH" >&2; exit 2; }
mkdir -p "$TPR_P0_BRANCH_OUTPUT_DIR"
if [[ "${TPR_P0_BRANCH_COMPARE_ONLY:-0}" != "1" ]]; then
  if [[ "$TPR_P0_BRANCH_OBJECTIVE" == "ppo" ]]; then
    # Reject missing actor fields BEFORE two expensive Qwen NPU loads.
    # For a rollout-only dump this is a DATA GATE, not a TPR test failure.
    python tests/models/mcore/tpr/parallel/inspect_real_tq_siblings.py \
      "$TPR_REAL_TQ_BATCH" --siblings "$TPR_P0_BRANCH_SIBLINGS" --require-ppo
  fi
  case "${TPR_P0_BRANCH_RUN_MODES:-both}" in
    both) modes=(full tpr) ;;
    full) modes=(full) ;;
    tpr) modes=(tpr) ;;
    *) echo "TPR_P0_BRANCH_RUN_MODES must be both,full,tpr" >&2; exit 2 ;;
  esac
  for mode in "${modes[@]}"; do
    export TPR_P0_BRANCH_MODE="$mode"
    if [[ "$mode" == "full" ]]; then port=$((base+1)); else port=$((base+2)); fi
    echo "P0-5 $mode true siblings=$TPR_P0_BRANCH_SIBLINGS objective=$TPR_P0_BRANCH_OBJECTIVE steps=$TPR_P0_BRANCH_STEPS localhost:$port" >&2
    torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
      -m pytest -x -q -s "$test_file"
  done
fi
left="$TPR_P0_BRANCH_OUTPUT_DIR/${TPR_P0_BRANCH_OBJECTIVE}_full_siblings${TPR_P0_BRANCH_SIBLINGS}_steps${TPR_P0_BRANCH_STEPS}.pt"
right="$TPR_P0_BRANCH_OUTPUT_DIR/${TPR_P0_BRANCH_OBJECTIVE}_tpr_siblings${TPR_P0_BRANCH_SIBLINGS}_steps${TPR_P0_BRANCH_STEPS}.pt"
if [[ -f "$left" && -f "$right" ]]; then
  python "$compare" "$TPR_P0_BRANCH_OUTPUT_DIR" --steps "$TPR_P0_BRANCH_STEPS" \
    --siblings "$TPR_P0_BRANCH_SIBLINGS" --objective "$TPR_P0_BRANCH_OBJECTIVE"
else
  if [[ "${TPR_P0_BRANCH_COMPARE_ONLY:-0}" == "1" ]]; then
    echo "Missing saved Full/TPR artifacts for comparison" >&2; exit 2
  fi
  echo "Only one mode completed, run the other mode later (does not overwrite first)" >&2
fi
