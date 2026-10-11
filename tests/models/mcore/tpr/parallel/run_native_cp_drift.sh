#!/usr/bin/env bash
# P0-1: run native MindSpeed CP=1/2/4 in the old /workspace/uni-agent/verl
# environment after rsyncing bridge/tests/models/mcore/tpr/ into it.
# HOST ONLY: cd /workspace/bridge && git pull --ff-only.
# INSIDE DOCKER ONLY: rsync -av /workspace/bridge/tests/models/mcore/tpr/ \
#   /workspace/uni-agent/verl/tests/models/mcore/tpr/
# INSIDE DOCKER: run this launcher from /workspace/uni-agent/verl.
# NEVER git pull inside Docker / old UniAgent.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."

test_path="tests/models/mcore/tpr/parallel/test_native_cp1_cp2_cp4_drift_npu.py"
if [[ ! -f "$test_path" ]]; then
  echo "Expected VERL test root with $test_path; current directory: $PWD" >&2
  exit 2
fi

export TPR_P0_NATIVE_CP=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_P0_CP_OUTPUT_DIR="${TPR_P0_CP_OUTPUT_DIR:-/tmp/tpr_p0_native_cp}"
port_base="${TPR_P0_CP_MASTER_PORT_BASE:-29560}"
if [[ ! "$port_base" =~ ^[0-9]+$ ]]; then
  echo "TPR_P0_CP_MASTER_PORT_BASE must be a decimal integer" >&2
  exit 2
fi
port_base=$((10#$port_base))
if (( port_base < 1024 || port_base + 4 > 65535 )); then
  echo "TPR_P0_CP_MASTER_PORT_BASE must be in [1024, 65531]" >&2
  exit 2
fi

compare_script="${TPR_P0_CP_COMPARE_SCRIPT:-/workspace/bridge/tools/compare_native_cp_drift.py}"
if [[ ! -f "$compare_script" ]]; then
  echo "Comparator not found: $compare_script" >&2
  echo "Your rsync copies tests only; set TPR_P0_CP_COMPARE_SCRIPT to a visible copy." >&2
  exit 2
fi

mkdir -p "$TPR_P0_CP_OUTPUT_DIR"
if [[ "${TPR_P0_CP_COMPARE_ONLY:-0}" == "1" ]]; then
  # Preserve already successful NPU evidence after a comparator-only failure.
  echo "P0-1 comparison only: reusing native_cp1.pt/native_cp2.pt/native_cp4.pt" >&2
  python "$compare_script" "$TPR_P0_CP_OUTPUT_DIR"
  exit 0
fi
for cp in 1 2 4; do
  port=$((port_base + cp))
  # Prevent comparison against stale outputs from a previous incomplete run.
  rm -f "$TPR_P0_CP_OUTPUT_DIR/native_cp${cp}.pt"
  echo "P0-1 native CP=$cp: 127.0.0.1:$port ($PWD)" >&2
  torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node="$cp" \
    -m pytest -x -q -s "$test_path"
done

python "$compare_script" "$TPR_P0_CP_OUTPUT_DIR"
