#!/usr/bin/env bash
# Host only: cd /workspace/bridge && git pull --ff-only.
# Inside Docker:
#   rsync -av /workspace/bridge/tests/models/mcore/tpr/ \
#     /workspace/uni-agent/verl/tests/models/mcore/tpr/
#   cd /workspace/uni-agent/verl
#   bash tests/models/mcore/tpr/parallel/run_native_cp_multistep.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test_path="tests/models/mcore/tpr/parallel/test_native_cp_multistep_npu.py"
compare_path="tests/models/mcore/tpr/parallel/compare_native_cp_multistep.py"
if [[ ! -f "$test_path" || ! -f "$compare_path" ]]; then
  echo "Run from the Docker VERL test checkout after rsync from host bridge" >&2
  exit 2
fi
export TPR_P0_CP_MULTISTEP=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_B_OUTPUT_DIR="${TPR_P0_B_OUTPUT_DIR:-/tmp/tpr_p0_native_cp_b}"
export TPR_P0_B_STEPS="${TPR_P0_B_STEPS:-50}"
export TPR_P0_B_GRAD_REDUCE="${TPR_P0_B_GRAD_REDUCE:-fp32}"
base="${TPR_P0_B_MASTER_PORT_BASE:-29660}"
if [[ ! "$base" =~ ^[0-9]+$ ]]; then
  echo "TPR_P0_B_MASTER_PORT_BASE must be an integer" >&2
  exit 2
fi
base=$((10#$base))
if (( base < 1024 || base + 4 > 65535 )); then
  echo "TPR_P0_B_MASTER_PORT_BASE must be in [1024,65531]" >&2
  exit 2
fi
if [[ ! -f "$TPR_REAL_TQ_BATCH" ]]; then
  echo "Missing real TQ data: $TPR_REAL_TQ_BATCH" >&2
  exit 2
fi
mkdir -p "$TPR_P0_B_OUTPUT_DIR"
if [[ "${TPR_P0_B_COMPARE_ONLY:-0}" == "1" ]]; then
  python "$compare_path" "$TPR_P0_B_OUTPUT_DIR" --steps "$TPR_P0_B_STEPS"
  exit 0
fi
for cp in 1 2 4; do
  port=$((base+cp))
  echo "P0_B launch CP=$cp localhost:127.0.0.1:$port" >&2
  torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node="$cp" \
    -m pytest -x -q -s "$test_path"
done
python "$compare_path" "$TPR_P0_B_OUTPUT_DIR" --steps "$TPR_P0_B_STEPS"
