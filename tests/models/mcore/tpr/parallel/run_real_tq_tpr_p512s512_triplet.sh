#!/usr/bin/env bash
# P0-3 true native CP1/CP2 vs TPR (P512/S512) real TQ row0 single-step.
# HOST ONLY: cd /workspace/bridge && git pull --ff-only
# DOCKER: rsync -av /workspace/bridge/tests/models/mcore/tpr/ \
#          /workspace/uni-agent/verl/tests/models/mcore/tpr/
#         cd /workspace/uni-agent/verl
#         bash tests/models/mcore/tpr/parallel/run_real_tq_tpr_p512s512_triplet.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test="tests/models/mcore/tpr/parallel/test_real_tq_tpr_p512s512_triplet_npu.py"
compare="tests/models/mcore/tpr/parallel/compare_real_tq_tpr_p512s512.py"
if [[ ! -f "$test" || ! -f "$compare" ]]; then
  echo "Missing P0-3 test/comparator. Docker must rsync bridge tests first." >&2
  exit 2
fi
export TPR_P0_TPR_P512S512=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_3_OUTPUT_DIR="${TPR_P0_3_OUTPUT_DIR:-/tmp/tpr_p0_tpr_p512s512}"
native_dir="${TPR_P0_3_NATIVE_DIR:-/tmp/tpr_p0_native_cp}"
port="${TPR_P0_3_MASTER_PORT:-29801}"

[[ -f "$TPR_REAL_TQ_BATCH" ]] || {
  echo "Real TQ batch missing: $TPR_REAL_TQ_BATCH" >&2; exit 2; }
for cp in 1 2; do
  [[ -f "$native_dir/native_cp${cp}.pt" ]] || {
    echo "Prior native P0-1 CP$cp evidence not found at $native_dir. Do not "
    echo "compare unrelated records: rerun P0-1 with the exact same TQ row0."
    exit 2
  }
done
if [[ "${TPR_P0_3_COMPARE_ONLY:-0}" == "1" ]]; then
  python "$compare" --native-dir "$native_dir" --tpr-dir "$TPR_P0_3_OUTPUT_DIR"
  exit 0
fi
if [[ ! "$port" =~ ^[0-9]+$ ]]; then
  echo "TPR_P0_3_MASTER_PORT must be decimal" >&2; exit 2
fi
port=$((10#$port))
if ((port < 1024 || port > 65535)); then
  echo "TPR_P0_3_MASTER_PORT must be 1024..65535" >&2; exit 2
fi
echo "P0-3 real P512/S512 native CP1/CP2 vs TPR (TPR CP=1) localhost:127.0.0.1:$port" >&2
torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
  -m pytest -x -q -s "$test"
python "$compare" --native-dir "$native_dir" --tpr-dir "$TPR_P0_3_OUTPUT_DIR"
