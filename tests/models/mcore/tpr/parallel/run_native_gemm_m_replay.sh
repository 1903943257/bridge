#!/usr/bin/env bash
# P0-2A real X/W/dY replay, one NPU, no production Megatron code edits.
# HOST: cd /workspace/bridge && git pull --ff-only
# DOCKER: rsync -av /workspace/bridge/tests/models/mcore/tpr/ \
#         /workspace/uni-agent/verl/tests/models/mcore/tpr/
#         cd /workspace/uni-agent/verl
#         bash tests/models/mcore/tpr/parallel/run_native_gemm_m_replay.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test_path="tests/models/mcore/tpr/parallel/test_native_gemm_m_replay_npu.py"
[[ -f "$test_path" ]] || { echo "Run inside the old UniAgent VERL checkout after rsync" >&2; exit 2; }

export TPR_P0_GEMM_REPLAY=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_2A_OUTPUT_DIR="${TPR_P0_2A_OUTPUT_DIR:-/tmp/tpr_p0_gemm_replay}"
port="${TPR_P0_2A_MASTER_PORT:-29761}"
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( 10#$port < 1024 || 10#$port > 65535 )); then
  echo "TPR_P0_2A_MASTER_PORT must be an integer 1024..65535" >&2
  exit 2
fi
if [[ ! -f "$TPR_REAL_TQ_BATCH" ]]; then
  echo "Real TQ data not found: $TPR_REAL_TQ_BATCH" >&2
  exit 2
fi
echo "P0-2A native real-X/W/dY GEMM replay on 127.0.0.1:$port" >&2
torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node=1 \
  -m pytest -x -q -s "$test_path"
