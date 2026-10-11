#!/usr/bin/env bash
# P0-2B official MindSpeed full vs CP2/CP4 Ring using frozen real Qwen Q/K/V/dO.
# HOST: cd /workspace/bridge && git pull --ff-only
# DOCKER: rsync -av /workspace/bridge/tests/models/mcore/tpr/ \
#         /workspace/uni-agent/verl/tests/models/mcore/tpr/
#         cd /workspace/uni-agent/verl
#         bash tests/models/mcore/tpr/parallel/run_native_attention_qkv_replay.sh
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../../../.."
test_path="tests/models/mcore/tpr/parallel/test_native_attention_qkv_replay_npu.py"
[[ -f "$test_path" ]] || { echo "Missing synced P0-2B test: $test_path" >&2; exit 2; }
export TPR_P0_NATIVE_ATTN_REPLAY=1
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_P0_2B_OUTPUT_DIR="${TPR_P0_2B_OUTPUT_DIR:-/tmp/tpr_p0_attn_replay}"
base="${TPR_P0_2B_MASTER_PORT_BASE:-29770}"
if [[ ! "$base" =~ ^[0-9]+$ ]]; then
  echo "TPR_P0_2B_MASTER_PORT_BASE must be decimal" >&2
  exit 2
fi
base=$((10#$base))
if ((base < 1024 || base + 4 > 65535)); then
  echo "TPR_P0_2B_MASTER_PORT_BASE must be 1024..65531" >&2
  exit 2
fi
if [[ ! -f "$TPR_REAL_TQ_BATCH" ]]; then
  echo "Missing real TQ dump needed for CP1 QKV capture: $TPR_REAL_TQ_BATCH" >&2
  exit 2
fi
mkdir -p "$TPR_P0_2B_OUTPUT_DIR"
for cp in 1 2 4; do
  port=$((base + cp))
  if [[ "$cp" == "1" && "${TPR_P0_2B_REUSE_CP1:-0}" == "1" ]]; then
    [[ -f "$TPR_P0_2B_OUTPUT_DIR/native_cp1_attention_oracle.pt" ]] || {
      echo "CP1 oracle missing: cannot reuse" >&2; exit 2; }
    echo "P0-2B reusing CP1 real-QKV oracle (ensure layers/checkpoint match)" >&2
    continue
  fi
  echo "P0-2B CP=$cp native attention localhost:127.0.0.1:$port" >&2
  torchrun --master_addr=127.0.0.1 --master_port="$port" --nproc_per_node="$cp" \
    -m pytest -x -q -s "$test_path"
done
