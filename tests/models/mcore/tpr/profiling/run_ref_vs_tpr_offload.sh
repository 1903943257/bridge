#!/usr/bin/env bash
# Ref vs TPR profile after Phase-B activation offload acceptance.
#
# Both paths use the same native MindSpeed activation offload and the same
# chunked LM head. N=2 matches the Phase-B tree: one shared Prefix, two Suffixes.
#
# Matrix:
#   CP2: P32K/S32K, P40K/S40K
#   CP4: P64K/S64K, P96K/S64K

set -u
set -o pipefail

TEST="tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py"
SUMMARY="tests/models/mcore/tpr/profiling/summarize_ref_vs_tpr_offload.py"
LOGDIR="${TPR_REF_TPR_LOGDIR:-tests/models/mcore/tpr/logs/ref_vs_tpr_offload}"
mkdir -p "$LOGDIR"

export TPR_RUN_QWEN_RING_CP_PROFILE=1
export TPR_QWEN_PROFILE_SIZE="${TPR_QWEN_PROFILE_SIZE:-1.7B}"
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"

export TPR_QWEN_RING_CP_PROFILE_OFFLOAD=1
export TPR_SWAP_MODULES="${TPR_SWAP_MODULES:-self_attention,mlp}"
export TPR_LOSS_CHUNK_SIZE="${TPR_LOSS_CHUNK_SIZE:-1024}"
export TPR_RING_COALESCE_PREFIX_FULL=1
export TPR_RING_COALESCE_PREFIX_QUERY=1

# Long-context formal samples: one warmup plus three measured samples.
export TPR_QWEN_RING_CP_PROFILE_WARMUP="${TPR_QWEN_RING_CP_PROFILE_WARMUP:-1}"
export TPR_QWEN_RING_CP_PROFILE_REPEATS="${TPR_QWEN_RING_CP_PROFILE_REPEATS:-3}"
export TPR_QWEN_RING_CP_PROFILE_BREAKDOWN=0

# Never carry correctness-only diagnostics into performance measurements.
unset TPR_OFFLOAD_B_FORCE_RESTORE_SYNC
unset TPR_OFFLOAD_B_OFF_STRESS
unset TPR_OFFLOAD_B_TIMING
unset ASCEND_LAUNCH_BLOCKING

port="${TPR_REF_TPR_BASE_PORT:-30020}"
timeout_value="${TPR_REF_TPR_TIMEOUT:-90m}"

cases=(
  "2 32768 32768 2"
  "2 40960 40960 2"
  "4 65536 65536 2"
  "4 98304 65536 2"
)

for spec in "${cases[@]}"; do
  read -r cp p s n <<< "$spec"
  log="$LOGDIR/cp${cp}_p${p}_s${s}_n${n}.log"

  echo "===== Ref vs TPR: CP=$cp P=$p S=$s N=$n ====="
  echo "log: $log"

  TPR_QWEN_RING_CP_PROFILE_CASES="${p}:${s}:${n}"   timeout "$timeout_value"     torchrun       --nproc_per_node="$cp"       --master_addr=127.0.0.1       --master_port="$port"       -m pytest -x -s -v       "$TEST::test_qwen3_reference_cp_vs_tpr_ring_cp_profile"       > "$log" 2>&1

  status=$?
  echo "END CP=$cp P=$p S=$s N=$n exit=$status"
  grep -E 'TPR_REF_TPR_(STAGE|RESULT)|Speedup \(median\)|Incremental-peak reduction|FAILED|PASSED'     "$log" | tail -n 20 || true
  port=$((port + 1))
done

echo
echo "===== Summary ====="
python "$SUMMARY" "$LOGDIR"   --csv "$LOGDIR/summary.csv"
