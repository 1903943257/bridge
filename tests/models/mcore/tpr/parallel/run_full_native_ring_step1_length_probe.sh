#!/usr/bin/env bash
# One independent 8-NPU process per selected Step-1 trajectory.
# Isolates single-trajectory full Native MindSpeed Ring memory ceilings,
# never changes the user's 20-step v2 replay snapshots.
set -euo pipefail
cd "$(dirname "$0")/../../../../.."

SOURCE="${1:-/workspace/tpr_offline_run5_full8_v2/session_first_batches/actor_update_step000001_session_first8.pt}"
OUT="${2:-/workspace/tpr_full_ring_length_probe_step1_$(date +%Y%m%d_%H%M%S)}"
[[ -f "$SOURCE" ]] || { echo "ERROR: merged Step-1 capture not found: $SOURCE" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "ERROR: probe destination exists (do not overwrite): $OUT" >&2; exit 2; }
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_REAL_ACTOR_MINIBATCH="$SOURCE"
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_OFFLINE_RUN_MODES=full
export TPR_OFFLINE_FULL_RING_BACKEND=native
export TPR_OFFLINE_SGD_LR=0
export TPR_OFFLINE_SAVE_FULL_GRADS=0
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128

python - <<'PY'
import inspect
from verl.models.mcore.tpr.parallel.ring_attention import ordinary_ring_cp_attention
assert "allow_trailing_padding" in inspect.signature(ordinary_ring_cp_attention).parameters
print("NATIVE_RING_SIGNATURE=PASS", flush=True)
PY

python - "$SOURCE" "$OUT" <<'PY'
import json,sys,torch
src,out=sys.argv[1:]
d=torch.load(src,map_location="cpu",weights_only=False)
data=d["tensordict"]
report=[]
for row in range(len(d["keys"])):
    report.append({
        "row":row,
        "source_key":d["keys"][row],
        "input_tokens":int(data["input_ids"][row].numel()),
        "ppo_tokens":int(data["response_mask"][row].bool().sum().item()),
        "source":d.get("source_row_provenance",[{}]*len(d["keys"]))[row],
    })
if len(report)!=8:
    raise SystemExit("Step-1 must contain exactly 8 selected Sessions")
with open(out+"/row_inventory.json","w") as f:
    json.dump(report,f,indent=2)
for x in report:
    print(f"PROBE_ROW row={x['row']} input_tokens={x['input_tokens']} ppo_tokens={x['ppo_tokens']}",flush=True)
PY

# Ordering: known shorter baseline; ~44K; two distinct 49,152-token
# Sessions; remaining controls. Every test restarts all 8 processes to
# guarantee fresh allocator state and no previous trajectory accumulation.
ROWS=(0 1 4 7 2 3 5 6)
echo -e "row\\tlength\\tstatus\\tpeak_alloc_gib\\tfwd_bwd_s" > "$OUT/summary.tsv"
failed=0
for row in "${ROWS[@]}"; do
  length="$(python - "$OUT/row_inventory.json" "$row" <<'PY'
import json,sys
xs=json.load(open(sys.argv[1]))
print(xs[int(sys.argv[2])]["input_tokens"])
PY
)"
  export TPR_OFFLINE_ROWS="$row"
  export TPR_OFFLINE_OUTPUT_DIR="$OUT/row${row}_len${length}"
  echo "FULL_NATIVE_RING_LENGTH_PROBE_START row=$row length=$length"
  status=PASS
  if bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
       > "$OUT/row${row}.log" 2>&1; then
    echo "FULL_NATIVE_RING_LENGTH_PROBE_PASS row=$row length=$length"
  else
    status=FAIL
    failed=1
    echo "FULL_NATIVE_RING_LENGTH_PROBE_FAIL row=$row length=$length log=$OUT/row${row}.log" >&2
    grep -E 'out of memory|RuntimeError|Error|peak_alloc_gib' "$OUT/row${row}.log" | tail -n 12 || true
  fi
  read -r peak elapsed < <(python - "$OUT/row${row}.log" <<'PY'
import re,sys
s=open(sys.argv[1],errors="replace").read()
peaks=re.findall(r"peak_alloc_gib=([0-9.]+)",s)
times=re.findall(r"fwd_bwd_sec=([0-9.]+)",s)
print((max(map(float,peaks)) if peaks else "NA"),
      (max(map(float,times)) if times else "NA"))
PY
)
  printf '%s\\t%s\\t%s\\t%s\\t%s\\n' "$row" "$length" "$status" "$peak" "$elapsed" >> "$OUT/summary.tsv"
done
cat "$OUT/summary.tsv"
echo "FULL_NATIVE_RING_PROBE_OUTPUT=$OUT"
exit "$failed"
