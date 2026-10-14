# TP2 / RingCP2 / DP2: one recorded PPO source step (TPR vs Full TP2/CP4)

Status: code added as an **isolated benchmark**, not yet executed on an eight-NPU node.
Do not confuse this with production VERL Trainer DP2 support or a native multi-step PPO run.

## Question

Compare a single previously captured, original eight-Session Qwen3-4B-128K
Actor PPO update using:

| | Existing Full | New TPR |
|---|---|---|
| Total NPUs | 8 | 8 |
| TP | 2 | 2 |
| CP | 4 (native MindSpeed Ring) | 2 (existing TPR Ring) |
| DP | 1 | 2 |
| Sample source | Existing session-first 40K manifest | Exactly the same source step and 8 rows |
| PPO | Original captured advantages, old logprobs, masks, temperature | Identical |
| Model | Identical real Qwen3-4B-128K checkpoint | Identical |
| Loss | Global token-mean vanilla PPO | Global token-mean vanilla PPO |
| Timing | F/B + gradient finalize | F/B + CP + DP gradient finalize |
| Optimizer/recompute | No updates / disabled | No updates / disabled |

The new topology intentionally changes both CP and DP: a speedup cannot be
ascribed to DP placement alone without additional controls.

## DP scheduling and normalization

The placement code is reused, unchanged, from PR #5
(feat/tpr-tp2-dp-placement-audit):
verl/models/mcore/tpr/dp_placement.py::plan_tpr_dta_dp.

1. Read all eight original trajectories and original keys once per rank.
2. Use TPR UID-scoped DFS minimax for two partitions. The result is a complete,
   disjoint assignment. The planner is allowed to choose unequal cardinalities
   (for example 5/3), unlike VERL's current ordinary equal-row Trainer dispatch.
3. Each TP2/CP2 group independently builds and executes an ordinary TPR forest
   on its own trajectories. No Prefix KV crosses DP replicas.
4. Keep metadata batch_num_tokens equal to **all eight trajectories' original
   supervised tokens**, dp_size=2 and global_batch_size=8 on both replicas.
   VERL's original token-mean PPO loss scales each local contribution by DP=2.
5. In the isolated benchmark only, finalize each parameter with a single
   Megatron DP-with-context-parallel SUM/4 over four weight replicas (the
   mathematical equivalent of CP SUM/2 then DP SUM/2, allowing BF16 reduction
   order differences). HCCL CP/DP/DP-with-CP communicators are prewarmed
   before model loading. Native Megatron TP collectives remain untouched.
   No per-segment DP collective is introduced.
6. After timing, reconstruct global PPO loss by averaging the two DP replicas
   and verify all eight ranks agree. Rank/row ownership, global token count,
   input digest and actual logical tree-token metrics are recorded.

The only Engine code change is the opt-in
TPR_EXPERIMENTAL_OFFLINE_DP2=1 and TPR_RUN_TP2CP2DP2_ONE_STEP=1 gates
for PPO TP2/CP2/DP2 and matching dp_size
metadata. DP1 production behavior and unsupported topologies are unchanged.

## Run

Sync the feature branch's relevant files into the actual Docker VERL checkout
using the same Bridge-sync method as the prior TP2CP4 experiments.

A comparable Full JSON must already exist from
run_tpr_fixed_shape_40k_benchmark.sh with the same 40K manifest/step.
The original full baseline outputs are named step6_full_fixed_shape.json,
step16_full_fixed_shape.json, etc.

Example:

    bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh \
      /workspace/40k_manifest.json \
      /workspace/tpr_dp2_step6_new \
      6 \
      /workspace/existing_full_fixed_shape/step6_full_fixed_shape.json

Defaults: physical NPU 8-15, Qwen3-4B-128K checkpoint, source step 6,
one warmup and three measured F/B passes, profiler off, captured logprobs on,
and TPR_PPO_LOGPROB_CHUNK_TOKENS=512 (experimental PPO logprob activation
checkpointing; set 0 to disable). This checkpoint only rematerializes
selected-logprob/softmax intermediate tensors on backward; it is NOT
Megatron per-layer recompute or MindSpeed swap-attention. Both modes use
the same original PPO labels/token normalization, but the chunk shape and
extra rematerialization may change BF16 numerics and runtime. Compare the
PPO loss gap before treating the speedup as a viable configuration.
TPR_BENCH_MEM_TRACE=1 enables a per-segment forward memory trace only for
debugging; leave off for timing.
Those timing defaults match the existing fixed-shape Full benchmark. They are
repeated execution of ONE source step without any weight update, not four
optimizer steps. To execute literally one forward/backward invocation, set
TPR_BENCH_WARMUPS=0 and TPR_BENCH_REPEATS=1; in that case the Full baseline
must have the same settings for strict comparison.

Without a fourth argument, the TPR result JSON is saved, and the baseline
comparison can be invoked separately:

    python tests/models/mcore/tpr/parallel/compare_tpr_tp2cp2dp2_one_step.py \
      FULL_STEP_JSON TPR_STEP_JSON COMPARISON_OUTPUT_JSON

Outputs:

- step6_tpr_tp2cp2dp2.json: per-rank DP rows, critical-path duration, peak
  allocated per NPU, global PPO loss, estimated/actual forest token costs.
- comparison.json: strict input SHA/checkpoint/step/timing settings check,
  median Full/TPR F/B+finalize speedup, peak-memory comparison, PPO loss gap.
- tpr_tp2cp2dp2_step6.log: torchrun/pytest output.

## Acceptance / interpretation

- CPU: PR #5 placement regressions and offline PPO parallel-size gate tests.
- Eight NPUs: must actually run before claiming that TP2CP2DP2 is supported.
- Checks: full/TPR input SHA equality, Session disjointness and completeness,
  all eight PPO-token sets, global loss agreement across TP/CP/DP ranks,
  finite timing and nonzero runtime fields.
- Report the global PPO loss gap to Full; **this benchmark does not compare
  complete Full vs TPR parameter-gradient tensors or AdamW updates**.
- Unequal DP partitions are deliberately experiment-only; do not infer that
  native VERL Trainer batch dispatch, no_sync/DDP, or distributed optimizer
  already supports unequal per-DP row counts.
- If CP2 OOMs for a particular 40K capture, report it as a capacity limitation
  of that configuration, not a speedup result. Native recompute remains
  unsupported by TPR's engine adapter. MindSpeed SwapAttention in the current
  TPR implementation requires TP=1 and cannot be enabled for TP2CP2DP2.
- The optional logprob checkpoint chunks do NOT reduce the base decoder
  activation footprint, KV cache, or full-model-output logits. If the
  backward peak remains above 60 GiB after chunking, this topology needs a
  deeper activation-memory change (e.g. validated TP-aware swap/recompute)
  or smaller single-trajectory length; changing capture length invalidates
  the original 40K fixed-source comparison.


## Isolated Step-6 30-Ki memory-capacity / DP correctness experiment

The TP2CP2DP2 40K Qwen3-4B experiment encountered a genuine
autograd-backward NPU OOM even after the optional PPO logprob chunks.
With CP2, per-card local sequence length is twice that of the Full
TP2CP4 baseline. 30 Ki tokens (30720) is a capacity hypothesis,
**not a guaranteed pass**.

Run the matched one-step experiment from the repo checkout:

    bash tests/models/mcore/tpr/parallel/run_tpr_dp2_vs_full_30k_one_step.sh \
      /workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json \
      /workspace/tpr_dp2_30k_step6_v1 \
      6

The script refuses to overwrite the root directory. It uses the same
physical NPU 8..15 and original Qwen3-4B-128K weights as the 40K tests.
It derives only Step 6: original 40K source capture is immutable, every
original prompt is retained, each response is cropped at the **right**
to make prompt+response <= 30720, and every retained PPO label/mask
keeps its original value. If the prompt alone exceeds 30720, or no
supervised PPO token would remain, preprocessing fails explicitly.

Derived manifest retains the original 20-step max_seq_len=40960 because
19 unchanged steps still reference the 40K source; it declares
experimental_sequence_cap=30720 and only_step_derived=6, and its
selected Step 6 item points at the new shortened capture and updated
PPO token count. Both Full and TPR gate this cap, and the strict
comparison checks exact derived source path, SHA, global supervised
tokens, warmups/repeats, and checkpoint. The **existing 40K Full JSON
must never be used as the baseline for this run**.

Outputs under ROOT:

- capture_manifest.json / step6_cap30720.pt / truncation_audit.json
- full/step6_full_fixed_shape.json (Full TP2CP4DP1)
- tpr/step6_tpr_tp2cp2dp2.json (TPR TP2CP2DP2)
- tpr/comparison.json: Full/TPR loss gap, median F/B+gradient-finalize
  timing, max-rank peak memory, DP partitions and duplicated tree tokens,
  and deterministic sampled gradient rel-L2 and cosine.
- CP/DP replica sampled weight gradients must be identical after finalize;
  this gate is run after timing. Sampled gradient comparisons are **not**
  full-parameter gradient equality. Neither job updates AdamW weights.

Use TPR_BENCH_WARMUPS=0 TPR_BENCH_REPEATS=1 for a quick functional
smoke test (new output directory); keep the default 1 warmup + 3 measured
iterations for timing. Both jobs inherit the same setting. If 30K still
OOMs, inspect per-rank DP placement and longest surviving tree before
changing sequence length again.
