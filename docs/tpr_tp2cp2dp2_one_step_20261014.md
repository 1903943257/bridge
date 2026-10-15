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



## 30K DP-local loss contribution vs global reporting (2026-10-14)

The 30K test reached the post-backward loss gate. Its old diagnostic
incorrectly asserted that the DP0 and DP1 contributions must be equal,
despite the DTA-DP placement assigning different trajectories to each
replica. In the observed run the two values were:

- DP0: 0.006869133561849594 on ranks 0,1,2,3
- DP1: 0.1446971446275711 on ranks 4,5,6,7

Their 0.1378280110657215 difference is **not** TP BF16 rounding;
it is grouped by DP replica. A global token-mean objective is additive
over the two disjoint DP partitions; unequal local contributions are
permitted. The earlier TP-drift explanation was incorrect.

The experiment now does:
1. Before model loading, ALL ranks exchange TP/CP/DP coordinates and
   validate Megatron's TP/CP/DP/DP-with-CP process-group memberships.
   Rank-coded HCCL probes also validate the collective SUM results.
   Group-size-only checks were insufficient.
2. The engine computes and CP-SUMs its *DP-local* PPO loss, with the
   original global-token denominator and VERL dp_size=2 scaling.
3. To **report only**, the benchmark world-SUMs the eight rank-local
   values and divides by eight, which is equivalent to averaging the
   two distinct DP contributions when their TP/CP views agree.
4. It checks CP agreement *within each (TP,DP)*, reports any TP shard
   drift separately, and requires all eight final reported global
   scalars to agree and match the reconstructed raw average.
   DP0 and DP1 raw loss contributions are NEVER required to match.
5. Gradients remain finalized by Megatron's DP-with-CP SUM/4
   and the sampled post-timing replicated-gradient gate stays strict.

The benchmark records `loss_report.dp0_local_loss`,
`dp1_local_loss`, `cp_replica_max_gap`,
`tp_local_max_gap`, and `global_reporting_max_gap`.
The comparison exports the complete `tpr_loss_consistency`.
This is NOT proof of full parameter-gradient parity or native multi-step
DP2 training correctness.


## Root cause of grouped "global" loss on Ascend 910B (2026-10-14)

The later 30K run reported two so-called *world-reduced* values:

- ranks 0..3: 0.0017172833904623985
- ranks 4..7: 0.036174286156892776

A genuine eight-rank SUM followed by /8 cannot return different values
on these two sets of ranks. The old benchmark used
`dtype=torch.float64` for **all three** HCCL reductions:
global-loss SUM, critical F/B-time MAX and peak-NPU-memory MAX.

Ascend torch_npu's ProcessGroupHCCL.cpp supports FP64 HCCL reduction
only on Ascend950/Atlas A5; on Ascend910B it is unsupported.
See https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp
(`checkSupportedDataType`; AllGather's type handling differs).
On this installed stack the scalar FP64 reductions evidently failed to
produce a true cross-rank aggregate. The exact backend error behavior
may vary by CANN/torch_npu version. No assertion tolerance relaxation
can make invalid FP64 reductions valid.

Fix: FP32 for ALL HCCL AllReduce report scalars (SUM and MAX) in
both DP2 TPR and the Full fixed-shape benchmark; FP64 remains permitted
for HCCL AllGather of fixed-width diagnostic records (no reduction).
Before loading model weights, run a rank-coded FP32 WORLD AllReduce:
SUM(rank+1) over ranks 0..7 must equal 36, and fail otherwise.
The DP2 per-rank AllGather records independent FP64-valued local loss
scalars; the loss-report gate compares the FP32 world-SUM/8 against
the FP64-precision reference reconstructed from ALL eight gathered
local values. It still verifies CP replica agreement, while allowing
different DP-local contributions.

**Previously produced Full/TPR time and peak-memory JSONs must be
considered untrusted if they used FP64 HCCL reductions.**
Both new JSON schemas contain
`"hccl_stats_reduce_dtype": "float32"`; the comparison script refuses
legacy outputs without this marker. To establish a fair 30K speedup,
rerun BOTH Full TP2CP4 and TPR TP2CP2DP2 on the *same* derived 30K
capture, with a clean output directory. A quick TPR-only smoke
test may reuse the derived manifest, but it cannot be used as a
reliable speedup against a stale Full result.


## Step 6 actual 30K performance and gradient-replica diagnostic

The 8-NPU Qwen3-4B 30Ki replay measured:

- Full TP2/CP4/DP1 median 16.361464 s, worst-rank peak 34.638 GiB
- TPR TP2/CP2/DP2 three measured durations:
  12.322243 / 12.332244 / 12.327386 s, median 12.327386 s;
  worst-rank peak 41.071 GiB
- Performance (indicative, with gradient witness *after* timing):
  speedup 16.361464/12.327386 ~= 1.3273x.
  TPR uses ~6.433 GiB extra peak vs Full (~18.6%).
- PPO loss Full 0.151394129 vs TPR ~0.15156628
  (absolute difference ~0.00017215). These are source Step 6,
  identical 30720-token cap with different CP/DP topologies.
- DTA-DP partitions ((2,3,0,6),(5,1,4,7)), tree costs
  (59493,70125). Local PPO scalar contributions 0.0137382671
  vs 0.289394289, correctly different because DP shards see
  different trajectories. CP replicas match, TP-local gap 0;
  global report agrees across ranks within 1.49e-8.
- The final gradient witness failed:
  TP2CP2DP2 sampled max_abs 1.52587891e-05
  (around 0.31%~0.36% sampled relative L2 spread);
  Full CP4 reference itself also had a 3.05175781e-05
  sampled max_abs (~0.47%~0.51% relative L2 spread).
  The highlighted TPR parameter was
  `decoder.layers.1.input_layernorm.weight`.

Because these differences appear *after* BF16 gradient SUM/divide,
but both Full and TPR show them, do **not** conclude gradient parity
is satisfied, or that TPR alone is defective. The previous diagnostic
used additional FP32 HCCL MIN/MAX AllReduce and may not independently
establish what each rank originally held. The revised witness:
1. FP32 all-gathers *read-only* original post-finalize gradient samples
   from the four CP×DP replicas, reconstructing min/max on CPU.
2. Logs the exact per-rank values and count of differing sampled
   elements; for a small worst offending parameter (<=8192 elements)
   also gathers its entire original BF16 gradient from all replicas.
3. Keeps `TPR_BENCH_GRAD_REPLICA_POLICY=strict` by default. Setting
   `TPR_BENCH_GRAD_REPLICA_POLICY=diagnostic` is an explicit
   diagnostic-only override that writes performance and sampled-gradient
   parity JSON while marking `gradient_replica_validation_passed=false`
   and `result_is_diagnostic_only=true` if discrepancies persist.
   It does NOT change the original gradient or falsely mark correctness
   as passed.

The next decision is grounded by these actual per-rank BF16 values:
if the samples are exactly identical, the old HCCL MIN/MAX-based
checker was faulty; if they differ, trace the BF16 gradient finalization
output and compare FP32 reduction / synchronization as a separate
controlled experiment, without altering the measured TPR algorithm
before diagnosis.


### Fast HCCL BF16 isolation probe (no Qwen3 checkpoint)

Before another expensive 30K replay, run:

    TPR_RUN_HCCL_BF16_GRAD_PROBE=1 \
    ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    torchrun --nproc_per_node=8 --master_port=29917 -m pytest -q -s -x \
      tests/models/mcore/tpr/parallel/test_hccl_bf16_replica_probe_npu.py

This initializes the identical Megatron TP2/CP2/DP2 groups without
loading model weights. For BF16 tensors with 2560 and 1048576 elements,
it runs the original DP-with-CP SUM/4 and gathers the post-reduction
BF16 tensors to all four group members. Exact bitwise identity is
required. A PASS excludes a simple isolated BF16 collective defect,
but does not prove autograd's long-sequence parameter-gradient
buffer or asynchronous execution is correct. A FAIL identifies
a backend/collective identity issue worth investigating first.


## BF16 HCCL gradient-collective replica mismatch reproduced (2026-10-14)

The standalone 8-NPU probe, with no model or autograd, failed on
its first 2560-element BF16 tensor:

- TP0 CP×DP group (0,2,4,6) and TP1 group (1,3,5,7)
- 197 of 2560 BF16 elements differ across post-finalize copies
- maximum absolute replica spread 6.103515625e-05

This shows that the observed small BF16 gradient-replica disagreement
can be reproduced **independently of TPR**. It does not yet isolate
whether the discrepancy is present directly after HCCL BF16
AllReduce(SUM) or only after the local BF16 in-place division by 4.

The revised standalone probe prints five individually synchronized,
raw-AllGather-verified stages on each four-rank group:
`bf16_sum_before_div`, `bf16_after_div`,
`fp32_sum_before_div`, `fp32_after_div`,
`fp32_mean_cast_to_bf16`. It also independently
AllGathers original BF16 inputs to build a CPU FP32 reference SUM.
It REQUIRES FP32 reduction and FP32->BF16 cast to produce identical
replica copies and match the input SUM; BF16 discrepancies are
reported without aborting early. The strict historical failure
can be requested using `TPR_PROBE_REQUIRE_BF16_EXACT=1`.

Experimental model-run workaround (not production VERL):
`TPR_OFFLINE_GRAD_SYNC_DTYPE=fp32` promotes one accumulated BF16
**weight parameter** gradient at a time to FP32, AllReduces and divides
in FP32, and copies the result back to its original dtype. It applies
only to this isolated TP2CP2DP2 benchmark and its matching Full CP4
baseline; the original BF16 path remains the default. It does not
alter TPR KV/dKV, model forward/backward, DP placement or PPO labels.
Each model-run JSON records `gradient_sync_reduce_dtype` and the
comparison rejects mixed or unspecified gradient-sync dtypes.

Sequence of validation:

1. Run the standalone dtype-stage probe first (same 8 physical NPUs);
   confirm all FP32 stages are bitwise consistent and correct vs CPU
   input-reconstruction. If FP32 fails, DO NOT run the Qwen FP32
   gradient-sync workaround.
2. To check weight gradient correctness, run TPR-only 30K
   with `TPR_OFFLINE_GRAD_SYNC_DTYPE=fp32`, strict
   `TPR_BENCH_GRAD_REPLICA_POLICY=strict`, and a NEW output directory.
3. Only if successful, rerun matched Full+TPR 30K with FP32 mode on
   BOTH. FP32 gradient collectives cost more bandwidth and workspace,
   so old BF16 ~1.33x performance is not a valid FP32 speedup claim.
4. Only sampled replica equality is checked; complete Full-vs-TPR
   gradient parity and multi-step AdamW training are still unverified.

Standalone:
    TPR_RUN_HCCL_BF16_GRAD_PROBE=1 \
    ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
    torchrun --nproc_per_node=8 --master_port=29917 -m pytest -q -s -x \
      tests/models/mcore/tpr/parallel/test_hccl_bf16_replica_probe_npu.py

FP32 TPR-only strict:
    TPR_OFFLINE_GRAD_SYNC_DTYPE=fp32 TPR_BENCH_GRAD_WITNESS=1 \
    TPR_BENCH_WARMUPS=0 TPR_BENCH_REPEATS=1 \
    bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp2dp2_one_step.sh \
      /workspace/tpr_dp2_30k_smoke_v2/capture_manifest.json \
      /workspace/tpr_dp2_30k_fp32_grad_sync_strict_v1 6

Matched pair (after successful probe and smoke test):
    TPR_OFFLINE_GRAD_SYNC_DTYPE=fp32 \
    bash tests/models/mcore/tpr/parallel/run_tpr_dp2_vs_full_30k_one_step.sh \
      /workspace/tpr_offline_run5_first8_40k_v1/capture_manifest.json \
      /workspace/tpr_dp2_30k_fp32_grad_sync_pair_v1 6


## Matched deterministic-BF16 30K Full vs TPR: forward pass/perf passes; gradient parity fails

Source step 6, exact same 30,720-token-cap capture, global PPO tokens 36,933.
Topology Full TP2/CP4/DP1 vs TPR TP2/CP2/DP2, fixed initial
Qwen3-4B-128K weights, HCCL_DETERMINISTIC=true,
TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16, no optimizer step.

Full F/B+manual CP4 gradient finalize median = 16.454547882080078 s.
TPR F/B+manual CP×DP4 gradient finalize median = 12.401572227478027 s.
Matched speedup = 1.3268114381x. Worst-rank mean allocated peak:
Full 34.6382561 GiB; TPR 41.0705605 GiB (+18.57%).
Neither benchmark uses production Megatron DDP bucket overlap;
the measured speedup must not be claimed against the fully
optimized production Megatron DDP path.

PPO loss Full 0.15139412879943848; TPR 0.151566281914711;
absolute difference 0.00017215311527252197.
TPR local DP contributions differ as expected
(0.013738267123699188 vs 0.2893942892551422),
while both TP and CP replica reporting gaps are zero.

Both sides' *sampled* post-finalize gradients are exactly replicated
across CP×DP groups (sampled replica max_abs=0).
**This is NOT the same as Full-vs-TPR gradient parity**:
290 parameter names, eight samples per parameter per TP shard,
4,640 gradient samples in all. Sampled gradient rel-L2=0.5027317891,
cosine=0.8703406694, max_abs=0.00291442871;
sample L2 norms Full=0.01367750568, TPR=0.01328762393.
This large error fails any meaningful algorithmic gradient gate.
Neither parameter updates nor multi-step training were tested.

The current Megatron schedule/TPR scaling equations agree in
expectation: Full local token-mean loss (dp_size=1) receives
CP4 backward scaling and manual CP4 SUM/4; TPR uses local
dp_size=2 plus SegmentExecutor's CP2 backward scaling and
manual DP×CP SUM/4. Thus the large gradient gap is not explained
by an obvious missing 1/2 factor in the inspected code. A global
scalar correction cannot repair the observed cosine of 0.8703.

Next *without any NPU rerun*, inspect attribution of the existing
gradient witnesses (per-layer, attention/MLP/norm, TP ranks,
single best-fit global scalar and residual):

    python -m tests.models.mcore.tpr.parallel.analyze_tpr_dp2_gradient_parity \
      /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1/full/step6_full_fixed_shape.json \
      /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1/tpr/step6_tpr_tp2cp2dp2.json \
      --top 25 --output /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1/tpr/grad_breakdown.json

Report is *sampled* error attribution, NOT whole-model rel-L2.
Depending on contributors, the next targeted experiment should compare
per-token logprobs/clip decisions, and, if necessary, BF16
per-segment parameter-gradient accumulation vs Full's microbatch path.
Do NOT certify TPR DP2 training correctness based on small Loss gap
or matching replica gradients alone.


### Step 6 deterministic-BF16 sampled-gradient attribution

The 4,640-value (8 evenly spaced positions per parameter per TP)
Full/TPR gradient witness was decomposed by the postprocessing script.
Global sampled rel-L2=0.502731789, cosine=0.870340669;
best scalar TPR~=0.845531325*Full still leaves rel-L2=0.478412667.
Thus the drift is **not** a uniform DP/CP scale factor.

Of the *sampled squared error*, 95.78% belongs to norm parameters,
4.03% to other attention params, 0.17% to QKV, 0.02% to MLP.
Important: the script samples the same eight indices of EACH parameter
regardless of tensor size; these proportions are not actual
whole-model squared-error shares! Tiny LayerNorm vectors receive the
same eight samples as giant matrices. Some LayerNorm samples are also
exactly repeated on the two TP shards. Do not infer the location
of the whole model's error from this unweighted sparse witness.

High-impact examples, per TP: layer13 input_layernorm sampled
Full L2 0.00476552 vs TPR 0.00186141 (cos ~0.996);
layer1 input_layernorm Full 0.000766952 vs TPR 0.00276119
(cos ~0.977). Their ratio differs in *opposite directions*,
so a global /CP or /DP correction cannot reconcile them.
Layer14 input_layernorm sampled cosine -0.73 indicates
at least some selected low-magnitude elements diverge in direction.
QKV/MLP sampled rel-L2 still ~35%-41%, so the issue cannot be
certified as exclusively RMSNorm either.

The analyzer now additionally reports `sampled_non_norm`,
`norm_share_of_full_sampled_gradient_energy`, and repeated
TP Norm sample counts using the ALREADY SAVED witnesses; zero
additional NPU run required.

Next diagnosis should independently isolate:
(1) same topology Full TP2CP4 vs TPR TP2CP4 on this SAME 30K capture;
(2) per-token PPO logprob/ratio and clipped-gradient indicator parity;
(3) BF16 local parameter-gradient accumulation across Full microbatches
vs TPR tree segments. FP32 communication alone only addresses the
final SUM, not the previous BF16 local accumulation.


## Additional isolated 30K TPR TP2CP4DP1 comparison (2026-10-15)

A third eight-NPU experiment is now available to isolate the
*incremental DP=2 benefit* from TPR's prefix reuse. Do NOT infer
this incremental gain from unrelated 40K/other-step measurements.

Experiment uses the SAME existing immutable Step 6 derived
30,720-token capture and frozen HF Qwen3-4B-128K checkpoint:
- A (existing): Native Full, TP2/CP4/DP1.
- B (NEW NPU run): **TPR, TP2/RingCP4/DP1**.
- C (existing): TPR, TP2/RingCP2/DP2 with DTA-style DP placement.

B uses the existing `test_tpr_fixed_shape_40k_benchmark_npu.py`
in `TPR_OFFLINE_MODE=tpr`. The test already accepts a derived
30K manifest and original eight rows, enables the same
`TPR_TREE_POLICY=qwen_turn` and
`TPR_PPO_LOGPROB_CHUNK_TOKENS=512`, runs 1 warmup + 3 measured
iterations and captures the same deterministic gradient-witness
layout after timing. It runs BF16 gradient SUM/4 with
`HCCL_DETERMINISTIC=true`, NO optimizer updates/recompute/rollout.
The CP4 TPR offline finalize has also been adapted to respect
`TPR_OFFLINE_GRAD_SYNC_DTYPE` in the SAME way as Full and DP2
without changing the default BF16 calculation.

Run exactly one new benchmark, reusing original Step6 Full + DP2 files:

    # First sync bridge main into /workspace/uni-agent-cc/verl
    # with the existing bridge-to-container sync flow. The container's
    # native verl git origin may point to upstream verl-project/verl,
    # NOT the writable bridge repository; do not git pull that origin.
    cd /workspace/uni-agent-cc/verl
    pytest -q tests/models/mcore/tpr/unit/test_tpr_30k_three_way_comparison.py
    HCCL_DETERMINISTIC=true TPR_OFFLINE_GRAD_SYNC_DTYPE=bf16 \
    bash tests/models/mcore/tpr/parallel/run_tpr_tp2cp4dp1_30k_one_step.sh \
      /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
      /workspace/tpr_30k_cp4_deterministic_v1 6

The launcher verifies both archived JSON inputs BEFORE any NPU model
loading and refuses to overwrite the output. B's result is at:

    /workspace/tpr_30k_cp4_deterministic_v1/step6_tpr_fixed_shape.json

The automatic three-way comparison is at:

    /workspace/tpr_30k_cp4_deterministic_v1/comparison_three_way.json

Its independent time ratios are:
- TPR algorithm contribution at fixed CP4:
  `full_cp4_median_s / tpr_cp4_median_s`.
- **Incremental DP switch** CP4→CP2DP2:
  `tpr_cp4_median_s / tpr_cp2dp2_median_s`.
- Total Full CP4→TPR CP2DP2:
  `full_cp4_median_s / tpr_cp2dp2_median_s`.

It also reports all three PPO losses, worst-rank allocated-memory
peaks, and sparse gradient comparisons A/B, A/C and B/C. This is
important since Full/DP2 was already at ~50% sampled gradient rel-L2;
matching gradient replicas do NOT establish algorithmic parity.

**Fairness caveat:** the archived Full and DP2 JSONs predate the
`hccl_deterministic` metadata marker. New CP4 JSON explicitly stamps
it. The paired baseline's deterministic setting was confirmed by
the original launch command, not independently recoverable solely
from old JSON fields; the three-way report marks these as
`not_recorded`. No invented NPU measurements or parity claims.


## Completed deterministic BF16 30K three-way Step 6 (2026-10-15)

All three artifacts use the SAME immutable source:
`/workspace/tpr_dp2_30k_bf16_deterministic_pair_v1/step6_cap30720.pt`;
SHA256 `1741463fa53ace991da29efd444d00b7a1272477b7fe789d9d27ddb0302b9d60`.
Qwen3-4B-128K, 8 NPUs, 36,933 supervised PPO tokens,
1 warmup + 3 timed replays, BF16 gradient SUM/4, no AdamW.
New CP4 TPR explicitly stamped HCCL_DETERMINISTIC=true;
prior paired Full and DP2 jobs were launched with this setting
but did not record it in their individual JSON files.

| Mode | Topology | Median F/B+finalize s | Worst-rank measured peak GiB | PPO loss |
|---|---|---:|---:|---:|
| Native Full | TP2CP4DP1 | 16.454548 | 34.638256 | 0.151394129 |
| TPR | TP2CP4DP1 | 15.035786 | 24.657299 | 0.151653767 |
| TPR + DTA-style DP placement | TP2CP2DP2 | 12.401572 | 41.070560 | 0.151566282 |

Measured speedups:
- TPR CP4 vs Full CP4: 1.094359x (save 1.419 seconds)
- TPR CP2DP2 vs TPR CP4: 1.212410x (save 2.634 seconds)
- TPR CP2DP2 vs Full CP4: 1.326811x (save 4.053 seconds)
- DP2 memory: +66.57% vs TPR CP4; +18.57% vs Full CP4.
- TPR CP4 memory: -28.81% vs Full CP4.

DP2 planner assigns rows (2,3,0,6) vs (5,1,4,7);
local tree costs 59,493 vs 70,125 (largest 17.87% above smaller);
local total 129,618 vs global TPR tree 112,089, hence
17,529 duplicated Prefix tokens (~15.64% of global tree).
Thus adding DP2 on same 8 NPUs also halves CP from 4 to 2,
loses cross-DP Prefix reuse, and may incur load imbalance.
It is NOT a pure extra 2x compute resource. Actual CP Ring,
GEMM, activation and communication effects require profiling;
tree-token counts alone cannot predict FLOP-level runtime.

**All three gradient witnesses have exact CP/DP replica agreement,
BUT numeric Full/TPR parity is NOT PASSED.**
Each witness uses only 4,640 deterministic, unweighted samples:
- Full CP4 vs TPR CP4 rel-L2=0.455813, cosine=0.898021
- Full CP4 vs TPR CP2DP2 rel-L2=0.502732, cosine=0.870341
- TPR CP4 vs TPR CP2DP2 rel-L2=0.416250, cosine=0.910348

The surprisingly large 41.63% *between TPR topologies* is critical:
DP/CP topology change, forest segmentation and BF16 numerical
paths produce substantial sampled gradient differences. This is
not the difference 50.27%-45.58%=4.69 percentage points interpreted
as an additive DP error, because L2 vector errors do not add.
The original Full-vs-TPR CP4 error is already ~45.58%.
Sampled gradients heavily overweight Norm vectors: 8 samples
per tiny Norm and 8 per huge matrix; no full-model tensor-level
error magnitude can be inferred. Small PPO loss gaps do not prove
gradient parity.

Next priorities:
1. Existing three witness JSONs can be fed into the two-way
   gradient breakdown analyzer three times (including CP4 vs DP2)
   at ZERO NPU cost, to see whether TP/CP changes specifically
   affect RMSNorm or attention / MLP sampled directions.
2. At equal CP4 topology, compare tokenwise PPO new logprobs,
   ratio, clip decisions (and relevant d(loss)/d(logprob)),
   against TPR CP4. Then compare TPR CP4 vs DP2.
3. For definitive gradient parity, collect full-parameter
   FP32 squared-error accumulation across runs (one TP shard
   each, TP replicas de-duplicated); don't rely on strided
   8-per-parameter samples alone.
4. Investigate why DP2 peak is 41.07 GiB while TPR CP4 is
   24.66 GiB: inspect rank-local and per-DP peaks and separately
   profile activations, Prefix-KV retention and HCCL group buffers.
   Current per-run JSON records only WORLD worst rank and
   cannot attribute memory delta by component.
5. Before optimizing placement, assess per-DP compute-only
   critical path separately from synchronous finalize; simple
   rank wall time that includes gradient AllReduce can obscure
   idle time waiting for the slower DP forest.

Results are isolated frozen-weights F/B+manual finalization only,
not proof of PPO e2e training correctness, reward, or production
Megatron-DDP performance.

 
## Controlled 10K TP/CP/DP correctness isolation (added 2026-10-15)

Goal: isolate *factor-by-factor* numerical effects of prefix reuse,
TP1→TP2, CP1→RingCP2, DP1→DTA-placement DP2, and CP2+DP2.
**These are correctness probes, not benchmarks or online training.**

Run exactly six isolated Qwen3-4B/Ascend NPU topology arms:

| Case | Model/algorithm | TP | CP | DP | NPU ranks |
|---|---|---:|---:|---:|---:|
| unshared_tp2cp1dp1 | unshared TPR execution (control, not native Megatron Full) | 2 | 1 | 1 | 2 |
| tpr_tp1cp1dp1 | shared TPR, no TP/CP/DP parallelism | 1 | 1 | 1 | 1 |
| tpr_tp2cp1dp1 | shared TPR, no CP | 2 | 1 | 1 | 2 |
| tpr_tp2cp2dp1 | shared TPR, Ring CP2 | 2 | 2 | 1 | 4 |
| tpr_tp2cp1dp2 | shared TPR, DP2 but **no CP** | 2 | 1 | 2 | 4 |
| tpr_tp2cp2dp2 | shared TPR, Ring CP2 plus DP2 | 2 | 2 | 2 | 8 |

The **TP1CP1DP1** case is included even though it may have greater
per-NPU memory demand. On a capacity error, report it explicitly,
not as a numerical-correctness failure or silent downgrade.

Use the existing Step6 matched *30K* frozen capture from
`/workspace/tpr_dp2_30k_bf16_deterministic_pair_v1`.
Construct ONE immutable shared 10,240-token controlled dataset and
reuse it in all arms. Preserve original PPO response
`old_log_probs`, `advantages` and masks for every retained token.

**Critical disclosure:** Original prompts may themselves exceed 10K.
In that case the 10K preparation retains at most the last 4,096
prompt tokens and the following leading response window. This changes
model conditioning and RoPE positions. The data is valid as
a **same-input fixed-label numerical experiment**, but is **not**
equivalent to the original online PPO run. Output
`parity_capture_audit.json` records each row's prompt/response
truncation and `changes_original_conditioning`.

The NPU launcher runs from the bridge-synchronized VERL source tree:

    cd /workspace/uni-agent-cc/verl
    # Synchronize new bridge/main commits into this container as usual.
    pytest -q tests/models/mcore/tpr/unit/test_tpr_10k_parallel_parity_cpu.py
    HCCL_DETERMINISTIC=true \
    bash tests/models/mcore/tpr/parallel/run_tpr_10k_parallel_parity.sh \
      /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
      /workspace/tpr_10k_axes_correctness_v1

The script is resumable: a completed case's `result.json` is reused;
a failed case without a valid result must be diagnosed, not
overwritten. Six-case CPU comparison runs automatically on success:

    /workspace/tpr_10k_axes_correctness_v1/comparison_10k.json

Pairwise checks:
- `unshared_vs_tpr_no_cp`: prefix-reuse effect without CP.
- `tp_only`: TP1→TP2 at CP=1, DP=1.
- `cp_only`: CP1→RingCP2 with TP=2, DP=1.
- `dp_only_cp1`: DP1→DP2 with **CP=1**, TP=2.
- `dp_only_cp2`: DP1→DP2 under TP2 RingCP2.
- `cp2dp2_vs_no_cp`: combined effect.

Each result preserves source fingerprint, global supervised PPO
token count, topology, DP original-row assignments, global PPO
loss, **every selected logical token new_logprob**, and 64 canonical
full-coordinate sparse parameter-gradient samples for each weight.
The comparator reconstructs host-FP64 PPO ratio/clip decisions and
effective dLoss/dLogprob from original saved labels. Reports:
loss gap, gradient relative-L2/cosine, per-token logprob difference,
Clip status switches and effective PPO-gradient relative-L2.

The canonical gradient witness accounts for TP-sharded QKV/FC1
gate+up/FC2/linear projection/embedding layouts; avoids counting
identical CP/DP replicas twice; audits CP/DP sampled replica exactness.
Q/K RMSNorm weights are replicated among TP ranks but get **per-head
local gradients**. Witness explicitly sums those TP contributions
for comparison against TP1; it ALSO reports
`qk_norm_tp_local_gradient_max_abs`. This **does not modify the
actual optimizer gradient**. Any necessary TP synchronization of the
physical Q/K Norm gradient must be addressed separately before
production optimizer updates.

All arms use BF16 gradients, HCCL_DETERMINISTIC=true, no activation
recompute, no optimizer/rollout. Experimental DP2 exception is
allowed **only** when both `TPR_RUN_10K_PARALLEL_PARITY=1` and
`TPR_10K_PARITY_ONLY=1` are set in addition to the separate
DP2 experimental flag. Production DP gate is otherwise unchanged.

Caveats: the control `unshared_tp2cp1dp1` disables shared prefixes
inside the SAME TPR Segment backend. It is not the independent native
Megatron Full schedule. The canonical 64-samples/parameter gradient
comparison is sparse (but comparable across TP1/TP2), not
full-model Rel L2. This matrix therefore isolates plausible sources
of error but does not alone certify full-gradient optimizer parity.

 
Additional safety gate: before each 10K arm, the exact eight-row
controlled input is independently planned with `qwen_turn`.
If `global_unique_tree_tokens >= global_logical_tokens`, the run
aborts rather than quietly claiming TPR correctness on data with no
actual shared Prefix. The reuse ratio is stored in
`global_tpr_reuse_audit` for each result.

 
Improved 10K correctness witness: **every** trainable Norm parameter
(`input_layernorm`, `pre_mlp_layernorm`, final Norm and
per-head Q/K Norm) is now compared using its complete gradient tensor,
not eight or 64 sparsely selected elements. The cross-topology report
prints `norm_full_rel_l2` / `norm_full_cos`, a genuine all-element
Norm-only metric. Non-Norm large matrices remain canonical sparse
(64 full-coordinate samples per parameter). The actual TP-shared
Q/K Norm optimizer-state synchronization remains out of scope.


### 10K TP1 OOM and safe resume protocol (2026-10-15)

Observed `unshared_tp2cp1dp1` succeeds and prints 22,680 PPO
tokens, loss `0.0142083429`, 18,560 canonical sparse gradient
elements and 22,680 token logprobs.
The next `tpr_tp1cp1dp1` fails in backward due to NPU OOM:
56.48 GiB active on a 60.95 GiB card, 1.74 GiB allocation
attempt, 384.51 MiB free. **Do NOT call TP parity PASS**:
a successful `unshared TP2` only establishes one runnable TP2
control; both shared TPR TP1 and shared TPR TP2 must have comparable
gradient/logprob witnesses to isolate TP.

The six-arm runner now supports opt-in `TPR_10K_SKIP_TP1_OOM=1`:
it REQUIRES `tpr_tp1cp1dp1.log` to contain the actual
"out of memory" text before skipping only TP1, and resumes
the four pending TP2 CP/DP topologies, reusing the finished
unshared result. The resulting `comparison_10k.json`
will mark `complete_matrix=false`,
`tp_correctness_status=NOT_EVALUATED_TP1_OOM`,
and exclude only the `tp_only` pair from measured metrics.
Other individual checks (unshared-vs-shared CP1, CP-only,
DP-only CP1, DP-only CP2) can still be quantified.

    # First sync updated bridge/main into the VERL container.
    cd /workspace/uni-agent-cc/verl
    TPR_10K_SKIP_TP1_OOM=1 HCCL_DETERMINISTIC=true \
      bash tests/models/mcore/tpr/parallel/run_tpr_10k_parallel_parity.sh \
        /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
        /workspace/tpr_10k_axes_correctness_v1

To finish TP1-vs-TP2 parity, create a separate much shorter
controlled shared capture (2K-4K per trajectory, or fewer complete
selected trees) and run both TP1CP1DP1 and TP2CP1DP1 on the SAME
short input. Preserve genuine shared Prefix, original retained PPO
labels, identical model weights, and corrected Q/K Norm TP gradient
semantics. The original 10K unshared TP2 PASS cannot substitute
for this paired numerical comparison.


### 2026-10-15 final 10K protocol: permanently remove TP1 case

**This section supersedes the earlier six-case and OOM-log skip protocols.**
Following TP1CP1DP1 backward OOM, the user elected to exclude TP
correctness isolation from this 10K matrix entirely.

The experiment now has exactly FIVE arms, ALL at TP2:
`unshared_tp2cp1dp1`, `tpr_tp2cp1dp1`,
`tpr_tp2cp2dp1`, `tpr_tp2cp1dp2`, and
`tpr_tp2cp2dp2`.

No `TPR_10K_SKIP_TP1_OOM` flag is needed, no TP1 OOM log is
read, and no TP1 artifacts are required. The five-way comparator
omits `tp_only` and explicitly marks
`tp_correctness_status=NOT_EVALUATED_TP_FIXED_AT_2`.
`complete_matrix=true` means all **five CP/DP arms** completed,
not that TP numerical correctness was established.

Run with the existing Step6 10K workspace, reusing all completed
results without overwriting or rebuilding the capture:

    cd /workspace/uni-agent-cc/verl
    # Sync latest bridge/main to container first.
    bash tests/models/mcore/tpr/parallel/run_tpr_10k_parallel_parity.sh \
      /workspace/tpr_dp2_30k_bf16_deterministic_pair_v1 \
      /workspace/tpr_10k_axes_correctness_v1

Remaining comparisons: unshared-vs-TPR with CP1, CP-only,
DP-only with CP1, DP-only with CP2, and CP2DP2-vs-CP1.
