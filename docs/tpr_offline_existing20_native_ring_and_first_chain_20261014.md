# 2026-10-14: Offline 20-step Full Native Ring vs Turn TPR, using existing captures

## Run E2 and E3 only — **no online rollout**

Update/sync the Bridge main TPR modules, tests and patches into the same
Docker VERL checkout as before. Do **not** git-pull inside the Docker VERL
tree. Locate the directory containing last night's
`actor_update_step000001_rank0_capture000000_batch0.pt` through
`actor_update_step000020_...pt`.

```bash
cd /workspace/uni-agent-cc/verl
bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/YOUR_ACTOR_CAPTURE_20STEP_DIR
```

The runner creates an isolated timestamped
`/workspace/tpr_20step_offline_only_YYYYmmdd_HHMMSS` directory,
builds `capture_manifest.json` from **20 distinct original steps**,
runs independent E2 Full and E3 Turn TPR, and compares every step.
Neither Ray nor online rollout is invoked.

To reuse an output directory and run modes separately (e.g. Full OOMs,
still run TPR):

```bash
bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/YOUR_ACTOR_CAPTURE_20STEP_DIR \
  /workspace/my_offline20 \
  full

bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/YOUR_ACTOR_CAPTURE_20STEP_DIR \
  /workspace/my_offline20 \
  tpr

bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/YOUR_ACTOR_CAPTURE_20STEP_DIR \
  /workspace/my_offline20 \
  compare
```

A strict preflight requires exactly one rank0 original actor minibatch
per step 1..20, nonempty original PPO fields, >=2 eligible whole
trajectories per step. It never silently fabricates/reuses a step.

The per-step row selector preserves genuine positive/negative Advantage
diversity when available; at most 3 whole sequences, each at most
49,152 tokens and <=130,000 aggregate tokens. This is still a *selection
from already-processed PPO labels*; it does not retroactively change
yesterday's online rewards.

## Which Ring is the Full reference?

Previously E2 used *the same custom* `_RingTPRAttention` as TPR,
but without Prefix caching. It was not native MindSpeed Ring.

New E2 explicitly sets `TPR_OFFLINE_FULL_RING_BACKEND=native`, calls
`ordinary_ring_cp_attention` ->
`mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel.ringattn_context_parallel`.
The native Ring only replaces the **core Attention** operator within the
same Full SegmentExecutor and Qwen TPRSelfAttention model. This is a
native-Ring-only reference, NOT native MegatronEngine/MindSpeed
forward/backward or production online optimizer.

Qwen3 TP2/CP4 real lengths often require right virtual padding to
Zigzag-CP8 alignment. For an independent full sequence the padding is
after all real causal tokens, so it cannot affect any preceding real
query's attention; supervised padding query mask is zero. The opt-in
`allow_trailing_padding=True` is only used on this native Full path.
**Native-vs-custom numerical and real NPU correctness still need
to be measured**, especially gradients. Do not apply this allowance
to cached-prefix TPR calls.

E3 continues custom TPR Ring with `TPR_TREE_POLICY=qwen_turn`.
Each snapshot records `ring_attention_impl`; the CPU comparer includes
the backend identifiers and exact global BF16 gradient norm,
logical logprobs, PPO loss, sampled gradient L2, AdamW master-weight
drift, per-segment times, allocator peaks and fixed overhead. Native
Ring-vs-TPR error now mixes native-vs-custom Ring numerical differences
with Prefix reuse physical-shape differences. Keep an independent
custom-Full control when attributing individual sources of error.

## Fix future multi-chain sessions before reward + TQ

The new `patches/apply_uniagent_session_hardcap.py` additionally handles
sessions for which multiple finalized Gateway trajectories already exist
despite the 48K session cap.

With `hard_stop_on_trajectory_capacity=True` it:

1. Checks the **original count before retention** after
   `finalize_session`.
2. For count >1, keeps only the original first finalized trajectory.
3. **Forces `reward_score=0.0` and `reward_info.reward=0.0`**,
   bypassing both any posted success reward and the RewardLoopWorker.
4. Marks `multi_chain_session=True` and
   `original_session_chain_count=N` on the remaining trajectory so
   the TQ and Actor capture record the reason.
5. Retains existing source masks/ids/logprobs; does NOT append a later
   chain, extend beyond 49,152 tokens or silently keep longest.

The upgraded patcher is compatible with files already patched by
last night's earlier version. Apply it to the Uni-Agent root, not VERL:

```bash
python /workspace/bridge/patches/apply_uniagent_session_hardcap.py \
  /workspace/uni-agent-cc
```

Use the actual copied Bridge patcher path if different.
The online training script invokes it automatically on its next run.
No overnight 20-step online rerun is needed for E2/E3.

**Very important:** Yesterday's 20 captured PPO minibatches already
contain original online `advantages` and `old_log_probs`. Simply
editing captured `rm_scores` to zero would NOT recompute GRPO
group-relative advantages and would corrupt causal attribution. The
new first-only/zero-reward policy affects **subsequent online generation**,
not the immutable historical 20 snapshots. Rerun rollout/GRPO if
requiring truly zeroed multi-chain Advantage labels. For existing
snapshots, leave all PPO tensors untouched and use them as historical
deterministic replay.

## Validation

Source and CPU regression code is in Bridge main. Full native whole
Ring with trailing virtual padding and the modified hardcap policy have
not yet passed your live Ray/NPU integration gates.

Measure real E2/E3 numerical parity and performance independently.
A non-zero PPO loss/logprob/gradnorm gap is useful diagnostic data;
no threshold has been relaxed.

## 2026-10-14 amendment: process historical multi-chain Sessions OFFLINE only

**No Uni-Agent patch or new rollout is required for run5.** For the run5
capture directory, invoke the revised stand-alone script:

```bash
cd /workspace/uni-agent-cc/verl
bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/tpr_actor_capture_4b_n8_run5 \
  /workspace/tpr_offline_run5_session_first \
  both
```

Preflight is CPU-only and uses *all* `actor_update_stepXXXXXX_rank0_captureYYYYYY_batchZ.pt`
files for each online step 1–20; `capture000020` is the 21st captured
actor mini-batch, **not** Session trajectory index 20. It does not
discard multiple Actor optimizer updates per online step.

Original `session_id` and `uid` in TensorDict form the session identity.
Before selecting up to 3 complete trajectories per captured mini-batch, the
preprocessor deduplicates **across all captures of that online step**:

- If the capture retained a true per-row `trajectory_index`,
  `trajectory_idx` or `tq_trajectory_index`, keep minimum original
  ordinal for each Session.
- If the original TQ key/ordinal has already been lost by Actor mini-batch
  shuffling, select the **first observed** in ascending capture_index,
  batch_idx, then original mini-batch row. **That is deterministic but
  not provably the Gateway's chronological first trajectory.**
  Preflight emits `TPR_SESSION_ORDER_WARNING` and records fallback counts.
- Never switch to a later chain merely because the selected first chain is
  too long. That session is excluded if its first selected trajectory is
  >49152 tokens or has no supervised PPO tokens.
- If a mini-batch has no selected first-of-session rows, skip that mini-batch
  with an explicit audit entry. Fail if an online step has no eligible
  trajectory at all; no missing online global step is hidden.
- Preserve the original actor-update boundaries of retained mini-batches,
  hence the number of *offline optimizer replay updates* may be >20.
  `capture_manifest.json` records both `online_steps=20` and
  `steps=<retained offline optimizer updates>`, the original source global
  step, capture index, source rows and multi-chain session counts.
- One retained trajectory in a mini-batch is valid for Full/TPR parity but
  provides no prefix reuse; no fake sibling is synthesized.
- Raw `.pt` files are never modified. Both E2 and E3 load exactly the same
  manifest, original old logprobs, advantages, temperature, and masks.

**Do not set historical reward to zero without re-deriving GRPO Advantage.**
These are captures taken *after* online GRPO reward processing. The
`session-first` replay performs deterministic row selection, not retroactive
reward relabeling. Changing just `rm_scores` would be numerically
inconsistent and is deliberately not implemented here.

If you require the true first *Gateway* trajectory rather than first observed
Actor row, use the original TQ key suffix `uid_session_index_trajectory_index`
or persist original `trajectory_index` in future captures. The old Actor
capture `keys` generated as `uid_tpr_row` do **not** encode the TQ
trajectory index.

**Important:** This new `v2` manifest is not compatible with an old
already-created `v1` manifest; use a fresh offline output directory to
regenerate preflight. No need to re-run the 20-step online experiment.

## 2026-10-14 final correction: preserve the FULL N=8 session set

The previously implemented **max 3 rows / 130,000 aggregate tokens**
was an overly conservative OOM-avoidance heuristic. It is REMOVED from
the current offline-only runner. The correct experimental sample for N=8
is one original first trajectory from EACH of eight `(uid,session_id)`
pairs, for EACH of 20 online global steps. No short-trajectory or
Advantage-sign prioritization takes place.

Multiple Actor `captureXXXXXXXX_batchY.pt` files in one online step
are **not different rollout steps**. The preflight groups all of them,
identifies the 8 original Sessions (or fails), selects each Session's
first available captured chain, verifies every one is <=49,152 tokens,
then concatenates all 8 **original TensorDict rows** into ONE newly saved
`session_first_batches/actor_update_stepXXXXXX_session_first8.pt`.
No old capture is overwritten. The one merged sample is passed to the
independent Full Native-Ring and Turn TPR replay as the SAME eight-row
minibatch; this preserves TPR prefix sharing across all eight children.

The manifest uses protocol `real_actor_session_first_8merged_v3`,
`online_steps=20`, `steps=20`, `expected_sessions=8`,
`max_rows=null` and `max_total_tokens=null`. Each item records source
row provenance from every input capture, original Session ids, total
logical tokens, old PPO token count, and the merged file path.
Missing Session or a first trajectory over the 48Ki-token limit
is an explicit error; no replacement with a later/shorter chain.

Use a NEW output directory to avoid reusing a v1/v2 manifest:

```bash
cd /workspace/uni-agent-cc/verl
python tests/models/mcore/tpr/parallel/prepare_real_actor_20step_manifest.py \
  /workspace/tpr_actor_capture_4b_n8_run5 \
  /workspace/tpr_offline_run5_full8_v1/capture_manifest.json \
  --steps 20 --max-seq-len 49152 --expected-sessions 8
bash tests/models/mcore/tpr/parallel/run_existing_20step_offline_full_tpr.sh \
  /workspace/tpr_actor_capture_4b_n8_run5 \
  /workspace/tpr_offline_run5_full8_v1 \
  both
```

Note: per-trajectory hard cap does NOT mathematically guarantee that
eight concurrently selected trajectories fit GPU memory. Full may OOM
even when all eight are <=49,152 tokens. Full and TPR modes can still
be launched separately to retain all failure/performance evidence.
The original TQ trajectory ordinal often is absent from Actor Capture,
so first-in-capture is a documented approximation rather than a proven
original Gateway chain order. The historical Advantage and rewards are
unaltered; this modification touches ONLY offline data selection.
