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
