# One-command Qwen3-4B nightly 20-step experiments

Bridge main has a **single launcher**:

```bash
cd /workspace/uni-agent-cc/verl
bash tests/models/mcore/tpr/integration/run_uniagent_20step_three_experiments.sh
```

**DO NOT `git pull` in the Docker VERL checkout.** Sync the current
Bridge main `verl/models/mcore/tpr`, `tests/models/mcore/tpr`, and the
`patches/apply_uniagent_session_hardcap.py` into the respective Docker
paths first. The launcher requires the patcher under either
`/workspace/uni-agent-cc/verl/patches` or `/workspace/bridge/patches`.
The script writes a fresh run directory under
`/workspace/tpr_nightly_20step_YYYYmmdd_HHMMSS/` and its path into
`/workspace/tpr_nightly_20step_latest.txt`.

## Three experiment phases

1. **E1 Online Native**: single node physical Ascend 8–15, Qwen3-4B-128K,
   N=8 real SWE rollouts, TP2/CP4/DP1, real vanilla PPO and GRPO rewards,
   20 training steps, 20 epochs for a single-prompt dataset, Native
   full Uniform(1) activation recompute, 48K session hardcap
   (16,384 prompt + 32,768 response tokens), no TPR.
   Original actor-update minibatches for each step are captured immediately
   before the true optimizer update. `skip.rollout_tq=cache` uses an entirely
   **fresh per-run TQ dump directory** to prevent stale cached rollout reuse.
2. **E2 Offline independent Full**: initialize the real Qwen3-4B-128K HF
   checkpoint and train for 20 steps consuming the 20 actor captures *in
   chronological order* with original old_log_probs/advantages. Full here
   means independent complete sequences through the same TPR-capable Ring
   backend; it is NOT production MindSpeed Native attention. Uses diagnostic
   CPU FP32-master AdamW (betas=0.9/0.95, eps=1e-8, lr=2e-5, wd=0,
   gradient clipping norm=1), TP2/CP4/DP1.
3. **E3 Offline Turn TPR**: starts from an independent copy of exactly the
   same HF weights, consumes the SAME ordered 20 captures/row selections,
   optimizer configuration, and computes shared-prefix Push/Pop/Visit under
   `TPR_TREE_POLICY=qwen_turn`. A failure/OOM of E2 does not block E3.
   If both finish, a CPU comparator outputs per-step PPO loss, logical
   logprob rel L2/max/mean absolute error, **exact distributed full-gradient
   BF16 norm and relative norm gap**, sampled gradient rel L2, cumulative
   sampled FP32-master update rel L2/cosine, peak VRAM, synchronized
   forward+backward+finalize critical-path time, and CPU optimizer time.

The offline runs are **sequential open-loop replay**, not online rollout
re-sampling after every independently updated replay model. They test
numerical stability and performance on identical actual Actor data, not
online Native vs TPR reward convergence or production-optimizer parity.
An all-zero-Advantage GRPO group is preserved and shown in preflight;
the AdamW replay does not fake nonzero rewards or gradients.

## Exact input and memory budget

`prepare_real_actor_20step_manifest.py` requires exactly one rank0
capture for every real global step 1..20 (strictly no missing/duplicate step).
For each captured step, selects up to three **whole trajectories** with
length <= 49,152 and aggregate full tokens <= 130,000, starting with the
shortest ones. At least two real eligible rows are required per step;
otherwise fail **before** starting any offline 8-NPU job. Never truncate
sequences or generate synthetic PPO fields. Its JSON manifest records each
step's chosen source file and **original row ids**, lengths and PPO tokens.
Both E2/E3 read exactly the same manifest. A step with all-zero rewards/
advantages is marked but not silently filtered.

If many steps have only one usable trajectory, the script will exit after
online capture preflight with a clear reason and keep all the capture data.
That is safer than silently claiming a 20-step comparison using unrelated or
repeated training data.

## New performance instrumentation

When `TPR_EXEC_PROFILE=1`, the PPO Engine records for every physical
Segment across all TP2/CP4 ranks:

- tree_id, segment_id, parent, kind=Push/Pop/Visit, depth, prefix and suffix
  token counts
- NPU-stream Event duration (ms) and CPU enqueue duration (ms)
- local allocator allocated and reserved bytes before/after, per-event
  peak allocated and reserved bytes
- fixed CPU/collective overhead: topology build time,
  TP schedule digest/collective, SegmentExecutor setup,
  segment result aggregation, complete entry host elapsed
- CP gradient finalize critical-path elapsed (separate),
  entire forward+backward+finalize **synchronized end-to-end elapsed**,
  and off-device FP32 AdamW update duration.

No `torch.npu.synchronize()` is inserted after individual segments.
The event clock tracks its recording stream; Ring/HCCL may execute on
additional device streams, so **never sum Segment event ms to infer total
critical-path GPU wall time**. Use synchronized `fwd_bwd_seconds`
(max rank) for speedup. The allocator peak reset at each segment is
reconstructed by maxima across recorded events, preserving overall peak
evidence. Instrumented performance is not directly identical to an
unprofiled run; report profiler overhead when interpreting tiny speedups.

Full per-parameter gradient relative L2 / cosine is optional, but disabled
by default to avoid multiple full 4B BF16 gradient disk dumps. The requested
primary correctness gates are logical logprob and distributed grad norm.
Sampled gradient L2 continues as a secondary signal; sampled SGD weight
difference is NOT used to declare parity.

## Outputs

The launcher prints the run root and records it at
`/workspace/tpr_nightly_20step_latest.txt`:

```text
logs/e1_online_20step.log
logs/capture_preflight.log
capture_manifest.json
captures/actor_update_step000001_rank0_capture000000_batch0.pt
...
captures/actor_update_step000020_rank0_*.pt
logs/e2_full.log
logs/e3_tpr.log
logs/final_comparison.log
offline/full_step1_rank0.pt ... full_step20_rank7.pt
offline/tpr_step1_rank0.pt  ... tpr_step20_rank7.pt
offline/multistep_comparison.json
```

If the online run fails, no replay occurs. If capture validation fails, no
replay occurs. Full and TPR replay each start independently; even if E2
fails/OOMs, E3 still runs and its memory/performance snapshots are kept.
No success is claimed until the associated job exits successfully.

## Recovery and known limitations

- `TPR_CAPTURE_ACTOR_MINIBATCH_DIR` and TQ skip cache directories are
  unique per run; never resume by overwriting the same capture.
- The online script is Native only; its recompute config is intentionally
  independent of the offline TPR (`recompute_granularity=None`).
- Online weights and optimizer states after each step are not checkpointed;
  the independent offline models start from identical HF checkpoints
  but follow their own FP32-master AdamW histories.
- Runtime NPU profiling and full 20-step execution **have not been run in
  this assistant environment**, which has no access to Ascend NPUs/Ray.
- The legacy 9-node exact Trie remains available for a separate A/B:
  `TPR_TREE_POLICY=exact`. This overnight run compares Full against
  the latest 6-ish-node Turn-constrained strategy.
