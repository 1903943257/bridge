# Stage 4.5 — AllGather CP2 Prefix Reuse performance

Scope: random full Qwen3.5-0.8B (24 layers, 18 GDN + 6 FA), BF16,
CP=2, TP=PP=1, batch=1, dropout=0, non-packed. Ring investigation is
deferred, not declared an upstream bug. No Ring/kernel/threshold changes.
This stage is the performance milestone, **not THD/remove-padding**.

## Execution and fairness

- Default matrix: N=2/4/8/16;
  (P,S)=(8192,1024)/(8192,8192), with 1k=1024 tokens.
  Ratios 8:1 / 1:1; total lengths 9216 / 16384; eight cases overall.
  CPU-seeded identical
  prefix/branch tokens, one unchanged model reused across all runs.
- Ref: N independent whole P+S calls via leaf execution; no cached state
  crosses trajectories. TPR: real `FixedTopologyScheduler` and
  `SegmentExecutor` Push(P), N Visit(S), Pop(P). This measures scheduler-level
  reuse, not request parsing/data loading/Engine integration or optimizer.
- Loss: suffix S[i] predicts S[i+1], i=0..S-2. No P loss and no P-last
  predicting S-first. Both objectives divide by N(S-1); cancel the Executor
  CP multiplier and perform exactly one parameter-gradient SUM per run.
- Both paths use identical TPR-capable stateful GDN primitives and the
  existing `install_whole_fa_allgather` FA control. **This gathers Q and KV
  and duplicates full FA compute on both ranks**; it is not optimized
  KV-only AllGather. Prefix KV is gathered again for each suffix. Results
  apply to this transport, not to every AllGather CP implementation.
- `cp_backend="ring"` only retains the already validated GDN/FA zigzag
  placement carrier. The FA callable is replaced with AllGather; RingP2P
  raises immediately if reached. No Ring attention implementation executes.
- No Linear shape control/clamp, optimizer updates or threshold relaxation.
  Loss gap is reported, finite loss/all parameter gradients and communication
  contracts are audited; benchmark completion is **not correctness PASS**.

## Timing and memory

Each case/path: clear gradients/cache before warmup, >=1 warmup, >=3 measured
repetitions. No cache flush between measured repeats. Alternate path order
across cases. Plan/token generation, initialization, zero-grad, barriers,
finite checks and reporting are outside timing. Total includes F+B,
scheduling/state lifecycle and one parameter SUM, ending with NPU synchronize.
Use **median of per-repetition rank-max wall time**; measured speedup is the
ratio of Ref/TPR medians. Lightweight communication counters are present on
both paths. No per-layer tensor copies, equality scans or fine timing probes
inside the main measurement.

Coarse asynchronous NPU events record `forward_device_ms`, `backward_device_ms`,
`prefix_recompute_device_ms` and `parameter_sum_device_ms` in these same runs,
with no per-call synchronization. Forward includes Push and Pop recompute;
recompute overlaps forward, so do not add it again. Device-stream elapsed time
includes stream waits/launch gaps, not just kernels, and does not partition
host total exactly. Detailed attribution is deliberately a separate pass.

Peak allocated/reserved and starting allocation/reservation are rank-max MiB,
captured before diagnostics. Also report incremental peak allocated above
the parameter-only starting allocation. Clear gradients before resetting
peaks; the model is shared and no reference model/gradient snapshot stays on
the NPU. Warm cached reservation is included, not subtracted from the peak.

A **separate**, equally warmed/repeated synchronized profile prints medians:

| Fields | Meaning |
| --- | --- |
| `profile_inclusive_model_forward_ms` | Forward including in-forward communications |
| `profile_inclusive_backward_ms` | Autograd backward including inverse collectives |
| `profile_exclusive_model_forward_ms` | Forward excluding nested communication; includes dispatch overhead |
| `profile_exclusive_backward_ms` | Backward excluding collectives; includes autograd overhead |
| `profile_exclusive_allgather/reduce_scatter/a2a_ms` | Synchronized collective call time, both directions for A2A |
| `profile_exclusive_parameter_sum_ms` | Parameter gradient finalization |
| `profile_inclusive_prefix_recompute_ms` | Pop(P) forward incl. model + communication, NOT additional additive cost |
| `profile_exclusive_state_ops_ms` | KV/GDN save/anchors/gradient accumulation/consume/release |
| `profile_exclusive_schedule_ms` | Remaining scheduler/host preparation and unclassified lifecycle overhead |
| `profile_exclusive_loss_compute_ms` | Cross-entropy construction/forward |

Nested exclusive buckets avoid double counting in each local sample.
Synchronizations perturb communication overlap and launch pacing: **do not
use profiled total for speedup**, or add inclusive recompute to forward time.
Independent rank maxima and medians of components need not sum to a median
total. These are attribution estimates, not pure device-kernel timings.

## Expected per-rank communication

| Counter | Ref | TPR |
| --- | ---: | ---: |
| Model forwards | N | N+2 (Push + N Visit + Pop recompute) |
| GDN cp2hp / hp2cp forward calls | 108N / 18N | 108(N+2) / 18(N+2) |
| FA AllGather backend calls | 6N | 6(N+2) |
| FA AllGather collectives | 18N | 30N+36 |
| FA backward ReduceScatter | 18N | 30N+18 |
| Ring P2P | 0 | 0 |

Push has no VJP. CP Pop includes a zero-logit loss root to keep collective
participation aligned, so the final prefix FA core still runs backward even
with no owned loss (unlike the connected-split diagnostic).

## Reading savings

Requested ideal token speedup: N(P+S)/(P+NS). This ignores recompute,
attention's nonlinear sequence cost, full-vocabulary head, communication,
state operations and launch overhead. Prefix is reused across branches but
is **not computed only once over training**: Push(P) is graph-free; Pop(P)
recomputes it once. With illustrative F:B=1:2, a recompute-adjusted token
estimate is 3N(P+S)/(4P+3NS), not a performance prediction.

`saving_realization_fraction = (1 - TPR_ms/Ref_ms) / (1 - 1/ideal_speedup)`
expresses how much ideal elapsed-time saving is realized. It can be negative
(slowdown) or >1; never clip it. Use the separate profile to examine recompute,
repeated prefix KV gather, A2A, parameter SUM and state/scheduler costs.
Do not attribute the gap to one component without the measurements.

## Run

From the server verl repository containing these files:

```bash
mkdir -p tests/models/mcore/tpr/logs
STAGE45_PERF=1 STAGE45_WARMUP=1 STAGE45_REPEATS=3 torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v --tb=short tests/models/mcore/tpr/parallel/test_qwen35_prefix_reuse_perf_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

This overwrites the log; use `>>` to append. CPU reporting unit test:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_prefix_reuse_metrics.py
```

Optional subset/sizing: `STAGE45_BRANCHES=2,4`,
`STAGE45_LENGTHS=256:768,512:512,768:256`. Lengths must be multiples of 64.
The model's max sequence length is set from the largest requested P+S.
A subset is a smoke test, not the complete matrix. The long default lengths
increase runtime and peak memory substantially, especially the full-vocabulary
logits and replicated whole-QKV FA. An OOM is reported as such, not silently
handled by shortening a case or changing the execution path.

Rank0 prints progress, one `STAGE45_CASE` JSON per case (median timing/memory,
profile medians, raw clean samples and communication counts), then a compact
8-row `STAGE45_SUMMARY`. Results: **NPU performance measurements pending**;
no speedup or memory saving is claimed before running on the server.

Local validation: five dependency-free reporting unit tests passed; all three
new Python files parsed successfully. Local PyTorch/pytest/NPU are unavailable,
so distributed execution, timing hooks and collective contracts remain to be
validated on the server.

## Long-sequence failure localization (not a performance run)

`STAGE45_SYNC_DIAG=1` logs both ranks with case, Ref/TPR, repetition and
warmup/repeat phase. Nested paths identify independent Ref branch, Push,
Visit, Pop, segment ID/length/prefix length, `_forward` (including no_grad),
loss construction and autograd backward. Each boundary prints PRE_SYNC,
BEGIN, POST_SYNC, END; FIRST_ERROR identifies pre_sync/body/post_sync.
Only the innermost first failure is tagged; the original exception is re-raised.
Pre-sync failure can belong to previously queued work, not the named body.

Diagnostics preserve warmup/repetition order but skip the separate fine profile
and suppress timing/speedup reports. Communication and finite checks remain.
No per-layer tensors are copied or printed. This locates a failing phase, not
the individual kernel within it. Synchronization may change reproducibility;
a diagnostic success does not prove the unsynchronized run is fixed.

```bash
mkdir -p tests/models/mcore/tpr/logs
STAGE45_PERF=1 STAGE45_SYNC_DIAG=1 STAGE45_BRANCHES=2 STAGE45_LENGTHS=8192:1024 STAGE45_WARMUP=1 STAGE45_REPEATS=3 torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v --tb=short tests/models/mcore/tpr/parallel/test_qwen35_prefix_reuse_perf_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Read the last END and first FIRST_ERROR per rank, particularly Visit/backward
(suffix initial-state gradient) vs Pop/backward (prefix final-state VJP).
After failed execution, test-level cleanup does not zero gradients or enter
the teardown barrier. Timers do not record end events/synchronize while an
exception is unwinding. Framework/HCCL destructor errors may still occur;
restart both worker processes after a device exception. These changes do not
recover a poisoned device context or suppress the original pytest failure.
Set `STAGE45_SYNC_DIAG=0` (default) for normal performance measurements.

For first-Visit stateful Conv/GDR backward isolation, see
[STAGE45_STATEFUL_OPS.md](STAGE45_STATEFUL_OPS.md): subprocess-isolated
initial-state/sequence-length A/B and opt-in `STAGE45_OPS_TRACE=1` backend
boundary/launch tracing. No OPS gate, kernel or tiling overrides are applied.
