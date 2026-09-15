# Stage 4.5 first Visit backward: stateful OPS audit

Status: static audit + diagnostic implementation. No kernel/gate/threshold
changes. NPU A/B results pending. First Visit S=1024 with P=8192 fails after
whole Ref completed; no sibling accumulation or Prefix Pop has run yet.

## Findings from local source (verify against server SHA)

1. **Concrete Conv dh0 global-buffer bounds defect candidate**:
   `arch32/triton/convolution.py` allocates
   `dh0[min(NT,ceil(W/BT)),B,D,W]`, but loops over all NT time tiles and stores
   at `dh0 + i_t*B*D*W + i_n*D*W + channel*W + slot` with only `mask=m_d`.
   The new global-time read mask zeros noncontributing tiles but does not
   protect the store. For T=1024,B=1,D=3072,W=4,BT=8: NT=128, allocated tile
   count=1; tiles 1..127 address outside the allocation even if writing zero.
   This defect is visible **before** commit `2d503ca` (the dh0 tile-read fix);
   that fix did not introduce or fix the unguarded store. It can affect short
   sequences too: a previous numeric PASS is not a memory-safety proof.
   This is a global-address argument, **not proof of the reported UB local
   address exception**. The actual first failing launch must be identified.
2. Initial-state presence, not `requires_grad`, selects dh0 allocation in both
   Conv and GDR. Neither inspected API skips dh0 via `ctx.needs_input_grad`.
   Frozen-state vs trainable-state is NOT a reliable kernel-branch toggle.
3. Both custom autograd APIs return final state and do not disable gradient
   materialization in the inspected source. Unused final state may therefore
   reach backward as a zero tensor, selecting `has_dht=True`. The probe logs
   actual dht presence, shape, dtype and nonzero count. Visit has no explicit
   final-state loss; that does not guarantee a None dht pointer.
4. Conv initial-state path: BT=min(8,nextpow2(ceil(max(16,B*T)/vector_cores))),
   BD=32, grid=(vector_cores,); no-state BT cap=32, BD depends on dht and
   parallelism. W=4,D=3072 (CP1 D=6144) satisfy D%BD. BT at T=64..1024 must
   be read from actual launches, not forced by changing vector core count.
   The combined state/final-gradient path uses static W loops and additional
   live FP32 buffers. Comments estimating UB use are not compiler proof.
5. GDR CP2: Q/K/V=[1,T,8,128], h0=[1,8,128,128] FP32. Backward dhu uses
   BT=64,BV=128, grid=(1,8), K<=256 assertion. For K=128 it holds two
   [64,128] FP32 recurrent-gradient blocks before other temporaries; compiler
   liveness/tiling may matter. dh0 has full h0 shape; the inspected final stores
   cover disjoint 64-row K blocks with boundary checks, unlike Conv's tile
   buffer mismatch. No analogous out-of-allocation formula established yet.

## Gate audit: distinguish backend availability from supported shape

- Native MindSpeed `GatedDeltaNet` constructor checks `HAVE_FLA`, populated by
  importing FLA Conv/GDR/L2norm. This is a backend availability gate, not an
  OPS T<=.../D<=... capability certificate. Local native code retains it;
  no `HAVE_FLA=True` assignment was found in the current Bridge TPR path.
  Server MindSpeed was dirty, so the precise server gate modification cannot
  be established from local source alone.
- TPR `_stage1_causal_conv1d` / `_stage1_gated_delta_rule` explicitly select
  stateful MindSpeed-Ops APIs instead of the layer's native FLA/backend
  selection. This bypasses native *backend selection*, not a demonstrated
  OPS shape rejection. It does not patch `is_arch35`, `get_vector_num`, BT/BD,
  or delete dtype/head-dimension checks in the current benchmark.
- OPS Conv rejects arch35 and enforces D divisibility; local arch32 Qwen
  D=3072,W=4,T=1024 passes these checks. GDR API checks matching QKV dtypes,
  rejects FP32 QKV, checks beta/layout/varlen constraints; dhu checks K<=256.
  BF16 QKV with FP32 g/h0, K=V=128 and non-packed T=1024 passes inspected
  gates. **No explicit T=1024 unsupported gate found** does not mean supported.
- The earlier `dh0_tile_fix.patch` changes read indexing; its UT overrides
  vector core count to exercise BT=1/2/4/8, scoped to that UT. Stage 4.5 does
  not run that override. The dw diagnostic patch changes test reporting only.
- Stage 1 GDR stateful test uses H=2,K=V=64: it does not certify H=8,K=V=128
  plus all longer sequence/state-gradient combinations.

## Minimal independent A/B (no model, TPR scheduler, CP or HCCL)

`run_stateful_ops_ab.py` launches a fresh Python child per combination, on
one NPU using the actual **post CP->HP** dimensions (time is not halved).
Default: Conv and GDR x T=64/128/256/512/1024 x initial=none/frozen/train,
final=unused: 30 isolated runs. Weights/inputs/state/upstream use identical
CPU seeds for each A/B; Conv BF16+SiLU,D=3072,W=4; GDR BF16 QKV,H=8,K=V=128,
FP32 g/h0, normalized Q/K. No capability flags are overridden.

`--finals off,unused,nonzero` expands to 90 cases: no requested final output,
requested-but-unused final output, and explicitly nonzero final-state VJP.
The actual logged dht, not just this label, decides the kernel branch.
Initial state is random, not captured from a P8192 execution; success cannot
rule out value/stride/history-sensitive failure on actual model tensors.
Use `--channels 6144` for an additional CP1-width Conv control.

```bash
mkdir -p tests/models/mcore/tpr/logs
python tests/models/mcore/tpr/parallel/run_stateful_ops_ab.py --device 0 > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

To prioritize the failure shape and combined-state branches first:

```bash
python tests/models/mcore/tpr/parallel/run_stateful_ops_ab.py --device 0 --lengths 1024 --finals off,unused,nonzero > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Each child logs actual source paths/SHA256, selected source gates, boundary
syncs, tensor shape/stride/dtype/requires_grad, dht zero/nonzero, Conv/dhu launch
grid/BT/BD/BV and conditional dh0 bounds. Parent prints each exit code and
returns failure if any case fails; no thresholds are relaxed. Failed children
exit without device destructors; no failure is retried in the same process.
Restart/recover devices if fresh child processes also cannot initialize.

## Actual Stage 4.5 backend-stage localization

```bash
STAGE45_PERF=1 STAGE45_SYNC_DIAG=1 STAGE45_OPS_TRACE=1 STAGE45_BRANCHES=2 STAGE45_LENGTHS=8192:1024 torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v --tb=short tests/models/mcore/tpr/parallel/test_qwen35_prefix_reuse_perf_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

This traces actual tensor arguments, without altering computation. GDR
backward boundaries include recompute w/u, fwd_h, dv_local, dhu, dqkwg and
prepare_wy; Conv logs preactivation recompute and backward launch. If GDR
fails at a synchronized boundary **before Conv backward begins**, Conv dh0
cannot be the first failing backward launch of that execution. If isolated
Conv fails with both frozen/train h0, inspect initial-state branch and dh0
store first. If only final=unused/nonzero fails, focus on actual dht-selected
path. Never infer a kernel fix from a pass at a different layout/shape.

Local CPU bounds tests demonstrate the conditional arithmetic discrepancy;
they cannot detect device UB errors. All server A/B results remain pending.
