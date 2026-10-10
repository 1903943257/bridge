# 2026-10-10 — Isolated TP2 Engine, DP2 and TP2×DP2 acceptance (NO E2E mutation)

This deliberately stays on the `feat/tpr-tp2-dp-placement-audit` **draft**
branch, leaving the actively changing `main` BF16/PPO/DTA and UniAgent
E2E work untouched. No production TPR, Megatron, loss, optimizer or
trainer source files were modified for this acceptance round.

## Evidence at the start

- Previous USER NPU test: pretrained Qwen3-1.7B, native Megatron TP2/SP
  off vs TPR thin adapter: CE loss 13.5967836 vs 13.5979404,
  grad rel-L2 0.0216998 PASS; PPO Forest *NLL specialization* loss
  12.6356554 vs 12.6377153, rel-L2 0.0313719 PASS. These are
  thresholded tests, not exact BF16 equivalence.
- Neither was a real `MegatronEngine.forward_backward_batch()` Phase-4
  **PPO auto-route** test: the latter had previously accessed
  `loss_mask` before the thin request branch.
- `main` at this date has active
  `docs/tpr_bf16_shape_drift_dta_reference_20261010.md` and
  `test_qwen3_1_7b_real_tq_weak_e2e_npu.py`. That weak E2E uses cropped
  real TQ, **torch AdamW FP32 masters**, recomputed old logprobs and
  optional diagnostic advantages; it is NOT real VERL/Megatron Actor
  end-to-end training. BF16/PPO numerical parity remains unresolved.
  Do not combine parallel acceptance with this work.

## Separate acceptance gates and actual scope

### Gate A — TP2 actual Engine Phase-4 routing (2×NPU)

`test_tpr_tp2_npu.py::test_tp2_real_qwen_engine_phase4_route_two_updates`

- Pretrained real Qwen3-1.7B, Megatron TP2, SP=False, CP/PP/EP/DP=1.
- Uses original two-row controlled NLL objective as a route/lifecycle
  signal, not production PPO (NO ratio/clip numeric comparison).
- Supplies *real Engine required* `loss_mask`, checks installed method
  `MegatronEngine.forward_backward_batch()` dynamically imports/calls
  `run_tpr_forward_backward_batch` exactly once per pass.
- Checks native Engine metadata `batch_num_tokens` and `dp_size`,
  one finalize per forest, non-finite gradients, two SGD updates, and
  that some weight actually changes. This DOES NOT assert native Megatron
  optimizer integration.
- If the deployed VERL Phase-4 Engine patch is missing, FAIL; do not
  silently fall back to direct thin adapter.
- **Not executed on user NPU yet.** A passing thin-adapter test is not
  evidence this route passed.

### Gate B1 — DP2 planner real-TQ CPU contract

`test_tpr_dp_real_tq_contract.py` loads the actual recorded 8-row
`tq_batch.pt`. It asserts exact row coverage, logical loss ownership,
unchanged existing `ForestExecutionPlan` execution cost, duplication
and equal-row eligibility. DTA minimax `enforce_equal_rows=False`
remains planning-only when partitions are uneven. This test does NOT
enable the controller/worker DP dispatch.

### Gate B2 — DP2 (TP1) existing module F/B on real NPU2

`test_tpr_dp_tpdp_module_npu.py` with `TPR_MODULE_TP_SIZE=1`:

- Real pretrained Qwen3-1.7B and the recorded real TQ tokens, not tiny
  random GPT or generated model inputs.
- Two DP replicas receive disjoint pairs of real TQ rows, each with
  a common 128-token prefix, then 32 tokens per leaf. Both run local
  existing `SegmentExecutor` / `FixedTopologyScheduler` and a
  separate full-trajectory same-checkpoint reference.
- Check local CE loss, local parameter grads (current 5% relative L2
  gate), native DP group membership, and unequal DP execution-tree
  row ownership. **No production Trainer, no native DDP optimizer sync.**

### Gate C — TP2×DP2 module F/B on real NPU4

Same test file with `TPR_MODULE_TP_SIZE=2`:

- Each DP replica's **two TP peers** receive identical real TQ tokens
  and identical SegmentPlan digest; its Megatron native TP projection/
  vocab collectives run inside its own TP group.
- **Different DP replicas** receive disjoint rows, independent Prefix
  KV/gradient lifetime and disjoint original row IDs. Token-identical
  trees can exist in different DP groups; never require distinct tree
  digests across all 4 world ranks, or move Prefix KV across DP.
- Check local CE and gradients against native independent full
  trajectories on the same TP-sharded real checkpoint.
- Test-only DP ProcessGroup handshake exchanges a small hash. No
  extra model/TP/DP kernel or production distributed gradient backend.
- **This is the TPR module-level TP×DP functional gate, NOT production
  mixed-parallel training parity or distributed optimizer integration.**

## Commands (always localhost + explicit port, never --standalone)

Host bridge branch:
```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/bridge
mkdir -p tmp/tpr-pr5
git fetch origin feat/tpr-tp2-dp-placement-audit
git archive origin/feat/tpr-tp2-dp-placement-audit \
  tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py \
  tests/models/mcore/tpr/parallel/_qwen3_tp_checkpoint.py \
  tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py \
  tests/models/mcore/tpr/integration/test_tpr_dp_real_tq_contract.py \
  | tar -x -C tmp/tpr-pr5
```

Docker (sync only these test files, with backups; no TPR/VERL production file):
```bash
cd /workspace/uni-agent/verl
for file in \
  tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py \
  tests/models/mcore/tpr/parallel/_qwen3_tp_checkpoint.py \
  tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py \
  tests/models/mcore/tpr/integration/test_tpr_dp_real_tq_contract.py
do
  mkdir -p "$(dirname "$file")"
  rsync -av --backup --suffix=.before-parallel-acceptance \
    "/workspace/bridge/tmp/tpr-pr5/$file" "$file"
done

# Gate A: run first; -k isolates the new test from previously passed ones.
TPR_RUN_TP2=1 TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B \
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29531 \
  -m pytest -vv -s tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py \
  -k test_tp2_real_qwen_engine_phase4_route_two_updates

# Gate B1: real TQ planning and native Forest contract.
python -m pytest -vv -s \
  tests/models/mcore/tpr/integration/test_tpr_dp_real_tq_contract.py

# Gate B2: 2 ranks, DP2/TP1 module F/B.
TPR_RUN_MODULE_DP=1 TPR_MODULE_TP_SIZE=1 \
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29532 \
  -m pytest -vv -s \
  tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py

# Gate C: 4 ranks, DP2/TP2 module F/B.
TPR_RUN_MODULE_DP=1 TPR_MODULE_TP_SIZE=2 \
torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29534 \
  -m pytest -vv -s \
  tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py
```

## Stop line before production E2E

- Do NOT disable the production `DP=1` guard simply because Gate B/C
  passes. Gate B/C bypass the high-level TPR PPO/VERL DP dispatcher
  intentionally. Genuine DP integration additionally needs controller
  placement, group-aware actor minibatches, global token normalization,
  per-segment VERL loss semantics, native DDP/optimizer finalization,
  and synchronized optimizer steps across DP replicas.
- On the recorded 8-rollout single-UID TQ, DP2 partitions may split
  a shared Prefix even with optimal scheduling; this duplicates computation
  but does not make the CE math wrong.
- Keep genuine UniAgent/VERL rollout-to-optimizer E2E **for after**
  the current single-DP baseline and BF16/old-vs-new logprob diagnosis
  stabilize in the other development thread. Do not touch existing
  E2E scripts, old/new logprobs, advantages, PPO clip ratio, or GEMM
  precision treatments from this branch.

## 2026-10-10 actual TP2 Engine smoke — token normalization root cause

First real two-rank TP2 Engine-route execution reached VERL's native
`batch_num_tokens` metadata and the TPR adapter. The test initially failed:

```
AssertionError: native Engine global token count mismatch: 128
assert 128 == (2 * 32)
```

The harness had incorrectly bound
`engine.get_data_parallel_group = lambda: None`.
In `torch.distributed.all_reduce`, `group=None` means **the world
group**, not a singleton/no-op: the TP2 peers both contribute the same
logical 64 supervised response tokens, producing 128.
`DP=1` must use its **real Megatron singleton DP group**. Corrected
`test_tpr_tp2_npu.py::_tpr_ppo_nll_run` to call
`parallel_state.get_data_parallel_group()` and to assert
`dist.get_world_size(group=dp_group)==1` before passing that group
through `engine.get_data_parallel_group`.
The test still insists on `batch_num_tokens == 64`; it must **not**
accept 128, because that would silently halve the NLL gradient scaling.

This is solely a test-double Engine group-wiring error, not evidence
that VERL's real MegatronEngine implements a world-group token reduction.
The correction has been committed; **NPU rerun pending**.


## 2026-10-10 NPU results — TP2 Engine and real-TQ DP planning

User's accepted output:

```
TPR_TP2_ENGINE_ROUTE status=PASS dispatch_hits=2 optimizer_steps=2
losses=[12.637715339660645, 7.607739448547363]
PASSED / PASSED

TPR_DP_REAL_TQ_PLACEMENT status=PASS rows=8 dp=2
per_dp_rows=[5,3] tree_costs=[59767,44228]
global_tree_tokens=86402 duplication=17593 equal_rows=False
trainer_dispatch=NOT_WIRED
PASSED
```

Thus A (Engine route, two controlled SGD steps) and B1 (real-TQ
DP2 offline placement) are PASS. B1's 5/3 rows violate VERL's
current equal-cardinality dispatch; do **not** wire this plan into
Trainer as-is. If later enabling real DP2, use a balanced native-UID
policy or a compatible equal-row/mini-batch policy.

First B2 isolated DP2 module attempt raised:

```
ValueError: cp_group must contain more than one rank, got 1
```

This was a **test harness only** issue: the new module test passed
Megatron's CP=1 singleton ProcessGroup to `SegmentExecutor`, but its
`cp_group` argument enables distributed CP and must be `None`
for CP=1. Fixed the test to assert the native CP group has size 1
while **passing `cp_group=None` to SegmentExecutor**. The production
executor's explicit CP group contract remains unchanged. B2 NPU
rerun pending; C (TP2×DP2) awaits B2 PASS.


## 2026-10-10 DP2 / TP2xDP2 first NPU runs: thresholded PASS, data diversity caveat

User's **first** real Qwen3-1.7B + real-TQ tests:

```
TPR_MODULE_PARALLEL status=PASS tp=1 dp=2 dp_rank=1
 real_tq_rows=(4,5) native_ce=5.7495232 tpr_ce=5.7314186
 grad_rel_l2=0.0445543
TPR_MODULE_PARALLEL status=PASS tp=1 dp=2 dp_rank=0
 real_tq_rows=(0,1) native_ce=5.7495232 tpr_ce=5.7314186
 grad_rel_l2=0.0445543

TPR_MODULE_PARALLEL status=PASS tp=2 dp=2 dp_rank=1
 real_tq_rows=(4,5) native_ce=5.7632203 tpr_ce=5.7735643
 grad_rel_l2=0.0322185
TPR_MODULE_PARALLEL status=PASS tp=2 dp=2 dp_rank=0
 real_tq_rows=(0,1) native_ce=5.7632203 tpr_ce=5.7735643
 grad_rel_l2=0.0322185
PASSED x4
```

Both CP1 standalone TPR module paths pass the current 5%-rel-L2
numerical gate. Important: the two DP replicas produced **exactly the same
loss and gradient-difference values**. Audit found the first test sliced
*only the first 128+32 tokens* of every chosen trajectory and confirmed
**only that different logical row IDs** were assigned, not that the
physical 160-token training windows differed. Real UniAgent TQ rows may
share those initial prompt tokens. Thus the first PASS validates separate
native DP process-group membership and local TP/TPR execution, **not a
demonstration of two different physical DP workloads**.

Committed strengthening, **NPU rerun needed**:
- On each DP replica, select from its disjoint real-TQ row range
  (`0..3` vs `4..7`) a pair of siblings with a **real first token
  divergence** (recorded full-token LCP) and >=128 shared prior tokens
  plus 32 post-divergence tokens.
- Use a genuine 160-token window around that actual first branch point,
  without invented token IDs or model parameters.
- Deterministically pick physically distinct fork windows across DP0/DP1
  and assert different SHA256 physical-token signatures using the native
  DP ProcessGroup. Fail explicitly if this recorded TQ does not contain
  qualifying physically distinct windows.
- Print `real_tq_rows`, `window_start`, `true_fork` and
  `physical_hash` as reproducible provenance. **This local window
  restarts position IDs from 0; it is an isolated short-context module
  test, NOT equivalence to full long-context original TQ training.**
- Keep existing 0.02 CE loss difference and 5% gradient-rel-L2
  gates unchanged; do not edit the BF16/PPO numerics.
- DP gradient synchronization and Trainer dispatch remain NOT_TESTED.

The existing 5/3 DTA offline placement plan is **not used by** this
TP/DP module test (which picks two local siblings per DP replica). It is
still not legal for native equal-row Trainer dispatch.


## 2026-10-10 heterogeneous real-TQ NPU result: DP2 passes, TP2×DP2 FAIL

User confirmed **TP1×DP2** physically disjoint real-token branches:
- DP0 rows (0,1): `window_start=17385`, `true_fork=17513`,
  hash `99804e90b786`, native CE 3.5415483, TPR 3.5555313,
  param grad rel-L2 **0.0406615** (<0.05, PASS).
- DP1 rows (4,5): `window_start=17361`, `true_fork=17489`,
  hash `43a848652718`, native CE 3.3814220, TPR 3.3785036,
  param grad rel-L2 **0.0410661** (<0.05, PASS).

So different DP replicas actually processed **different physical input
tokens**, as checked by recorded fork positions and hashes.

**TP2×DP2**, same heterogeneous TQ windows, is **NOT ACCEPTED**:
different TP ranks of DP1 produced two gradient relative L2
failures **0.0739336** and **0.0566149**, both strictly greater than
the existing 5% gate. Full four-rank result cannot be marked PASS.
We have **no grounded evidence yet** whether their excess error is
BF16 shape/GEMM sensitivity, local TP sharding/grad reduction, or a
specific layer/parameter problem. Do not weaken the threshold.

The later HCCL `hcclCommInitRootInfoConfig` connectivity timeout is
likely a **secondary collective-order failure**, not proof of a broken
physical cluster: the old test asserted rank-local gradient tolerance
**before** a native DP-group `all_gather` that surviving peers still
tried to execute. Early failure could strand the latter ranks waiting
for a communicator peer. A separate fresh run is still appropriate
if HCCL errors occur without early Python assertions.

**Test-only fix, no numerical arithmetic change:**
- Delay all loss / gradient threshold assertions until all native TP
  and DP communications finish and WORLD diagnostic metrics are
  exchanged. Use a common global failure decision on every rank.
- Record rank-local grad relative L2 and proper **TP-shard weighted**
  relative L2 (`sqrt(sum(error_sq)/sum(ref_sq))`) separately.
  Retain the original `relative_l2 < 0.05` **per-rank** acceptance gate.
  Aggregated TP metrics are diagnostics, NOT a replacement criterion.
- On failed ranks, print 8 largest per-parameter contributions to the
  total local error, with each parameter's own relative L2 and norm.
- Print a four-rank result matrix before the unified FAIL, so the other
  TP/DP ranks are not hidden by the first crashing rank.
- The test does NOT add TP/DP gradient synchronization, alter
  `SegmentExecutor`, change native VERL training, or touch PPO/GEMM.

Rerun **only TP2×DP2** after syncing
`tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py`.
If the same DP1 parameter groups dominate across both TP shards,
investigate numerical shape sensitivity there *separately* from the
ongoing single-DP E2E work. If failures track rank/shard ownership or
incorrectly synchronized replicated parameters, check Megatron TP2
reference and TPR gradient finalization contracts. Do not infer the
cause without the attribution output.


## 2026-10-10 repeatability and dedicated TP2×DP1 same-input control

The improved collective-safe TP2×DP2 test reproduced **exactly** the
same failure on global ranks 2 and 3, the two TP ranks of DP1:
`grad_rel_l2=0.0566149` (TP0), `0.0739336` (TP1);
both above the **unchanged 0.05 gate**.
Both saw **native CE 3.3734765 vs TPR CE 3.3696692**,
difference 0.0038073 (passes the 0.02 CE gate).
The unified failure path completed communication without a new HCCL
timeout, so the earlier HCCL event was plausibly a side effect of
premature rank-local assertions. This does **not** prove or disprove
a TP×DP implementation issue.

**Next isolating experiment** (test-only):
Run **exactly the same DP1 real-TQ fork window** (rows 4,5,
`true_fork=17489`, `window_start=17361`) with the original native TP2
group but **DP=1 (2 NPUs)**. New optional
`TPR_MODULE_DP_SIZE=1 TPR_MODULE_TQ_PAIR=1` reuses the same
`_reference`, `_tpr`, checkpoint/weights, segment plan, scalar
aggregation and rank-local 0.05 gradient gate. Only the number of
DP replica groups changes; the singleton DP group performs no
inter-replica ownership check.

```bash
TPR_RUN_MODULE_DP=1 TPR_MODULE_TP_SIZE=2 \
TPR_MODULE_DP_SIZE=1 TPR_MODULE_TQ_PAIR=1 \
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29536 \
  -m pytest -vv -s \
  tests/models/mcore/tpr/parallel/test_tpr_dp_tpdp_module_npu.py
```

- If TP2×DP1 on **identical real-token input** also reproduces
  the >5% gradient discrepancy, the excess error does **not require**
  DP2 topology and may be a TP2 + BF16 segmented-kernel numerical
  issue; inspect `TPR_MODULE_PARALLEL_DIAG top_contributors` to isolate
  parameters instead of weakening the threshold.
- If TP2×DP1 passes while TP2×DP2 reproducibly fails on the
  same input and checkpoint, inspect TP/DP group mapping, process
  initialization and rank-specific gradient assumptions; do not
  immediately blame numerical shape sensitivity.
- This **new control is not a full production E2E test** and carries no
  claim of any NPU result until actually run. Neither TP2×DP2 nor
  strict numerical parity is accepted yet.

The original `TPR_MODULE_DP_SIZE` default stays **2** for existing
two- and four-NPU DP2 tests.
