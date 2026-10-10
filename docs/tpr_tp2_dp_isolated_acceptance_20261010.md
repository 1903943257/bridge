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
