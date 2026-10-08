# TPR Phase 4/5: VERL native PPO and real Qwen3-1.7B

## Supported / deliberately unsupported

* Training target: **real Qwen3-1.7B checkpoint**, no synthetic model.
* Trajectory source: the recorded 8-row django11163 `tq_batch.pt`, not fabricated token sequences.
* Phase-4 PPO: vanilla, token-mean, TP=PP=CP=EP=DP=1.
* Fused LM head, Megatron native recompute, router replay, MoE, and MTP are not enabled for this first integration.
* The old CE request path is retained as a regression oracle; it is not the production PPO entry.
* This is a correctness gate, **not yet a full Uni-Agent RL rollout-to-optimizer loop**.

## Engine integration contract

`MegatronEngine.forward_backward_batch` first attaches global `batch_num_tokens`
and `dp_size`. Before `prepare_micro_batches`, training batches with
`tpr_enabled` dispatch to:

```python
run_tpr_forward_backward_batch(engine, data, loss_function, forward_only=False)
```

The adapter resolves `tpr_trajectory_keys` (preferred) or a per-row `uid`,
builds `ForestExecutionPlan`, binds each tree's PPO objective, and runs one
`FixedTopologyScheduler` per tree. The executor performs Push/Visit/Pop
and forwards prefix gradients. Native `finalize_model_grads_func` runs **once**
per complete mini-batch; `BaseEngine.train_batch` retains zero_grad and the
optimizer step. **There is no artificial factor of physical segment count**.

Important: to obtain prefix reuse across the recorded 8 trajectories, the actor
update mini-batch must contain them together. The `GBS1_N8` directory label
alone does not prove this; inspect actor `mini_batch_td` and the actual keys.

## Apply integration to an existing VERL checkout

`bridge` is the source repository for the added TPR files and
`patches/3_5_megatron_engine.patch`. Do not run `git pull` inside
`/workspace/uni-agent/verl` based on these instructions.

Ensure the target VERL checkout already includes the **latest** TPR modules.
For an unpatched checkout, inspect:

```bash
cd /workspace/uni-agent/verl
git apply --check /path/to/bridge/patches/3_5_megatron_engine.patch
```

If the old CE patch was already applied, `git apply` of the whole updated patch
will likely fail: add only the new PPO dispatch hunk after global token
metadata. Do not reinstall the earlier changes. Verify:

```bash
grep -n 'run_tpr_forward_backward_batch' \
  verl/workers/engine/megatron/transformer_impl.py
```

## Real TQ CPU Phase-4 tests

```bash
python -m pytest -s -q \
  tests/models/mcore/tpr/integration/test_real_tq_trajectory_tree.py \
  tests/models/mcore/tpr/integration/test_real_tq_tree_plan_builder.py \
  tests/models/mcore/tpr/integration/test_real_tq_phase4_training_keys.py
```

If the dump lives elsewhere, set `TPR_REAL_TQ_BATCH` to its absolute path.
These tests require the real dump and skip if absent; no synthetic substitution.

## Real Qwen3-1.7B Phase-5 numerical test

```bash
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
export TPR_QWEN17_PPO_PROMPT=64
export TPR_QWEN17_PPO_RESPONSE=64

python -m pytest -s -q \
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

The test loads real Qwen3-1.7B weights and checks the actual model parameter
count. It uses authentic token IDs from the TQ trajectories and compares
native row-wise forward/PPO backward with the patched Engine's TPR forest.
To keep the first numerical test affordable on one NPU, it retains the final
64 real prompt tokens and the first 64 real response tokens per row. This
is a **cropped real-data test**, not an exact full-length trajectory run.

`old_log_probs` in the first numerical gate are recomputed with the **actual
frozen checkpoint**. If the TQ dump lacks `advantages` (it usually predates
actor update), the test uses deterministic nonzero probe coefficients to
expose wrong gradients. **Those are not actual RL advantages** and the test
must not be described as end-to-end RL correctness.

For the final production gate, capture `mini_batch_td` at
`TrainingWorker.train_mini_batch`, including actual `old_log_probs`,
`advantages`, `response_mask`, `loss_mask`, `temperature`, and stable
trajectory identities. Then validate the unmodified actor update loss,
gradients, optimizer step and weight sync. Keep both TQ and actor-update
captures as distinct fixtures.

## Not yet validated

* No server NPU run has been performed by the GitHub edits themselves.
* The actual Qwen3-1.7B and native/TPR numerical gate may need NPU/runtime
  compatibility fixes once real logs are available.
* Full ~17k prompt trajectories may need LM-head chunking and activation
  memory work before they can fit; the crop is intentionally much smaller.
* CP>1, TP>1, DP>1 and full actor-update integration remain separate work.
