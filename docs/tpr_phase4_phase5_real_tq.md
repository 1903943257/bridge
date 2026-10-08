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

If the old CE patch was already applied, **do not reapply the whole patch**.
Use the focused incremental hunk instead, after checking compatibility:

```bash
git apply --check /path/to/bridge/patches/3_5_megatron_engine_phase4_incremental.patch
git apply /path/to/bridge/patches/3_5_megatron_engine_phase4_incremental.patch
```

If either check fails because your native file differs in the area around
`routed_num_tokens`, patch that location manually; don't discard local
modifications or force the patch. Verify:

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

## Real Qwen3-1.7B Phase-5 numerical gate

Use the **complete 8 recorded variable-length trajectories by default**.
No synthetic GPT proxy. Do **not** set `TPR_QWEN17_PPO_PROMPT` or
`TPR_QWEN17_PPO_RESPONSE` for the whole-TQ correctness test.

```bash
cd /workspace/uni-agent/verl
export TPR_RUN_QWEN17_PPO=1
export TPR_QWEN_PROFILE_SIZE=1.7B
export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
unset TPR_QWEN17_PPO_PROMPT TPR_QWEN17_PPO_RESPONSE

python -m pytest -x -vv -s --tb=long \\
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

This may exceed single-NPU memory because a native row can span more
than 40k tokens and the checkpoint vocabulary is 151,936. Memory failure
is a real capacity finding, **not** Phase-5 numerical correctness PASS.

For a *separate*, explicitly labeled cropped-real-data diagnostic:

```bash
export TPR_QWEN17_PPO_PROMPT=64
export TPR_QWEN17_PPO_RESPONSE=64
python -m pytest -x -vv -s --tb=long \\
  tests/models/mcore/tpr/correctness/test_qwen3_1_7b_real_tq_ppo_npu.py
```

Both windows are copied from recorded TQ samples, but cropping changes
causal context and tree topology and does **not** verify the full TQ.

Qwen3-1.7B normally has 40,960 maximum positions; the recorded TQ
contains a 41,029-token trajectory. The shared real-Qwen test fixture now
honors requested test length (RoPE extrapolation with unchanged theta),
and Phase5 reports actual maximum length.

`old_log_probs` in the diagnostic are recomputed with the frozen real
checkpoint. If the TQ dump lacks `advantages` (before actor update),
deterministic signed coefficients are used **solely** for probing the
mathematical gradient equivalence. They are *not* true rollout advantages.
To validate production PPO, capture `mini_batch_td` at actor update with
actual `old_log_probs`, `advantages`, `response_mask`, `loss_mask`,
temperature and stable trajectory identity.

The Phase5 test also sanitizes an empty-named field from MindSpeed's
`get_full_args()`, as is already done in the real-Qwen Ring profile.
If a dataclass error remains, save the **full** `--tb=long` traceback.

## Not yet validated

* No server NPU run has been performed by the GitHub edits themselves.
* The actual Qwen3-1.7B and native/TPR numerical gate may need NPU/runtime
  compatibility fixes once real logs are available.
* Full ~17k prompt trajectories may need LM-head chunking and activation
  memory work before they can fit; the crop is intentionally much smaller.
* CP>1, TP>1, DP>1 and full actor-update integration remain separate work.
