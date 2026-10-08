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

When the cropped real-TQ numerical check fails, do **not** loosen
`rtol=0.02, atol=0.2` automatically. The Phase5 test now prints:

- `NATIVE REPEATABILITY`: same Native weights/tokens, no-grad versus backward-enabled
  forward; estimates the numerical baseline noise floor.
- `TPR LOGPROB DIAG`: per-real-row and root/nonroot mismatch counts, followed
  by the worst 20 logical tokens with absolute query coordinates, segment
  span, Native logprob, TPR logprob, and exact tolerance.
- `TPR MODEL NATIVE-FORWARD`: *same TPR checkpoint* in non-tree
  SelfAttention mode versus Native reference. This isolates checkpoint,
  model-spec, or ordinary-forward differences.
- `WHOLE-SEGMENT TPR`: same TPR model and attention kernel with an
  unsplit single physical segment for the worst mismatched real row; compare
  its error with the Forest error to diagnose decomposition/shape effects.
- PPO scalar loss and sampled parameter gradient relative-L2 and cosine
  are reported even if the logprob gate fails.

### Native physical-segment cutoff oracle (Phase-5 accuracy)

The 64+64 real-TQ experiment confirmed two positions at which the Native
**truncated-prefix** result exactly equals TPR, while Native full 128
differs:

| Absolute query | Native full 128 | Native cutoff | TPR forest | Cutoff length |
| --- | ---: | ---: | ---: | ---: |
| 69 | -4.79739285 | -4.78902054 | -4.78902054 | 70 |
| 76 | -1.70170808 | -2.12716579 | -2.12716579 | 94 |

These positions prove that TPR-specific KV/gradient mistakes are **not
required** to produce the observed divergence. They do not prove every
logical token and every branch is correct.

To extend the causal control to **every real TQ objective reference**:

```bash
export TPR_QWEN17_PPO_SEGMENT_ORACLE=1
```

For each compressed-tree physical Segment the Phase-5 test now performs
a Native forward on the original real tokens up to the Segment end and
scores all of that Segment's original logical targets, including
different child targets for a shared parent query. It prints
`NATIVE FULL vs NATIVE PER-SEGMENT CUTOFF` and
`TPR FOREST vs NATIVE PER-SEGMENT CUTOFF`, and per-Segment max errors.

If `TPR FOREST vs NATIVE PER-SEGMENT CUTOFF` is small for all tokens,
the large original Native-Full/Forest discrepancy is predominantly the
precision-dependent sequence-shape effect. If not, pursue the residual
TPR-specific discrepancy in attention/KV. This cutoff oracle is only
for short **real-token debug runs** (<=256 total tokens) and will skip
full-length real TQ to avoid reproducing its observed OOM. It does not
relax the strict original Native-full gate.

### Root/branch BF16 numerical diagnosis

For the Qwen3-1.7B cropped-real-TQ case, current evidence includes:

- Same checkpoint Native / TPR module in **context-free** mode: logprob max diff zero.
- One **unsplit** TPR segment of length 128: logprob max diff zero.
- 70-token root and smaller rectangular queries: layer 1 matches at
  selected queries, layer 2 begins to differ by BF16-size units, later
  layers amplify the difference.
- A 0.425 logprob drop can move the PPO ratio to approximately 0.65,
  changing clipping. Therefore matching PPO gradients cannot be achieved
  merely by making the loss aggregation or gradient relay correct.

**Causal truncation control (opt-in diagnostic enabled by default with
layer tracing):**

```bash
export TPR_QWEN17_PPO_TRACE_LAYERS=1
export TPR_QWEN17_PPO_NATIVE_CUTOFF=1
```

The NPU test compares at the exact *real* row-0 token positions:

1. Native full 128-token forward (the canonical reference);
2. Native forward truncated to the real segment's `position_end`
   (e.g. 70 for query 69 and 94 for query 76);
3. TPR single, **unsplit** segment with exactly the same cutoff length;
4. TPR Forest's segmented Prefix-KV execution.

Look for `NATIVE TRUNCATION LOGPROB` and
`TPR SINGLE-CUTOFF LOGPROB` alongside the already reported Forest
per-token error. If Native itself changes substantially with causal
sequence length, the underlying BF16 kernel shape sensitivity must be
quantified before attributing the drift to a tree-specific bug. If Native
is invariant at that cutoff but Forest differs, investigate rectangular
FA, post-RoPE K/V and ancestor-prefix state. This probe changes **no**
model arithmetic, inference defaults, or pass/fail tolerance.

When the 64+64 TQ crop gives repeated mismatches in nonroot Segment 1
(e.g. real absolute query 76) while whole-segment TPR is bit-identical,
the next gate is **attention-layer-local**. Set
`TPR_QWEN17_PPO_TRACE_LAYERS=1` (default) to compare the actual real row-0
queries at response offsets 6 and 13 across all Qwen layers. For each layer
the test reports the relative L2 and maximum absolute differences of
`self_attention` **input and output**. It maps physical Segment ownership
from `ForestExecutionPlan`, not from hard-coded node IDs; sibling branches
at the same absolute position cannot contaminate the trace.

- First-layer input mismatch: investigate embeddings, sequence positions,
  Qwen model/row mismatch.
- First-layer input matches but output differs: investigate TPR attention
  projection, RoPE, rectangular FA and NPU shape sensitivity.
- First-layer output close but drift grows through later layers: quantify
  BF16 shape-dependent accumulation before considering tolerances.

Also, this diagnostic's PPO advantages are intentionally signed (+1.0/-0.5)
rather than an exact +1/-1 cancellation: previously native PPO loss was 0
by construction and hid loss-relative deviation. This is a mathematical
probe, not real rollout advantage data.

The numerical gate remains strict; none of these diagnostics changes
acceptance thresholds or marks mismatches as passing.

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
