# P0-5 — Real SWE siblings, shared Prefix KV + backward (2026-10-11)

**State: test/runner/comparator committed to bridge/main. User Ascend
measurement pending. Production TPR/MindSpeed code remains unchanged.**

## Decision after P0-4

The P0-4 Full-vs-TPR 50-step BF16 training comparison established that
both complete 50 actual FP32-master AdamW steps on matched real masked-NLL
windows. Both have similar heldout Full-oracle NLL (Full 0.613095485,
TPR 0.617147543 at step50); sampled FP32-master cumulative update
vector relative L2 is **0.24937**. This permits *temporarily removing
strict FP/bfloat16 bitwise parity as an engineering blocker*, but does
**not** make the sampled gradients or parameter updates numerically
identical. Keep monitoring token tails and optimizer drift.

## Why real trajectory data is not automatically real PPO

The real `tq_batch.pt` guarantees real agent SWE trajectories,
true token IDs, prompt/response boundaries and response_mask.

**True actor PPO update replay additionally needs:**
- Original, actual behavior-policy `old_log_probs` aligned to each
  original response token (not arbitrary zeros or current-model
  recomputation presented as the rollout old policy).
- Original actor-computed `advantages` aligned to each response
  token. Never substitute alternating +1 / -0.5 test coefficients.
- Original policy/checkpoint provenance, loss/clip, temperature,
  advantages/returns/normalization, KL handling, and sampling
  semantics for an on-policy training-equivalence claim.

The older real-TQ correctness loader explicitly **generates probe
advantages if they are missing**, and writes zero placeholder
old_log_probs before recomputing a reference. That route is a
*mathematical PPO-style diagnostic*, not authentic recorded PPO
optimization. Furthermore, the SWE TQ path name refers to Qwen3-8B
while this probe uses pretrained Qwen3-1.7B; even matching old and
advantage tensors from the dump would be a fixed-objective PPO
**replay** with unverified/mismatched behavior-policy provenance,
not automatically on-policy PPO training.

The new P0-5 test defaults to **real masked NLL**, and an explicitly
selected `TPR_P0_BRANCH_OBJECTIVE=ppo` refuses to run unless the
unmodified original TQ dump has **both** `old_log_probs` (or
`old_logprobs`) and `advantages`, matching the full response lengths
and containing finite values. It does NOT fabricate either field or
pretend that a missing old policy is available.

With both fields present, it implements fixed-data clipped vanilla
PPO surrogate `-minimum(ratio*adv, clamp(ratio,1±0.2)*adv)`
normalized over all real supervised tokens. It is a controlled
**recorded-field PPO objective replay**, not the full VERL actor
update (which can include other loss modes, KL, entropy, and metadata).

## Authentic multi-sibling workload

From the real dump, find 2 or 3 **different original** trajectory
rows with identical *last 512 prompt tokens*, each at least
512 real response tokens, supervised response[0] and supervised
response[1..511]. Each row's `input_ids == prompt + response`
must hold. The first 512 response tokens must NOT be identical
across the selected rows; prefer branches that diverge at response[0].
Require same source prefix **bitwise**, do not substitute a synthetic
prefix, paste independent prompts, or create fake response samples.

Cropping is transparent: this test uses the **actual final 512
prompt tokens plus the actual first 512 response tokens** of
each original row, with local RoPE positions 0..1023. It
does not claim to reproduce the complete original 16k+
position context or exact production actor logits.

The schedule is:

```text
FULL:
  independent Full(P512+S512, sibling A) backward
  independent Full(P512+S512, sibling B) backward
  -> sum parameter.grad -> FP32-master AdamW step

TPR:
  Push(P512)                 # exactly once, shared
     Visit(A S512)           # backward, dK_A/dV_A into PrefixState
     Visit(B S512)           # backward, dK_B/dV_B into PrefixState
     [verify per-layer observed BF16 accumulated dK/dV samples
      equal exactly-one per-child contributions]
  Pop(P512)                  # one replay, root's own fork-token loss
  -> FP32-master AdamW step
```

Every sample's first response token is predicted by **the same
root-last logit** but with its *own actual* target and, in PPO
mode, its own old-logprob and advantage; do not collapse duplicate
fork query owners into one loss. Child S512 query0..510
predicts response offsets 1..511. The denominator is the sum
of original true response_mask counts across all siblings.

Tests audit each of the 28 Prefix layer accumulation calls per
leaf (no duplicate/missing calls); verify 32 deterministic
BF16 dK and dV elements per layer against the sequential sum
of actual contributions; then verify 28 gradient roots relayed
at the single root Pop. This is a **bounded numerical witness**,
not a full all-elements independent autograd derivative proof.

The Full comparison uses a separate unsegmented full causal pass
per true sibling with the same Qwen weights and objective, and
one shared FP32-master AdamW update. It records before-clip
sampled parameter gradients, per-trajectory response logprobs,
clipped-loss objective and sampled optimizer weight updates.

Both trained models are evaluated through the SAME unsegmented
Full-M1024 backend; Step0 original sampled FP32/BF16 weights
and shared-eval logits must match exactly. Step1 comparison
is a same-initial-weight full-vs-tree numerical control; later
steps incorporate differing policy histories. Default steps=1;
optional 2..20 fixed-data replay.

## Run

**Host** (only bridge checkout receives `git pull`):

```bash
cd /workspace/bridge
git pull --ff-only
```

**Inside Ascend Docker:**

```bash
rsync -av /workspace/bridge/tests/models/mcore/tpr/ /workspace/uni-agent/verl/tests/models/mcore/tpr/
cd /workspace/uni-agent/verl

# No export needed if the same default TQ file exists;
# otherwise use the original, unchanged TQ dump:
# export TPR_REAL_TQ_BATCH=/path/to/tq_batch.pt

# Cheap CPU-only eligibility / real-PPO-field check (no Qwen init):
python tests/models/mcore/tpr/parallel/inspect_real_tq_siblings.py \
  "${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"

# First: NLL multi-branch one-step gradient + one real AdamW update:
TPR_P0_BRANCH_STEPS=1 \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh

# Optional NLL multiple updates (same sibling group and fixed real trajectories):
TPR_P0_BRANCH_STEPS=5 \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh

# Optional third REAL sibling if the data truly contains 3 distinct suffixes:
TPR_P0_BRANCH_SIBLINGS=3 \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh

# Only if ORIGINAL raw TQ contains BOTH old_log_probs and advantages:
TPR_P0_BRANCH_OBJECTIVE=ppo \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh
```

If the real dump cannot supply 2 or 3 different genuine
response branches with the same exact prompt-tail512 prefix,
the test **fails explicitly**. This does NOT mean TPR has
failed; it means the selected real TQ evidence cannot support
that fixed-shape branch probe.

Both Full and TPR are single-NPU, sequential `torchrun` processes
with explicit `127.0.0.1` ports 29841/29842 (no `--standalone`).
Artifacts saved under `/tmp/tpr_p0_real_siblings`, named for
objective, mode, sibling count and optimizer step count.

To run just the failed TPR half without overwriting Full:
```bash
TPR_P0_BRANCH_RUN_MODES=tpr \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh
```

## What is accepted vs pending

P0-5 completes only when real sibling token/prompt hashes agree,
at least two suffixes differ, all supervised owners are accounted
for, the measured root is pushed and popped exactly once, each
sibling contributes exactly once to every Prefix KV gradient,
and the Full vs TPR comparator reports complete single-step
gradient and sampled optimizer update metrics.

A successful test means **multi-sibling scheduling/gradient
accounting control completed**, not strict Full/TPR numerical parity.
Judge sampled gradient rel L2/cosine and shared-Full evaluation NLL
and response-logprob outliers explicitly before upgrading status.

Future: use the actual full 8-trajectory tree from
`build_tree_execution_plans` and a genuine captured actor-update
batch with verified old-policy provenance, advantages and VERL PPO
configuration to establish training parity on real multi-level trees.
