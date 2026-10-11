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


## P0-5 measured real-sibling NLL outcome (user Ascend, 2026-10-11)

**Real siblings [0,1] found**, same original last512 prompt token
prefix hash `36bc65deeefa00840c6a7201536451a4f3db3e3ad08d99f1682051bd097ed15a`,
response suffixes first diverge at **offset 30** (so the actual
maximal common prefix is at least 512+30=542 tokens, while this fixed
P512 experiment deliberately only reuses 512). There are **966
supervised response tokens** across two real branches.

Original rollout fields: `has_actual_advantages=False`,
`has_recorded_old_logprobs=False`. Therefore **real PPO replay
was not run**. The user's attempted `OBJECTIVE=ppo` correctly
raised `REAL PPO FIELDS MISSING`. This is an expected DATA
failure, not a TPR optimizer/attention/gradient failure.

P0-5 NLL **20 actual FP32-master AdamW steps** completed for
independent Full(2x1024) and TPR Push(512)/Visit(512)x2/Pop(512).
All 28 layer Prefix dKV updates were observed once per sibling
and sampled BF16 accumulations matched their contributions
exactly (`prefix_kv_sum_verified=True` at each reported step).

| Step | Common-Full NLL Full | Common-Full NLL TPR | TPR-Full NLL | Sampled master update rel L2 |
|---:|---:|---:|---:|---:|
| 1 | 0.602285936 | 0.606591672 | +0.004305735 | 0.328631988 |
| 2 | 0.360511988 | 0.363266543 | +0.002754554 | 0.244783621 |
| 5 | 0.067441724 | 0.076127026 | +0.008685302 | 0.211572911 |
| 10 | 0.022494603 | 0.030985202 | +0.008490599 | 0.241915577 |
| 20 | **0.011431768** | **0.008580486** | **-0.002851283** | **0.400043659** |

The two policies successfully trained to very low NLL on the
**same two repeatedly replayed training trajectories**, not a
trajectory-disjoint validation set. This is a convergence/schedule
smoke rather than a generalization claim.

The NLL agreement does NOT guarantee logprob tail or gradient
equivalence. For example, step10 Full-oracle logprob absolute
max error 12.44 but p95 only 0.00201: a few extreme token
outliers dominate an otherwise small upper-95-percentile error.
At step20 response MAE 0.0031478, max 1.9990 and p95
0.0010908. The sampled master update rel L2 grows to 40%;
preclip gradient rel L2 step20 0.9381, cosine 0.3492,
but at that point the two trained parameter states differ
and most training token losses are very small. Do NOT read
that as 94% same-parameter TPR backward error.

**Bounded P0-5 acceptance:** original real sibling extraction,
the 28-layer graph-free KV relay accumulation count/sampled
sum, and 20 NLL updates all succeed. Strict Full equivalence,
real PPO, multi-level maximal trie, heldout generalization and
actual runtime speedup are separate open gates.

## Actual actor-update mini-batch capture, not rollout TQ

The repository **already contains** the opt-in capture helper
`verl/models/mcore/tpr/actor_capture.py`, the worker insertion
helper `patches/apply_tpr_actor_capture.py`, and contract
`tests/models/mcore/tpr/integration/test_real_actor_update_minibatch_contract.py`.
Do not attempt to populate PPO fields on the old rollout dump;
capture the later **actual actor-update mini_batch_td** containing
advantages, old_log_probs, response_mask and temperature.

On the Docker checkout used to **run actual UniAgent training**:

```bash
# Check whether the capture patch is installed in this VERL worker source:
python /workspace/bridge/patches/apply_tpr_actor_capture.py --check \
  /workspace/uni-agent/verl/verl/workers/engine_workers.py

# Only if the check reports "patched" rather than "already_applied",
# apply it without --check:
python /workspace/bridge/patches/apply_tpr_actor_capture.py \
  /workspace/uni-agent/verl/verl/workers/engine_workers.py

# Make this variable available to the ACTOR WORKER when launching the
# existing, real UniAgent RL actor-update command:
export TPR_CAPTURE_ACTOR_MINIBATCH_DIR=/tmp/tpr_actor_real
```

Run a normal real rollout -> advantage processing -> actor update
so `TrainingWorker.train_mini_batch` is reached. The capture helper
writes `actor_update_rank<rank>_batch<batch_idx>.pt`.
The patch is opt-in; if the environment variable is absent, it
does nothing. It does not synthesize old logprobs or advantages.

Once a fresh file exists, validate its **actor-update contract**
before attempting an NPU PPO replay:

```bash
export TPR_REAL_ACTOR_MINIBATCH=/tmp/tpr_actor_real/actor_update_rank0_batch0.pt
TPR_RUN_REAL_ACTOR_MINIBATCH_CONTRACT=1 \
pytest -q -s tests/models/mcore/tpr/integration/test_real_actor_update_minibatch_contract.py

# CPU preflight, including authentic siblings and source PPO fields:
python tests/models/mcore/tpr/parallel/inspect_real_tq_siblings.py \
  "$TPR_REAL_ACTOR_MINIBATCH" --siblings 2 --require-ppo

# Only after BOTH checks pass (P0-5 strict recorded-field PPO replay):
TPR_P0_BRANCH_OBJECTIVE=ppo \
TPR_REAL_ACTOR_MINIBATCH="$TPR_REAL_ACTOR_MINIBATCH" \
TPR_P0_BRANCH_STEPS=1 \
  bash tests/models/mcore/tpr/parallel/run_real_tq_multisibling_prefix.sh
```

The P0-5 runner automatically uses `TPR_REAL_ACTOR_MINIBATCH`
as its PPO input and runs the **CPU preflight before model load**.
If it cannot find 2 genuinely different response suffixes with the
same 512 prompt tokens in this particular actor minibatch,
it fails with a data-eligibility error. This is not a TPR failure;
capture a suitable complete original actor update batch instead
of manufacturing branches.

**P0-5 actual-PPO-field replay restrictions**: requires provenance
marker `capture_stage=actor_update_mini_batch`, original unique
trajectory keys, aligned loss_mask/response_mask and true original
old_log_probs/advantages; current implementation only supports
temperature exactly `1.0`. A Qwen3-8B rollout/actor capture
used to replay Qwen3-1.7B still has an actor-model mismatch
and cannot be claimed as on-policy PPO training.
