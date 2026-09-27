# TPR Phase C1: CP1 Prefix backward graph offload

Status: implementation acceptance harness.  Phase C1 is deliberately limited
to Dense Full Attention with CP=TP=PP=EP=1.  It does not claim Ring CP2/CP4,
GDN/Hybrid, Prefix KV/dKV offload, optimizer/gradient offload, native
checkpoint/recompute compatibility, or a new prefetch policy.

The development checkout can only perform static and dependency-free checks.
No NPU result is claimed by this document yet: correctness, transfer direction,
forward counts, memory stability and latency remain pending execution on the
target Ascend server.

## Behavior under test

The `recompute` control retains the Phase B lifecycle:

```text
Push no-grad forward
Visit differentiable forward/backward
Pop differentiable Prefix forward/backward
```

The `offload` policy instead owns one retained native session per active
Prefix:

```text
Push with-grad forward + loss/KV roots
  -> native saved activations D2H
  -> detached compact KV enters KVStack
Visit uses only detached Prefix state
Pop restores saved activations
  -> original Push graph backward
  -> graph/session/pinned payload release
```

The implementation continues to use MindSpeed `SwapPrefetch` tensor selection,
pinned buffers, transfer stream and unpack callbacks.  Prefix graph records add
cross-Push/Pop ownership only.  Prefix KV and accumulated dKV remain resident on
NPU.  Transformer dropout is fixed to zero for comparison with recomputation.

`offload` requires native activation swap to be enabled and fails closed
otherwise.  A Push-to-Pop interval must not contain an optimizer step, an
in-place parameter mutation, a parameter-storage replacement, or a train/eval
mode change.

The formal engine entry selects the policy with:

```yaml
tpr_enabled: true
tpr_prefix_backward_policy: offload
```

The default remains `recompute`; the policy is an engine execution setting and
is not injected into Megatron's `TransformerConfig`.

## C1 acceptance coverage

`test_activation_offload_phase_c_npu.py` builds the real configured Qwen3
checkpoint and runs the same weights and plan with `recompute` and `offload`.
It keeps the original `rtol=2e-3`, `atol=2e-4` strict gate for:

- summed normalized loss;
- every owned per-term logprob;
- per-layer Prefix key/value dKV at every non-leaf Pop boundary.

Parameter gradients use the existing Phase B per-tensor repeat-noise gate from
`_offload_b_gate.py`. Two extra `recompute` repeats calibrate that baseline;
two `offload` runs and a final `recompute_after` run are then judged against
the recompute-only baseline. Offload results never contribute to calibration,
and no Phase C-specific tolerance is introduced.

There are two topology cases:

- `flat`: one Prefix with two sibling leaves;
- `nested`: root Prefix, child Prefix with two leaves, and a root sibling leaf.

The nested case verifies that a child Pop uses its original parent anchors and
relays their gradients into the parent Prefix before the parent Pop.  Both
policies run the real native fused CE path.  The test does not use a synthetic
or GDN model.

Each Push/Visit/Pop emits one `TPR_PHASE_C_STAGE` JSON row and enforces:

| Policy | Stage | model forward | D2H | H2D |
|---|---|---:|---:|---:|
| recompute | Push | 1 | 0 | 0 |
| recompute | Visit | 1 | positive | positive |
| recompute | Pop | 1 | positive | positive |
| offload | Push | 1 | positive | 0 |
| offload | Visit | 1 | positive | positive |
| offload | Pop | 0 | 0 | positive |

For a single Prefix with `N` sibling leaves, the complete-model forward count
must be `N + 2` under `recompute` and `N + 1` under `offload`.  Counting the GPT
model entry separately avoids misreporting a possible LM-head checkpoint replay
as a complete Prefix forward.

The lifecycle case warms native/pinned allocators, then runs three more trees.
After every tree it checks the executor graph store is empty, no observed pinned
payload tensor remains live, and settled NPU allocated/reserved memory, process
RSS and `VmPin` stay within bounded drift.  This is a leak detector rather than
a capacity benchmark; Phase C capacity/latency measurements should be added
only after C1 correctness closes.

## Run C1 correctness

Run from the VERL repository root in a fresh one-rank process:

```sh
TPR_RUN_OFFLOAD_C=1 \
TPR_QWEN_PROFILE_SIZE=1.7B \
TPR_QWEN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B \
TPR_SWAP_MODULES=self_attention,mlp \
torchrun --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29566 \
  -m pytest -x -s -v \
  tests/models/mcore/tpr/profiling/test_activation_offload_phase_c_npu.py \
  -k correctness
```

The default correctness shape is P1K/S1K.  Override it without editing the
test using `TPR_PHASE_C_PREFIX` and `TPR_PHASE_C_SUFFIX`.

## Run parameter-gradient repeat diagnostics

If strict correctness reaches the retained Pop backward but only parameter
gradients differ, keep the production implementation unchanged and characterize
repeat noise first:

```
TPR_RUN_OFFLOAD_C=1 \
TPR_PHASE_C_GRAD_DIAGNOSTIC=1 \
TPR_QWEN_PROFILE_SIZE=1.7B \
TPR_QWEN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B \
TPR_SWAP_MODULES=self_attention,mlp \
torchrun --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=29564 \
  -m pytest -x -s -v \
  tests/models/mcore/tpr/profiling/test_activation_offload_phase_c_npu.py \
  -k parameter_gradient_repeat_diagnostics
```

This runs `recompute1/recompute2/offload1/offload2` on identical weights and
prints `TPR_PHASE_C_GRAD_REPEAT` for repeat-vs-repeat and cross-policy
comparisons. It also traces a small representative set of parameters around
Pop and prints `TPR_PHASE_C_PREFIX_PARAM` for the isolated Prefix-backward
contribution. Override that set with the comma-separated
`TPR_PHASE_C_TRACE_PARAMETERS` variable. The diagnostic asserts finiteness
only; it does not widen the C1 correctness tolerance or accept a noisy result.

## Run repeated lifecycle/leak acceptance

Use a fresh process so RSS and pinned-memory observations are attributable to
this case:

```sh
TPR_RUN_OFFLOAD_C=1 \
TPR_QWEN_PROFILE_SIZE=1.7B \
TPR_QWEN_MODEL_PATH=/workspace/hf_models/Qwen3-1.7B \
TPR_SWAP_MODULES=self_attention,mlp \
torchrun --standalone --nproc_per_node=1 \
  -m pytest -x -s -v \
  tests/models/mcore/tpr/profiling/test_activation_offload_phase_c_npu.py \
  -k repeated_lifecycle
```

Defaults are P512/S512 and three measured repetitions after one warmup.  The
following environment variables are available for controlled diagnosis:

```text
TPR_PHASE_C_LEAK_PREFIX=512
TPR_PHASE_C_LEAK_SUFFIX=512
TPR_PHASE_C_LEAK_REPEATS=3
TPR_PHASE_C_NPU_LEAK_TOLERANCE_MIB=64
TPR_PHASE_C_CPU_LEAK_TOLERANCE_MIB=128
```

Increasing a tolerance is not a correctness workaround.  If settled memory
grows per iteration, preserve the emitted `TPR_PHASE_C_MEMORY` rows and inspect
graph/session cleanup and live pinned payloads first.

## Phase C2 boundary

No C2 behavior is accepted by this file.  Ring CP2/CP4, streaming Ring,
QKV/Prefix merge, padding/non-divisible lengths and zero-local-loss ranks need a
separate distributed matrix after C1 graph/session ownership is stable.  C2
must retain strict loss/logprob/Prefix-dKV gates and the existing
baseline-aware Ring parameter-gradient gate without broadening tolerance.
