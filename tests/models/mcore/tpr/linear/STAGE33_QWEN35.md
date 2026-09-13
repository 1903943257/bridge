# Stage 3.3: complete Qwen3.5 Hybrid puncture

Scope: random Qwen3.5-0.8B, 24 layers (18 GDN + 6 Full Attention), BF16,
SiLU, CP/TP/PP/DP=1, non-packed, dropout=0. No weights or Bridge provider needed.
P, S1 and S2 each contain 64 tokens; this does not certify BT=1 or Stage 4.
Keep the MindSpeed-Ops dh0 tile fix that made Stage 3.2 pass.

## Execution and comparisons

- Materialized: two independent native forwards/backwards, P+S1 and P+S2.
  The same model/parameters are reused unchanged; without a TPR context the
  FA/GDN wrapper classes delegate to native forwards. Native GDN uses the
  validated stateful primitives via the existing baseline binding.
- TPR: real `MegatronEngineWithLMHead.forward_backward_batch` tree request,
  then the production executor/scheduler: Push(P), Visit(S1), Visit(S2), Pop(P).
  The Engine fixture supplies the random model directly; checkpoint loading,
  distributed optimizer initialization and optimizer stepping are not tested.
- Connected control: shared prefix graph and identity-connected boundary
  clones. Measures pure external GDN-state/FA-KV gradients independently from
  prefix-internal uses of the same tensors.
- Unsplit TPR-context control: each independent P+Si path is still 128 tokens,
  but runs inside a zero-prefix TPR context. There are no anchors, Push/Pop, or
  final-state backward roots. This separates native-vs-wrapper execution
  differences from full-vs-split continuation differences.

Prefix internal next-token loss has weight 2, with one boundary target per
branch. Suffix internal loss has weight 1. Global denominator is 254.
Push is graph-free and computes no loss; each segment's owned loss is evaluated
once (internal prefix at Pop). Both state caches must be empty after execution.

Diagnostics are printed before numeric assertions: loss, per-segment target
logprob max-absolute difference, all-parameter gradients, GDN and FA boundary
gradient norms/relative-L2/cosine. A2A and Ring must both remain zero.

Additional diagnostic comparisons (no new acceptance thresholds):

1. `native-full-vs-TPR-context-full`: same full sequences and independent
   backwards, different native/TPR GDN and FA execution paths.
2. `TPR-context-full-vs-connected-split`: same TPR wrappers, full paths versus
   a shared-prefix connected graph (includes splitting/sharing/reduction-order
   effects; not an isolated kernel proof).

Both print loss/logprob/parameter metrics and all 24 decoder-layer output and
output-gradient metrics, separately for P/S1/S2. The full-path prefix gradients
are **summed across both branches**, not averaged; prefix output uses path 1
and full-path prefix repeat max-abs is also printed. Layer snapshots are on CPU.
`first-nonzero` is diagnostic only, not a significance threshold or bug verdict.

Every original numerical gate reports its own PASS/FAIL, including each of the
48 boundary tensors. The test fails at the end if any gate fails. An early
native parameter-gradient mismatch no longer prevents relay gates from running.

Acceptance criteria (fixed before the first server run):

- Materialized vs TPR: target logprob atol=0.08/rtol=0.02; loss
  atol=0.02/rtol=0.02; global parameter-gradient rel-L2 <=0.10, cosine >=0.995
  (the existing full-Hybrid BF16 envelope).
- Connected vs TPR: global parameter-gradient and every one of the 48 boundary
  tensors rel-L2 <=0.02, cosine >=0.999; no missing/non-finite gradients.
- Architecture, loss ownership, graph-free Push, release, and communication
  checks must all pass. Thresholds must not be relaxed just to obtain PASS.

## Sync and run

Sync these bridge paths into the corresponding server `/workspace/uni-agent/verl`
paths, preserving the existing server environment and other dirty modifications:

- `verl/models/mcore/tpr/{attention,gated_delta_net,module_spec,segment_executor}.py`
- `tests/models/mcore/baseline/_qwen35_baseline_utils.py`
- `tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py`
- `tests/models/mcore/tpr/unit/{test_hybrid_segment_executor,test_module_spec,test_tpr_self_attention}.py`

The existing TPR Engine entry patch and Stage 3.1/3.2 modules must already be
installed. No new Engine-source patch or MindSpeed/Megatron/Ops edit is needed.

Run unit regressions in a separate process:

```bash
python -m pytest -v \
  tests/models/mcore/tpr/unit/test_gdn_prefix_state.py \
  tests/models/mcore/tpr/unit/test_segment_executor.py \
  tests/models/mcore/tpr/unit/test_hybrid_segment_executor.py \
  tests/models/mcore/tpr/unit/test_module_spec.py \
  tests/models/mcore/tpr/unit/test_tpr_self_attention.py
```

Then the full-model NPU puncture:

```bash
torchrun --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
  -m pytest -s -v \
  tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py
```

Previous server result: native-vs-TPR parameter rel-L2=0.14378,
cosine=0.989669 (FAIL); connected-vs-TPR aggregate rel-L2=0.01307,
cosine=0.999915. Individual relay gates had not yet run because the first
parameter gate stopped the test. The new unsplit control/layer diagnostics and
independent gate reporting await server execution; local checks are not an NPU
correctness PASS.

## First-layer shape diagnostic

After locating the full/split difference before the first FA, run this narrower
probe instead of another full 24-layer backward. It builds the same random
model, runs the real embedding and first GDN Transformer layer only, and prints:

- Embedding 128-token prefix vs independently embedded 64-token prefix.
- Layer runs with canonical identical prefix inputs (the full embedding's
  first 64 positions), autograd enabled, no initial states or packed metadata.
- Input norm, in-projection, conv, GDR q/k/v/g/beta, gated norm, out-projection,
  attention residual, MLP norm/fc1/fc2 (fc2 input exposes activation output),
  and final layer output. Metrics compare only the first 64 token positions.
- First nonzero difference, without declaring that any nonzero value is a bug.
- If that boundary is a captured callable output, replay the callable using
  canonical full-run input prefixes and the same deterministic output-gradient
  seed. The full call's suffix seed is zero; state outputs receive no VJP seed.
  Print isolated output/input/parameter-gradient differences. If the first
  difference is an uncaptured residual/input seam, report it explicitly rather
  than incorrectly blaming the next operator.

No thresholds or production/kernel implementation are changed. Successful
diagnostic execution is NOT a Stage 3.3 correctness PASS. Sync the new script
alongside the existing Stage 3.3 test (its runtime/plan helpers are reused):

```bash
torchrun --master_addr=127.0.0.1 --master_port=29564 --nproc_per_node=1 \
  -m pytest -s -v \
  tests/models/mcore/tpr/linear/test_qwen35_first_layer_shape_probe_npu.py
```

## First-layer out-projection causal control (opt-in)

`STAGE33_OUT_PROJ_CHUNK64=1` temporarily changes only layer 1 GDN out_proj:
128-token inputs are projected as two contiguous 64-token calls and concatenated
with autograd intact; existing 64-token calls are unchanged. This applies to
native-full and TPR-context-full equally. Other layers, weights, dtype, state
relay, optimizer and acceptance thresholds are unchanged. Chunking also changes
this projection's backward reduction order, so this is an execution-shape
intervention, not a forward-rounding-only proof.

The default (`0` or unset) remains the original test. The control prints its
mode and full/short call counts and restores the original method on exit.
A PASS with this control enabled does NOT certify the unmodified baseline.

Sync `_first_layer_projection_control.py`, both updated Qwen test scripts and,
optionally, `unit/test_first_layer_projection_control.py` before running.
First compare the controlled first-layer trace with the existing baseline log:

```bash
STAGE33_OUT_PROJ_CHUNK64=1 torchrun \
  --master_addr=127.0.0.1 --master_port=29564 --nproc_per_node=1 \
  -m pytest -s -v \
  tests/models/mcore/tpr/linear/test_qwen35_first_layer_shape_probe_npu.py
```

Then measure the remaining full-model difference with all original gates:

```bash
STAGE33_OUT_PROJ_CHUNK64=1 torchrun \
  --master_addr=127.0.0.1 --master_port=29563 --nproc_per_node=1 \
  -m pytest -s -v \
  tests/models/mcore/tpr/linear/test_qwen35_hybrid_push_branch_pop_npu.py
```

Inspect whether the first-layer output becomes exact, the next first-nonzero
boundary if it does not, and the full-model parameter-gradient/relay changes.
Do not assume removing the earliest perturbation removes later shape effects.
