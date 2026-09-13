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

Prefix internal next-token loss has weight 2, with one boundary target per
branch. Suffix internal loss has weight 1. Global denominator is 254.
Push is graph-free and computes no loss; each segment's owned loss is evaluated
once (internal prefix at Pop). Both state caches must be empty after execution.

Diagnostics are printed before numeric assertions: loss, per-segment target
logprob max-absolute difference, all-parameter gradients, GDN and FA boundary
gradient norms/relative-L2/cosine. A2A and Ring must both remain zero.

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

Initial status: server NPU execution pending; local syntax checks are not a
correctness PASS. Report the printed metrics together with PASS/FAIL.
