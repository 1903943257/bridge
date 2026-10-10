# 2026-10-10 | TPR BF16 shape drift: evidence and DTA-style reproduction

> **Handoff / new-chat starting point.** Repo: `1903943257/bridge`, use **bridge checkout** for pull/sync (never run `git pull` inside the vendored `/workspace/uni-agent/verl` directory). This is an NPU Qwen3-1.7B **cropped real TQ** numerical diagnostic, **not** a production actor step. The present numerical equivalence gate is **NOT PASS**.

## 1. Precise scope

- Model: pretrained Qwen3-1.7B (28 layers), Ascend 910B2C, BF16 *default Linear*, controlled CANN attention on the Megatron side; no test-only FP32 forward GEMM, tile padding, GMM, split-dW intervention.
- Input: 8 real UniAgent SWE TQ trajectories, cropped to **prompt 128 + response 64 = 192 tokens** each; single rank, 11 physical Forest segments; `row=5, response_offset=62`, logical absolute query `189`, physical leaf `segment_id=5 [189:192]`.
- Optimizer in the weak test: `torch.optim.AdamW` with FP32 masters. This is **not** Megatron's/VERL's production optimizer. The batch uses **recomputed Native Full old_logprobs** and deterministic diagnostic advantages when missing: no claim of production PPO correctness.
- Native and TPR models share the same loaded checkpoint. The Native oracle intentionally uses the same **controlled CANN core** as TPR, not untouched Megatron `DotProductAttention`.

### Baseline results (weak PPO diagnostic)

| Metric | Native Full | TPR Forest |
|---|---:|---:|
| PPO loss | -0.25 | -0.247665048 |
| Gradient norm | 33.4427567 | 32.0759048 |
| Approx. fwd/bwd | 1.15 s | 1.98–2.05 s |
| PPO tokens clipped out of 512 | 0 | 25 |
| Sampled clipped gradient relative L2 / cosine | reference | 0.748188 / 0.685975 |
| Sampled AdamW update relative L2 / cosine | reference | 1.020061 / 0.479735 |

The bounded sample is **strided up to 512 values per parameter**, not whole-model gradient cosine; first-step AdamW sign changes in tiny gradients can amplify the sample discrepancy. Both optimizers actually step and updated models run, **but numerical parity is not met**.

Logprobs: Full Native vs TPR mean absolute error **0.032839**, max **0.749377**. Example: `row5 response62` Native Full/Cutoff `-7.50056601`, Forest `-6.75118923`. This is a `~exp(0.749)=2.12x` token probability ratio and matters for clipping/optimization.

## 2. What the controlled experiments proved

### A. Same post-RoPE Q/K/V isolates the FA kernel

`CORE` mode captures the real Native core Q/K/V, runs CANN square `192x192` vs rectangular `58x192`, `34x192`, `14x192`, `3x192` for Transformer layers **1,2,3,4,14,28**. All **24** tested shape/layer outputs: `max_abs=0`, `rel_l2=0`. **Conclusion**: for these fixed inputs, shape-changing CANN FA alone does not cause divergence. It does **not** verify actual Forest K/V.

### B. Exact physical Segment trace identifies first divergence

`TRACE` gates on **exact `segment_id=5`**, not shape (sibling nodes may share `[189:192]`). It compares 28 × 11 = **308** module stages against the Native Full row.

Layer-0 `input_norm`, `qkv`, `q_norm`, `k_norm` for query189: exact (`max_abs=0`); first visible difference at **`attn_proj` input** `max_abs=6.103515625e-5`, projection output `4.8828125e-4`. Transformer layer outputs max abs: layer0 `0.00390625`, layer1 `0.015625`, layer2 `0.0625`, layer3 `0.125`. These tensors have different scales; their maxima alone are **not a causal growth-rate proof**.

### C. Actual cached K/V compared with Native

On the leaf path `0[0:134] -> 1[134:158] -> 2[158:166] -> 3[166:189] -> 5[189:192]`:

- Layer1 root `K_POST_ROPE`: `max_abs=0.015625`, `rel_l2=3.29e-6`, 6/134 tokens differ.
- Layer1 root `V_RAW`: `max_abs=0.0009765625`, `rel_l2=2.14e-5`, 34/134 tokens differ.
- Layer1 leaf `K` is exact; leaf `V` differs `4.77e-7` max.
- Layer2 root K/V have larger differences, including K `max_abs=1`; 91/134 tokens differ. This is **after** earlier layer differences have propagated, and absolute K magnitude can be large.

### D. First-layer attention replay: output shift is caused by V in this case

Replay identical original CANN forward and compare the **actual** model projection-input witnesses. All 3 witnesses exactly match replay (`witness_valid=True`):

| Q/K/V swap relative to Native | FA output delta max |
|---|---:|
| TPR Q only | 0 |
| TPR K only | 0 |
| TPR V only | `6.103515625e-5` |
| TPR K+V | `6.103515625e-5` |
| TPR all | `6.103515625e-5` |

Thus **this specific first-layer query** difference is entirely explained by V differences; does not establish that later final logprob `0.749` is caused only by the root V.

### E. Three-way Root V provenance: M shape, not KVStack error

Same BF16 weight and QKV-projection input for first 134 positions:

| First-layer Root comparison | input max error | V max error | changed tokens |
|---|---:|---:|---:|
| Native Full (`M=192`) vs Native cutoff (`M=134`) | 0 | `0.0009765625` | 34/134 |
| Native cutoff (`M=134`) vs actual TPR Root (`M=134`) | 0 | **0** | **0/134** |
| Native Full (`M=192`) vs TPR Root (`M=134`) | 0 | `0.0009765625` | 34/134 |

**Proved locally:** at layer1 Root, BF16 `linear_qkv` result depends on the physical GEMM M despite identical prefix inputs. TPR Root computes the same V as Native at **the same M=134**. This is strong evidence against incorrect first-layer Root KV caching/concatenation. Do **not** extrapolate that all other layers/segments or the backward are bug-free.

### F. Native itself is cutoff-shape-sensitive

Token-level oracle found `Native Full → Native cutoff` max logprob delta `0.481088`, with `Native Cutoff → Forest` max `0.749377` across **different** logical owners. Examples: segment2 `[158:166]` Forest→Cutoff `7.57e-5`, but Full→Cutoff `0.481`; segment5 `[189:192]` Full→Cutoff `0`, Forest→Cutoff `0.749`. Therefore **the exact target 0.749 is not explained away by the Root M comparison**. Additional segmented effects, other layers' GEMMs, and/or real Forest state differences remain to be isolated.

## 3. Why AReaL-DTA is relevant, and what its code *actually* does

Verified against AReaL `feat/dta` at commit `a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6`:

- [areal/experimental/dta/dta_engine.py](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/dta_engine.py): independent cache buffers, `TokenTrie`/LCP DFS traversal. `push` builds ancestor HF `DynamicCache`; `pop` recreates prefix views with `detach().requires_grad_(True)`, **recomputes** suffix via `self.model(tokens_to_pop, past_key_values=prefix_cache, use_cache=True)`, backpropagates loss/segment new KV, accumulates gradients to ancestor KV buffers. Uses `pop_byblock` for block-size limits; handles fork-position logits separately to connect token logprobs. This is genuine physical prefix/suffix separation with changing Linear M.
- [areal/experimental/dta/token_trie.py](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/token_trie.py): lexicographic sort, adjacent LCP, merges fully-overlapping prefix trajectories; distinct `forward_permute` and `backward_permute`.
- [areal/experimental/dta/wrapper.py](https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/wrapper.py): HF model with `past_key_values`, `use_cache=True`; separate no-grad logprob and grad-carrying backward entry.
- This **is not the same** as AReaL's [packed-tree / FlexAttention path](https://github.com/areal-project/AReaL/blob/feat/dta/areal/models/tree_attn/tree.py), which constructs a single padded tree-packed microbatch and special attention mask. Do **not** conflate packed-tree and DTA KV-recompute implementations.

DTA's [Archon vs FSDP test](https://github.com/areal-project/AReaL/blob/feat/dta/tests/experimental/archon/test_dta.py) asserts forward **mean absolute diff < 0.25**, and grad-norm relative gap **< 0.25**, but parameter-delta mismatches `atol=1e-8, rtol=0.3` are logged rather than made fatal. This is a relatively loose smoke/consistency test, **not** proof DTA has no per-token 0.7 drift. Conversely it is **not** proof DTA suffers such drift.

[DTA Qwen3-1.7B example](https://github.com/areal-project/AReaL/blob/feat/dta/examples/tau2/dta/config_1.7b_airline_dta.yaml): `dtype: bfloat16`, `gradient_checkpointing: false`, `eps_clip: 0.4`, `recompute_logprob: true`, `use_decoupled_loss: true`. Our diagnostic used `eps_clip=0.2`, recomputed Native Full old_logprobs, and vanilla PPO; **not comparable as RL update parity**.

**Open question, not answered by the current evidence:** Does AReaL-DTA's **HF DynamicCache** implementation, running on the *same Ascend NPU*, show materially smaller segmented-forward drift than our Megatron TPR? If yes, which HF/Megatron/kernel/schedule distinction accounts for it?

## 4. Plan: independent DTA-style control inside this repo

Keep our production TPR / Megatron untouched. Port a **small independent HF `DynamicCache` reference**, following actual DTA `TokenTrie`, no-grad cache Push and suffix forward, with adjacent LCP and fork-token logprobs. Use **HF pretrained Qwen3-1.7B checkpoint** and the **same recorded 8 real TQ token sequences**; compare *within HF* (HF Full vs HF DTA-style) on **one Ascend NPU**, BF16, no injected FP32 GEMM.

Compare separately:
1. HF Full vs HF repeated prefix-cached segmentation (same HF model/checkpoint).
2. Megatron controlled Native Full vs Megatron TPR (existing baseline).
3. Optional HF Full vs Megatron Full to quantify **model/framework delta**, not attribute that delta to DTA or TPR.

To keep it honest, compute *per-token* max/mean/p95 abs-logprob, worst token and its physical chunk length, and the first-layer Root V `M=192` vs `M=134` when hooks are enabled. Pay special attention to small segments and the target row5 query `189`. Use same checkpoint and minimal precision interventions.

The initial HF port is a **forward numerical control**, not yet the full DTA training algorithm: full pop/recompute + KV/logprob gradient relay, gradient parity, and AdamW parity are separately gated follow-ups. Do not label it `DTA training PASS` before those tests exist.

### Existing NPU reproduction

```bash
cd /workspace/bridge  # or your actual bridge-backed VERL directory
# Pre-existing environment expects TPR_REAL_TQ_BATCH and TPR_QWEN_1_7B_PATH.
python -m pytest -q tests/models/mcore/tpr/unit/test_qwen17_weak_e2e_trace.py
TPR_QWEN17_WEAK_E2E_MODES=core bash tests/models/mcore/tpr/correctness/run_qwen17_real_tq_weak_e2e.sh
TPR_QWEN17_WEAK_E2E_MODES=trace bash tests/models/mcore/tpr/correctness/run_qwen17_real_tq_weak_e2e.sh
```

See `tests/models/mcore/tpr/correctness/_qwen17_weak_e2e_trace.py`, `test_qwen3_1_7b_real_tq_weak_e2e_npu.py`, and `run_qwen17_real_tq_weak_e2e.sh`.

### Unresolved / safeguards

- No source change to production TPR or Megatron should be mixed with oracle experiments.
- `TRACE status=PASS` means **probe completed**, not numerical equivalence.
- Core square/rectangular exactness holds *for identical post-RoPE Q/K/V*, not for real Forest inputs.
- BF16 GEMM M-dependence is established for Root `linear_qkv` at M134 vs M192; the **0.749 Forest vs Full difference remains unresolved**.
- DTA stable reward curves / non-strict numerical unit tests do not settle the strict logprob question. Compare under **same model/hardware/inputs** and include per-token errors, not only mean reward/loss.
- When this document is extended with results, record exact commits, NPU/CANN/torch/transformers versions, shapes, test gates and failures.


## 5. Implemented follow-up: HF DTA-style forward oracle (committed, **NPU pending**)

Added in `tests/models/mcore/tpr/correctness/`:
- `_qwen17_dta_style_reference.py`: stand-alone `DynamicCache` adapter, lexicographic LCP/DFS prefix persistence, shifted fork-token logprobs, full-HF and fixed-chunk baselines; **no dependency on the TPR executor**.
- `test_qwen3_1_7b_hf_dta_reference_npu.py`: real Qwen3-1.7B HF BF16 checkpoint, eight cropped real TQ trajectories on Ascend, `HF_FULL` vs `HF_LCP_DFS` vs `HF_FIXED_PATH`. Records all-token `DFS_SUMMARY`, separately **512 response logprobs** in `RESPONSE_SUMMARY`, `FIXED_PATH` for row5 query189, and first-layer HF `ROOT_V_SHAPE` / `ROOT_CUTOFF_TO_DTA` triple.
- `../unit/test_qwen17_dta_style_reference.py`: CPU-only fake HF `DynamicCache` oracle for shifted fork logits, nested prefixes, duplicates, no sharing, and fixed chunks.
- `run_qwen17_hf_dta_reference.sh`: one entrypoint to CPU tests and opt-in NPU.

**Run from the bridge-backed VERL repository root (not a stale vendored checkout):**

```bash
export TPR_REAL_TQ_BATCH="${TPR_REAL_TQ_BATCH:-/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt}"
export TPR_QWEN_1_7B_PATH="${TPR_QWEN_1_7B_PATH:-/workspace/hf_models/Qwen3-1.7B}"
TPR_QWEN17_DTA_HF_ATTN=sdpa \
  bash tests/models/mcore/tpr/correctness/run_qwen17_hf_dta_reference.sh
```

Optional `TPR_QWEN17_DTA_HF_ATTN=eager` rerun if the specific transformers/torch_npu version does not support cached `sdpa`; **record this as an attention-backend change**, not the same kernel.

**Interpretation after a real run:**
- `RESPONSE_SUMMARY` (HF Full vs HF LCP/DFS) gives DTA-style same-framework actor-token drift. Comparing only aggregate prompt+response mean would mask PPO-sensitive tails.
- `FIXED_PATH` compares HF Full to the **same five segment lengths** as TPR target row5. If `FIXED_PATH` is much worse than `DFS`, scheduling/physical M distribution is important even within the same HF backend.
- `ROOT_V_SHAPE` compares HF Full `M=192` against HF cutoff `M=134`; `ROOT_CUTOFF_TO_DTA` compares cutoff against the **actual fixed-path HF DTA root** `M=134`. This mirrors the Megatron triple-control.
- **Do not directly subtract** HF DTA logprob from Megatron TPR logprob to attribute algorithmic error; that mixes framework/weight mapping/attention implementation differences.
- `status=PASS execution=FORWARD_CONTROL` means the probe completed, **not** a strong tolerance gate.

**Not implemented:** AReaL's `backward_permute`, `pop_byblock`, suffix recompute, KV/logprob/fork-logit gradient relay, and full AdamW/ppo training. The current port is intentionally **Phase 1 (forward)**. A full DTA training comparison will need separate CPU unit gates and same-checkpoint NPU gradient tests. Current session cannot run the user's NPU or HF checkpoint, so no claim of observed HF DTA numerical results is made.


## 6. **NPU result 2026-10-10: independent HF DTA-style forward run succeeded, not parity**

Real user run, HF Qwen3-1.7B, 8×192 cropped real TQ rows, same NPU/checkpoint/weights **within** HF; compare with previously recorded Megatron TPR results (which have different framework/controlled FA). Log: `/tmp/tpr_qwen17_hf_dta.h1VmOo/hf_dta.log` on user's host.

| Metric (shifted response tokens) | Megatron Full vs TPR Forest | HF Full vs HF DTA-style LCP/DFS |
|---|---:|---:|
| Number of response tokens | 512 | 512 |
| Mean absolute logprob delta | **0.032839** | **0.0309696756** |
| Max absolute logprob delta | **0.74937677** | **0.445066452** |
| p95 absolute delta | not recorded in previous report | **0.24168916** |
| Tokens with abs delta > 0.2 | not recorded here | **50 / 512** |

Full HF DFS logprob comparison including prompt positions: 1528 comparisons, max `0.445066452`, mean `0.0103772739`, p95 `0.0411890894`, >0.2 count 50. Per-row maxes: row0 `0.25`, row1 `0.370897293`, row2 `0.445066452`, row3 `0.445066452`, **row4 `0`**, row5 `0.25`, row6/7 `0.254723549`. Some duplicate rows naturally share outcomes.

**Important:** The independently written HF-DTA-style DFS forward shows meaningful numerical drift; it is **false** that changing physical M with BF16 prefix reuse cannot cause significant output drift. But the experiment is a forward-only approximation of DTA, **not a full reproduction of the original AReaL DTA training schedule or a GPU-vs-NPU parity claim**.

DFS execution, as implemented in this reference:
- `physical_m=[192,58,34,14,26,3]`
- `physical_starts=[0,134,158,178,166,189]`
- Saved forward token processing: 1209 vs independent dense forwards, in this input-specific sorted traversal.

**Critical schedule distinction:** Our **forward-only** HF DFS initializes the cache by forwarding the entire *first* trajectory with `M=192` (row4, which is exact by construction). The tested TPR path stores Root cache after `M=134`. Actual AReaL `DTAEngine.backward()` uses `push(..., cache_len=...)`, `build_cache()`, and `pop_byblock()`, which can compute the first Root with `M=134` instead. Do not label the current HF DFS schedule equivalent to **DTA's training execution**. This matters more than whether each path uses DynamicCache.

### Same-M root triple, HF vs Megatron

HF first-layer `v_proj`: identical input in all comparisons. `M192` Full vs `M134` Root: `V max_abs=0.001953125`, relative L2 `3.73479061e-5`, 20/134 changed tokens. `M134` Native cutoff vs `M134` DTA fixed-path root: **V max_abs=0**.

Megatron TPR corresponding `linear_qkv` full M192 vs cutoff/root M134: `V max_abs=0.0009765625`, 34/134 changed tokens; cutoff vs TPR root: **0**.

**Grounded conclusion:** BF16 GEMM physical-M dependence reproduces in both HF and Megatron. It is not evidence of uniquely faulty TPR root KV storage. This does not prove full-model/TQ equivalence.

### **Critical unresolved difference: row5, query189**

Same fixed physical TPR root-to-leaf path `[0,134,158,166,189,192]` reproduced using independent HF `DynamicCache`:
- HF Full vs HF fixed path (row5): max logprob delta **0.249992371**, mean **0.0335460231**, p95 **0.124519587**; 5 positions above 0.2; worst logical query **179**.
- At **query189**, HF Full vs HF fixed path delta **0**.
- Earlier Megatron Full vs TPR Forest, corresponding `row5 query189` logprob delta **0.749376774**.

Therefore **the overall mean drift being similar does not explain the particular outlier**. The same physical five-chunk boundaries give exact logprob at query189 in HF, but a serious discrepancy in the Megatron TPR tree. This could be caused by different HF/Megatron model+kernel numerical paths or another TPR-specific effect; current evidence cannot discriminate. Do not claim the 0.749 error is proven to be inevitable BF16 amplification.

### Exact next experiments

1. **DTA training-shape forward schedule** (priority): port AReaL's `backward_permute()` + `push(cache_len)` + `build_cache` + `pop_byblock` *forward computation* while holding the same HF Qwen3 weights and BF16. Compare this output with HF Full, HF DFS-forward-only and the fixed path, logging physical M for every Push and Pop and token logprob drift. The current sorted DFS of `forward_permute` semantics can differ from actual backward. Do not equate the two.
2. **Strict within-Megatron 5-segment reference**: physically split the same Native checkpoint via HF-style KV-cache semantics but without the existing TPR tree scheduler, if feasible. Compare precisely query189 and per-layer outputs to separate schedule/state issues from native kernel shape sensitivity. Keep **same model/checkpoint/attention backend**.
3. **Full AReaL DTA training port** if needed after step1: actual KV/logprob/entropy/fork-logit gradient accumulation, parameter gradients, independent AdamW and PPO clip checks; no backward gradient or optimizer parity has yet been demonstrated for HF-style reference.
4. For every experiment, preserve **response-only** max/mean/p95 and especially `row5 query189`; no artificial FP32/padded GEMM interventions until causal attribution is complete.

**Handoff verdict:** Independent HF DTA-style forward shows BF16 shape drift of comparable *average* magnitude, falsifying “only our TPR numerical path drifts”. Nevertheless the unexplained `query189: HF fixed path 0 vs Megatron TPR 0.749` is still a material blocker to claiming TPR parity, and the official DTA training algorithm has not yet been reproduced.

## 7. More faithful AReaL-DTA forward control (2026-10-10 follow-up)

**Motivation:** Original HF lexical DFS reproduced material BF16 forward drift, but
was missing the optimized AReaL forward order and fixed KV storage. Both the
old lexical DFS and the exact row5 five-segment comparison remain intact.

### Observed in the previous real NPU run (NOT the new oracle)

- HF lexical DFS response (512 values): max_abs=0.445066452,
  mean_abs=0.0309696756, p95_abs=0.24168916, 50 values above 0.2.
- Lexical physical starts=[0,134,158,178,166,189],
  M=[192,58,34,14,26,3], saved_forward_tokens=1209.
- Full M192 to Root M134 HF first-layer V max_abs=0.001953125,
  with 20/134 changed token positions; cutoff M134 to fixed-path DTA root
  M134 V difference exactly zero.
- Most important: HF fixed-path row5 query189 abs logprob error=0 while
  Megatron Native Full vs TPR Forest at the same logical position was
  0.749377. The HF result does NOT explain the Megatron tail.

### New isolated implementation

- _qwen17_areal_dta_reference.py: ports AReaL feat/dta a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6
  forward-only TokenTrie sorted leafization (duplicates and prefix
  attachments), CompressedTrie chain-priority forward_permute, persistent
  per-layer fixed K/V buffers, rebuilding DynamicCache from prefix buffer
  views, and storing fork logits with shifted token labels.
- unit/test_qwen17_areal_exact_reference.py: CPU fake HF coverage of
  duplicate/nested prefix, no sharing, fork labels and KV buffer overwrite.
- Existing HF NPU driver now compares four modes on the same HF BF16 model:
  HF_FULL, HF_LCP_DFS (lexical), HF_AREAL_FORWARD (optimized + persistent
  buffers), and HF_FIXED_PATH (five chunks for row5).
- New logs: AREAL_ROW, AREAL_SUMMARY, AREAL_RESPONSE_SUMMARY,
  AREAL_VS_LEXICAL, AREAL_OUTLIER (response token and actual visit/M),
  AREAL_TARGET (row5 query189 all four HF values).
- All existing ROOT_V_SHAPE and ROOT_CUTOFF_TO_DTA controls remain.

### NPU run from the bridge-backed VERL checkout

    export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
    export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
    TPR_QWEN17_DTA_HF_ATTN=sdpa bash tests/models/mcore/tpr/correctness/run_qwen17_hf_dta_reference.sh

### Interpretation and limitations

- Compare within HF, using each path's difference from the same HF Full.
- The AReaL-style mode changes both optimized traversal order AND fixed KV
  buffer layout. These must be separated by further ablations if drift changes.
- This is a forward-only oracle. DTA backward_permute, pop_byblock, suffix
  recomputation, Prefix KV/fork-logit gradient relay and PPO/optimizer
  parity remain NOT IMPLEMENTED; success means the probe ran, not parity.
- New AReaL-faithful NPU numerical results are PENDING user execution.

### Additional two-axis numerical ablation (same checkpoint)

To avoid attributing a schedule change to the KV storage layout, the new
NPU test also runs AReaL-compatible persistent KV storage *without*
forward_permute. It prints:

- AREAL_LEXICAL_BUFFER: HF Full vs lexical/leafized persistent-KV mode.
- AREAL_BUFFER_ABLATION: old lexical last-leaf cache vs lexical persistent KV.
- AREAL_PERMUTE_ABLATION: lexical persistent KV vs optimized persistent KV.
- AREAL_SUMMARY and AREAL_RESPONSE_SUMMARY: Full vs optimized persistent KV.

Interpret these as measured numerical differences; all tests remain
diagnostic-only, with no automatic strict parity gate or NPU execution claim.
CPU fake-cache tests for the forward plan and both ordering modes: 5 passed
locally on CPU (not a substitute for real BF16 NPU kernel testing).


## 8. Actual AReaL training-schedule loss Forward control (2026-10-10)

The previous oracle covers AReaL `DTAEngine.forward()/push_forward_only()`.
**It does NOT cover the training-time Forward calls inside
`DTAEngine.backward()`.** The new test-only numerical oracle now ports
those Forward calls, while deliberately NOT running backward gradients.

Source pinned to AReaL `feat/dta` commit
`a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6`:

- `TokenTrie.backward_permute()`: compressed trie, leaf-priority traversal,
  reverse DFS, plus duplicate/nested-prefix attachment handling.
- `DTAEngine.backward()`: block-size calculation, `lcp_next`,
  `cache_len`, and `cut_f1_tail`. Important: cache_len can be below the
  current start, in which case Push does not call build_cache.
- `push()/build_cache()`: no_grad, fixed K/V buffers, logits and fork
  positions at both branch and block boundaries.
- `pop_byblock()/pop()`: reverse order; reconstruct prefix DynamicCache
  using detached prefix K/V with requires_grad=True; execute the model
  in torch.enable_grad() rather than no_grad; concatenate the prefix,
  fork-token and suffix logprobs seen by the real loss function. The probe
  detaches those output logprobs and releases the graph without backward.
- Log provenance traces actual physical **CACHE vs POP** events to the
  specific row/query token. Any outlier can be linked to event start/M,
  including fork-connection logits sourced from earlier cache Forward.

Implementation:
`tests/models/mcore/tpr/correctness/_qwen17_areal_training_forward_reference.py`

NPU diagnostic:
`tests/models/mcore/tpr/correctness/test_qwen3_1_7b_hf_dta_training_forward_npu.py`

CPU scheduler tests:
`tests/models/mcore/tpr/unit/test_qwen17_areal_training_forward.py`

One-shot runner:
`tests/models/mcore/tpr/correctness/run_qwen17_hf_dta_training_forward.sh`

Command from the bridge-backed VERL root:

    export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
    export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
    TPR_QWEN17_DTA_HF_ATTN=sdpa \
      bash tests/models/mcore/tpr/correctness/run_qwen17_hf_dta_training_forward.sh

Default block sizes `64,-1` compare blockwise and unchunked Pop.
The environment `TPR_QWEN17_DTA_TRAIN_BLOCK_SIZES` accepts a comma-separated
list; `TPR_QWEN17_DTA_TRAIN_CUT_F1_TAIL=1` preserves AReaL's
backward() default; and `TPR_QWEN17_DTA_TRAIN_MAX_SEQ_LEN` controls
persistent KV buffer allocation (default 192 for these cropped inputs).

The NPU test keeps the *same HF checkpoint* across HF Full,
AReaL Forward-only, and AReaL training-Pop Forward, uses model.train()
with BF16/SDPA, and reports `DTA_TRAIN RESPONSE_SUMMARY`,
`VS_FWD_ONLY`, `EVENT`, `OUTLIER`, and `TARGET` (row5/query189).

Interpretation: execution=TRAIN_LOSS_FORWARD_CONTROL is **not**
numeric parity PASS, DTA gradient parity, or complete DTA Backward.
This oracle does not execute autograd.backward, KV/logprob/entropy/fork-logit
gradient injections, loss scaling, or optimizer updates.
Full backward requires a separate correctness project.
Actual Ascend NPU results for this new training path are pending.


## 9. Full DTA Backward / PPO-step gradient experiment and GEMM interventions

Code added directly to `main` (test-only, does not modify VERL or
production TPR):

- `tests/models/mcore/tpr/correctness/_qwen17_areal_native_dta_backward.py`:
  Apache-2.0 original DTAEngine body pinned at
  `areal-project/AReaL feat/dta a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6`,
  including real `torch.autograd.backward`, KV gradient relays, fork logits,
  saved logprob and entropy gradient relays, `pop_byblock`, `build_cache`,
  and `cut_f1_tail`. **Explicit HF-only numerical API adaptation:** AReaL's
  original `gather_logprobs_entropy(logits=[1,B,V], labels=[1,B-1])` sites
  have mismatched label sequence lengths when fed unmodified Transformers
  HF logits. Local helper uses `logits[:, :-1]` for next-token logprobs
  and all B positions for entropy. It does not change the backward relay.
- `tests/models/mcore/tpr/unit/test_qwen17_areal_native_backward.py`:
  differentiable two-layer causal-KV toy with prefix K/V influencing output;
  checks per-token PPO-like loss, every parameter gradient, post-SGD weights,
  true prefix K/V nonzero gradients; 6 block sizes × cut-tail on/off.
- `tests/models/mcore/tpr/correctness/test_qwen3_1_7b_areal_backward_ppo_npu.py`:
  BF16 Qwen3-1.7B eight cropped real SWE trajectories; reference:
  eight independent HF Full trajectories. DTA: original per-Pop
  `DTAEngine.backward`. Both use one single *identical* clipped-PPO
  surrogate loss, entropy coefficient, advantage, old logprob, and a
  fresh identical AdamW optimizer; make one actual optimizer step.
  Report pre-step PPO loss, per-token logprobs, clip fraction, full
  parameter gradient relative-L2 / cosine / worst parameters, and selected
  post-AdamW parameter-change mismatch.
- `tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh`:
  one-shot CPU gate + NPU experiment from Docker VERL root.

Ablations apply to DTA model for this first controlled comparison:
`native` (unmodified BF16), `m_split` (chunk Linear M dimension,
default tile=32), `fp32_linear` (FP32 GEMM of Linear inputs/weights/bias,
cast output back to BF16), and optional `m_split_fp32`.
Parameter storage, attention, RMSNorm, KV and loss remain in the original
configured dtypes. These modes do NOT amount to end-to-end FP32 training.
The baseline is identical unpatched BF16 HF Full and reused for all DTA
modes. The script records the number of patched Linear modules.
The sampled BF16 optimizer-step mismatch is affected by weight rounding;
gradient metrics remain separately reported.

**Important PPO source contract:** `advantages` and `old_log_probs` are
read from the TQ TensorDict only when present as exact `[8,64]` tensors.
If either field is missing, the result is labeled `PPO_PROXY`: missing
advantages are deterministic signed values; missing old logprobs are
current-checkpoint HF Full. They are not claimed to be real rollout GAE or
old policy data. Set `TPR_PPO_REQUIRE_RECORDED=1` to fail rather than use
proxy data. This is a one-step controlled PPO objective experiment, NOT a
full UniAgent rollout/reward/critic/advantage pipeline.

Run, after syncing host `bridge/main` to the Docker-mounted bridge tree
and rsync to Docker's `/workspace/uni-agent/verl/tests/models/mcore/tpr/`:

    cd /workspace/uni-agent/verl
    export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
    export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
    TPR_DTA_BWD_MODES=native,m_split,fp32_linear \
      bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

Start by setting `TPR_DTA_BWD_MODES=native` if memory is limited, then
run `m_split` and `fp32_linear` separately with the same checkpoint/data.

**No result claims before NPU**: merging scripts/tests does not establish
gradient parity or reduction of numerical error. The runner prints
`numerical_parity=DIAGNOSTIC_ONLY`; CPU and actual NPU gates have not
been executed on the author's local Ascend host from this environment.

Original AReaL DTA integration check on `feat/dta`
`tests/experimental/archon/test_dta.py` allows `mean_diff < 0.25`
in Forward and `grad_norm_rel_gap < 0.25` in train;
per-parameter update mismatches are logged instead of failing the test.
These are not exact parity gates and are CUDA-based; do not infer equivalent
Ascend behavior or claim the paper is invalid from local outliers alone.


## 10. NPU actual DTA Backward diagnostic (2026-10-10)

The user executed the Qwen3-1.7B BF16 Ascend NPU test with 8 real
128+64 cropped TQ rows, SDPA, DTA Pop block=64, AdamW lr=1e-4,
and clipped PPO **proxy** (old_log_probs generated from same-model HF
Full, advantages deterministic rather than recorded).

Reported against unpatched BF16 HF Full:

| DTA mode | response logp mean/max | full-param grad rel-L2/cosine | clip fraction | sampled BF16 step rel-L2 |
|---|---|---|---|---|
| native | 0.029684 / 0.458695 | 0.593944 / 0.805095 | 0.09180 | 0.982568 |
| m_split (tile=32) | 0.036054 / 0.600080 | 0.567760 / 0.882624 | 0.08984 | 0.834743 |
| fp32_linear | 0.035510 / 0.553312 | 0.533855 / 0.859047 | 0.10742 | 0.869125 |

The full baseline proxy loss was -0.08978709. Native DTA loss was
-0.09137118; M split -0.08439070; FP32 linear -0.08329321.
All 311 parameter gradients were present, and AdamW stepped, but
`numerical_parity=UNVERIFIED`: no strict gradient parity.

**Interpretation caveats:** clip fraction is the ratio-outside-band
fraction, NOT the percentage of token gradients actually blocked by the
PPO clipped surrogate. Missing real rollout old policy values means Full
has ratio=1 and clip=0 by construction; DTA does not. A single BF16
AdamW step measured on sampled BF16 model weights includes weight-rounding
and first-step sign effects. Do not extrapolate the sampled ~0.98 step
relative error to long-horizon training. AReaL original CUDA tests have
loose gradient-norm gates; the NPU results do not prove the paper wrong.

**New controlled isolation (defaults preserved):**
- `TPR_DTA_BWD_OBJECTIVE=fixed_logprob` uses a linear loss
  `-sum(advantage * logprob)/(N*S)`. The logprob derivative per token
  is identical for Full/DTA, removing PPO clipping and ratio derivative
  changes. **This is NOT PPO.** With `entropy_coef=0`, it isolates
  execution/numerical backward drift from changes in the PPO objective.
- `TPR_DTA_BWD_PAIRED_FULL=1` adds, for M split and FP32 modes, an
  independent HF Full run patched with the *same* GEMM mode. It reports
  `PAIRED_FULL grad_rel_l2/grad_cosine/response_logp_mean`.
  This isolates whether the intervention makes DTA intrinsically more
  similar to a Full model **at matching precision and GEMM shape policy**,
  rather than to the original BF16 Full.
- `effective_clip_frac` counts only PPO-relevant clipped-gradient token
  cases (positive advantage ratio above upper bound; negative advantage
  ratio below lower bound).
- `GRAD_GROUP` aggregates absolute gradient error by functional module,
  to distinguish high relative error on tiny norm parameters from the
  contributors to global gradient error.
- Corrected entropy alignment to query positions `p-1 : p+s-1`; this
  does not affect the reported run because `entropy_coef=0`.

Recommended **sanity**:
 
    TPR_DTA_BWD_ROWS=1 TPR_DTA_BWD_BLOCK=-1 \
    TPR_DTA_BWD_MODES=native TPR_DTA_BWD_OBJECTIVE=fixed_logprob \
    bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

This should eliminate all tree-sharing and Pop chunking: both the reference
Full and the DTA final Pop perform a single 192-token Forward. If gradients
still significantly disagree, investigate DTA backward/adapter correctness
rather than BF16 shape drift.

Recommended **isolation**:

    TPR_DTA_BWD_ROWS=8 TPR_DTA_BWD_BLOCK=64 \
    TPR_DTA_BWD_MODES=native,m_split,fp32_linear \
    TPR_DTA_BWD_OBJECTIVE=fixed_logprob \
    TPR_DTA_BWD_PAIRED_FULL=1 \
    bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

Compare this fixed-token-gradient experiment with the prior clipped PPO
proxy run before attempting more GEMM modes, new optimizer heuristics, or
precision changes. All new controls remain **NPU unverified** pending run.


## 11. Fixed-logprob gradient isolation on real Ascend NPU (2026-10-10)

Two independent cases were supplied by the user:

**Single full 192-token sequence, block_size=-1, objective=fixed_logprob:**
All 311 parameter gradients bitwise aligned with HF Full
(global grad relative L2 = 0, cosine = 1); logprob drift 0 and sampled
AdamW parameter step drift 0. This checks the native DTA no-sharing
full-Forward/Backward path, **not** prefix reuse or KV relay under forks.

**Eight real trajectories, block_size=64, objective=fixed_logprob:**
- Native BF16: grad relative L2 **0.0456423**, cosine **0.998987**,
  response logprob mean **0.0296836**, max **0.458695**,
  sampled BF16 AdamW step relative L2 **0.301296**.
- BF16 M split, tile=32: global grad relative L2 vs original BF16 Full
  **0.0536512**, cosine **0.998732**; vs **equivalently M-split Full**
  grad relative L2 **0.0218131**, cosine **0.999769**,
  response logprob mean **0.0104654**.
- FP32 Linear: vs original BF16 Full global grad relative L2
  **0.0704752**, cosine **0.997905**; vs equivalently FP32-Linear Full
  relative L2 **0.0536680**, cosine **0.998666**,
  response logprob mean **0.0219767**.

Comparison to the preceding clipped-PPO-proxy run in the same native
configuration: gradient relative L2 fell from **0.593944** to
**0.0456423**, while response logprob drift was unchanged.
**Important:** `fixed_logprob` removes both PPO clipping AND
importance-ratio reweighting. It cannot identify clipping as the sole
source of the additional gradient mismatch.

Added an intermediate `TPR_DTA_BWD_OBJECTIVE=ppo_unclipped` to isolate:
- `ppo`: `-min(ratio * advantage, clamp(ratio)*advantage)`
- `ppo_unclipped`: `-ratio * advantage`
- `fixed_logprob`: `-logprob * advantage`

The `effective_clip_frac` still reports which tokens would be
clipped under PPO; `active_clip_frac` is set to zero for the latter
two objectives and reflects actual clipping only under `ppo`.

Suggested next runs after the usual host bridge git pull and Docker rsync:

    TPR_DTA_BWD_ROWS=8 TPR_DTA_BWD_BLOCK=64 \
    TPR_DTA_BWD_MODES=native TPR_DTA_BWD_OBJECTIVE=ppo_unclipped \
    bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

    TPR_DTA_BWD_ROWS=1 TPR_DTA_BWD_BLOCK=64 \
    TPR_DTA_BWD_MODES=native TPR_DTA_BWD_OBJECTIVE=fixed_logprob \
    bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

    TPR_DTA_BWD_ROWS=8 TPR_DTA_BWD_BLOCK=-1 \
    TPR_DTA_BWD_MODES=native TPR_DTA_BWD_OBJECTIVE=fixed_logprob \
    bash tests/models/mcore/tpr/correctness/run_qwen17_areal_full_backward_ppo.sh

The first distinguishes ratio reweighting from clipping; the second
isolates chunked Pop on a single logical sequence; the third tests
shared-prefix paths with the last full-sequence Pop unchunked.
These new modes are NPU UNVERIFIED until executed.


## 12. PPO ratio, Pop chunking, and triplet Full/DTA/TPR (2026-10-10)

New NPU observations from the real cropped SWE-TQ batch (8x192,
Qwen3-1.7B BF16, SDPA):

| AReaL DTA Pop | Objective | Response mean abs | Param gradient rel L2 | cosine |
|---|---|---:|---:|---:|
| 8 rows, block=64 | fixed_logprob | 0.029684 | 0.0456423 | 0.998987 |
| 8 rows, block=64 | ppo_unclipped | 0.029684 | 0.127322 | 0.992397 |
| 8 rows, block=64 | ppo | 0.029684 | 0.593944 | 0.805095 |
| 1 row, block=-1 | fixed_logprob | 0 | 0 | 1 |
| 1 row, block=64 | fixed_logprob | 0.022037 | 0.0732935 | 0.997724 |
| 8 rows, block=-1 | fixed_logprob | 0.025984 | 0.0361715 | 0.999371 |

These establish: fixed_logprob **does not repair Forward logprobs**;
it replaces the PPO derivative with a constant -advantage per token.
The unclipped ratio retains its logprob-dependent derivative and
increases gradient differences; active clipping further increases them.
The error percentages cannot be decomposed additively. The old-policy
logprobs here equal *HF Full same-checkpoint*, which sets Full ratio=1,
making this a strong numerical sensitivity test but not a real rollout
old policy measurement.

### Actual Megatron TPR third arm

To avoid conflating HF model/backends with Megatron, the new three-way
runner executes separate processes with exactly the same batch and
objective. It compares:

1. HF Full vs independent AReaL-DTA real full backward (all-parameter
   gradient metrics).
2. Megatron Native vs the **real** TPR Forest routed through
   \`MegatronEngineWithLMHead.forward_backward_batch\`
   (all named trainable parameters, first 4096 flattened gradient
   entries per parameter + actual BF16 AdamW step).
3. HF Full vs Megatron Native RESPONSE logprob before attempting any
   cross-framework TPR attribution. It also displays raw HF Full vs
   Megatron TPR gap, explicitly labeled as including framework drift.

\`tests/models/mcore/tpr/correctness/test_qwen3_1_7b_tpr_dta_triplet_npu.py\`
implements the TPR path using the same
\`_load_real_tq_probe(prompt_length=128,response_length=64)\`,
\`_make_qwen_model\`, \`_native_response_logprobs\`, and
\`SegmentPPOObjectiveAdapter\` as the existing production-routed TPR
PPO correctness test. A single custom loss callable supports both packed
native and compact segment-local query positions:
\`fixed_logprob\`, \`ppo_unclipped\`, \`ppo\`. No production changes.

\`tests/models/mcore/tpr/correctness/compare_qwen17_dta_tpr_triplet.py\`
rejects mismatched objective, shapes, or model checkpoint paths and prints
clearly scoped comparison lines. **Do not equate** the all-parameter HF
gradient relative L2 with the sampled Megatron relative L2. No HF-to-
Megatron gradient parameter mapping is claimed.

Run from Docker VERL root after host bridge git pull / Docker rsync:

    cd /workspace/uni-agent/verl
    export TPR_REAL_TQ_BATCH=/workspace/tq_dump/django11163/swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/GBS1_N8_in16384_out114688/1/0/tq_batch.pt
    export TPR_QWEN_1_7B_PATH=/workspace/hf_models/Qwen3-1.7B
    TPR_DTA_BWD_OBJECTIVE=ppo_unclipped \
      bash tests/models/mcore/tpr/correctness/run_qwen17_dta_tpr_triplet.sh

Default Pop block=64 and HF DTA mode=native. The HF subprocess
exports exact input tokens, old logprobs and advantages; the Megatron
Native/TPR subprocess REUSES these values and explicitly rejects
mismatches. If TQ lacks exact [8,64] rollout fields, the same
deterministic signed proxy advantages and HF Full old logprobs are used
across ALL four paths. The objective is truly matched; separate
HF-Full/Megatron-Native Forward divergence is still reported as a
framework/weight-conversion floor. Cross-backend raw ratios are now
relative to the SAME old policy, but cross-backend parameter gradients
and optimizer steps are still NOT directly comparable without a
validated HF-to-Megatron parameter-gradient mapping.

**New triplet NPU status:** not yet executed. This must not be reported
as TPR gradient/optimizer numerical parity before physical NPU run.
