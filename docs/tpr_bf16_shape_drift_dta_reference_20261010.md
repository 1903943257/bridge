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
