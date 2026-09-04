# MindSpeed-Ops Stateful Training Audit for TPR Linear Attention

Audit date: 2026-09-04

This is a static source audit. No Uni-Agent/VERL, Megatron-LM, or MindSpeed Core production code was changed, and no NPU result is claimed by this document.

## 1. Source snapshots

### MindSpeed-Ops

- Repository: `https://gitcode.com/Ascend/MindSpeed-Ops.git`
- Local checkout: `D:/project/MindSpeed-Ops`
- Branch: `master`
- Commit: `9076690409ca92790700eb0bbfceee03d206fd05`
- Commit subject: `fix(triton): support BF16 gates with TA 3.6.0`
- Package namespace: `mindspeed_ops`
- Current package version default: `26.2.0` in `setup.py:434`
- Current declared stack:
  - CANN 9.1.0: `README.md:14`, `docs/zh/release_notes.md:42`
  - PyTorch >= 2.7.1: `README.md:15`
  - matching `torch_npu`: `docs/zh/release_notes.md:42`
  - `triton-ascend==3.2.2`: `requirements.txt:1`, `pyproject.toml:10`
  - Python >= 3.10: `README.md:17`, `setup.py:443`
  - compiled CANN components require package versions >= 9.0.0: `mindspeed_ops/csrc/version.info:3-10`

The master metadata is internally inconsistent with its newest commit subject: dependencies still pin Triton-Ascend 3.2.2, while HEAD explicitly contains a TA 3.6.0 compatibility fix. Master is therefore an active development snapshot, not a suitable server pin.

### MindSpeed Core

- Local checkout: `D:/project/MindSpeed`
- Branch snapshot: `core_r0.16.0` (detached local checkout)
- Commit: `376e9cc302a2c00bfcc84e49a557b47fec941c87`

### Known server environment

| Component | Known value | Audit result |
|---|---:|---|
| Python | 3.11.14/3.11.15 observed in prior logs | Satisfies current `python_requires>=3.10`; historical README validated Python 3.10.x, so 3.11 still needs import/kernel smoke tests. |
| PyTorch | Unknown | Must be collected before choosing an Ops commit. |
| torch_npu | Unknown | Must match PyTorch/CANN; collect before installation. |
| CANN | Unknown | `8c6d73c` documents 9.0.0; current master documents 9.1.0. |
| Triton-Ascend | 3.2.0 | Incompatible with the declared requirements of every official commit containing the public GDR API. |
| MindSpeed Core | `core_r0.16.0@376e9cc3` | Existing local GDN and Triton GDR implementation are used; no external `mindspeed_ops` import exists. |
| MindSpeed-Ops | Not installed/pinned yet | Do not install master into the server environment. |

Verify that the reported `3.2.0` is `triton-ascend`, not an unrelated upstream `triton` distribution, with `python -m pip show triton-ascend triton`.

## 2. Version history and recommendation

| Revision | Relevant content | Declared stack | Suitability |
|---|---|---|---|
| `9076690` master | CausalConv + public GDR; active development | CANN 9.1, torch >=2.7.1, Triton-Ascend 3.2.2, Python >=3.10 | No: server Triton is 3.2.0 and HEAD already contains TA 3.6-oriented changes. |
| `v26.1.0` / `39dc950` | Released tag with both operators | README says Triton 3.2.2, but tag `requirements.txt` pins 3.2.1 | No: neither value matches 3.2.0; metadata is inconsistent. |
| `babee85` | First public `chunk_gated_delta_rule` API | CANN 9.0, torch/torch_npu >=2.7.1,<2.8, Triton-Ascend 3.2.1, Python 3.10.x | No for current server: first full GDR API already requires 3.2.1. |
| `125c8cd` | First standalone GDR `fwd_h` operator | CANN 9.0, torch/torch_npu 2.7.1, Triton-Ascend 3.2.1 | No for current server. |
| `8c6d73c` | CausalConv public autograd wrapper, stateful forward/backward tests | CANN 9.0, torch/torch_npu 2.7.1, Triton-Ascend 3.2.0, Python 3.10.x | Best conditional candidate for a CausalConv-only server probe. It does not contain the public GDR API. |

Recommendation:

1. Do not install or vendor MindSpeed-Ops master.
2. First record exact server torch, torch_npu, CANN, SoC, and package identities.
3. If the server is torch/torch_npu 2.7.1 + CANN 9.0 + Triton-Ascend 3.2.0, use `8c6d73c` only for isolated CausalConv validation. Install with `--no-deps`; do not let pip replace the server stack.
4. There is no official MindSpeed-Ops commit that is both declared compatible with Triton-Ascend 3.2.0 and contains the full public GDR API. Keep the current MindSpeed Core GDR path for the first probe instead of forcing a whole-package integration.
5. `8c6d73c` predates later CausalConv fixes (`0c9fb40` backward optimization, `532a02a` removal of the D/BD divisibility restriction, and arch35 support). Use model-realistic dimensions divisible by the older tile width and treat it as a compatibility probe, not the final long-term version.

## 3. CausalConv stateful-training audit

### Call chain

`mindspeed_ops.api.triton.convolution.causal_conv1d`

→ `CausalConv1dFunction.apply` (`mindspeed_ops/api/triton/convolution.py:104-123`)

→ `CausalConv1dFunction.forward` (`:15-70`)

→ arch32 `causal_conv1d_fwd_impl` (`mindspeed_ops/arch32/triton/convolution.py:525-595`)

→ `causal_conv1d_fwd_kernel` and, when requested, `causal_conv1d_update_states`

Backward:

`CausalConv1dFunction.backward(dy, d_final_state)` (`mindspeed_ops/api/triton/convolution.py:72-101`)

→ arch32 `causal_conv1d_bwd_impl(..., dht=d_final_state, initial_state=...)` (`mindspeed_ops/arch32/triton/convolution.py:598-730`)

→ `causal_conv1d_bwd_kernel`

→ returns `dh0` in the fifth autograd input slot, exactly matching `initial_state`.

### Capability table

| Requirement | Static result | Evidence |
|---|---|---|
| Non-None `initial_state` in forward | Yes | API forwards it at `convolution.py:44-52`; arch32 launcher forwards it at `convolution.py:566-584`; kernel loads it in the head region at `convolution.py:34, 137`. |
| `output_final_state=True` | Yes | `convolution.py:586-593` calls the state-update kernel and returns `[N,D,W]`. |
| Backward receives initial state | Yes | Saved in `ctx` at API `:56-67`, restored at `:79-85`, passed at `:87-97`. |
| Computes `dh0` | Yes | arch32 launcher allocates/reduces it at `convolution.py:687-695,727-730`; the backward kernel writes contributions around `:460-480`. |
| Final-state gradient `dht` | Yes | Autograd backward accepts `d_final_state` at API `:73`, passes it as `dht` at `:90`; kernel launcher forwards it at `convolution.py:699-720`. |
| Returns `dh0` to `initial_state` | Yes | API backward return tuple at `convolution.py:99-101`. |
| Inference-only gate | No code gate | Documentation calls state an inference feature, but public custom autograd and tests explicitly exercise training gradients. |
| Stateful training | Implemented in source on arch32 | Public autograd wrapper and direct backward exist. Must still run on the target NPU stack. |
| Varlen + initial state | Statically wired, not sufficiently tested | Kernels carry both flags and index state per sequence, but official varlen tests use `initial_state=None`. Do not claim this combination supported until an NPU test passes. |

### Shapes, dtypes, and layout

- Input: `x [B,T,D]`, contiguous after `input_guard`.
- Weight: `[W,D]` in arch32 public path.
- Initial/final state: `[N,D,W]`; `N=B` for equal length and `N=len(cu_seqlens)-1` for packed varlen.
- Tests cover FP32, FP16, and BF16 for forward and BF16/FP16 autograd cases. TPR should validate BF16 because it is the intended training dtype.
- Activations: `None`, `silu`, `swish`; bias and residual are optional.
- Master tests include W=2/4/8 and small T. Historical `8c6d73c` should initially use W=4 and a D compatible with its older tiling restriction.
- `input_guard` makes tensor arguments contiguous (`mindspeed_ops/api/triton/utils.py:63-94`), so state identity is not guaranteed, but autograd still receives the returned `dh0` for the original input slot.

### Official tests

`tests/unit_tests/triton/test_causal_conv1d/test_backward.py` contains:

- direct `dh0` checks: `test_grad_with_initial_state*` around `:133-144`;
- split/full forward state continuity: `test_state_continuity` at `:445`;
- direct `dht` propagation: `test_backward_with_output_final_state` at `:490`;
- public wrapper autograd with initial state: `_grad_test`/`test_autograd` at `:583-630`;
- public wrapper final-state gradient: `test_autograd_with_final_state_grad` at `:646`;
- varlen autograd with no initial state at `:664-704`.

Arch35 has a package-wide skip except explicitly marked cases (`tests/unit_tests/triton/test_causal_conv1d/conftest.py:21-28`), so the positive conclusion is strongest for arch32/Ascend 910B-class paths.

### Verdict

For arch32 and non-varlen inputs, the source contains the complete required chain:

`Prefix -> final_conv_state -> Suffix(initial_state) -> backward -> dInitialConvState -> Prefix backward`.

No custom CausalConv backward kernel should be written before the official operator has been tested on the server. The missing work is integration and target-stack validation, not an obvious missing derivative.

## 4. Chunk Gated Delta Rule stateful-training audit

### Call chain

Public API:

`mindspeed_ops.api.triton.chunk_gated_delta_rule.chunk_gated_delta_rule` (`mindspeed_ops/api/triton/chunk_gated_delta_rule.py:239-361`)

→ public initial-state gate at `:329-330`

→ `ChunkGatedDeltaRuleFunction.forward` (`:171-211`)

→ `chunk_gated_delta_rule_fwd` (`:28-74`)

→ arch32 `chunk_gated_delta_rule_fwd_h` imported directly at `:15`

→ second initial-state gate at `mindspeed_ops/arch32/triton/gdn/chunk_gated_delta_rule_fwd_h.py:252-253`

→ Triton fwd-h kernel.

Backward:

`ChunkGatedDeltaRuleFunction.backward(do,dht)` (`chunk_gated_delta_rule.py:213-235`)

→ `chunk_gated_delta_rule_bwd` (`:77-168`)

→ recomputed fwd-h with the saved initial state (`:99-108`)

→ arch-specific `chunk_gated_delta_rule_bwd_dhu(..., h0=initial_state, dht=dht)` (`:118-145`)

→ returns `dh0` (`:168`)

→ public autograd returns `dh0` in the seventh slot matching `initial_state` (`:235`).

### Capability table

| Requirement | Static result | Evidence and caveat |
|---|---|---|
| Public API accepts non-None state | No | Explicit gate at `chunk_gated_delta_rule.py:329-330`. |
| Arch32 launcher accepts non-None state | No | Second gate at `arch32/.../chunk_gated_delta_rule_fwd_h.py:252-253`. Removing only the public gate is insufficient. |
| Arch32 fwd kernel reads `h0` | Code exists | `USE_INITIAL_STATE` heuristic at `fwd_h.py:18`; state loads at `:92-109`; launcher already passes `h0` at `:289`. This path is deliberately unreachable and untested on arch32. |
| Produces final state | Code exists | Launcher allocates FP32 `[N,H,K,V]` at `fwd_h.py:272`, kernel stores it near `:225-237`. |
| Backward computes `dh0` | Code exists and has direct kernel tests | Heuristics at `bwd_dhu.py:17-18`; initializes from `dht` at `:94-105`; writes `dh0` at `:223-234`; launcher allocates/returns it at `:237-296`. |
| Public autograd returns `dh0` | Yes in source | `chunk_gated_delta_rule.py:216-235`. |
| Final-state `dht` enters backward | Yes in source | `backward(ctx,do,dht)` at `:216`; forwarded at `:228`; consumed by bwd-dhu. |
| Initial state saved in ctx | Yes | `ctx.save_for_backward(... initial_state ...)` at `:207`. |
| Stateful top-level F/B tested | No | `tests/unit_tests/triton/test_chunk_gated_delta_rule.py:163-260` calls the public API without state. |
| Arch32 fwd-h state tested | No | `test_chunk_gated_delta_rule_fwd_h.py:544-545` explicitly skips initial state on non-arch35. |
| Arch32 bwd `h0+dht` tested | Yes, direct kernel only | `test_chunk_gated_delta_rule_bwd_dhu.py:726-735` includes equal-length `has_dht=True, has_h0=True`; varlen case has neither. |
| Varlen + state | Not established | Indexing exists in fwd/bwd kernels, but the public docs state varlen requires `initial_state=None`, and tests do not cover the combined path. |

### Classification

For arch32 the result is **A-static, not A-supported**:

- forward kernel code reads nonzero state;
- backward kernel code computes `dh0` and consumes `dht`;
- the custom autograd return chain is structurally complete;
- but two explicit gates make the path unreachable, and official arch32 tests intentionally skip it.

This means no new backward kernel capability is obviously required. The minimum experimental change would be limited to the two launcher gates, followed by numeric tests. However, a failed NPU probe may still reveal a real Triton compile, UB, layout, or numerical defect; static code alone cannot justify deleting the gates in production.

For arch35, the separate fwd-h implementation visibly supports state, but the top-level GDR module imports the arch32 fwd-h directly at `chunk_gated_delta_rule.py:15` rather than using the dispatcher in `mindspeed_ops/api/triton/chunk_gated_delta_rule_fwd_h.py:12-65`. That is an additional routing concern. It does not affect an arch32 server but prevents a blanket all-SoC conclusion.

### Gate history

- Lower fwd-h gate introduced by `125c8cd9` on 2026-05-29.
- Public API gate introduced by `babee85c` on 2026-06-23 with the first overall GDN API.
- The merge-request text only says that the public GDN API and UT/ATK were added. No commit comment or local issue reference explains a specific kernel defect.

### Other constraints

- Public API rejects FP32 (`chunk_gated_delta_rule.py:322-323`); use BF16 for real probes.
- Q/K/V dtypes must match (`:318-321`).
- Arch32 fwd-h allocates `final_state` as FP32 (`fwd_h.py:272`) and bwd-dhu allocates `dh0` as FP32 (`bwd_dhu.py:266`). Official direct state tests also construct FP32 `h0`. Start TPR recurrent-state storage in FP32 even though the public doc example shows BF16, then test any lower-precision state policy separately.
- Layout is `[B,T,H,K/V]`, `head_first=False` is the intended path.
- `beta` must be rank 3 (`:324-327`).
- `K <= 256` in arch32 fwd-h and bwd-dhu (`fwd_h.py:269`, `bwd_dhu.py:256`).
- Varlen requires `B=1` (`chunk_gated_delta_rule.py:344-349`).
- Chunk size is effectively 64 in bwd-dhu (`bwd_dhu.py:255`); tests should start at multiples/non-multiples around 64.

## 5. Current MindSpeed Core relationship

MindSpeed Core `376e9cc3` has not integrated the external `mindspeed_ops` package for GDN. Similar local variable names such as `mindspeed_ops = Builder().load()` refer to Core's own compiled extension, not `import mindspeed_ops`.

### Actual backend routing

`mindspeed/features_manager/megatron_basic/requirements_basic.py:60-70` registers patches into the FLA namespace:

- if `fla_npu` imports:
  - GDR becomes `mindspeed.core.ssm.ops.flash_gated_delta_rule.flash_gated_delta_rule`;
  - causal conv becomes `mindspeed.core.ssm.ops.npu_causal_conv1d.causal_conv1d`;
- if `fla_npu` is absent:
  - GDR becomes `mindspeed.core.ssm.chunk_gated_delta_rule.chunk_gated_delta_rule`;
  - causal conv is not replaced there and remains the installed FLA implementation.

`mindspeed/core/ssm/gated_delta_net.py:37-47` imports both symbols from FLA, then stores the selected GDR callable in `self.gated_delta_rule` at `:194-198`.

The current Core Triton GDR uses its internal code:

- high-level API: `mindspeed/core/ssm/chunk_gated_delta_rule.py`;
- fwd-h/bwd-dhu: `mindspeed/ops/triton/chunk_delta_h.py`.

Its fwd kernel also contains real `h0` loads (`chunk_delta_h.py:92-109`) and its backward kernel contains `dh0/dht`, but its launcher raises `The training does not support initial_state` at `chunk_delta_h.py:251-252`. This is the exact gate hit by the existing TPR forward relay test.

The Core NPU causal-conv wrapper only installs a custom autograd function for the no-state/no-final-state training path (`mindspeed/core/ssm/ops/npu_causal_conv1d.py:254-257,281-283`). Stateful calls bypass that wrapper and call `torch.ops.npu.npu_causal_conv1d` directly (`:259-270,285-299`); this file does not explicitly return a `dInitialState`. Therefore it is not a proven replacement for MindSpeed-Ops CausalConv stateful training.

### Where state is fixed to None

`mindspeed/core/ssm/gated_delta_net.py` hardcodes:

- causal conv `initial_state=None, output_final_state=False` at `:438-447`;
- GDR `initial_state=None, output_final_state=False` at `:466-478`;
- the returned recurrent final state is not exposed; the module returns only `(out, out_bias)` at `:516-521`.

### GDN CP path

The GDN CP implementation is not a ring-attention KV exchange. It transforms layouts:

1. `in_proj` at `gated_delta_net.py:329-332`.
2. CP→HP All-to-All at `:334-378`, using `tensor_a2a_cp2hp` (`:938-1000`) and Megatron `_all_to_all_cp2hp` at `:995`.
3. Each rank now processes the full time dimension for a subset of head/channels; causal conv and GDR execute at `:399-478`.
4. HP→CP All-to-All at `:493-514`, using `tensor_a2a_hp2cp` (`:1003-1070`) and `_all_to_all_hp2cp` at `:1068`.

Consequently, Linear Prefix State should be stored rank-locally in HP/head-sharded layout after CP→HP. A separate raw cross-rank state relay is normally unnecessary as long as every segment uses the same CP group and mapping. CP correctness work must verify rank ownership, mapping stability, collective ordering, and autograd through both A2As.

## 6. Largest actual gaps

The primary missing operator capability is not CausalConv backward. It is a supported, reachable, numerically validated arch32 GDR stateful-training path on the server's Triton 3.2.0 stack.

The primary model-integration gap is that `GatedDeltaNet.forward` has no TPR state ingress/egress contract and discards both final states. Even after the operator gate is validated, TPR needs a non-invasive model adapter/context that:

- supplies per-layer conv and recurrent initial states;
- requests and collects both final states;
- preserves ordinary `ctx=None` behavior exactly;
- leaves CP→HP/HP→CP communication intact;
- returns state gradients to the TPR executor for branch accumulation and prefix recompute/backward.

## 7. Updated minimum development plan

1. **Environment fingerprint.** Record exact Python, torch, torch_npu, CANN, Triton distribution/version, and SoC. Do not change packages.
2. **CausalConv-only Ops probe.** Conditionally checkout `8c6d73c`, install/import with `--no-deps`, and run public-API BF16 full-vs-split forward plus `dh0/dht` tests using real GDN dimensions. If the environment is not torch/torch_npu 2.7.1 + CANN 9.0-compatible, stop and report instead of upgrading.
3. **Current Core GDR ordinary F/B baseline.** Confirm the existing no-state path still matches the Torch reference. This separates general GDR correctness from state support.
4. **Arch32 GDR state kernel probe.** In a test-only experimental checkout/helper, bypass both launcher gates without modifying installed MindSpeed Core. Compare full vs Prefix→state→Suffix for output, final state, Q/K/V/g/beta gradients, `dh0`, and `dht`. Test BF16 equal-length first. Do not start with varlen.
5. **LinearPrefixState CP=1 composition.** Combine official CausalConv state and validated recurrent state. Compare full vs split output, loss, input gradients, all relevant parameter gradients, both `dInitialState`s, and prefix recompute/backward.
6. **GDN model adapter, CP=1.** Add a TPR-side state context/subclass or ModuleSpec replacement later, not a MindSpeed Core source edit. Preserve the original path for `ctx=None`.
7. **CP=2 state-layout test.** Reuse MindSpeed's existing CP→HP/HP→CP A2As. Verify that each layer's Prefix State stays on the same head-owning rank across Prefix and Suffix; no standalone cross-rank state send/receive should be added unless this test proves it necessary.
8. **TPR branches under CP=2.** Only after single-chain CP=2 equivalence: detach shared Prefix State, run sibling branches, locally sum each branch's state gradients on the owning rank, recompute Prefix, inject the summed state gradients, and compare loss/logprob/all parameter gradients against the CP reference.
9. **Varlen/packed and deeper trees last.** These combinations lack adequate official state tests and should not block the first equal-length CP=1/CP=2 proof.

Removed or reduced work compared with the old plan:

- do not implement a new CausalConv backward;
- do not integrate all of MindSpeed-Ops master;
- do not add a separate two-rank Prefix State transport before checking the existing HP ownership;
- do not begin CP=2 until both state derivatives pass CP=1.

## 8. NPU server test checklist

### A. Environment and imports

- Print `platform.machine()`, Python, `torch.__version__`, `torch_npu.__version__`, `triton.__version__`, `pip show triton-ascend triton`, `torch.npu.get_device_name()`, and CANN environment/package versions.
- Confirm imported module paths for GDN, causal conv, and GDR with `callable.__module__` and `inspect.getsourcefile`.
- Confirm the target is arch32 or arch35 before interpreting skips/results.

### B. MindSpeed-Ops CausalConv

Run in this order, BF16 plus a small FP32 diagnostic where supported:

1. import-only smoke test at the pinned commit;
2. ordinary forward/backward versus Torch reference;
3. nonzero initial state changes early outputs correctly;
4. full forward versus Prefix→final_state→Suffix forward;
5. suffix loss produces finite, nonzero `initial_state.grad` matching full reference;
6. a loss on `final_state` propagates `dht` to input and initial state;
7. combined loss on suffix output and suffix final state;
8. real GDN shape/dtype/activation (`W=4`, BF16);
9. only after all above: varlen + one initial state per packed sequence.

### C. GDR

1. current Core ordinary no-state F/B versus its Torch reference;
2. prove the expected public/launcher gates are the only observed Python failure before kernel launch;
3. test-only arch32 state launcher: nonzero `h0` forward and final state versus Torch;
4. suffix output loss: verify finite, nonzero `dh0` and Q/K/V/g/beta gradients;
5. final-state loss: verify `dht` contribution;
6. combined output + final-state loss;
7. full versus split Prefix/Suffix output, loss, final state, and all gradients;
8. repeat with actual GDN H/K/V/chunk size;
9. varlen + state only after equal-length passes.

### D. Composed GDN/TPR

1. CP=1 full versus split `LinearPrefixState`;
2. CP=1 shared Prefix with two siblings and summed state gradients;
3. CP=2 full sequence reference with A2A call counters;
4. CP=2 split sequence with per-rank state shape/data-owner checks;
5. CP=2 two-sibling TPR versus CP reference: loss, logprob, every parameter gradient, finite checks, and collective-order trace.
