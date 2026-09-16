# Stateful causal_conv1d backward reproducer

Run from the Bridge repository root in the Ascend environment with MindSpeed-Ops
installed. The script imports that installed package directly and prints its
actual source path and commit; it does not import Bridge/VERL:

```bash
python -u repro_causal_conv1d_initial_state_bwd_oob.py > dh0_repro.log 2>&1
```

The default launches three fresh Python processes sequentially, continuing even
if a child fails. Each process uses one NPU; there is no distributed setup.
Alternatively:

```bash
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case none
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case state_no_grad
python -u repro_causal_conv1d_initial_state_bwd_oob.py --case state_grad
```

All cases use BF16 x `[1,64,3072]`, weight `[4,3072]`, and (when present)
initial_state `[1,3072,4]`. x and weight always require gradients. This is the
actual operator BTD/WD layout, also used by the Bridge stage-1 adapter. Bias,
residual, activation and cu_seqlens are None; output_final_state is False to
isolate initial-state backward. The loss is `y.float().sum()`. No upper model
layers, fixtures, custom autograd implementation or UniAgent patches are loaded.
No wrapper gate bypass is necessary in the inspected public API.

## Confirmed call chain

```text
reproducer
 -> mindspeed_ops.api.triton.convolution.causal_conv1d
 -> CausalConv1dFunction.apply / forward
 -> arch32.triton.convolution.causal_conv1d_fwd_impl
 -> loss.backward()
 -> CausalConv1dFunction.backward
 -> arch32.triton.convolution.causal_conv1d_bwd_impl
 -> causal_conv1d_bwd_kernel
 -> USE_INITIAL_STATE branch / dh0 store
```

The API selects arch32 when `is_arch35()` is False and rejects arch35. The
script prints loaded module paths, source hash, git commit/status, signature,
device, package versions and relevant store guard source lines. A Python trace
observes the real backward function's BT, NT, eff_NT and dh0.shape before launch;
it does not replace the kernel or change its inputs. This is a diagnostic script,
not a timing benchmark. The pre-call formula is explicitly labeled as derived
from the inspected checkout; observed locals are authoritative if code changes.
Python tracing is best-effort: autograd can execute backward on another thread,
outside the scope of `sys.settrace`. Missing trace events produce a diagnostic
message, not a failed case. In that situation actual runtime path/BT/allocation
remain unverified. Successful backward plus NPU synchronization and required
gradient presence determine execution PASS; they do not establish numerical
correctness or verify the internal dispatch path.

## Important source-version distinction

The local checkout at preparation was `c6d457d` ("bound initial-state gradient
stores to head tiles"). It already has:

```python
if USE_INITIAL_STATE and i_t * BT < W - 1:
```

The parent `a9a97b58df5ff72867e74ed6687c087075d5f445` has the unguarded
`if USE_INITIAL_STATE:` at this store. No operator source was changed or reverted
for this reproducer. Run the standalone script against the exact affected server
checkout to reproduce the original failure; do not infer failure from tile
counts on a guarded checkout. This script introduces no dh0 numerical patch.

With observed BT=2: eff_NT=32 and allocated state-gradient tiles=2. There are
30 excess time-tile indices **if stores are unguarded**. This is a potential
addressing violation, not proof that a particular device exception has that
cause. Runtime error text is preserved, and no further device operations are
attempted after an exception. PASS only means forward/backward completed and
required gradients exist, not that their numerical values are correct.

## Actual local validation (2026-09-16)

Python 3.12.9 on Windows. torch, torch_npu and triton are absent; no Ascend
device is available in this execution environment. Syntax validation passed.
The default command launched all three child processes; each stopped at Setup
with `ModuleNotFoundError: No module named 'torch'` (exit 2). Thus:

| Case | Local device result |
|---|---|
| initial_state=None | NOT RUN: missing dependencies/device |
| state requires_grad=False | NOT RUN: missing dependencies/device |
| state requires_grad=True | NOT RUN: missing dependencies/device |

The reported Ascend 910B2C / arch32 PASS, FAIL, FAIL and 507035/UB error remain
user-provided observations, not measurements obtained here. Server execution
and its resulting log are still required for a verified standalone reproduction.
