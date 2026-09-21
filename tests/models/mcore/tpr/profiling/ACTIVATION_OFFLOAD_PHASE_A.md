# TPR Phase A: CP1 native MindSpeed activation swap

Status: implementation and server acceptance harness; **not NPU-validated locally**.
The Windows development environment has neither PyTorch nor an NPU. No memory,
latency, maximum sequence length or numerical pass result is claimed.

## Implementation and configuration

`verl/models/mcore/tpr/activation_offload.py` installs the existing
`mindspeed.core.memory.swap_attention.prefetch.SwapPrefetch` around each complete
Visit/Pop forward-backward operation. `segment_executor.py` is the only execution
integration point. Push remains no-grad; Pop still recomputes the Prefix.

Use the existing `swap_attention=True` and `swap_modules` MindSpeed configuration,
passed through the launcher's normal MindSpeed repatch/args path. No new TPR
configuration schema or independent transfer manager is introduced. An explicit
`model.config.swap_attention` overrides the initialized global flag, useful for
controlled on/off tests. `model.config.swap_modules`, when present, similarly
overrides the native args value. Do not blindly pass new kwargs to a vanilla
TransformerConfig constructor: its accepted fields depend on the installed stack.

The native default selection is `input_norm,self_attention,post_attention_norm`;
names must match actual Transformer submodules. `self_attention,mlp` additionally
covers ordinary MLP activations and is the acceptance harness default. These are
native selection names, not a new activation policy. No matched targets is an
error; selected targets alone do not prove that any tensor passes native filters.

Reused unchanged: native size/view/leaf filters (including the roughly 512-KiB
minimum tensor size), pinned CPU allocation, native transfer stream, D2H/storage
release, H2D and native PP1 same-layer scheduling. This implementation does not
invent threshold/fraction/pool settings missing from this native revision.

Necessary adaptations:

- VERL bypasses native `setup_model_and_optimizer`: temporarily install the native
  saved-tensor wrappers and layer hooks on the already-built model.
- VERL can initialize MindSpeed args without Megatron training globals: bind the
  native module's argument accessor for the operation, then restore it. Missing
  training-only eval fields default to disabled evaluation scheduling.
- Megatron-Core-only installations do not ship `megatron.training`. For native
  MindSpeed versions that import its `get_args` at module scope, provide that
  accessor from MindSpeed only while importing the installed prefetch module,
  then remove the temporary training module. Other missing dependencies still
  raise their original errors. The NPU fixture uses MindSpeed args directly.
- KV/GDN exports are external live references and Pop gradient roots. Exclude
  their entire storages, including aliases, before native storage release.
  Detached Prefix state and accumulated gradients are not offloaded.
- Direct Pop roots can bypass a layer backward hook. At unpack, invoke native
  same-layer H2D and wait on its stream. Duplicate native handles do not own an
  independent recorded event, so waiting on each handle's event is unsafe.
- Drain the native stream and dispose operation-local queues on success/error;
  restore wrappers/accessor. Reject an existing native installation to avoid
  double wrapping. Execution is sequential, not concurrent across model threads.

The state-storage exclusion synchronizes already-submitted copies when necessary;
this is a correctness cost to measure, not an asynchronous performance claim.
Untraversed branches are discarded when the operation ends. TP/PP/EP must also be
1 in this first implementation. CP>1, graph capture, existing CPU offload,
adaptive swap/recompute and native checkpoint/recompute combinations fail closed.

## Local checks

```sh
python -B tests/models/mcore/tpr/unit/test_activation_offload_adapter.py
```

These dependency-free tests exercise adapter decisions and cleanup using stubs;
they do **not** establish native stream correctness or numerical equivalence.

## NPU acceptance

Sync the bridge overlay into the server's complete VERL checkout. Use the same
Megatron/MindSpeed versions and the verified MindSpeed-Ops PR169 baseline as the
existing TPR tests. Run from that checkout with its usual package paths. Capture
all repository revisions, torch/torch_npu/CANN versions and device type alongside
results; this harness deliberately does not enforce old PR-before Git hashes.

Run this module in a fresh pytest process as shown below. Its runtime installs
MindSpeed before importing model/spec helpers: importing the helpers first can
cache a GPT spec module without `TESpecProvider` on NPU. Dense FA configurations
explicitly set `experimental_attention_variant=None`, `linear_attention_freq=None`
and `transformer_impl="local"`, overriding the runtime's global GDN defaults;
the hybrid configuration retains `linear_attention_freq=4`. The fixture supplies
offload args only to the adapter/native swap accessor and does not replace the
global MindSpeed model-construction arguments with a partial namespace.

```sh
TPR_RUN_OFFLOAD=1 torchrun --nproc_per_node=1 --master_port=29571 \
  -m pytest -sv tests/models/mcore/tpr/profiling/test_activation_offload_phase_a_npu.py -k correctness
```

Four cases: dense FA and 4-layer 3-GDN/1-FA hybrid, each with/without root-owned
loss. Compare off/off/on/on/off using identical weights: loss, per-term logprob,
parameter gradients, Prefix dKV and (hybrid) dConv/dRecurrent state. The no-owned
case exercises Pop with only state/KV backward roots. Repeated operations exercise
queue reuse/cleanup. The on/off tolerance is rtol=2e-3, atol=2e-4 and must not be
relaxed merely to obtain a pass.

`TPR_OFFLOAD_COMPARE` reports `off_repeat`, `on_first`, `on_repeat`, and `off_after`
against the first offload-disabled run, for all four comparison categories. A
numerical mismatch does not stop subsequent comparisons: every failed tensor
reports max absolute error, relative L2, element count and finiteness, then the
test fails at the end with the original tolerance. This distinguishes baseline
repeat variation from differences introduced during/after offload. Hybrid uses
BF16 and tied embedding/output weights; small embedding errors alone do not
establish either nondeterminism or an offload correctness bug. The mismatch
fraction printed as 0.0% by PyTorch is rounded, not zero. Chunked CPU diagnostics
avoid allocating full FP32 copies of the large embedding tensor. These extra
runs and diagnostics apply only to correctness, not capacity/latency timing.

If `off_repeat` already fails, first stabilize the baseline with the existing
MindSpeed operator-level deterministic initialization in a fresh process:

```sh
HCCL_DETERMINISTIC=True CLOSE_MATMUL_K_SHIFT=1 PYTHONHASHSEED=123 \
TPR_RUN_OFFLOAD=1 TPR_OFFLOAD_DETERMINISTIC=1 \
torchrun --nproc_per_node=1 --master_port=29571 -m pytest -sv \
  tests/models/mcore/tpr/profiling/test_activation_offload_phase_a_npu.py \
  -k 'correctness and hybrid'
```

This calls MindSpeed `extend_seed_all` (including strict PyTorch deterministic
algorithms), preserves the original tolerance, and restores its flags/environment
after the module. Model-level `deterministic_mode` stays false: enabling that
flag can select a different GDN implementation and would change the comparison.
Custom Triton kernels may still be nondeterministic; this control is not a
guarantee of a repeatable baseline. Unsupported deterministic operations fail
explicitly. If the baseline still fails, retain the full `TPR_OFFLOAD_COMPARE`
numeric records for baseline/on/off-after before attributing differences to
offload. Do not interpret a baseline failure as an offload pass. Run performance
on/off comparisons with identical deterministic settings; profiles log that flag.

Test-only probes observe real native calls: Push has no transfers; each Visit and
Pop must release positive bytes (storage size actually becomes zero, host buffer
is pinned) and reload positive bytes. `released_bytes` is cumulative release
volume, **not peak memory saved or all D2H bytes**; excluded exports may have been
copied before exclusion. `h2d_bytes` counts submitted native reload volume.
Transfer duration and exposed wait are not available here; use a separate native
profiler trace if needed, without changing the transfer implementation.

Run each profile in a fresh process so CPU high-water RSS and caching allocator
measurements do not contaminate the opposite configuration:

```sh
for suffix in 4096 16384; do
  for offload in 0 1; do
    TPR_RUN_OFFLOAD=1 TPR_OFFLOAD_PROFILE=1 TPR_OFFLOAD=$offload \
    TPR_PREFIX=16384 TPR_SUFFIX=$suffix \
    torchrun --nproc_per_node=1 --master_port=29572 -m pytest -sv \
      tests/models/mcore/tpr/profiling/test_activation_offload_phase_a_npu.py -k capacity
  done
done
```

The profile uses the existing synthetic dense ~0.6B/32-layer model, two siblings,
one warmup and three measured complete Push/Visit/Visit/Pop iterations. It reports
NPU peak allocated/reserved bytes, Linux process CPU high-water RSS (KiB), and
mean synchronized end-to-end latency. No CPU gradient snapshots or instrumentation
wrappers run inside the timed region. An OOM is a failed capacity result, not a
skip/pass. This proxy cannot establish capacity for the real full hybrid training
model or optimizer footprint: repeat the same on/off sweep in that training job.

Also rerun existing non-offload tests (with the baseline suite's existing runtime
requirements): `equivalence/test_tpr_engine_reference_equivalence_npu.py`,
`linear/test_qwen35_hybrid_push_branch_pop_npu.py`, and the existing Ring/QKV-merge
regression suite. No changes were made to Ring/QKV-merge code.

## Phase B boundary

CP is explicitly rejected when offload is enabled. Before expanding it, verify
communication-buffer lifetimes/aliases, multi-root Ring backward ordering versus
native layer queue indices, collective/transfer stream dependencies, and peak
memory from native same-layer restore. No source-wise restore, prefetch policy,
Prefix graph retention, Prefix KV/dKV offload or checkpoint compatibility was
implemented. Offload reduces selected saved activations, not detached Prefix
storage, logits, parameters or optimizer state; it need not solve every OOM.
