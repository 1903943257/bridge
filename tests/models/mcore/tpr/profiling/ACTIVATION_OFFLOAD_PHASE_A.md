# TPR Phase A: CP1 Full-Attention native MindSpeed activation swap

Status: implementation plus NPU acceptance harness. Phase A is intentionally
**Full-Attention only**. GDN/Hybrid support is not implemented or accepted here.

## Scope

Phase A reuses MindSpeed's native
`mindspeed.core.memory.swap_attention.prefetch.SwapPrefetch` for one complete
TPR Visit/Pop forward-backward operation. Push remains graph-free/no-grad and
never enters the native swap queue; Pop keeps the existing TPR Prefix recompute
semantics.

The supported model shape for this phase is CP=TP=PP=EP=1 Dense Full Attention.
If activation offload is enabled on a model containing a TPR GDN layer, validation
fails closed with a Full-Attention-only error. Hybrid is deliberately deferred
rather than partially supported.

The adapter protects only FA Prefix/current KV storages exported through the TPR
context. `_iter_extra_external_tensors(context)` is an intentionally empty
extension seam for a later non-FA/Hybrid phase. Phase A does not protect, offload,
or otherwise interpret GDN Conv/Recurrent state.

## Native implementation reused

`activation_offload.py` installs the existing native saved-tensor wrappers and
layer hooks on VERL's already-built model because VERL bypasses MindSpeed's normal
`setup_model_and_optimizer` installation path. The implementation keeps native:

- `swap_attention=True` and `swap_modules` configuration;
- tensor-size/view/leaf filters and the native minimum-size policy;
- pinned CPU allocation and transfer stream;
- D2H, NPU-storage release, H2D and PP1 same-layer restore scheduling.

No independent TPR transfer manager, threshold, pool or prefetch policy is added.
`model.config.swap_attention` and `model.config.swap_modules`, when explicitly
present, are used for controlled tests without changing the normal launcher path.

Necessary adapter seams are limited to: temporarily installing/restoring native
wrappers, bridging the native args accessor in Megatron-Core-only environments,
protecting live FA KV aliases from native `resize_(0)`, and forcing native
same-layer H2D when Pop's direct dKV roots bypass a layer backward hook. Native
streams/queues are drained and cleared on both success and failure.

Phase A rejects CP>1, TP/PP/EP>1, graph capture, existing CPU offload,
fine-grained/adaptive swap/recompute, native checkpoint/recompute combinations,
VPP and LoRA combinations already rejected by the adapter.

## Local checks

```sh
python -B tests/models/mcore/tpr/unit/test_activation_offload_adapter.py
```

The unit suite is dependency-free. It verifies lifecycle cleanup, Full-Attention
scope rejection for GDN/Hybrid, KV-storage protection, the reserved external-state
extension seam, direct-root H2D fallback and Core-only MindSpeed import behavior.
It does not establish NPU numerical or stream correctness.

## NPU correctness acceptance

Run the Phase A module in a fresh process after syncing this branch into the
server VERL checkout:

```sh
TPR_RUN_OFFLOAD=1 torchrun --nproc_per_node=1 --master_port=29571 \\
  -m pytest -sv \\
  tests/models/mcore/tpr/profiling/test_activation_offload_phase_a_npu.py \\
  -k correctness
```

There are exactly two correctness cases: FA with root-owned loss and FA without
root-owned loss. The latter forces Pop to run from relayed dKV roots without a
Prefix-owned loss term.

Each case uses the same model/weights and runs:

```text
off reference -> off_repeat -> on_first -> on_repeat -> off_after
```

The test compares loss, per-term logprob, all parameter gradients and Prefix dKV.
It uses the existing rtol=2e-3/atol=2e-4 gate; a mismatch is reported with max
absolute error, relative L2, element count and finiteness, while remaining
comparisons continue so cleanup/on-off behavior is still visible. No Hybrid or
GDN model is constructed by this Phase A test.

The transfer probe wraps real native `SwapTensor.wait_d2h_finished` and
`launch_h2d`. Push must transfer zero bytes. With offload enabled, every Visit
and Pop must release and reload positive native-selected activation volume.
`released_bytes` and `h2d_bytes` are cumulative traffic counters, not peak
memory savings.

## Capacity and latency

Run off/on in separate fresh processes:

```sh
for suffix in 4096 16384; do
  for offload in 0 1; do
    TPR_RUN_OFFLOAD=1 TPR_OFFLOAD_PROFILE=1 TPR_OFFLOAD=$offload \\
    TPR_PREFIX=16384 TPR_SUFFIX=$suffix \\
    torchrun --nproc_per_node=1 --master_port=29572 -m pytest -sv \\
      tests/models/mcore/tpr/profiling/test_activation_offload_phase_a_npu.py \\
      -k capacity
  done
done
```

The capacity test uses the existing synthetic dense Full-Attention ~0.6B/32-layer
model with two siblings, one warmup and three measured Push/Visit/Visit/Pop
iterations. It reports synchronized mean latency, NPU peak allocated/reserved
bytes and process CPU high-water RSS. An OOM is a failed capacity result.

The proxy does not claim optimizer/full-training capacity. Once Phase A FA
correctness and capacity are closed, end-to-end FA training should be measured
with the same native swap configuration.

## Deferred extension boundaries

Two independent extensions are deliberately left outside Phase A:

1. **Context parallelism**: validate communication-buffer lifetime/aliasing,
   Ring backward ordering, collective/transfer-stream dependencies and native
   restore scheduling before enabling CP>1.
2. **Hybrid/GDN**: define ownership and lifetime of Conv/Recurrent Prefix state,
   decide which non-FA exported storages must stay resident, and then implement
   that policy through the reserved external-state seam. No Hybrid correctness
   or performance gate belongs to Phase A.

Phase A offloads selected saved activations only. Detached Prefix KV, accumulated
dKV, logits, parameters and optimizer state remain outside native activation swap.
