# Stateful Conv dh0 compact-buffer store fix

Scope: fix compact dh0 writes in the current arch32 non-packed stateful
backward. No gate override, no threshold change, no GDR/Ring/scheduler edit.

For B=1,T=64,D=3072,W=4,BT=2, allocation is [2,1,3072,4] but the old kernel
iterates i_t=0..31 and stores all those partials. Masking dy loads to zero
does not make an out-of-bounds store valid. The previous global-time read fix
keeps correct summands but did not address this already-existing store bug.

Only suffix t < W-1 can depend on h0. The new guard
`USE_INITIAL_STATE and i_t * BT < W - 1` restricts the **whole dh0 block** to
contributing head tiles. Allocation and final tile SUM remain unchanged.
Every valid contribution stays in exactly one tile; unused slot0 remains
zero. dx/dw/db and final-state dht-to-dx code are outside this guard and remain
unchanged. Frozen h0 still takes the kernel state branch and computes dh0;
autograd discards that gradient. Optimizing this is deferred, not used as fix.

For T=1024,BT=8, allocation has one tile and only i_t=0 writes; all other
tiles still compute their normal non-dh0 gradients. This repairs a provable
global-memory indexing defect; the server must verify whether it also removes
the reported 507035 UB exception. Packed/multi-sequence layout is not certified
by this regression; in particular B vs number-of-sequences strides need a
separate audit before THD rollout.

## Deployment

This is an **incremental** patch after the earlier dh0 global-time read fix.
From the server MindSpeed-Ops repository (adjust sibling bridge path):

```bash
git apply --check ../bridge/patches/mindspeed_ops_causal_conv_dh0_store_guard.patch
git apply ../bridge/patches/mindspeed_ops_causal_conv_dh0_store_guard.patch
```

If the server keeps the kernel inside `api/triton/convolution.py` instead of
`arch32/triton/convolution.py`, do not force a failed patch: use OPS-SOURCE
to identify the actual kernel and port the same guard there. Do not apply to
an API forwarding function. Restart failed worker processes after updating.

## Validation order

1. Repeat standalone OPS A/B (especially frozen/train, unused/nonzero dht).
2. Run canary + CPU reference tests from the server verl test repository:

```bash
mkdir -p tests/models/mcore/tpr/logs
python -m pytest -s -v tests/models/mcore/tpr/parallel/test_conv_dh0_bounds_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Thirty cases cover T=64/128/256/512/1024, frozen/trainable h0, final output
off/unused/nonzero. Four additional cases force BT=1/2/4/8 via test-only core
count control. Guard zones reserve the old erroneous address extent, so an
old unguarded store is detected as exact sentinel corruption without relying
on an allocator-dependent crash. dx/dw/db/dh0 compare to independent CPU FP32
SiLU autograd with existing OPS BF16 atol=rtol=0.05; canaries and unused h0
slot0 must be exact. Other known dw failures are not waived.

3. Rerun Stage 3.2 and Stage 4.5 N=2,P=8192,S=1024 with sync diagnosis.
4. Only after successful diagnostic regression rerun performance without
   instrumentation. No performance or training PASS is claimed from a patch.

Local validation: static syntax, CPU head-tile coverage/bounds arithmetic and
patch application checks; NPU execution remains pending.
