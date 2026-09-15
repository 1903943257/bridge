# Ordinary whole Ring alignment

The explicit production entry `ordinary_ring_cp_attention` delegates to paired
MindSpeed 376e9cc's native `ringattn_context_parallel` (SBH, causal, no padding,
dropout zero, full KV cache). MindSpeed owns RingP2P, streaming step order,
2-call CP2 grouping, output/stat cast boundaries, reverse backward and owner dKV.
No kernel changes, clamp, custom accumulation or additional gradient SUM.

`ring_cp_attention` remains the old TPR implementation. Empty prefix is NOT an
automatic dispatch condition: segmented root P also has empty prefix. The test
installer selects the new entry only for an explicitly whole execution and
rejects external prefix blocks. Packed/padded and segmented migration is deferred.

Reference: MCore 55ac708's TE pin 5671fd36 (v2.12) confirms the causal grouping
and reverse replay structure; native MindSpeed, not TE GPU kernels, executes.

## Server commands (from verl repository)

CPU adapter contracts:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_ordinary_ring_adapter.py
```

Fixed-QKV diagnostic (CP1, AllGather, old Ring, native Ring):

```bash
mkdir -p tests/models/mcore/tpr/logs
torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v tests/models/mcore/tpr/parallel/test_ordinary_ring_schedule_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Full 24-layer whole trajectory, same initial tensors and suffix-only objective:

```bash
STAGE44_WHOLE_ONLY=1 STAGE44_MATRIX_LINEAR_CONTROL=1 torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v tests/models/mcore/tpr/parallel/test_qwen35_trajectory_transport_npu.py >> tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Keep the existing CP1 projection control identical across all transports. It is
reported explicitly; this is not an unmodified-model PASS claim. No FA clamp.

Single-layer assertions: old 5 calls vs native 2; native shapes rank0
64x64 causal then 32x64 full, rank1 64x64 causal then 64x32 full; backward reverse.
Log output/dQ/dK/dV and native merged LSE vs CP1. LSE rather than raw max/sum
is compared because equivalent decompositions can use different max offsets.
Full-model logs loss/logprob/input/parameter gradients; GDN A2A and FA/P2P
contracts remain asserted. Numeric diagnostics do not imply training PASS and
do not relax existing correctness thresholds. NPU results pending server run.
