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

The whole trajectory additionally prints `TRAJECTORY-L4` for every transport.
`native_ring` requires one native-entry hit, two SBH FA calls and reverse-order
backward with the same shapes as the single-layer test. `ring` requires five TND
calls. Trace keeps metadata only, tags L4 autograd contexts for backward, and
does not trace segmented execution. CP1/AllGather automatic kernel backward
may not appear in the Python-level `npu_fusion_attention_grad` trace; an empty
list there is not evidence that autograd skipped backward.

## Single-layer layout matrix

The same `test_ordinary_ring_schedule_npu.py` command now also executes
`cp1_sbh` (one whole SBH call) and `native_tnd` (native two-step schedule with
test-only SBH/TND kernel-boundary conversion). Production Ring is unchanged.
Both additions use the same QKV and full-query upstream as existing paths.
Native TND preserves native communication, correction/casts and reverse VJP;
traces record the actual TND kernel arguments, not the shim's SBH arguments.

Printed paired comparisons: whole TND/SBH, native 2-call TND/SBH, old TND
5-call/native TND 2-call, whole SBH/native SBH. Every pair reports output,
dQ/dK/dV and LSE. The old/new TND pair still includes native merge and reduction
differences; it is not a pure FA-call-count-only ablation. No thresholds changed.

## Native output oracle / official-style GQA

`test_native_ring_output_oracle_npu.py` is standalone from VERL attention and
runtime helpers. Its HCCL WORLD is CP2; singleton inner window, no overlap,
cache_policy=None matches ordinary official UT configuration. Reference is
whole SBH FA. QKV/upstream use the prior matrix seed, BF16, full-query VJP.
Original official test uses MHA/longer sequences and is marked skipped upstream;
this is an adapted GQA control, not a claim of reproducing an upstream PASS.

```bash
mkdir -p tests/models/mcore/tpr/logs
torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v tests/models/mcore/tpr/parallel/test_native_ring_output_oracle_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Two tests: direct official-style output/dQ/dK/dV metrics; captured two-step
output oracle on CPU FP32 (no float64). Compare native vs same-kernel FP32
correction both before/after final BF16 rounding, kernel context merge vs FP32
block computation, and FP32 decomposition vs whole FP32. Save CPU captures to
`logs/native_output_oracle_rank{0,1}.pt` (replaced on rerun). Only metadata/metrics
are printed. Shapes and finiteness are asserted; numerical metrics are diagnostic,
not relaxed correctness gates. NPU execution remains pending server validation.
