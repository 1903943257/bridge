# Prefix FULL block coalescing

## Status

KV coalescing is an opt-in first-stage optimization. Default is OFF:
`TPR_RING_COALESCE_PREFIX_FULL=0`. ON is `1`. The setting is captured in each
attention invocation's saved config, so backward uses the forward decision.
Optional second-stage Q coalescing is described below. Neither stage changes
masks, checkpointing, scheduler or transport.

Local validation executes the actual production forward/backward dispatch
loops with shape-only tensor/FA/transport stubs, for every rank in CP=2/4.
It proves dispatch counts and view selection, not numerical equivalence or
NPU memory/latency. NPU correctness and performance have NOT been run locally
(PyTorch/NPU unavailable). Do not treat this as a validated performance win.

## Baseline decomposition from the executor

`_circulate_kv` first gathers source buffers in a source-indexed tuple. It
retains all source buffers for backward. FA execution then iterates query
chunk -> segment -> source rank -> KV chunk. Thus “ring_step” in the trace
means receive origin `(rank-source_rank) % CP`, not FA execution order.
This existing transport/storage behavior is unchanged.

For CP=4 and P=S=16384, each chunk contains 2048 tokens. Rank 0 owns Q0
`[0,2048)` and Q7 `[14336,16384)` in suffix-local coordinates (add 16384
for trajectory positions). Every source buffer contains two physical chunks:

| Receive step | Source | Prefix/current chunks | Prefix FA OFF/ON | Current FA |
|---:|---:|---|---:|---:|
| 0 | 0 | K0 [0,2048), K7 [14336,16384) | 4 / 2 | 3 |
| 1 | 3 | K3 [6144,8192), K4 [8192,10240) | 4 / 2 | 2 |
| 2 | 2 | K2 [4096,6144), K5 [10240,12288) | 4 / 2 | 2 |
| 3 | 1 | K1 [2048,4096), K6 [12288,14336) | 4 / 2 | 2 |

Each Q can see all Prefix chunks: 2 Q x 4 sources x 2 KV chunks = 16.
Current Q0 sees K0 (causal) and skips K1..K7; Q7 sees K0..K6 (FULL)
and K7 (causal): 9 calls. Other ranks have the same total via zigzag balance.
`_block_attention_forward` returns output, softmax max and sum;
`_merge_attention` updates the FP32 online-softmax accumulator after each
visible block (the first block initializes it). There was no internal
Prefix coalescing.

For 28 layers and 8 leaves, per-rank expected counts are:

| Phase | OFF FWD | ON FWD | OFF BWD | ON BWD |
|---|---:|---:|---:|---:|
| Push | 252 | 252 | 0 | 0 |
| Leaves | 5600 | 3808 | 5600 | 3808 |
| Pop | 252 | 252 | 252 | 252 |
| Total | **6104** | **4312** | **5852** | **4060** |

Reference remains 8 x 28 x 9 = 2016 FWD and BWD. Leaf changes 25 -> 17.
Those totals match the supplied baseline; optimized totals are dispatch-test
results, not measured NPU FA counters yet.

## Implementation and memory

`_prefix_full_slices` uses the entire source K/V tensor as a view when that
Prefix segment has no padding. Its two non-adjacent logical ranges already
occupy adjacent physical rows. FULL visibility permits one FA over those rows
without positional masking. Query chunks stay separate. Synthetic range
metadata represents packed positions only and is never used for Current
causal attention. Trace reports the original logical ranges/chunk IDs.

Backward consumes the same view with the final global attention output and
softmax statistics. Its dK/dV rows already have the original source-buffer
order; one `add_` to the full local slice replaces the two slice updates.
Existing Ring dKV reduction and Prefix gradient accumulation are unchanged.

Padded Prefix segments fall back to original per-chunk dispatch. There is no
new `cat`, packing workspace or persistent KV copy. FA's internal workspace
can still change with the larger block; actual allocated/reserved peaks must
be measured. Root Push/Pop in the fixed single-root benchmark have no external
Prefix and retain the original dispatch.

## Correctness commands (CP=2, then CP=4)

Run each command with both process counts before timing:

```bash
torchrun --nproc_per_node=2 --master_port=29563 -m pytest -s -v -x \
  tests/models/mcore/tpr/parallel/test_ring_cp_attention_npu.py

TPR_RUN_RING_COALESCING_TREE=1 \
torchrun --nproc_per_node=2 --master_port=29564 -m pytest -s -v -x \
  tests/models/mcore/tpr/parallel/test_ring_coalescing_tree_npu.py
```

Single-layer tests compare OFF/ON output, dQ/dK/dV and the independent dense
oracle, require nonzero Prefix gradients, and check actual FA call reductions.
They include multiple Prefix segments and a padded fallback. Tree tests compare
loss, logprobs, all parameter gradients and Prefix KV gradients against OFF and
independent segmented trajectories, including a repeated ON iteration. Existing
pointwise and global numerical tolerances are reused unchanged.

## Fixed performance experiment (only after correctness passes)

`_PROFILE_CASES` in the profile file now contains only the requested case:

```python
_PROFILE_CASES = (_ProfileCase(16384, 16384, 8),)
```

Run OFF, then repeat with `TPR_RING_COALESCE_PREFIX_FULL=1`, keeping all other
settings unchanged:

```bash
TPR_RUN_QWEN_RING_CP_PROFILE=1 TPR_QWEN_PROFILE_SIZE=1.7B \
TPR_RING_COALESCE_PREFIX_FULL=0 TPR_RING_BLOCK_TRACE=1 \
torchrun --nproc_per_node=4 --master_port=29565 -m pytest -s -v -x \
  tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py
```

The untimed probe prints actual dispatch counts by Push/Visit/Pop, FWD/BWD,
Prefix/Current. Optional detailed trace prints only rank 0, first layer and
first leaf with external Prefix: receive step, source, Q/KV chunk IDs/ranges,
FULL/CAUSAL/SKIP, physical lengths, and whether FA is dispatched. Timing runs
do not install the trace callback. Preserve both logs and compare Reference/TPR
latency, speedup, FA counts, Ring communication, FA, merge/framework, complete
attention time, peak allocated and peak reserved. No optimized timing or peak
memory numbers are available yet.

## Second stage: zero-copy Prefix Q coalescing

Enable `TPR_RING_COALESCE_PREFIX_FULL=1` and
`TPR_RING_COALESCE_PREFIX_QUERY=1`. Q coalescing also defaults OFF.

### Layout assessment

In the existing forward, `query.squeeze(1).contiguous()` creates `query_tnd`;
`_iter_range_slices` takes adjacent row views from it. For the fixed Qwen3-1.7B
case, each CP rank has 4096 local query rows, 16 heads and head_dim 128:

```text
Q TND shape: [4096, 16, 128]
contiguous stride: [2048, 128, 1] (elements)
Q0 physical rows: [0:2048],     offset = base_offset
Q1 physical rows: [2048:4096],  offset = base_offset + 4194304 elements
Q01: entire existing TND buffer; no cat or Q packing workspace
```

The two logical ranges are non-adjacent, but physical rows are adjacent. Their
RoPE positions were already applied. The FULL kernel uses TND with
`actual_seq_qlen=[4096]`, `actual_seq_kvlen=[source_local_length]`. Query rows
do not attend to each other, so combining them does not change visibility.
The detailed untimed probe prints actual storage pointer, offset, shape and
stride for the first rank-0/layer-1 leaf. Numeric addresses are runtime values;
no NPU layout measurement has been obtained locally.

Eligibility checks the ORIGINAL `query.squeeze(1)` before the baseline
contiguous conversion. If it is not contiguous, any segment is padded, KV
merge is OFF, or no external Prefix exists, Q coalescing falls back. For
unpadded Prefix + padded Current this is the 17-block KV-only path; padded
Prefix also retains its original per-chunk KV fallback. No Q packing is used
to force eligibility.

### Forward/backward and final statistics

Prefix FULL execution uses all local Q rows once per Prefix source. One online
softmax accumulator spans those rows; it is then split into chunk views for
the unchanged Current causal loop. Splitting output/dQ along sequence rows is
zero-copy. FA backward sums contributions to shared dK/dV from both query
chunks, and each source's dKV buffer receives that result exactly once.

CANN TND statistics expose shape `[T,H,8]` with single-sequence **head-major**
storage `[H,T,8]`. They cannot be split with `stats[:T/2]`. The helper slices
the query axis of `[H,T,8]` and makes a contiguous small chunk statistic. The
final full-query max/sum pair is assembled once after Current attention and
saved in place of the old per-chunk pairs. Backward uses these FINAL global
statistics and output, not the Prefix-only intermediate normalizer. It then
slices the final result for the unchanged Current backward calls.

The returned full output already existed in the baseline and is reused as
the saved full output. There is no extra output concat, Q workspace or saved
per-source Prefix output. The saved max/sum element count is unchanged. The
fixed case's full FP32 max/sum pair totals 4 MiB; conversion and larger FA
outputs can affect transient peaks. Zero-copy Q does NOT establish that peak
memory or runtime are unchanged: allocated/reserved peaks and kernel workspace
must be measured. No memory/performance improvement is claimed before that.

### Dispatch and transport gates

The dependency-free test executes the production dispatch loops using shape
stubs and checks every CP=2/4 rank, Q-buffer aliasing, and communication wrapper
calls. CP=4 expectations for the fixed case are:

| Metric | KV-only | Q+KV |
|---|---:|---:|
| Prefix FA per leaf/layer | 8 | 4 |
| Current FA per leaf/layer | 9 | 9 |
| Total per leaf/layer | 17 | 13 |
| Prefix FA across leaves, each direction | 1792 | 896 |
| Total forward FA | 4312 | 3416 |
| Total backward FA | 4060 | 3164 |
| Forward communication wrappers | 504 | 504 |
| Backward communication wrappers | 476 | 476 |

The shape tests passed locally; the NPU tests have NOT been run. The existing
single-layer test now includes Q OFF/ON with Qwen-like GQA/head_dim, multiple
Prefix segments, padded Prefix/Current and non-contiguous Q fallback. Both
modes are checked against the dense oracle and each other with existing
tolerances, including nonzero Prefix gradients and unchanged transport counts.
Tree tests now parameterize KV-only and Q-merge comparisons, using the same
loss/logprob/parameter/Prefix-gradient tolerances and repeated ON iteration.

Run the two correctness commands above with CP=2 and then CP=4; no additional
Q switch is needed because these tests set it internally. Also run the
head-major layout contract checks where PyTorch is installed:

```bash
python -m pytest -q tests/models/mcore/tpr/parallel/test_ring_cp_attention_contract.py
```

After all correctness gates pass, keep the fixed case/model/CP and compare:

```bash
mkdir -p tests/models/mcore/tpr/log1
for QMERGE in 0 1; do
  TPR_RUN_QWEN_RING_CP_PROFILE=1 TPR_QWEN_PROFILE_SIZE=1.7B \
  TPR_RING_COALESCE_PREFIX_FULL=1 TPR_RING_COALESCE_PREFIX_QUERY=$QMERGE \
  TPR_RING_BLOCK_TRACE=1 TPR_QWEN_RING_CP_PROFILE_BREAKDOWN=1 \
  torchrun --nproc_per_node=4 --master_port=29565 -m pytest -s -v -x \
    tests/models/mcore/tpr/profiling/test_tpr_qwen3_ring_cp_profile_npu.py \
    > tests/models/mcore/tpr/log1/query_merge_${QMERGE}.log 2>&1 || break
done
```

Compare TPR median, attention and FA forward/backward, actual FA counters,
merge/framework, Ring communication, peak allocated and reserved. The Q switch
must change neither Reference dispatch nor Ring transport counts. At present
there are no measured Q-merge latency or peak-memory results.
