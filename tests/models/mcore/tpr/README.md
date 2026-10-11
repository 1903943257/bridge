# TPR tests

- `unit/`: CPU-friendly component contracts and scheduler behavior.
- `equivalence/`: NPU forward/backward equivalence through attention, model, and engine layers.
- `correctness/`: full-result and multi-step optimizer comparisons.
- `profiling/`: opt-in latency, memory, capacity, and long-context measurements.
- `parallel/`: phase-two Context Parallel and Linear Attention capability gates.

Run tests from the verl repository root so package-relative test helpers resolve consistently.

## P0-1 native CP numerical control

On the **host** run `cd /workspace/bridge && git pull --ff-only`.
Then **inside Docker** synchronize
`/workspace/bridge/tests/models/mcore/tpr/` into
`/workspace/uni-agent/verl/tests/models/mcore/tpr/` using `rsync -av`.
Still inside Docker, run
`cd /workspace/uni-agent/verl` and
`bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh`.
Never pull Git repositories from the Docker runtime checkout.
The launcher uses `127.0.0.1` plus explicit ports for CP1/2/4;
it never uses standalone rendezvous. See
`/workspace/bridge/docs/p0_native_cp_drift_20261011.md`.

## P0-B real-TQ 50-step native CP training control

On **host** `cd /workspace/bridge && git pull --ff-only`. Inside
**Docker** `rsync -av /workspace/bridge/tests/models/mcore/tpr/
/workspace/uni-agent/verl/tests/models/mcore/tpr/`, then in
`/workspace/uni-agent/verl`, set `TPR_REAL_TQ_BATCH` and run
`bash tests/models/mcore/tpr/parallel/run_native_cp_multistep.sh`.
This is NLL with FP32 master AdamW and fixed real TQ row order,
**not** production PPO or a Megatron DistributedOptimizer benchmark.
See `/workspace/bridge/docs/p0_native_cp_multistep_20261011.md`.
