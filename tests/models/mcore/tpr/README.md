# TPR tests

- `unit/`: CPU-friendly component contracts and scheduler behavior.
- `equivalence/`: NPU forward/backward equivalence through attention, model, and engine layers.
- `correctness/`: full-result and multi-step optimizer comparisons.
- `profiling/`: opt-in latency, memory, capacity, and long-context measurements.
- `parallel/`: phase-two Context Parallel and Linear Attention capability gates.

Run tests from the verl repository root so package-relative test helpers resolve consistently.

## P0-1 native CP numerical control

Update only `/workspace/bridge` using `git pull --ff-only`, then synchronize
`/workspace/bridge/tests/models/mcore/tpr/` into
`/workspace/uni-agent/verl/tests/models/mcore/tpr/` with `rsync -av`.
In `/workspace/uni-agent/verl` run
`bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh`.
The launcher uses `127.0.0.1` plus explicit ports for CP1/2/4;
it never uses standalone rendezvous. See
`/workspace/bridge/docs/p0_native_cp_drift_20261011.md`.
