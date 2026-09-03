# TPR tests

- `unit/`: CPU-friendly component contracts and scheduler behavior.
- `equivalence/`: NPU forward/backward equivalence through attention, model, and engine layers.
- `correctness/`: full-result and multi-step optimizer comparisons.
- `profiling/`: opt-in latency, memory, capacity, and long-context measurements.
- `parallel/`: phase-two Context Parallel and Linear Attention capability gates.

Run tests from the verl repository root so package-relative test helpers resolve consistently.
