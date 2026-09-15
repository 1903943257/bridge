# MindSpeed-native Qwen3.5-0.8B CP baseline

`test_mindspeed_native_qwen35_cp_npu.py` uses the existing native P3 factory:
TransformerConfig + GPTModel + native experimental-attention ModelSpec,
24 layers / 18 GDN / 6 FA, random identical weights, BF16, dropout zero.
Bridge is not needed/installed on the server; no TPR ModelSpec, Engine,
scheduler, state adapter, Ring backend, AllGather or Linear shape control.

CP1 uses singleton CP groups. CP2 uses native GDN A2A and MindSpeed DPA's
`ringattn_context_parallel` directly, verified by function identity and six
actual calls. CP1 must have zero A2A/Ring; CP2 must have 108/18 GDN A2A.
No stateful primitive replacement is installed. Server dirty native GDN/Ops
environment remains authoritative and is printed by the existing runtime audit.

The objective is next-token CE on the whole sequence (excluding its final
token), global denominator S-1. CP2 parameter SUM occurs once. Models execute
sequentially using the same CPU initial-state snapshot. Layer inputs/outputs
and gradients are gathered into logical sequence order and compared on CPU.
Print 24 compact layer lines and aggregate parameter rel/cos/norm ratio/worst.
Layer boundary first divergence is not an operator-level first-divergence claim.

All original P3 numerical thresholds are preserved; no diagnostic clamp.
This is a native whole-sequence baseline, not the previous suffix-only TPR
trajectory objective, and not a multi-step convergence certification.

Run from server verl root (128 tokens explicitly):

```bash
mkdir -p tests/models/mcore/tpr/logs
STAGE34_SEQUENCE_LENGTH=128 torchrun --master_addr=127.0.0.1 --master_port=29557 --nproc_per_node=2 -m pytest -s -v tests/models/mcore/baseline/test_mindspeed_native_qwen35_cp_npu.py > tests/models/mcore/tpr/logs/stage4_4_5.logs 2>&1
```

Local AST validation only; NPU correctness metrics pending server execution.
