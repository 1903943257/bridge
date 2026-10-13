# TP2+SP selective integration on current main (2026-10-13)

Ported from PR #5 at 38b4f713 onto main 37e73bc4e4fce2a7b54c4165881b6d2947013e41. This does not merge PR #5's history.

## Included
- Native Megatron TP2 projection/head/vocab-parallel CE, optional native SP.
- TP-group schedule consistency guard for the formal and PPO tree runner.
- TP2+CP1 path; legacy TP1+CP1/CP2 Ring remains unchanged.
- CPU regression and real pretrained Qwen3-1.7B opt-in 2-NPU tests.

## Deferred
- DP planner, DP worker/trainer dispatch, Radix tree builder and optimizer integration.
- TP2+CP2 production entry and multi-dimensional parallel training.
- Non-divisible SP segment padding; fail closed instead of silently truncating.

## Acceptance after syncing integrated branch to NPU
```bash
pytest -q tests/models/mcore/tpr/unit/test_tp_validation.py tests/models/mcore/tpr/unit/test_tpr_self_attention.py tests/models/mcore/tpr/unit/test_module_spec.py
TPR_RUN_TP2=1 torchrun --nproc_per_node=2 -m pytest -q -s tests/models/mcore/tpr/parallel/test_tpr_tp2_npu.py
TPR_RUN_TP_SP=1 torchrun --nproc_per_node=2 -m pytest -q -s tests/models/mcore/tpr/parallel/test_tpr_tp2_sp_npu.py
```

NPU tests have not yet been rerun on this integrated main branch. Do not assume production PPO E2E, TP+CP, DP, or BF16 strict numeric equivalence.
