# Uni-Agent-CC / VERL 3efe38c: TPR integration patch

This patch targets **the exact upstream VERL commit**
`3efe38c759c14622fd1b2c9e3679f2d02f86bdac` used by
`/workspace/uni-agent-cc/verl`. It is an additive integration patch;
it **does not install packages, modify GEMM, or turn on TPR**.

## Included

- `verl/workers/config/engine.py`: add `tpr_enabled=False` and `tpr_cp_backend=None` to `McoreEngineConfig`.
- `verl/workers/engine/megatron/transformer_impl.py`: install the opt-in TPR attention Module Spec before model creation; route TPR PPO after global token/DP metadata and before native micro-batch splitting.
- `verl/workers/engine_workers.py`: optional real actor-update mini-batch capture immediately before `train_batch`; disabled unless `TPR_CAPTURE_ACTOR_MINIBATCH_DIR` is set.

The TPR implementation and `actor_capture.py` must already be copied to the
target checkout at `verl/models/mcore/tpr/` from this Bridge repository.

## Apply on the server

**Do not use `git apply` inside the Bridge overlay**. Apply from the
**new Uni-Agent-CC VERL repository root**:

```bash
cd /workspace/uni-agent-cc/verl
PATCH=/workspace/bridge/patches/uniagent_cc_verl_3efe38c_tpr.patch

git status --short
git apply --check "$PATCH"
git apply "$PATCH"
git diff --check

python -m py_compile \
  verl/workers/config/engine.py \
  verl/workers/engine/megatron/transformer_impl.py \
  verl/workers/engine_workers.py

grep -nE 'tpr_enabled|install_tpr_module_spec|run_tpr_forward_backward_batch|TPR P0 real actor-update capture' \
  verl/workers/config/engine.py \
  verl/workers/engine/megatron/transformer_impl.py \
  verl/workers/engine_workers.py
```

`git apply --check` is a dry run; the following `git apply` **writes** the patch.
Do not use `git apply` again after successful application. To inspect a
reverse-check for an already applied patch: `git apply -R --check "$PATCH"`.

**Do not additionally run** `patches/apply_tpr_phase4.py` or
`patches/apply_tpr_actor_capture.py` against the same checkout: their hooks
are included here. Unrelated local modifications, including
`verl/utils/device.py`, are not touched.

## Limits: wiring, not completed E2E

- **Native remains unchanged** when `tpr_enabled=False`.
- The current Bridge `install_tpr_module_spec(provider)` requires a
  **non-None Megatron-Bridge provider**. The actual 2026-10-11 Ascend
  environment uses `vanilla_mbridge=True`, `mbridge=0.15.1`, and has
  **no `megatron.bridge` package**. TPR cannot yet construct its model in
  that mode. Adapt the vanilla bridge spec before enabling TPR in that stack;
  do not blindly upgrade NPU Megatron-Core.
- The current `bridge/main` Phase-4 PPO Adapter rejects TP=4 and native
  full uniform recompute. The already-passing 8B Native script uses both.
  First TPR smoke must select a supported topology and recompute setting
  or separately extend these capabilities.
- Existing Native one-step Uni-Agent-CC E2E has already been observed;
  this patch does not establish TPR numerical equivalence or end-to-end
  optimizer correctness.

The `feat/tpr-tp2-dp-placement-audit` branch is independent and not
merged by this patch.
