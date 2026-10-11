# Uni-Agent-CC / VERL 3efe38c: TPR integration

The initial unified diff `uniagent_cc_verl_3efe38c_tpr.patch` is retained
for reference, but **its `engine_workers.py` hunk can fail on the modified
server checkout** even though the actual Actor train call is present.

**Use the semantic installer below instead of `git apply`**. It reuses the
existing `apply_tpr_phase4.py` and `apply_tpr_actor_capture.py` transformers
plus explicit AST checks for config and model construction. It changes only:

- `verl/workers/config/engine.py`: add `tpr_enabled=False` and `tpr_cp_backend=None`.
- `verl/workers/engine/megatron/transformer_impl.py`: opt-in TPR Module Spec installation and PPO Forest dispatch after global token metadata, before microbatch packing.
- `verl/workers/engine_workers.py`: opt-in genuine Actor mini-batch capture just before `train_batch` (preserves local profiling edits).

## Server commands

First sync the updated Bridge `main` files into `/workspace/bridge`.
The failing `git apply --check` **did not modify any file**.

```bash
python /workspace/bridge/patches/apply_uniagent_cc_tpr.py \
  --check /workspace/uni-agent-cc/verl

# Only if the check finishes successfully:
python /workspace/bridge/patches/apply_uniagent_cc_tpr.py \
  /workspace/uni-agent-cc/verl

cd /workspace/uni-agent-cc/verl
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

The installer preflights all three targets before writing, refuses missing
or ambiguous anchors, checks that each file has not changed before replacing
it, and is idempotent (repeat runs do not duplicate hooks). It uses atomic
per-file replacement and attempts rollback of its own writes on an error.
No installation command, Git pull, or manual file overwrite is performed.

**Do not additionally apply** `uniagent_cc_verl_3efe38c_tpr.patch`,
`apply_tpr_phase4.py`, or `apply_tpr_actor_capture.py` to the same working tree.

## Scope / follow-up

- `tpr_enabled=False` keeps the Native engine path.
- `tpr_enabled=True` currently still requires a non-None
  Megatron-Bridge provider; the proven NPU Native script uses
  `vanilla_mbridge=True` and Megatron-Core 0.16.1. This is a separate
  model-construction compatibility task. Do not blindly upgrade core.
- The existing `bridge/main` Phase-4 PPO path rejects TP=4 and native full
  uniform recompute. The successful Native Qwen3-8B smoke uses both.
- Installing these hooks is **not** a TPR optimizer-step or numerical parity
  acceptance. The separate `feat/tpr-tp2-dp-placement-audit` branch is untouched.

## Vanilla MBridge v0.15.1 support (isolated feature branch)

The TPR adapter now accepts `install_tpr_module_spec(provider, bridge=bridge)`.
When the NVIDIA provider is absent and the local `bridge` is Vanilla MBridge,
it wraps **only the bridge instance's** `_get_transformer_layer_spec`.
This is invoked lazily by MBridge's model provider before constructing
`GPTModel`; a deep-copied Transformer layer/block spec substitutes
`TPRSelfAttention` without changing MBridge source, its classes, native
model instances or QKV/MLP kernel precision.

For installations that **already applied** the earlier semantic patch,
the updated installer upgrades the single provider-only call to
`install_tpr_module_spec(self.provider, bridge=self.bridge)`.
Apply the same installer in check mode, then real mode, after copying
the latest TPR modules from this branch to Uni-Agent-CC:

```bash
python /workspace/bridge/patches/apply_uniagent_cc_tpr.py --check /workspace/uni-agent-cc/verl
python /workspace/bridge/patches/apply_uniagent_cc_tpr.py /workspace/uni-agent-cc/verl
```

Check the resulting source and first verify the built model actually
contains `TPRSelfAttention` on a single-rank smoke test. This feature does
**not** add PPO TP4, recompute, or full-length 8B support.
