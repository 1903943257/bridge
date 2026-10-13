# Opt-in Qwen3 Turn-aware TPR tree construction (2026-10-13)

The legacy Token-exact Trie remains the default; select
`TPR_TREE_POLICY=qwen_turn` for a *physical execution* tree that shares only
full Qwen3 ChatML message/turn boundaries, not arbitrary partial shared token
prefixes. Short adjacent complete messages are grouped until at least
`TPR_TURN_MIN_SEGMENT_TOKENS` tokens separate physical cut points (default 128).
The original prompt boundary also qualifies as an execution cut even if an
assistant header is open. The final partial message is a single leaf segment.

A real, completed ChatML message is identified by Qwen3's `<|im_end|>`
token ID `151645`. This is Qwen3-specific; do not use with non-Qwen
tokenization or preprocessed data that strips ChatML delimiters.

**Important:** ChatML *message* boundaries are used as a safe *turn proxy*
for Agent trajectories. A higher-level agent turn might span assistant tool
calls, tool responses, and later assistant continuations. This policy does not
infer higher-level task/Agent turn semantics.

No hardcoded prompt-root restriction: if two trajectories split within the
first complete message and do not share a complete prompt, the builder emits
separate full trees. Original rows, masks, PPO refs, query offsets, and labels
remain token-exact. This is a topology strategy; it changes BF16 physical
shapes and requires independent numerical gates. No Fixed Tile.

## CPU preflight on the five real PPO trajectories

After syncing `bridge/main` code into your Docker VERL checkout, run:

```bash
cd /workspace/uni-agent-cc/verl
python tests/models/mcore/tpr/parallel/inspect_real_actor_turn_tree.py \
    /workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt \
    --rows 0,1,2,4,6 --min-segment-tokens 128
```

The script compares old exact Trie with the Turn-constrained forest, prints
each physical Segment, and **asserts** exact reconstruction and complete
original PPO query/target ownership. Inspect the actual number of Qwen3
`<|im_end|>` markers; absent markers reduce reuse safely rather than
synthesizing boundaries.

CPU regression:

```bash
python -m pytest -q tests/models/mcore/tpr/unit/test_trajectory_tree_turn.py
```

## NPU Full vs TPR single-step benchmark

Use the same 5 real captured trajectories and same Ring backend:

```bash
cd /workspace/uni-agent-cc/verl
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt
export TPR_OFFLINE_ROWS=0,1,2,4,6
export TPR_TREE_POLICY=qwen_turn
export TPR_TURN_MIN_SEGMENT_TOKENS=128
export TPR_OFFLINE_RUN_MODES=both
export TPR_OFFLINE_SAVE_FULL_GRADS=0
export TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_offline_qwen4b_turn5_v1
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

Keep old `exact` 9-node result as baseline. Explicitly set
`TPR_TREE_POLICY=exact` or unset it to compare with the original.
The Full-unshared reference continues to use five independent complete
trajectory roots in either policy. TPR's Engine builds the same turn-aware
forest from the unchanged input batch; objective ownership is recomputed
against the selected execution segments.

Relevant files:

- `verl/models/mcore/tpr/trajectory_tree_turn.py`
- `verl/models/mcore/tpr/tree_plan_builder.py`
- `verl/models/mcore/tpr/megatron_adapter.py`
- `tests/models/mcore/tpr/unit/test_trajectory_tree_turn.py`
- `tests/models/mcore/tpr/parallel/inspect_real_actor_turn_tree.py`

**Validation status:** CPU unit tests and real-capture preflight have been
authored, but must run in your environment; 8-NPU correctness and performance
are not yet verified. This is opt-in, not a production-default change.
