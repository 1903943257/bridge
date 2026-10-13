# Uni-Agent Qwen3-4B: Session-level 48K hard stop (2026-10-13)

## Why

The Gateway trajectory_capacity = rollout.prompt_length + rollout.response_length
is a per-chain training-buffer limit, not an Agent episode limit. At exactly
49,152 tokens, it materializes the existing chain with materialization_reason =
max_trajectory_length. The Session remains ACTIVE, so Claude Code can create a
new training chain from existing conversation history.

This can change rollout.n=8 into more than eight actual training samples per
prompt group when trajectory_selection=all. Switching to longest does not
guarantee retaining the first segment.

Observed sample: trajectory 0 prompt=17483 + response=31669 = 49152;
trajectory 1 prompt=43211 + response=796.

## Change

patches/apply_uniagent_session_hardcap.py installs a narrowly scoped opt-in
patch into the local Uni-Agent checkout, not VERL or TPR.

- GatewaySession reads hard_stop_on_trajectory_capacity from session metadata,
  records hard_cap_reached, closes the chain at the exact generation filling
  capacity, and rejects subsequent generations with HTTP 410. If context
  expansion exhausts capacity, close the active chain without backend calls.
- GatewayManager forwards the existing actor get_session_state() method.
- GatewayAgentFramework polls the Gateway snapshot for opt-in ray_task
  sessions. On hard cap, it cancels the remote Ray Runner with existing
  cleanup/force-kill behavior, finalizes (does not abort) the training
  trajectory, and emits TaskResult(reward=0.0, accuracy=0.0, finished=False)
  with extra_info.session_length_truncated=True.
- If the hard-capped session does not return exactly ONE trajectory labeled
  max_trajectory_length, fail loudly rather than duplicating GRPO samples.
- Default upstream behavior remains unchanged when the flag is false.

The Qwen3-4B training script now sets rollout token capacity to
16384 + 32768 = 49152; model checkpoint remains Qwen3-4B-128K.
trajectory_selection=all remains intentionally unchanged, because the session
itself is now stopped at the budget.

## Usage

From the Bridge checkout, run the script normally; it installs the
opt-in patch into the Uni-Agent checkout BEFORE starting Ray. Manual preflight:

    python patches/apply_uniagent_session_hardcap.py --check /workspace/uni-agent-cc
    python patches/apply_uniagent_session_hardcap.py /workspace/uni-agent-cc
    python patches/apply_uniagent_session_hardcap.py --check /workspace/uni-agent-cc

The final check should report already-patched. It is idempotent and fails
closed when local Uni-Agent anchors differ. No git pull inside VERL.

If the startup script was copied outside Bridge, set BRIDGE_ROOT to the
actual Bridge checkout or UNIAGENT_SESSION_PATCHER to the absolute script
path. SESSION_HARD_CAP=False disables patch installation and restores default
Uni-Agent semantics for this training script.

## Acceptance (to run in user's Ray/NPU environment)

1. 49,152-token materialization produces one trajectory with
   materialization_reason=max_trajectory_length, finished=False, reward=0,
   mask/logprob alignment intact.
2. No second trajectory or new model generation is possible after the hard
   cap. Ray Runner and Sandbox are cancelled/cleaned up.
3. Sessions ending before 48K keep actual SWE reward and normal runner flow.
4. For rollout.n=8, expect eight training samples per prompt, not more,
   except that genuinely failed sessions may yield no model tokens.
5. Re-running patcher produces no diff. Check hard_cap_reached and
   session_length_truncated diagnostics.
6. Native vs TPR multistep replay uses the same recorded Actor minibatches;
   separately sampled Rollouts are not strict numerical-parity tests.

The patcher checks source anchors and compiles patched Python via ast.parse.
A live Claude Code + Ray + NPU integration test has NOT been run here; the
assistant cannot directly access the user's local /workspace checkout.

## Training interpretation

A truncated episode receives an explicit synthetic reward=0; it is NOT
a successful, fully evaluated SWE-bench task.
mask_unfinished_episode=False remains intentionally enabled for this GRPO
experiment; unfinished capped failures can be compared to successful samples.
An all-zero-reward GRPO group has zero advantages, but executed AdamW updates
could still move parameters via historical momentum or weight decay, and
colocate_async Rollouts may use stale weight versions.
