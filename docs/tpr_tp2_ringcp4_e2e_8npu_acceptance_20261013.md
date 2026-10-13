# Qwen3-4B TPR TP2 × RingCP4 × DP1 E2E smoke (2026-10-13)

The prior 4-NPU TP2 × RingCP2 Qwen3-1.7B **module-level** check was reported PASS. This change allows TP2 × RingCP2/4 in the formal and PPO Engine entries. It does NOT claim TP2 × RingCP4 real PPO E2E or long-context numerical parity has passed.

- Current test target: one 8-card replica, physical NPU IDs 8,9,10,11,12,13,14,15 (visible local devices 0..7). Uses `ASCEND_RT_VISIBLE_DEVICES` before Ray and torch-npu imports.
- Script: `tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh`. Filename is historical: script runs **Qwen3-4B-128K**.
- TPR=True, TP=2, CP=4 with `ring`, PP=EP=DP=1, N=8 by default. Full response budget preserved; native SP+CP not supported.
- PPO mini-batch defaults to all N sibling trajectories so TPR Forest can attempt Prefix reuse; no new DP dispatcher, no artificial rewards.
- Existing `TPR_CAPTURE_ACTOR_MINIBATCH_DIR` snapshots are intentionally never overwritten.

## Run commands

On HOST, in Bridge:
```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/bridge
git fetch origin feat/tpr-tp2-ringcp4-e2e-npu8-15-20261013
mkdir -p tmp/tpr-tpcp4
git archive origin/feat/tpr-tp2-ringcp4-e2e-npu8-15-20261013 verl/models/mcore/tpr tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh tests/models/mcore/tpr/unit/test_tp_cp_engine_gate.py | tar -x -C tmp/tpr-tpcp4
```

Inside Docker:
```bash
cd /workspace/uni-agent-cc/verl
rsync -av --backup --suffix=.before-tpcp4 /workspace/bridge/tmp/tpr-tpcp4/verl/models/mcore/tpr/ verl/models/mcore/tpr/
rsync -av --backup --suffix=.before-tpcp4 /workspace/bridge/tmp/tpr-tpcp4/tests/models/mcore/tpr/ tests/models/mcore/tpr/
python -m pytest -q tests/models/mcore/tpr/unit/test_tp_cp_engine_gate.py
cd /workspace/uni-agent-cc
bash verl/tests/models/mcore/tpr/integration/run_uniagent_cc_qwen17_swe_tpr_ppo_cp2.sh
```

## Acceptance / known risks

1. Require all eight visible NPUs, verify startup `TPR_NPU_VISIBILITY`, and `TRAIN_DP=1`.
2. Confirm TPR interception/Forest actually executed (`tpr/forest_trees`, `tpr/physical_segments`, `tpr/topology_token_reuse_ratio`), not Native fallback.
3. Check finite PPO actor loss, real nonzero gradient and a completed optimizer update.
4. Distinguish model memory OOM, Ring numerical drift, and unmatched TP/SP/CP collectives. The 4-card TP2+CP2 module gate does **not** prove a 128K Qwen3-4B E2E run fits memory.
5. DO NOT enable DP2, TP+SP+CP, native recompute, or Fixed-Tile oracle by changing this gate.
