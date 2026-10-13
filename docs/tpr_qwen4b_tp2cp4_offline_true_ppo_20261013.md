# Real Actor PPO TP2 × RingCP4 two-sibling offline replay

Branch: `feat/tpr-real-actor-tp2cp4-offline-20261013`. Built from main, independent from PR #12's online TPR script.

## Scope

- Actual Qwen3-4B-128K HF checkpoint (36L, 2560 hidden, 32 Q / 8 KV heads), native Megatron TP2 weight partition, real 8-rank Ring CP4.
- Full data: saved `actor_update_rank0_batch0.pt`, choose original rows 4 and 6 (18,342 and 37,629 tokens, 4,616 real PPO targets), no cropping, no fake advantages or recomputed old logprob.
- Two sequential, separate 8-NPU jobs: `full` builds two **independent complete-trajectory** roots, `tpr` builds a shared compressed prefix tree; both use **the same explicit Ring CP + TP model backend and VERL vanilla PPO loss**.
- **Not yet a production Native MindSpeed-vs-TPR numerical comparison**: full uses the same TPR-capable model and Ring backend with no sharing. This removes backend variance first.
- CP parameter gradient reduction is a test-time CP SUM/4 and **SGD in BF16 with no momentum**. This is a one-step gradient/update diagnostic, not the original Precision-Aware AdamW / optimizer offload acceptance. Same model weights are independently loaded per mode.
- Records original logical token logprobs, PPO loss, bounded sampled gradients, sampled pre/post-SGD model parameters, worst NPU allocated/reserved peak, critical-path forward/backward time and update time. OOM is a failed run, not a parity PASS.

## Host: archive branch files

```bash
cd /mnt/pipeline-data/j00972288/project/uniagent_linear/bridge
git fetch origin feat/tpr-real-actor-tp2cp4-offline-20261013
mkdir -p tmp/tpr-offline-tp2cp4
git archive origin/feat/tpr-real-actor-tp2cp4-offline-20261013 \
  verl/models/mcore/tpr \
  tests/models/mcore/tpr/parallel/_qwen3_tp_checkpoint.py \
  tests/models/mcore/tpr/parallel/test_real_actor_tp2cp4_offline_ppo_npu.py \
  tests/models/mcore/tpr/parallel/compare_real_actor_tp2cp4_offline_ppo.py \
  tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh \
  tests/models/mcore/tpr/unit/test_tp2_cp4_gate.py \
  | tar -x -C tmp/tpr-offline-tp2cp4
```

## Docker: sync and execute

```bash
cd /workspace/uni-agent-cc/verl
rsync -av --backup --suffix=.before-offline \
  /workspace/bridge/tmp/tpr-offline-tp2cp4/verl/models/mcore/tpr/ \
  verl/models/mcore/tpr/
rsync -av --backup --suffix=.before-offline \
  /workspace/bridge/tmp/tpr-offline-tp2cp4/tests/models/mcore/tpr/ \
  tests/models/mcore/tpr/
python -m pytest -q tests/models/mcore/tpr/unit/test_tp2_cp4_gate.py
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt
export TPR_QWEN_4B_PATH=/workspace/hf_models/Qwen3-4B-128K
export TPR_OFFLINE_ROWS=4,6
export TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_offline_qwen4b_tp2cp4_row4_row6
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

Run controls: `TPR_OFFLINE_RUN_MODES=full` or `tpr` or `compare`; `TPR_OFFLINE_SGD_LR=0` to skip update. The launcher refuses to overwrite existing per-rank snapshots, so use a NEW output directory for retries.

## Acceptance sequence

1. Gate unit tests + capture contract PASS.
2. Verify all 8 per-rank Full snapshots, no OOM; then all 8 TPR snapshots, no OOM.
3. `comparison.log`: exact initial parameter fingerprints, identical input SHA and original PPO fields, 4,616 owned logprobs, loss, sampled gradient and optimizer delta, peak memory and time.
4. A later production Native-vs-TPR experiment must replace the controlled Full reference with actual unmodified MindSpeed attention/optimizer and verify its own numerical and memory behavior.
5. For 5-step trajectory collection and 20-50 replay later, preserve immutable capture/model checkpoint snapshots and original PPO coefficients, and do not conflate fixed-data replay with on-policy updated rollouts.
