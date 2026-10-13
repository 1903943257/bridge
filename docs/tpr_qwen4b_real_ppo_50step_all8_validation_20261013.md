# 2026-10-13: Qwen3-4B TP2×RingCP4 recorded-PPO diagnostics and replay

All tools are on `main`. Experimental inputs already available in NPU Docker:

- Pair: `/workspace/tpr_actor_capture_4b_n8_run1/actor_update_siblings_row4_row6.pt`; select `TPR_OFFLINE_ROWS=0,1`, 2 rows, 18,342/37,629 tokens, 4,616 real PPO tokens, one shared tree with three segments.
- Full original 8-row capture: `/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt`; select `TPR_OFFLINE_ROWS=all`, 8 rows, 89,665 supervised PPO tokens; no trimming or replacing old logprobs/advantages.
- Real Qwen3-4B-128K, TP2, Ring CP4, DP1 on physical NPU 8-15.
- **Full** in this first-level parity experiment = separate complete trajectories on *the same TPR-capable Ring backend*. Not unmodified MindSpeed Native production Full.

## A. CPU-only effective PPO gradient-weight attribution

No NPU and no model rerun. Use the existing one-step outputs:

```bash
cd /workspace/uni-agent-cc/verl
python tests/models/mcore/tpr/parallel/analyze_real_actor_ppo_adjoint.py \
    /workspace/tpr_offline_qwen4b_tp2cp4_pair_run1 \
    /workspace/tpr_actor_capture_4b_n8_run1/actor_update_siblings_row4_row6.pt
```

Computes VERL vanilla PPO's `d(loss)/d(new_logprob)` by CPU torch autograd with the **recorded original** old logprobs and advantages. Includes exp(clamp(delta,-20,20)), ratio clip 0.8/1.2, dual-clip 3.0 for negative advantages, and optional rollout importance weights. Reports gradient-adjoint L2/cosine, active upstream tokens, clip branch flips, and top 25 effective-difference tokens. This is CPU FP32 attribution from recorded values; it does not replay full backward through 4B weights or diagnose any one kernel on its own.

## B. Exact full-gradient norm / full-gradient relative L2

New snapshots record FP32-accumulated norm over all *finalized BF16* gradient tensors, counting each TP shard once, CP copies once and TP replicas once.

To obtain **true full-parameter** gradient rel L2 / cosine, explicitly opt in to lossless BF16 gradient shard files (large disk footprint; two modes together approximately one gradient copy per checkpoint per mode):

```bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_siblings_row4_row6.pt
export TPR_OFFLINE_ROWS=0,1
export TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_offline_qwen4b_tp2cp4_pair_exactgrads_v2
export TPR_OFFLINE_SAVE_FULL_GRADS=1
unset TPR_MEM_TRACE
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

Complete gradients are written only by each TP shard's CP rank0, **not 8 copies**. Comparator reports exact original-BF16 finalized gradient norm / relative L2 / cosine and top contributors; no sampled extrapolation. Previously captured old one-step snapshots cannot recover unsaved full gradients; rerun is required.

## C. All 8 trajectories, one step, independent Full/TPR, keep OOM evidence

```bash
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_rank0_batch0.pt
export TPR_OFFLINE_ROWS=all
export TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_offline_qwen4b_tp2cp4_all8_run1
unset TPR_OFFLINE_SAVE_FULL_GRADS
export TPR_OFFLINE_RUN_MODES=both
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_offline_ppo.sh
```

The launcher now **continues to the TPR job even if Full OOMs**. Both modes have independent logs and per-rank success snapshots. When either mode fails, parity comparison is not attempted; a difference in OOM outcome remains useful, but is not gradient-parity PASS. For repeat runs, use a different output directory to avoid old checkpoints. If you only want TPR: `TPR_OFFLINE_RUN_MODES=tpr`.

## D. Fixed-data true PPO 1–50 step drift with CPU FP32-master AdamW

Uses the same HF checkpoint and original saved PPO fields at every step for each mode. No fresh rollout, no synthetic advantages, no claimed on-policy PPO reward convergence.

Model weights BF16 on NPUs; FP32 optimizer master/moments on **CPU CP leader only** for each TP shard, broadcasting updated BF16 weights to its CP peers. CPU RAM cost is substantial (roughly 3 FP32 copies per local TP shard across two leaders) and CPU optimizer time is reported separately; this is *diagnostic* AdamW, not production VERL distributed optimizer/offload.

Preflight on two trajectories and two steps:

```bash
cd /workspace/uni-agent-cc/verl
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export TPR_REAL_ACTOR_MINIBATCH=/workspace/tpr_actor_capture_4b_n8_run1/actor_update_siblings_row4_row6.pt
export TPR_OFFLINE_ROWS=0,1
export TPR_OFFLINE_STEPS=2
export TPR_OFFLINE_OUTPUT_DIR=/workspace/tpr_offline_qwen4b_tp2cp4_fixed_2step_v1
export TPR_OFFLINE_RUN_MODES=both
bash tests/models/mcore/tpr/parallel/run_real_actor_tp2cp4_multistep.sh
```

Then use fresh directory and `TPR_OFFLINE_STEPS=50`. Comparator verifies initial FP32 master equality, input SHA and matching objective/optimizer, and prints loss/logprob drift, global grad norm, sampled gradient rel L2 and **cumulative FP32-master update rel L2/cosine** at steps 1,2,5,10,20,30,40,50. It writes `multistep_comparison.json`.

Optionally record complete BF16 gradient shards at the first and last milestones:

```bash
export TPR_OFFLINE_SAVE_FULL_GRADS=1
export TPR_OFFLINE_FULL_GRAD_STEPS=1,50
```

This consumes large disk capacity and has additional CPU I/O cost. The gradient snapshots are measured **before** clipping; optimizer applies matching global clip norm=1 to both paths.

**Limitations:** 50 steps repeatedly reuse the *same frozen minibatch*, so a low/divergence-free training PPO loss could simply reflect clipping saturation. A common same-Full-backend held-out evaluation and *five genuine new actor steps with independently collected trajectories* remain separate next-stage gates. Avoid conflating BF16 model-weight similarity with real FP32 master update parity. Do not enable Fixed Tile.
