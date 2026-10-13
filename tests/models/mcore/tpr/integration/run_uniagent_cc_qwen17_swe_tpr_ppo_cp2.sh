#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# Uni-Agent Qwen3-4B SWE end-to-end: rollout -> TQ -> GRPO/PPO -> actor update.
# Retains the established Uni-Agent-CC V1 Trainer path and SWE task.
# TP2*CP4=8 NPU ranks per replica. Default 16 NPUs -> DP2; set
# N_GPUS_PER_NODE=8 to run the smaller DP1 smoke. Native only.
# NOTE: current Bridge main includes TP2 CP1 but still rejects TP2 CP4 and DP2, so this is
# a NATIVE baseline + real Actor PPO capture, NOT a TPR TP2/CP4 acceptance.
# ============================================================
ROOT=/workspace/uni-agent-cc
# ------------------------------------------------------------
# Same model / data / task as the verified inference E2E
# ------------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-/workspace/hf_models/Qwen3-4B-128K}
MODEL_NAME=Qwen3-4B
TRAIN_FILE=/workspace/data/uniagent-cc/django_11163_smoke_py311.parquet
TEST_FILE=/workspace/data/uniagent-cc/django_11163_smoke_py311.parquet
TASK_CONFIG=${ROOT}/examples/claude_code_swe_task/task_config_claude_code_django11163.yaml
# tpr
# Deliberately native until TP2 x CP4 TPR production guard is implemented.
TPR_ENABLED=False
# ------------------------------------------------------------
# Same rollout config as run_e2e.sh
# ------------------------------------------------------------
N=${N:-8}
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
TOOL_PARSER=hermes
# Model checkpoint supports 128K, but ONE AGENT SESSION is capped at 48K
# total training tokens (prompt + response + intermediate tool/context tokens).
# Gateway enforcement uses rollout.prompt_length + rollout.response_length.
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=32768
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_TP=2
ROLLOUT_GPU_MEMORY_UTILIZATION=0.60
GATEWAY_COUNT=1
CONCURRENCY=2
# 一个 prompt group 即可。
NUM_AGENT_WORKERS=1
# Session-level hard stop prevents split materializations; keep ALL to avoid
# silently choosing the wrong member of a group if a regression occurs.
TRAJECTORY_SELECTION=all
SESSION_HARD_CAP=${SESSION_HARD_CAP:-True}
MASK_UNFINISHED_EPISODE=False
# ------------------------------------------------------------
# Training side: native Megatron TP2 x CP4 smoke + genuine PPO capture
# ------------------------------------------------------------
NNODES=1
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
TRAIN_TP=2
TRAIN_PP=1
TRAIN_CP=4
TPR_CP_BACKEND=ring
TRAIN_DP=$((N_GPUS_PER_NODE / (TRAIN_TP * TRAIN_PP * TRAIN_CP)))
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-$TRAIN_DP}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$TRAIN_DP}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_DP}
PPO_MICRO_BATCH_SIZE_PER_GPU=1
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
# 与当前旧训练栈匹配：
# mbridge 0.15.1 + megatron-core 0.16.1 + MindSpeed 0.16
USE_MBRIDGE=True
VANILLA_MBRIDGE=True
PARAM_OFFLOAD=True
OPTIMIZER_OFFLOAD=True
# Capture an authentic actor update first; do not pretend zero-advantage means PPO parity.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# E1 Native only: opt-in full activation recompute to reduce online OOM.
# Offline Native/TPR Replay must disable recompute independently.
NATIVE_RECOMPUTE=${NATIVE_RECOMPUTE:-False}
RECOMPUTE_ARGS=()
case "${NATIVE_RECOMPUTE}" in
  True|true|1)
    RECOMPUTE_ARGS=(
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full"
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform"
      "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1"
    )
    ;;
  False|false|0) ;;
  *) echo "NATIVE_RECOMPUTE must be True or False" >&2; exit 2 ;;
esac
# 不做 validation/checkpoint。
SAVE_FREQ=-1
TEST_FREQ=-1
# 不预取额外 prompt。
NUM_WARMUP_BATCHES=0
# ------------------------------------------------------------
# Output
# ------------------------------------------------------------
RUN_NAME=${RUN_NAME:-swe-django-11163-qwen3-4b-n${N}-tp2cp4-long-response}
AGENT_LOG_DIR=/workspace/logs/${RUN_NAME}/agent
TRAIN_LOG_DIR=/workspace/logs/${RUN_NAME}
TQ_DUMP_DIR=${TQ_DUMP_DIR:-/workspace/tq_dump/django11163_4b_tp2cp4}
CKPTS_DIR=/workspace/ckpts/${RUN_NAME}
# ============================================================
# Environment
# ============================================================
cd "${ROOT}"
if [ -f /workspace/.venvs/uniagent-cc/bin/activate ]; then
    source /workspace/.venvs/uniagent-cc/bin/activate
fi
# Uni-Agent upstream implements per-trajectory materialization, NOT a hard
# Session token stop. Install Bridge's strict, opt-in patch before Ray starts.
if [[ "${SESSION_HARD_CAP}" == "True" || "${SESSION_HARD_CAP}" == "true" || "${SESSION_HARD_CAP}" == "1" ]]; then
    BRIDGE_ROOT="${BRIDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
    SESSION_PATCHER="${UNIAGENT_SESSION_PATCHER:-${BRIDGE_ROOT}/patches/apply_uniagent_session_hardcap.py}"
    if [[ ! -f "${SESSION_PATCHER}" ]]; then
        echo "ERROR: session-cap patcher not found: ${SESSION_PATCHER}; set UNIAGENT_SESSION_PATCHER" >&2
        exit 2
    fi
    python "${SESSION_PATCHER}" "${ROOT}" || exit 2
fi
# IMPORTANT:
# 不加入 /workspace/Megatron-Bridge/src。
# 本次明确走 vanilla mbridge。
export PYTHONPATH="/workspace/vllm-ascend:${ROOT}/verl:${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=2400
# The existing patched TrainingWorker captures ORIGINAL actor mini_batch_td.
# Use a distinct capture directory: actor_capture.py refuses overwrites.
export TPR_CAPTURE_ACTOR_MINIBATCH_DIR="${TPR_CAPTURE_ACTOR_MINIBATCH_DIR:-/workspace/tpr_actor_capture_4b_tp2cp4}"
unset TPR_REAL_ACTOR_CAPTURE || true
mkdir -p "${AGENT_LOG_DIR}"
mkdir -p "${TRAIN_LOG_DIR}"
mkdir -p "${TQ_DUMP_DIR}"
mkdir -p "${CKPTS_DIR}"
mkdir -p "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}"
# ============================================================
# Preflight
# ============================================================
test -d "${MODEL_PATH}" || {
    echo "ERROR: model not found: ${MODEL_PATH}"
    exit 1
}
test -f "${TRAIN_FILE}" || {
    echo "ERROR: dataset not found: ${TRAIN_FILE}"
    exit 1
}
test -f "${TASK_CONFIG}" || {
    echo "ERROR: task config not found: ${TASK_CONFIG}"
    exit 1
}
# The official Qwen3-4B is native 32K / config 40960.
# Full 131072 tokens require a model config with YaRN enabled for BOTH
# rollout and actor, not just a larger max_model_len CLI override.
MODEL_PATH="${MODEL_PATH}" MAX_MODEL_LEN="${MAX_MODEL_LEN}" python - <<'PY_CTX_CHECK'
import json, os
from pathlib import Path
p = Path(os.environ["MODEL_PATH"]) / "config.json"
if not p.is_file():
    raise SystemExit(f"ERROR: no model config: {p}")
c = json.loads(p.read_text())
limit = int(os.environ["MAX_MODEL_LEN"])
rope = c.get("rope_scaling") or {}
if limit > 40960:
    if (rope.get("rope_type", rope.get("type")) != "yarn"
        or float(rope.get("factor", 0)) < 4.0):
        raise SystemExit(
            "ERROR: 131072-token Qwen3-4B needs YaRN in the model "
            "config.json (rope_type=yarn, factor>=4, "
            "original_max_position_embeddings=32768). "
            "Prepare a separate 128K config/checkpoint directory and set MODEL_PATH."
        )
print(f"LONG_CONTEXT_CHECK max_model_len={limit} rope_scaling={rope}")
PY_CTX_CHECK
if [[ "${N_GPUS_PER_NODE}" != "8" && "${N_GPUS_PER_NODE}" != "16" ]]; then
    echo "ERROR: TP2 x CP4 requires 8 or 16 training NPUs (DP1 or DP2)" >&2
    exit 2
fi
if (( TRAIN_PROMPT_BSZ % TRAIN_DP != 0 || GEN_BATCH_SIZE % TRAIN_DP != 0 || PPO_MINI_BATCH_SIZE % TRAIN_DP != 0 )); then
    echo "ERROR: train/gen/ppo mini batch sizes must be divisible by TRAIN_DP=${TRAIN_DP}." >&2
    exit 2
fi
if (( TRAIN_DP > 1 )); then
    echo "NOTE: DP${TRAIN_DP} will distribute actor mini-batches; verify per-DP sibling grouping and reward diversity." >&2
fi
if [[ "${TPR_ENABLED}" != "False" ]]; then
    echo "ERROR: TPR TP2 x CP4 PPO is not enabled in the production Engine. Run Native first." >&2
    exit 2
fi
if find "${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}" -maxdepth 1 -name 'actor_update_*.pt' | grep -q .; then
    echo "ERROR: Actor capture directory already has .pt snapshots; choose a NEW directory." >&2
    exit 2
fi
echo "============================================================"
echo "Uni-Agent TRAIN/TQ E2E"
echo "============================================================"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "MODEL_NAME        = ${MODEL_NAME}"
echo "TPR_ENABLED       = ${TPR_ENABLED} (Native baseline; TPR TP2 x CP4 not wired)"
echo "TRAIN_FILE        = ${TRAIN_FILE}"
echo "TASK_CONFIG       = ${TASK_CONFIG}"
echo
echo "SESSION_HARD_CAP  = ${SESSION_HARD_CAP}"
echo "N                 = ${N}"
echo "TEMPERATURE       = ${TEMPERATURE}"
echo "TOP_P             = ${TOP_P}"
echo "TOP_K             = ${TOP_K}"
echo "TOOL_PARSER       = ${TOOL_PARSER}"
echo
echo "MAX_PROMPT        = ${MAX_PROMPT_LENGTH}"
echo "MAX_RESPONSE      = ${MAX_RESPONSE_LENGTH}"
echo "MAX_MODEL_LEN     = ${MAX_MODEL_LEN}"
echo
echo "ROLLOUT_TP        = ${ROLLOUT_TP}"
echo "TRAIN_TP          = ${TRAIN_TP}"
echo "TRAIN_PP          = ${TRAIN_PP}"
echo "TRAIN_CP          = ${TRAIN_CP}"
echo "TPR_CP_BACKEND    = ${TPR_CP_BACKEND}"
echo
echo "NNODES            = ${NNODES}"
echo "N_GPUS_PER_NODE   = ${N_GPUS_PER_NODE}"
echo "TRAIN_DP          = ${TRAIN_DP}"
echo "TRAIN_PROMPT_BSZ  = ${TRAIN_PROMPT_BSZ}"
echo "GEN_BATCH_SIZE    = ${GEN_BATCH_SIZE}"
echo "PPO_MINI_BSZ      = ${PPO_MINI_BATCH_SIZE}"
echo "NATIVE_RECOMPUTE  = ${NATIVE_RECOMPUTE}"
echo "TOTAL_STEPS       = ${TOTAL_TRAINING_STEPS}"
echo "TOTAL_EPOCHS      = ${TOTAL_EPOCHS}"
echo
echo "USE_MBRIDGE       = ${USE_MBRIDGE}"
echo "VANILLA_MBRIDGE   = ${VANILLA_MBRIDGE}"
echo
echo "TQ_DUMP_DIR       = ${TQ_DUMP_DIR}"
echo "AGENT_LOG_DIR     = ${AGENT_LOG_DIR}"
echo "ACTOR_CAPTURE_DIR = ${TPR_CAPTURE_ACTOR_MINIBATCH_DIR}"
echo "============================================================"
python - <<'PY'
import importlib.util
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__)
for m in [
    "torch_npu",
    "verl",
    "uni_agent",
    "mbridge",
    "megatron.core",
    "megatron.bridge",
]:
    try:
        print(m, "=>", importlib.util.find_spec(m))
    except Exception as e:
        print(m, "=> ERROR:", repr(e))
PY
# ============================================================
# Clean stale local Ray
# ============================================================
ray stop --force >/dev/null 2>&1 || true
# ============================================================
# Run normal VERL / Uni-Agent V1 training flow
# ============================================================
python -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    \
    transfer_queue.enable=True \
    transfer_queue.metrics.enabled=True \
    \
    trainer.device=npu \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${MAX_MODEL_LEN}" \
    \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${TEST_FILE}']" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    data.gen_batch_size="${GEN_BATCH_SIZE}" \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=False \
    \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.checkpoint.strict=False \
    \
    actor_rollout_ref.actor.megatron.use_mbridge="${USE_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.vanilla_mbridge="${VANILLA_MBRIDGE}" \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    "${RECOMPUTE_ARGS[@]}" \
    +actor_rollout_ref.actor.megatron.tpr_enabled="${TPR_ENABLED}" \
    +actor_rollout_ref.actor.megatron.tpr_cp_backend="${TPR_CP_BACKEND}" \
    \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
    \
    actor_rollout_ref.rollout.n="${N}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${CONCURRENCY}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection="${TRAJECTORY_SELECTION}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.hard_stop_on_trajectory_capacity="${SESSION_HARD_CAP}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}" \
    \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    \
    'trainer.logger=["console"]' \
    trainer.project_name=uniagent-tq-smoke \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.val_before_train=False \
    trainer.resume_mode=disable \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    \
    skip.rollout_tq.enable=True \
    skip.rollout_tq.dump_dir="${TQ_DUMP_DIR}" \
    "skip.rollout_tq.steps=[1]" \
    skip.rollout_tq.action=cache
echo
echo "============================================================"
echo "Trainer exited"
echo "TQ dumps:"
find "${TQ_DUMP_DIR}" -name tq_batch.pt -print || true
echo "============================================================"