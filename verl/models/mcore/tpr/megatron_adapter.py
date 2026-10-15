# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin integration helpers between verl's MegatronEngine and TPR."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from megatron.core.utils import get_model_config

from verl.utils.megatron_utils import unwrap_model

from .attention import TPRSelfAttention
from .engine_adapter import TPRForwardBackwardRequest
from .fixed_topology_scheduler import FixedTopologyScheduler
from .module_spec import install_tpr_vanilla_mbridge_spec, make_tpr_module_spec_provider
from .parallel.engine_runtime import aggregate_cp_loss, resolve_engine_cp_runtime
from .segment_executor import SegmentExecutor

if TYPE_CHECKING:
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def install_tpr_module_spec(provider: Any, *, bridge: Any | None = None) -> None:
    """Install the TPR attention spec for the actual model-construction path.

    Non-vanilla uses the provider. Vanilla MBridge builds its layer spec lazily
    through the bridge instance, so patch that instance before get_model().
    """
    if provider is None:
        if bridge is None:
            raise NotImplementedError(
                "TPR model construction requires either a non-vanilla provider "
                "or the vanilla bridge instance"
            )
        install_tpr_vanilla_mbridge_spec(bridge)
        return
    provider.transformer_layer_spec = make_tpr_module_spec_provider(
        provider.transformer_layer_spec
    )



def _validate_tpr_parallel_sizes(sizes: dict[str, int], *, ppo: bool) -> int:
    """Keep DP1 production scope; explicitly gate isolated DP2 parity experiments."""
    import os

    tp, cp = sizes["TP"], sizes["CP"]
    existing_dp2 = (
        ppo and tp == 2 and cp == 2 and sizes["DP"] == 2
        and os.getenv("TPR_EXPERIMENTAL_OFFLINE_DP2") == "1"
        and os.getenv("TPR_RUN_TP2CP2DP2_ONE_STEP") == "1"
    )
    # Isolated 10K correctness matrix only. This deliberately does NOT
    # expand the supported production VERL DP topology.
    parity_10k_dp2 = (
        ppo and tp in (1, 2) and cp in (1, 2) and sizes["DP"] == 2
        and os.getenv("TPR_EXPERIMENTAL_OFFLINE_DP2") == "1"
        and os.getenv("TPR_RUN_10K_PARALLEL_PARITY") == "1"
        and os.getenv("TPR_10K_PARITY_ONLY") == "1"
    )
    experimental_dp2 = existing_dp2 or parity_10k_dp2
    supported_cp = (
        (cp in (1, 2) if tp == 1 else cp in (1, 2, 4))
        if ppo else (cp >= 1 if tp == 1 else cp in (1, 2, 4))
    )
    if (tp not in (1, 2) or not supported_cp
            or any(sizes[k] != 1 for k in ("PP", "EP"))
            or (sizes["DP"] != 1 and not experimental_dp2)):
        raise NotImplementedError(
            "TPR PPO: TP1 CP1/2 or TP2 CP1/2/4; "
            "thin entry: TP1 CP>=1 or TP2 CP1/2/4; "
            f"PP=EP=1, DP=1 (or explicitly gated offline DP2 parity), got {sizes}"
        )
    return tp


def _validate_tpr_cp_combination(
    tp_size: int, cp_runtime: Any, config: Any, *, ppo: bool
) -> None:
    if cp_runtime.size <= 1:
        return
    if (ppo or tp_size > 1) and cp_runtime.backend_name != "ring":
        raise NotImplementedError(
            "TPR TP+CP PPO requires the tested Ring backend; "
            f"got {cp_runtime.backend_name!r}"
        )
    if tp_size > 1 and getattr(config, "sequence_parallel", False):
        raise NotImplementedError(
            "TP2+CP requires sequence_parallel=False; combined TP+SP+CP "
            "has not been validated"
        )


def run_tpr_forward_backward(
    engine: MegatronEngine,
    request: TPRForwardBackwardRequest,
    *,
    forward_only: bool,
) -> dict[str, Any]:
    """Run TPR instead of Megatron's linear schedule on the initialized CP topology."""

    if not isinstance(request, TPRForwardBackwardRequest):
        raise TypeError(
            "TPR request must be a TPRForwardBackwardRequest, "
            f"got {type(request).__name__}"
        )
    if forward_only:
        raise NotImplementedError("TPR thin entry only supports training forward/backward")
    if not engine.engine_config.tpr_enabled:
        raise RuntimeError("TPR request requires engine_config.tpr_enabled=True before model construction")
    if len(engine.module) != 1:
        raise NotImplementedError("TPR MVP requires exactly one Megatron model chunk")

    parallel_sizes = {
        "TP": engine.engine_config.tensor_model_parallel_size,
        "CP": engine.engine_config.context_parallel_size,
        "PP": engine.engine_config.pipeline_model_parallel_size,
        "EP": engine.engine_config.expert_model_parallel_size,
        "DP": engine.get_data_parallel_size(),
    }
    tp_size = _validate_tpr_parallel_sizes(parallel_sizes, ppo=False)
    if engine.engine_config.virtual_pipeline_model_parallel_size is not None:
        raise NotImplementedError("TPR MVP does not support virtual pipeline parallelism")
    if engine.model_config.mtp.enable:
        raise NotImplementedError("TPR MVP does not support MTP")
    if engine.enable_routing_replay:
        raise NotImplementedError("TPR MVP does not support router replay")

    wrapped_model = engine.module[0]
    model = unwrap_model(wrapped_model)
    config = get_model_config(wrapped_model)
    restrictions = {
        "calculate_per_token_loss": getattr(config, "calculate_per_token_loss", False),
        "activation recomputation": getattr(config, "recompute_granularity", None) is not None,
        "CUDA graph": getattr(config, "cuda_graph_impl", "none") not in (None, "none"),
        "CPU offload": getattr(config, "cpu_offloading", False),
        "FP8": getattr(config, "fp8", None) is not None,
        "MoE": getattr(config, "num_moe_experts", None) is not None,
    }
    active_restrictions = [name for name, active in restrictions.items() if active]
    if active_restrictions:
        raise NotImplementedError(f"TPR MVP does not support: {', '.join(active_restrictions)}")
    if not model.training:
        raise RuntimeError("TPR thin entry requires model.train()")

    tpr_attentions = [module for module in model.modules() if isinstance(module, TPRSelfAttention)]
    if not tpr_attentions:
        raise TypeError("TPR thin entry requires a model built with TPRSelfAttention")
    layer_numbers = tuple(sorted(attention.layer_number for attention in tpr_attentions))
    cp_runtime = resolve_engine_cp_runtime(engine, config)
    _validate_tpr_cp_combination(tp_size, cp_runtime, config, ppo=False)
    executor = SegmentExecutor(
        model,
        request.plan,
        expected_layer_numbers=layer_numbers,
        loss_scale_func=getattr(config, "grad_scale_func", None),
        cp_group=cp_runtime.group,
        cp_backend=cp_runtime.backend,
    )
    scheduler = FixedTopologyScheduler(request.plan, executor, events=request.events)
    if tp_size > 1:
        from .tp_validation import assert_tp_plan_agreement, digest_segment_plan

        assert_tp_plan_agreement(
            digest_segment_plan(request.plan, scheduler.events), tp_size=tp_size
        )

    no_sync_func = getattr(config, "no_sync_func", None)
    if isinstance(no_sync_func, list):
        raise NotImplementedError("TPR MVP does not support per-model-chunk no_sync lists")
    no_sync_context = nullcontext() if no_sync_func is None else no_sync_func()
    with no_sync_context:
        result = scheduler.run()

    finalize_model_grads_func = getattr(config, "finalize_model_grads_func", None)
    if finalize_model_grads_func is None:
        raise RuntimeError("TPR training requires config.finalize_model_grads_func")
    finalize_model_grads_func(engine.module, None, force_all_reduce=True)

    global_loss_sum, global_normalized_loss = aggregate_cp_loss(
        result.loss_sum,
        result.normalized_loss,
        cp_runtime,
    )
    loss = global_normalized_loss.item()
    return {
        "model_output": {},
        "loss": loss,
        "metrics": {
            "tpr_loss": loss,
            "tpr_loss_sum": global_loss_sum.item(),
            "tpr_cp_size": cp_runtime.size,
            "tpr_cp_backend": cp_runtime.backend_name,
            "tpr_peak_path_tokens": result.peak_path_tokens,
            "tpr_segment_count": result.executed_segment_count,
            "tpr_direct_leaf_count": result.direct_leaf_count,
        },
    }


def _trajectory_keys_from_minibatch(data: Any) -> tuple[str, ...]:
    """Resolve stable per-row identities WITHOUT asking the VERL Trainer to build a plan.

    tpr_trajectory_keys preserves exact TQ identity when propagated to the actor
    mini-batch. The usual VERL uid column is an acceptable fallback: it
    provides the shared-sample grouping boundary, while row numbers guarantee
    unique trajectory keys after mini-batch sampling/shuffling.
    """
    from verl.utils import tensordict_utils as tu

    count = len(data["input_ids"])
    explicit = tu.get_non_tensor_data(data, key="tpr_trajectory_keys", default=None)
    if explicit is not None:
        if hasattr(explicit, "tolist"):
            explicit = explicit.tolist()
        keys = tuple(explicit)
        if len(keys) != count or not all(isinstance(key, str) for key in keys):
            raise ValueError("tpr_trajectory_keys must contain one string per mini-batch row")
        return keys

    uid = data.get("uid", None)
    if uid is None:
        raise ValueError(
            "TPR PPO requires tpr_trajectory_keys or a per-row uid in the actor "
            "mini-batch. Preserve the TQ identity across VERL preprocessing."
        )
    if hasattr(uid, "tolist"):
        values = uid.tolist()
    elif hasattr(uid, "unbind"):
        values = list(uid.unbind())
    else:
        values = list(uid)
    if len(values) != count:
        raise ValueError(f"uid row count {len(values)} != input_ids rows {count}")
    result = []
    for row, value in enumerate(values):
        value = tu.unwrap_non_tensor_data(value)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, (str, int)) or not str(value):
            raise ValueError(f"uid row {row} must be a nonempty string or integer")
        result.append(f"{value}_tpr_{row}")
    return tuple(result)


def run_tpr_forward_backward_batch(
    engine: MegatronEngine,
    data: Any,
    loss_function: Any,
    *,
    forward_only: bool = False,
) -> dict[str, Any]:
    """Native VERL PPO mini-batch -> TPR Forest schedule -> one gradient finalize.

    The Engine MUST compute batch_num_tokens / dp_size first. It MUST invoke
    this function BEFORE prepare_micro_batches, so shared trajectories remain
    in the same logical mini-batch. Current training scope: dense Qwen/GPT,
    TP=1/2, CP=1/2/4, PP=EP=DP=1 in production. Only the isolated
    TP2/CP2/DP2 replay may opt in through TPR_EXPERIMENTAL_OFFLINE_DP2=1.
    Ring required for CP>1,
    and native SP is restricted to CP=1 at TP2.
    Vanilla token-mean PPO; no native /M schedule
    scaling is applied because SegmentExecutor performs autograd itself.
    """
    import torch

    from verl.utils import tensordict_utils as tu
    from verl.utils.megatron_utils import unwrap_model

    from .objective_adapter import SegmentPPOObjectiveAdapter
    from .tpr_batch_runner import TPRBatchRunner
    from .tree_plan_builder import build_tree_execution_plans

    if forward_only:
        raise NotImplementedError("TPR PPO schedule is training-only; use native forward_only")
    if not engine.engine_config.tpr_enabled:
        raise RuntimeError("TPR PPO requires engine_config.tpr_enabled=True at model creation")
    if loss_function is None:
        raise ValueError("TPR PPO requires the original VERL actor loss_function")
    if len(engine.module) != 1:
        raise NotImplementedError("TPR PPO requires exactly one Megatron model chunk (PP=VPP=1)")

    sizes = {
        "TP": engine.engine_config.tensor_model_parallel_size,
        "PP": engine.engine_config.pipeline_model_parallel_size,
        "EP": engine.engine_config.expert_model_parallel_size,
        "CP": engine.engine_config.context_parallel_size,
        "DP": engine.get_data_parallel_size(),
    }
    tp_size = _validate_tpr_parallel_sizes(sizes, ppo=True)
    if engine.engine_config.virtual_pipeline_model_parallel_size not in (None, 1):
        raise NotImplementedError("TPR PPO does not support virtual PP")
    if engine.model_config.mtp.enable or engine.enable_routing_replay:
        raise NotImplementedError("TPR PPO does not yet support MTP or router replay")

    wrapped_model = engine.module[0]
    model = unwrap_model(wrapped_model)
    config = get_model_config(wrapped_model)
    if not model.training:
        raise RuntimeError("TPR PPO requires model.train()")
    unsupported = {
        "Megatron recompute": getattr(config, "recompute_granularity", None) is not None,
        "calculate_per_token_loss": getattr(config, "calculate_per_token_loss", False),
        "FP8": getattr(config, "fp8", None) not in (None, False),
        "MoE": getattr(config, "num_moe_experts", None) is not None,
        "CUDA graph": getattr(config, "cuda_graph_impl", None) not in (None, "none"),
        "native CPU offload": getattr(config, "cpu_offloading", False),
        "fused LM head": bool(getattr(engine.engine_config, "use_fused_kernels", False)),
    }
    if any(unsupported.values()):
        raise NotImplementedError(
            "TPR PPO phase-4 does not support: "
            + ", ".join(k for k, enabled in unsupported.items() if enabled)
        )
    attentions = [module for module in model.modules() if isinstance(module, TPRSelfAttention)]
    if not attentions:
        raise TypeError("TPR PPO requires TPRSelfAttention installed during model initialization")
    layer_numbers = tuple(sorted(attention.layer_number for attention in attentions))

    token_count = tu.get_non_tensor_data(data, key="batch_num_tokens", default=None)
    dp_size = tu.get_non_tensor_data(data, key="dp_size", default=None)
    if (token_count is None or token_count <= 0
            or dp_size != sizes["DP"]):
        raise ValueError(
            "Engine must attach positive GLOBAL batch_num_tokens and the "
            f"actual dp_size={sizes['DP']} before TPR routing"
        )

    import os
    import time

    detailed_profile = os.getenv("TPR_EXEC_PROFILE", "0") == "1"
    stage_ms = {}
    profile_origin = time.perf_counter()
    tree_policy = os.getenv("TPR_TREE_POLICY", "exact")
    try:
        turn_min_tokens = int(os.getenv("TPR_TURN_MIN_SEGMENT_TOKENS", "128"))
    except ValueError as exc:
        raise ValueError("TPR_TURN_MIN_SEGMENT_TOKENS must be a positive integer") from exc
    keys = _trajectory_keys_from_minibatch(data)
    forest = build_tree_execution_plans(
        keys, data, tree_policy=tree_policy,
        turn_min_segment_tokens=turn_min_tokens,
    )
    if not forest.trees or not forest.logical_loss_tokens:
        raise ValueError("TPR PPO mini-batch has no supervised response tokens")
    if detailed_profile:
        stage_ms["tree_build_cpu_ms"] = (time.perf_counter()-profile_origin)*1000
    tp_check_start = time.perf_counter()
    if tp_size > 1:
        from .tp_validation import assert_tp_plan_agreement, digest_forest_plan

        assert_tp_plan_agreement(digest_forest_plan(forest), tp_size=tp_size)
    if detailed_profile:
        stage_ms["tp_plan_collective_ms"] = (time.perf_counter()-tp_check_start)*1000
    cp_runtime = resolve_engine_cp_runtime(engine, config)
    _validate_tpr_cp_combination(tp_size, cp_runtime, config, ppo=True)
    # DP2+CP2 has twice the per-rank local sequence of the CP4 baseline.
    # Default this *isolated* experiment to recomputable PPO logprob chunks;
    # ordinary TPR and all other topologies remain unchanged.
    default_lp_chunk = "512" if sizes["DP"] == 2 and sizes["CP"] == 2 else "0"
    try:
        lp_chunk_tokens = int(os.getenv("TPR_PPO_LOGPROB_CHUNK_TOKENS", default_lp_chunk))
    except ValueError as exc:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be a nonnegative integer") from exc
    if lp_chunk_tokens < 0:
        raise ValueError("TPR_PPO_LOGPROB_CHUNK_TOKENS must be nonnegative")
    objective = SegmentPPOObjectiveAdapter(
        data, loss_function, dp_group=engine.get_data_parallel_group(),
        log_prob_chunk_size=lp_chunk_tokens or None,
    )

    # Native Megatron uses /M inside its PP schedule. That schedule is bypassed;
    # the objective already carries the full VERL global denominator, so do
    # NOT multiply or divide by the number of physical tree segments here.
    no_sync_func = getattr(config, "no_sync_func", None)
    if isinstance(no_sync_func, list):
        raise NotImplementedError("TPR PPO does not support per-chunk no_sync callbacks")
    finalize = getattr(config, "finalize_model_grads_func", None)
    if not callable(finalize):
        raise RuntimeError("TPR PPO requires native finalize_model_grads_func")

    metric_lists: dict[str, list[Any]] = {}
    from .segment_event_profiler import SegmentEventProfiler
    event_profiler = SegmentEventProfiler() if detailed_profile else None
    setup_ms = 0.0
    metrics_ms = 0.0
    tree_number = 0

    def run_one_tree(tree_plan):
        nonlocal setup_ms, metrics_ms, tree_number
        setup_start = time.perf_counter() if detailed_profile else None
        shard_for_segment = None if cp_runtime.backend is None else lambda segment: cp_runtime.backend.make_sequence_shard(segment.length)
        callback, counts = objective.bind_tree(tree_plan, shard_for_segment=shard_for_segment)
        executor = SegmentExecutor(
            model,
            tree_plan.segment_plan,
            expected_layer_numbers=layer_numbers,
            loss_scale_func=getattr(config, "grad_scale_func", None),
            segment_loss_fn=callback,
            segment_loss_term_counts=counts,
            cp_group=cp_runtime.group,
            cp_backend=cp_runtime.backend,
        )
        if detailed_profile:
            setup_ms += (time.perf_counter()-setup_start)*1000
        result = FixedTopologyScheduler(
            tree_plan.segment_plan, executor,
            event_profiler=event_profiler, tree_index=tree_number,
        ).run()
        tree_number += 1
        metrics_start = time.perf_counter() if detailed_profile else None
        if not bool(torch.isfinite(result.normalized_loss).item()):
            raise FloatingPointError("non-finite TPR PPO tree loss; gradients will not be finalized")
        from verl.utils.metric import AggregationType, Metric

        for segment_id, metrics in executor.segment_loss_metrics:
            token_weight = counts[segment_id] / forest.logical_loss_tokens
            for name, value in metrics.items():
                if isinstance(value, Metric) and value.aggregation is AggregationType.MEAN:
                    # PPO reports e.g. clipfrac and approx-KL as per-segment
                    # token means. A simple mean across physical segments
                    # would be biased by their extremely unequal lengths.
                    # Convert to additive weighted contributions, preserving
                    # the original logical token-mean reporting semantics.
                    value = Metric(
                        aggregation=AggregationType.SUM,
                        value=value.aggregate() * token_weight,
                    )
                elif isinstance(value, Metric) and value.aggregation not in (
                    AggregationType.SUM, AggregationType.MEAN
                ):
                    raise NotImplementedError(
                        f"TPR PPO metric {name} requires unsupported {value.aggregation}"
                    )
                metric_lists.setdefault(name, []).append(value)
        if detailed_profile:
            metrics_ms += (time.perf_counter()-metrics_start)*1000
        return result

    runner = TPRBatchRunner(
        run_tree=run_one_tree,
        finalize_gradients=lambda: finalize(engine.module, None, force_all_reduce=True),
        no_sync_context=no_sync_func,
    )
    run_start = time.perf_counter() if detailed_profile else None
    result = runner.run(forest)
    if detailed_profile:
        stage_ms["executor_setup_cpu_ms"] = setup_ms
        stage_ms["segment_metrics_cpu_ms"] = metrics_ms
        stage_ms["tree_run_and_finalize_host_ms"] = (time.perf_counter()-run_start)*1000
    if not bool(torch.isfinite(result.normalized_loss).item()):
        raise FloatingPointError("non-finite TPR PPO mini-batch loss")
    if objective.capture_log_probs:
        # Explicitly opt-in through tpr_capture_log_probs for numerical gates.
        # Never materialize all per-token logprobs on the host during training.
        engine._tpr_captured_log_probs = dict(objective.debug_new_log_probs)
    if objective.capture_logit_parts:
        engine._tpr_captured_logit_parts = dict(objective.debug_logit_parts)

    logical_input_tokens = sum(int(row.numel()) for row in data["input_ids"].unbind())
    unique_tree_tokens = sum(
        segment.length for tree in forest.trees
        for segment in tree.segment_plan.segments.values()
    )
    # TPRBatchResult exposes normalized_loss only; all segment PPO losses
    # already share the original global batch denominator. The helper's
    # first (loss_sum) return is unused here, so pass the same additive
    # local objective to both reduction slots.
    _, global_normalized_loss = aggregate_cp_loss(
        result.normalized_loss, result.normalized_loss, cp_runtime
    )
    loss = float(global_normalized_loss.item())
    if detailed_profile:
        stage_ms["whole_entry_host_ms"] = (time.perf_counter()-profile_origin)*1000
        # NPU events must only be materialized after caller's one final NPU sync.
        engine._tpr_segment_event_profiler = event_profiler
        engine._tpr_stage_profile = stage_ms
        engine._tpr_segment_peak_allocated_bytes = event_profiler.max_allocated_bytes
        engine._tpr_segment_peak_reserved_bytes = event_profiler.max_reserved_bytes
    metric_lists.update({
        "tpr/forest_trees": [result.tree_count],
        "tpr/physical_segments": [result.segment_count],
        "tpr/logical_loss_tokens": [result.logical_loss_tokens],
        "tpr/logical_input_tokens": [logical_input_tokens],
        "tpr/unique_tree_tokens": [unique_tree_tokens],
        "tpr/topology_token_reuse_ratio": [logical_input_tokens / unique_tree_tokens],
    })
    # The V1 Trainer reduces list-valued metrics via numpy.mean, which cannot
    # add VERL Metric wrappers. Collapse weighted segment contributions to
    # numeric scalars before returning them across the worker boundary.
    for name, values in metric_lists.items():
        from verl.utils.metric import Metric

        if any(isinstance(value, Metric) for value in values):
            if not all(isinstance(value, Metric) for value in values):
                raise TypeError(f"TPR metric {name} mixes Metric and numeric values")
            # Segment Metric(MEAN) values were converted to SUM contributions
            # weighted by their share of the original global response tokens.
            # The trainer later uses np.mean(list), so return ONE aggregate
            # numeric value, not separate weighted contributions.
            metric_lists[name] = [sum(float(value.aggregate()) for value in values)]
        else:
            metric_lists[name] = [float(value) for value in values]

    # VERL postprocess expects list-valued micro-batch metrics and a list of
    # loss contributions. The sum of all physical segment objectives is the
    # entire mini-batch objective, with no additional micro-batch factor.
    return {
        "model_output": {},
        "loss": [loss],
        "metrics": metric_lists,
    }
