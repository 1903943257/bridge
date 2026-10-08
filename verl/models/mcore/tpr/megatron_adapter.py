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
from .module_spec import make_tpr_module_spec_provider
from .parallel.engine_runtime import aggregate_cp_loss, resolve_engine_cp_runtime
from .segment_executor import SegmentExecutor

if TYPE_CHECKING:
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def install_tpr_module_spec(provider: Any) -> None:
    """Install the opt-in TPR SelfAttention spec before model construction."""

    if provider is None:
        raise NotImplementedError("TPR model construction requires the non-vanilla Megatron-Bridge provider")
    provider.transformer_layer_spec = make_tpr_module_spec_provider(provider.transformer_layer_spec)


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
        "PP": engine.engine_config.pipeline_model_parallel_size,
        "EP": engine.engine_config.expert_model_parallel_size,
        "DP": engine.get_data_parallel_size(),
    }
    unsupported_sizes = {name: size for name, size in parallel_sizes.items() if size != 1}
    if unsupported_sizes:
        raise NotImplementedError(
            "TPR formal CP entry currently requires TP/PP/EP/DP=1, "
            f"got {unsupported_sizes}"
        )
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
    executor = SegmentExecutor(
        model,
        request.plan,
        expected_layer_numbers=layer_numbers,
        loss_scale_func=getattr(config, "grad_scale_func", None),
        cp_group=cp_runtime.group,
        cp_backend=cp_runtime.backend,
    )
    scheduler = FixedTopologyScheduler(request.plan, executor, events=request.events)

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
    PP=TP=EP=CP=DP=1, vanilla token-mean PPO; no native Megatron /M schedule
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
    if any(size != 1 for size in sizes.values()):
        raise NotImplementedError(f"TPR PPO phase-4 is single-rank only, got {sizes}")
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
    if token_count is None or token_count <= 0 or dp_size != 1:
        raise ValueError("Engine must attach positive batch_num_tokens and dp_size=1 before TPR routing")

    keys = _trajectory_keys_from_minibatch(data)
    forest = build_tree_execution_plans(keys, data)
    if not forest.trees or not forest.logical_loss_tokens:
        raise ValueError("TPR PPO mini-batch has no supervised response tokens")
    objective = SegmentPPOObjectiveAdapter(
        data, loss_function, dp_group=engine.get_data_parallel_group()
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

    def run_one_tree(tree_plan):
        callback, counts = objective.bind_tree(tree_plan)
        executor = SegmentExecutor(
            model,
            tree_plan.segment_plan,
            expected_layer_numbers=layer_numbers,
            loss_scale_func=getattr(config, "grad_scale_func", None),
            segment_loss_fn=callback,
            segment_loss_term_counts=counts,
        )
        result = FixedTopologyScheduler(tree_plan.segment_plan, executor).run()
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
        return result

    runner = TPRBatchRunner(
        run_tree=run_one_tree,
        finalize_gradients=lambda: finalize(engine.module, None, force_all_reduce=True),
        no_sync_context=no_sync_func,
    )
    result = runner.run(forest)
    if not bool(torch.isfinite(result.normalized_loss).item()):
        raise FloatingPointError("non-finite TPR PPO mini-batch loss")
    if objective.capture_log_probs:
        # Explicitly opt-in through tpr_capture_log_probs for numerical gates.
        # Never materialize all per-token logprobs on the host during training.
        engine._tpr_captured_log_probs = dict(objective.debug_new_log_probs)

    logical_input_tokens = sum(int(row.numel()) for row in data["input_ids"].unbind())
    unique_tree_tokens = sum(
        segment.length for tree in forest.trees
        for segment in tree.segment_plan.segments.values()
    )
    loss = float(result.normalized_loss.item())
    metric_lists.update({
        "tpr/forest_trees": [result.tree_count],
        "tpr/physical_segments": [result.segment_count],
        "tpr/logical_loss_tokens": [result.logical_loss_tokens],
        "tpr/logical_input_tokens": [logical_input_tokens],
        "tpr/unique_tree_tokens": [unique_tree_tokens],
        "tpr/topology_token_reuse_ratio": [logical_input_tokens / unique_tree_tokens],
    })
    # VERL postprocess expects list-valued micro-batch metrics and a list of
    # loss contributions. The sum of all physical segment objectives is the
    # entire mini-batch objective, with no additional micro-batch factor.
    return {
        "model_output": {},
        "loss": [loss],
        "metrics": metric_lists,
    }
