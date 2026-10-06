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
