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

"""Thin integration helpers between verl's MegatronEngine and DTA."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from megatron.core.utils import get_model_config

from verl.utils.megatron_utils import unwrap_model

from .attention import DTASelfAttention
from .engine_adapter import DTAForwardBackwardRequest
from .fixed_topology_scheduler import FixedTopologyScheduler
from .module_spec import make_dta_module_spec_provider
from .segment_executor import SegmentExecutor

if TYPE_CHECKING:
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine


def install_dta_module_spec(provider: Any) -> None:
    """Install the opt-in DTA SelfAttention spec before model construction."""

    if provider is None:
        raise NotImplementedError("DTA model construction requires the non-vanilla Megatron-Bridge provider")
    provider.transformer_layer_spec = make_dta_module_spec_provider(provider.transformer_layer_spec)


def run_dta_forward_backward(
    engine: MegatronEngine,
    request: DTAForwardBackwardRequest,
    *,
    forward_only: bool,
) -> dict[str, Any]:
    """Run the single-rank DTA MVP instead of Megatron's linear schedule."""

    if not isinstance(request, DTAForwardBackwardRequest):
        raise TypeError(
            "dta_forward_backward_request must be a DTAForwardBackwardRequest, "
            f"got {type(request).__name__}"
        )
    if forward_only:
        raise NotImplementedError("DTA thin entry only supports training forward/backward")
    if not engine.engine_config.dta_enabled:
        raise RuntimeError("DTA request requires engine_config.dta_enabled=True before model construction")
    if len(engine.module) != 1:
        raise NotImplementedError("DTA MVP requires exactly one Megatron model chunk")

    parallel_sizes = {
        "TP": engine.engine_config.tensor_model_parallel_size,
        "PP": engine.engine_config.pipeline_model_parallel_size,
        "CP": engine.engine_config.context_parallel_size,
        "EP": engine.engine_config.expert_model_parallel_size,
        "DP": engine.get_data_parallel_size(),
    }
    unsupported_sizes = {name: size for name, size in parallel_sizes.items() if size != 1}
    if unsupported_sizes:
        raise NotImplementedError(f"DTA MVP requires all parallel sizes to be 1, got {unsupported_sizes}")
    if engine.engine_config.virtual_pipeline_model_parallel_size is not None:
        raise NotImplementedError("DTA MVP does not support virtual pipeline parallelism")
    if engine.model_config.mtp.enable:
        raise NotImplementedError("DTA MVP does not support MTP")
    if engine.enable_routing_replay:
        raise NotImplementedError("DTA MVP does not support router replay")

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
        raise NotImplementedError(f"DTA MVP does not support: {', '.join(active_restrictions)}")
    if not model.training:
        raise RuntimeError("DTA thin entry requires model.train()")

    dta_attentions = [module for module in model.modules() if isinstance(module, DTASelfAttention)]
    if not dta_attentions:
        raise TypeError("DTA thin entry requires a model built with DTASelfAttention")
    layer_numbers = tuple(sorted(attention.layer_number for attention in dta_attentions))
    executor = SegmentExecutor(
        model,
        request.plan,
        expected_layer_numbers=layer_numbers,
        loss_scale_func=getattr(config, "grad_scale_func", None),
    )
    scheduler = FixedTopologyScheduler(request.plan, executor, events=request.events)

    no_sync_func = getattr(config, "no_sync_func", None)
    if isinstance(no_sync_func, list):
        raise NotImplementedError("DTA MVP does not support per-model-chunk no_sync lists")
    no_sync_context = nullcontext() if no_sync_func is None else no_sync_func()
    with no_sync_context:
        result = scheduler.run()

    finalize_model_grads_func = getattr(config, "finalize_model_grads_func", None)
    if finalize_model_grads_func is None:
        raise RuntimeError("DTA training requires config.finalize_model_grads_func")
    finalize_model_grads_func(engine.module, None, force_all_reduce=True)

    loss = result.normalized_loss.item()
    return {
        "model_output": {},
        "loss": loss,
        "metrics": {
            "dta_loss": loss,
            "dta_loss_sum": result.loss_sum.item(),
            "dta_peak_path_tokens": result.peak_path_tokens,
            "dta_segment_count": result.executed_segment_count,
            "dta_direct_leaf_count": result.direct_leaf_count,
        },
    }
