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

"""ModuleSpec transformation for installing TPR self-attention."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec, get_module

from .attention import TPRSelfAttention

_SpecT = TypeVar("_SpecT")


def replace_self_attention_with_tpr(transformer_layer_spec: _SpecT) -> _SpecT:
    """Copy a spec, dispatching FA/GDN to their respective TPR attention classes.

    Both forms used by the local verl model builders are accepted: a single
    TransformerLayer ``ModuleSpec`` and a TransformerBlock submodule object
    containing ``layer_specs``.  Only the nested attention module class is
    changed; its params, submodules, and metainfo are preserved.
    """

    if transformer_layer_spec is None:
        raise TypeError("transformer_layer_spec must not be None")

    copied_spec = copy.deepcopy(transformer_layer_spec)
    layer_specs = _get_layer_specs(copied_spec)

    for index, layer_spec in enumerate(layer_specs):
        if not isinstance(layer_spec, ModuleSpec):
            raise TypeError(
                f"layer_specs[{index}] must be a ModuleSpec, "
                f"got {type(layer_spec).__name__}"
            )
        layer_submodules = layer_spec.submodules
        if layer_submodules is None or not hasattr(layer_submodules, "self_attention"):
            raise ValueError(f"layer_specs[{index}] does not define self_attention submodules")

        attention_spec = layer_submodules.self_attention
        if not isinstance(attention_spec, ModuleSpec):
            raise TypeError(
                f"layer_specs[{index}].submodules.self_attention must be a ModuleSpec, "
                f"got {type(attention_spec).__name__}"
            )

        attention_module = get_module(attention_spec)
        if attention_module is TPRSelfAttention:
            continue
        if getattr(attention_module, "tpr_state_kind", None) == "gdn":
            continue
        if getattr(attention_module, "__name__", None) == "GatedDeltaNet":
            # Import only after the NPU adaptor has installed the MindSpeed spec.
            from mindspeed.core.ssm.gated_delta_net import GatedDeltaNet

            from .gated_delta_net import TPRGatedDeltaNet

            if attention_module is not GatedDeltaNet:
                raise TypeError("TPR GDN requires the patched MindSpeed GatedDeltaNet spec")
            layer_submodules.self_attention = replace(attention_spec, module=TPRGatedDeltaNet)
            continue
        if attention_module is not SelfAttention:
            module_name = getattr(attention_module, "__name__", repr(attention_module))
            raise TypeError(
                f"layer_specs[{index}] uses unsupported attention module {module_name}; "
                "TPR requires Megatron SelfAttention or patched MindSpeed GatedDeltaNet"
            )

        layer_submodules.self_attention = replace(attention_spec, module=TPRSelfAttention)

    return copied_spec


def make_tpr_module_spec_provider(
    original: _SpecT | Callable[..., _SpecT],
) -> _SpecT | Callable[..., _SpecT]:
    """Wrap a Megatron-Bridge layer-spec provider with the TPR replacement."""

    if not callable(original):
        return replace_self_attention_with_tpr(original)
    accepts_vp_stage = "vp_stage" in inspect.signature(original).parameters

    def tpr_spec_provider(config: Any, vp_stage: int | None = None) -> _SpecT:
        spec = original(config, vp_stage=vp_stage) if accepts_vp_stage else original(config)
        return replace_self_attention_with_tpr(spec)

    return tpr_spec_provider


def _get_layer_specs(transformer_layer_spec: object) -> list[ModuleSpec]:
    if isinstance(transformer_layer_spec, ModuleSpec):
        submodules = transformer_layer_spec.submodules
        if submodules is not None and hasattr(submodules, "layer_specs"):
            layer_specs = submodules.layer_specs
        elif submodules is not None and hasattr(submodules, "self_attention"):
            layer_specs = [transformer_layer_spec]
        else:
            raise ValueError(
                "ModuleSpec must describe either a TransformerLayer with self_attention "
                "or a TransformerBlock with layer_specs"
            )
    elif hasattr(transformer_layer_spec, "layer_specs"):
        layer_specs = transformer_layer_spec.layer_specs
    else:
        raise TypeError(
            "transformer_layer_spec must be a TransformerLayer ModuleSpec or an object "
            f"containing layer_specs, got {type(transformer_layer_spec).__name__}"
        )

    if not isinstance(layer_specs, list) or not layer_specs:
        raise ValueError("layer_specs must be a non-empty list")
    return layer_specs
