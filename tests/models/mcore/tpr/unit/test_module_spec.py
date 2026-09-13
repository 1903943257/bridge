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

from types import SimpleNamespace

import pytest

from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from verl.models.mcore.tpr import (
    TPRSelfAttention,
    make_tpr_module_spec_provider,
    replace_self_attention_with_tpr,
)


class _TransformerLayer:
    pass


class _TransformerBlock:
    pass


class _LinearQKV:
    pass


class _CoreAttention:
    pass


class _LinearProj:
    pass


class _UnsupportedAttention:
    pass


def _layer_spec(attention_module=SelfAttention):
    attention_submodules = SimpleNamespace(
        linear_qkv=ModuleSpec(module=_LinearQKV),
        core_attention=ModuleSpec(module=_CoreAttention),
        linear_proj=ModuleSpec(module=_LinearProj),
        q_layernorm=None,
        k_layernorm=None,
    )
    attention_spec = ModuleSpec(
        module=attention_module,
        params={"attn_mask_type": "causal"},
        submodules=attention_submodules,
        metainfo={"source": "original"},
    )
    return ModuleSpec(
        module=_TransformerLayer,
        params={"layer_param": 1},
        submodules=SimpleNamespace(
            self_attention=attention_spec,
            mlp=ModuleSpec(module=object),
        ),
        metainfo={"layer": "metadata"},
    )


def _attention_spec(layer_spec):
    return layer_spec.submodules.self_attention


def test_replaces_single_transformer_layer_without_mutating_input():
    original = _layer_spec()

    converted = replace_self_attention_with_tpr(original)

    assert converted is not original
    assert _attention_spec(original).module is SelfAttention
    assert _attention_spec(converted).module is TPRSelfAttention


def test_preserves_attention_and_layer_spec_configuration():
    original = _layer_spec()

    converted = replace_self_attention_with_tpr(original)
    original_attention = _attention_spec(original)
    converted_attention = _attention_spec(converted)

    assert converted.params == original.params
    assert converted.metainfo == original.metainfo
    assert converted_attention.params == original_attention.params
    assert converted_attention.metainfo == original_attention.metainfo
    assert converted_attention.submodules.linear_qkv.module is _LinearQKV
    assert converted_attention.submodules.core_attention.module is _CoreAttention
    assert converted_attention.submodules.linear_proj.module is _LinearProj
    assert converted.submodules.mlp.module is object


def test_replaces_every_layer_in_transformer_block_submodules():
    original_layers = [_layer_spec(), _layer_spec(), _layer_spec()]
    block_submodules = SimpleNamespace(layer_specs=original_layers, layer_norm=object)

    converted = replace_self_attention_with_tpr(block_submodules)

    assert converted is not block_submodules
    assert all(_attention_spec(layer).module is TPRSelfAttention for layer in converted.layer_specs)
    assert all(_attention_spec(layer).module is SelfAttention for layer in original_layers)
    assert converted.layer_norm is object


def test_replaces_layers_nested_in_transformer_block_module_spec():
    block_spec = ModuleSpec(
        module=_TransformerBlock,
        submodules=SimpleNamespace(layer_specs=[_layer_spec(), _layer_spec()]),
    )

    converted = replace_self_attention_with_tpr(block_spec)

    assert all(
        _attention_spec(layer).module is TPRSelfAttention
        for layer in converted.submodules.layer_specs
    )
    assert all(
        _attention_spec(layer).module is SelfAttention
        for layer in block_spec.submodules.layer_specs
    )


def test_replacement_is_idempotent():
    converted_once = replace_self_attention_with_tpr(_layer_spec())
    converted_twice = replace_self_attention_with_tpr(converted_once)

    assert _attention_spec(converted_twice).module is TPRSelfAttention
    assert _attention_spec(converted_twice).params == _attention_spec(converted_once).params


def test_hybrid_spec_dispatch_preserves_pattern_and_is_idempotent(monkeypatch):
    import sys
    from types import ModuleType

    class GatedDeltaNet:
        pass

    class TPRGatedDeltaNet(GatedDeltaNet):
        tpr_state_kind = "gdn"

    native = ModuleType("mindspeed.core.ssm.gated_delta_net")
    native.GatedDeltaNet = GatedDeltaNet
    adapted = ModuleType("verl.models.mcore.tpr.gated_delta_net")
    adapted.TPRGatedDeltaNet = TPRGatedDeltaNet
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, adapted.__name__, adapted)
    original = SimpleNamespace(layer_specs=[
        _layer_spec(cls) for cls in (GatedDeltaNet,) * 3 + (SelfAttention,)
    ])
    converted = replace_self_attention_with_tpr(original)
    twice = replace_self_attention_with_tpr(converted)
    assert [_attention_spec(layer).module for layer in twice.layer_specs] == [
        TPRGatedDeltaNet, TPRGatedDeltaNet, TPRGatedDeltaNet, TPRSelfAttention,
    ]
    assert _attention_spec(original.layer_specs[0]).module is GatedDeltaNet
    for before, after in zip(original.layer_specs, converted.layer_specs, strict=True):
        assert _attention_spec(before).params == _attention_spec(after).params


def test_rejects_unsupported_attention_module():
    with pytest.raises(TypeError, match="unsupported attention module"):
        replace_self_attention_with_tpr(_layer_spec(_UnsupportedAttention))


@pytest.mark.parametrize(
    ("spec", "exception", "match"),
    [
        (None, TypeError, "must not be None"),
        (SimpleNamespace(layer_specs=[]), ValueError, "non-empty list"),
        (SimpleNamespace(layer_specs=[object()]), TypeError, "must be a ModuleSpec"),
        (ModuleSpec(module=_TransformerLayer), ValueError, "must describe either"),
    ],
)
def test_rejects_invalid_spec_structures(spec, exception, match):
    with pytest.raises(exception, match=match):
        replace_self_attention_with_tpr(spec)


def test_tpr_spec_provider_wraps_callable_without_vp_stage():
    original = _layer_spec()

    def provider(config):
        assert config == "config"
        return original

    wrapped = make_tpr_module_spec_provider(provider)
    converted = wrapped("config")

    assert _attention_spec(converted).module is TPRSelfAttention
    assert _attention_spec(original).module is SelfAttention


def test_tpr_spec_provider_forwards_vp_stage():
    original = _layer_spec()
    calls = []

    def provider(config, vp_stage=None):
        calls.append((config, vp_stage))
        return original

    wrapped = make_tpr_module_spec_provider(provider)
    converted = wrapped("config", vp_stage=3)

    assert calls == [("config", 3)]
    assert _attention_spec(converted).module is TPRSelfAttention
