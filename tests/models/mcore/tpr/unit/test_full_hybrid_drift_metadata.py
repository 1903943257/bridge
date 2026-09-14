"""CPU checks for drift diagnostic grouping and objective fingerprints."""

from types import SimpleNamespace

import pytest

from ..parallel._full_hybrid_drift_diagnostic import parameter_category, plan_digest


@pytest.mark.parametrize("name,expected", [
    ("decoder.layers.0.self_attention.A_log", "GDN"),
    ("decoder.layers.3.self_attention.linear_qkv.weight", "FA"),
    ("decoder.layers.23.self_attention.linear_proj.weight", "FA"),
    ("decoder.layers.0.mlp.linear_fc2.weight", "MLP"),
    ("decoder.layers.0.self_attention.in_proj.layer_norm_weight", "norm"),
    ("decoder.final_layernorm.weight", "norm"),
    ("embedding.word_embeddings.weight", "embedding/output"),
    ("output_layer.weight", "embedding/output"),
])
def test_parameter_category(name, expected):
    assert parameter_category(name) == expected


def test_plan_digest_includes_position_label_and_weight():
    term = SimpleNamespace(query_offset=0, target_token_id=7, weight=2.)
    segment = SimpleNamespace(segment_id=0, parent_id=None, position_start=0,
                              prefix_length=0, token_ids=SimpleNamespace(tolist=lambda: [3, 7]),
                              loss_terms=[term])
    plan = SimpleNamespace(segments={0: segment})
    baseline = plan_digest(plan)
    assert plan_digest(plan) == baseline
    for target, field, value in ((term, "target_token_id", 8), (term, "weight", 1.),
                                  (segment, "position_start", 1)):
        original = getattr(target, field)
        setattr(target, field, value)
        assert plan_digest(plan) != baseline
        setattr(target, field, original)
    assert plan_digest(plan) == baseline
