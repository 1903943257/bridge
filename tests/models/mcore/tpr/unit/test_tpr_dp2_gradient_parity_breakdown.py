"""CPU tests for analyzing frozen TP2 Full-vs-TPR gradient witness JSON."""
import math

import pytest

from ..parallel.analyze_tpr_dp2_gradient_parity import analyze_gradient_witness


def _results(full_shards, tpr_shards, names=None, counts=None):
    names = names or ["decoder.layers.0.input_layernorm.weight", "decoder.layers.1.mlp.linear_fc1.weight"]
    counts = counts or [2, 2]
    base = {
        "names": names, "shapes": [[2] for _ in names],
        "samples_per_name": counts, "samples_per_parameter": 2,
        "replica_exact": True,
    }
    full = {"source_step": 6, "input_sha256": "sha", "runs": [{"loss": 0.15}],
            "gradient_witness": {**base, "tp_shards": full_shards}}
    tpr = {"source_step": 6, "input_sha256": "sha", "runs": [{"loss": 0.15001}],
           "gradient_witness": {**base, "tp_shards": tpr_shards}}
    return full, tpr


def test_exact_gradients_have_zero_errors():
    v = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]
    r = analyze_gradient_witness(*_results(v, v))
    assert r["global_sampled"]["relative_l2"] == pytest.approx(0)
    assert r["global_sampled"]["cosine"] == pytest.approx(1)
    assert r["global_sampled"]["best_tpr_over_full_scale"] == pytest.approx(1)
    assert all(x["share_of_total_error_sq"] == 0 for x in r["all_parameters_ranked"])


def test_global_scale_error_is_identified_as_scale_not_direction():
    v = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]]
    t = [[2*x for x in a] for a in v]
    r = analyze_gradient_witness(*_results(v, t))
    assert r["global_sampled"]["relative_l2"] == pytest.approx(1)
    assert r["global_sampled"]["cosine"] == pytest.approx(1)
    assert r["global_sampled"]["best_tpr_over_full_scale"] == pytest.approx(2)
    assert r["global_sampled"]["relative_l2_after_best_scalar_rescale"] == pytest.approx(0)


def test_localized_gradient_error_is_sorted_by_squared_error_and_layer():
    f = [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
    t = [[1.0, 0.0, 5.0, 0.0], [1.0, 0.0, 1.0, 0.0]]
    r = analyze_gradient_witness(*_results(f, t))
    p = r["top_parameters"][0]
    assert p["tp_rank"] == 0
    assert p["name"] == "decoder.layers.1.mlp.linear_fc1.weight"
    assert p["layer"] == "1"
    assert p["kind"] == "mlp"
    assert p["share_of_total_error_sq"] == pytest.approx(1)
    assert r["per_layer"]["1"]["share_of_total_error_sq"] == pytest.approx(1)


def test_missing_witness_and_layout_mismatch_fail():
    f, t = _results([[1,2,3,4], [1,2,3,4]], [[1,2,3,4], [1,2,3,4]])
    with pytest.raises(ValueError, match="gradient_witness"):
        analyze_gradient_witness({}, t)
    t["gradient_witness"]["samples_per_name"] = [4, 4]
    with pytest.raises(AssertionError, match="samples_per_name"):
        analyze_gradient_witness(f, t)



def test_non_norm_breakdown_and_repeated_tp_norm_samples_are_explicit():
    f = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 5.0, 6.0]]
    t = [[2.0, 4.0, 3.0, 4.0], [2.0, 4.0, 5.0, 6.0]]
    report = analyze_gradient_witness(*_results(f, t))
    assert report["sampled_non_norm"]["relative_l2"] == 0
    assert report["norm_share_of_sampled_squared_error"] == pytest.approx(1.0)
    assert report["identical_norm_sample_pairs_across_tp"]["count"] == 1
    assert report["norm_share_of_full_sampled_gradient_energy"] > 0
    assert "NOT full-gradient rel-L2" in report["sampling_warning"]
