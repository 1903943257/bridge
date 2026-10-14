"""CPU-only regression for Full-reference vs DP2 gradient witness reporting."""

import pytest

from ..parallel.compare_tpr_tp2cp2dp2_one_step import compare_sampled_gradients


def _witness(*, replica_error=0.0):
    return {
        "names": ["w0", "w1"],
        "shapes": [[8], [16]],
        "samples_per_name": [2, 2],
        "samples_per_parameter": 2,
        "tp_shards": [[0.25, 0.5, 1.0, -1.0], [0.25, 0.5, 1.0, -1.0]],
        "replica_sample_max_abs": replica_error,
    }


def test_full_reference_replica_roundoff_does_not_masquerade_as_dp2_failure():
    full = {"gradient_witness": _witness(replica_error=3.05175781e-05)}
    tpr = {"gradient_witness": _witness()}
    result = compare_sampled_gradients(full, tpr)
    assert result["sampled_gradient_rel_l2"] == 0
    assert result["full_reference_replica_exact"] is False
    assert result["full_reference_replica_sample_max_abs"] == pytest.approx(3.05175781e-05)
    assert result["tpr_dp2_replica_exact"] is True
    assert "Full CP4" in result["sampled_gradient_parity_caveat"]


def test_tpr_dp2_replica_disagreement_remains_a_hard_failure():
    full = {"gradient_witness": _witness()}
    tpr = {"gradient_witness": _witness(replica_error=3.05175781e-05)}
    with pytest.raises(AssertionError, match="TPR DP2 replica"):
        compare_sampled_gradients(full, tpr)


def test_both_modes_must_provide_witness_when_requested():
    with pytest.raises(AssertionError, match="both"):
        compare_sampled_gradients({"gradient_witness": _witness()}, {})
