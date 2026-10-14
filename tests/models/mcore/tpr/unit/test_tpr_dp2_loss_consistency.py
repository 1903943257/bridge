"""CPU-only gates for reporting losses in TP2 CP2 DP2 benchmark."""

import pytest

from ..parallel.tpr_dp2_loss_consistency import analyze_dp2_loss_report


def _ranks(tp0=0.125, tp1=0.125, *, corrupt=None):
    rows = []
    for tp, value in ((0, tp0), (1, tp1)):
        for cp in range(2):
            for dp in range(2):
                rows.append({
                    "rank": len(rows), "tp_rank": tp, "cp_rank": cp,
                    "dp_rank": dp, "global_loss_pre_tp": value,
                    "global_loss": (tp0 + tp1) / 2,
                })
    if corrupt is not None:
        rows[corrupt["row"]][corrupt["key"]] = corrupt["value"]
    return rows


def test_tp_shard_numerical_drift_is_measured_not_misclassified_as_dp_divergence():
    # The old cross-world comparison at 1e-4 rejected this legitimate
    # per-TP reporting discrepancy before a performance JSON was written.
    r = analyze_dp2_loss_report(_ranks(0.125, 0.12525))
    assert r["tp_shard_loss_gap"] == pytest.approx(0.00025)
    assert r["tp_shard_loss_gap_exceeds_1e_4"]
    assert r["cpdp_pre_tp_max_gap"] == 0
    assert r["global_reporting_max_gap"] == 0
    assert r["global_reporting_loss"] == pytest.approx(0.125125)
    assert r["training_objective_modified"] is False


def test_cpdp_replica_disagreement_is_still_rejected():
    with pytest.raises(AssertionError, match="CP/DP replicas"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 1, "key": "global_loss_pre_tp", "value": 0.5}
        ))


def test_wrong_tp_cp_dp_coordinates_are_rejected():
    with pytest.raises(AssertionError, match="exactly once"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 1, "key": "dp_rank", "value": 0}
        ))


def test_failed_tp_reporting_reduction_is_rejected():
    with pytest.raises(AssertionError, match="TP-averaged"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 7, "key": "global_loss", "value": 2.0}
        ))


def test_nan_scalar_is_rejected():
    with pytest.raises(AssertionError, match="Non-finite"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 5, "key": "global_loss_pre_tp", "value": float("nan")}
        ))
