"""CPU-only gates for TP2 CP2 DP2 PPO loss reporting.

Different DP replicas own different forests; *local* PPO loss must not
be mistaken for replicated global loss.
"""

import pytest

from ..parallel.tpr_dp2_loss_consistency import analyze_dp2_loss_report


def _ranks(dp0=0.006869133561849594, dp1=0.1446971446275711,
           tp1_offset=0.0, *, corrupt=None):
    rows = []
    expected = (dp0 + dp1) / 2 + tp1_offset / 2
    for dp in range(2):
        for cp in range(2):
            for tp in range(2):
                rows.append({
                    "rank": len(rows), "tp_rank": tp, "cp_rank": cp,
                    "dp_rank": dp,
                    "local_loss_dp_scaled": (dp0 if dp == 0 else dp1) +
                                           (tp1_offset if tp == 1 else 0),
                    "global_loss": expected,
                })
    if corrupt is not None:
        rows[corrupt["row"]][corrupt["key"]] = corrupt["value"]
    return rows


def test_distinct_dp_local_forest_losses_are_expected_not_a_replica_mismatch():
    result = analyze_dp2_loss_report(_ranks())
    assert result["dp0_local_loss"] == pytest.approx(0.006869133561849594)
    assert result["dp1_local_loss"] == pytest.approx(0.1446971446275711)
    assert result["dp_local_contribution_gap"] == pytest.approx(0.1378280110657215)
    assert result["cp_replica_max_gap"] == 0
    assert result["global_reporting_max_gap"] == 0
    assert result["global_reporting_loss"] == pytest.approx(
        (0.006869133561849594 + 0.1446971446275711)/2
    )


def test_tp_shard_drift_is_reported_without_misclassifying_dp_partition():
    result = analyze_dp2_loss_report(_ranks(tp1_offset=0.00025))
    assert result["tp_shard_loss_gap"] == pytest.approx(0.00025)
    assert result["tp_local_max_gap"] == pytest.approx(0.00025)
    assert result["global_reporting_max_gap"] == 0


def test_cp_replica_mismatch_remains_hard_error():
    with pytest.raises(AssertionError, match="CP replicas"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 2, "key": "local_loss_dp_scaled", "value": 0.5}
        ))


def test_wrong_tp_cp_dp_coordinates_are_rejected():
    with pytest.raises(AssertionError, match="Invalid or repeated"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 1, "key": "dp_rank", "value": 1}
        ))


def test_wrong_global_reporting_collective_is_rejected():
    with pytest.raises(AssertionError, match="World-reduced"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 7, "key": "global_loss", "value": 2.0}
        ))


def test_nan_scalar_is_rejected():
    with pytest.raises(AssertionError, match="Non-finite"):
        analyze_dp2_loss_report(_ranks(
            corrupt={"row": 5, "key": "local_loss_dp_scaled", "value": float("nan")}
        ))
