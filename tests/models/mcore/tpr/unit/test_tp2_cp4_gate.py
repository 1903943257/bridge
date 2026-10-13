from types import SimpleNamespace
import pytest

from verl.models.mcore.tpr.megatron_adapter import (
    _validate_tpr_parallel_sizes,
    _validate_tpr_cp_combination,
)

@pytest.mark.parametrize("tp,cp", [(1,1),(1,2),(2,1),(2,2),(2,4)])
def test_accepted_tp_ringcp_dp1(tp, cp):
    for ppo in (False, True):
        assert _validate_tpr_parallel_sizes(
            {"TP":tp,"CP":cp,"PP":1,"EP":1,"DP":1}, ppo=ppo
        ) == tp

@pytest.mark.parametrize("tp,cp,dp", [(2,8,1),(2,4,2),(1,8,1),(4,1,1)])
def test_rejected_unvalidated_topologies(tp, cp, dp):
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(
            {"TP":tp,"CP":cp,"PP":1,"EP":1,"DP":dp}, ppo=True
        )

def test_ring_backend_and_sp_fail_closed():
    for backend, sp in (("allgather", False), ("hybrid", False), ("ring", True)):
        with pytest.raises(NotImplementedError):
            _validate_tpr_cp_combination(
                2, SimpleNamespace(size=4, backend_name=backend),
                SimpleNamespace(sequence_parallel=sp), ppo=True,
            )
    _validate_tpr_cp_combination(
        2, SimpleNamespace(size=4, backend_name="ring"),
        SimpleNamespace(sequence_parallel=False), ppo=True,
    )

def test_existing_tp1_non_ring_thin_entry_retained():
    assert _validate_tpr_parallel_sizes(
        {"TP":1,"CP":8,"PP":1,"EP":1,"DP":1}, ppo=False
    ) == 1
    _validate_tpr_cp_combination(
        1, SimpleNamespace(size=4, backend_name="hybrid"),
        SimpleNamespace(sequence_parallel=False), ppo=False,
    )

def test_tp1_cp4_ppo_not_silently_enabled():
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(
            {"TP":1,"CP":4,"PP":1,"EP":1,"DP":1}, ppo=True
        )
