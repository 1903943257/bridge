from types import SimpleNamespace

import pytest

from verl.models.mcore.tpr.megatron_adapter import (
    _validate_tpr_tp_cp_runtime,
    _validate_tpr_tp_cp_sizes,
)


@pytest.mark.parametrize("tp,cp,ppo", [
    (1, 1, True), (1, 2, True), (1, 4, True),
    (2, 1, True), (2, 2, True), (2, 4, True),
    (2, 2, False), (2, 4, False), (1, 8, False),
])
def test_supported_tp_cp_combinations(tp, cp, ppo):
    _validate_tpr_tp_cp_sizes(
        {"TP": tp, "CP": cp, "PP": 1, "EP": 1, "DP": 1},
        ppo=ppo,
    )


@pytest.mark.parametrize("tp,cp,dp,ppo", [
    (2, 8, 1, True), (2, 8, 1, False),
    (1, 8, 1, True), (2, 4, 2, True),
    (1, 2, 2, False), (4, 1, 1, True),
])
def test_unsupported_tp_cp_dp(tp, cp, dp, ppo):
    with pytest.raises(NotImplementedError):
        _validate_tpr_tp_cp_sizes(
            {"TP": tp, "CP": cp, "PP": 1, "EP": 1, "DP": dp},
            ppo=ppo,
        )


@pytest.mark.parametrize("tp,cp,ppo,backend", [
    (2, 2, True, "ring"), (2, 4, True, "ring"),
    (2, 4, False, "ring"), (1, 4, True, "ring"),
    (1, 4, False, "hybrid"), (2, 1, True, "none"),
])
def test_tp_cp_runtime_accepts_valid_backend(tp, cp, ppo, backend):
    _validate_tpr_tp_cp_runtime(
        tp, SimpleNamespace(size=cp, backend_name=backend),
        SimpleNamespace(sequence_parallel=False),
        ppo=ppo,
    )


@pytest.mark.parametrize("tp,cp,ppo,backend,sp", [
    (2, 4, True, "hybrid", False),
    (1, 4, True, "allgather", False),
    (2, 2, False, "allgather", False),
    (2, 4, True, "ring", True),
    (2, 2, False, "ring", True),
])
def test_tp_cp_runtime_rejects_untested_backend_or_sp(tp, cp, ppo, backend, sp):
    with pytest.raises(NotImplementedError):
        _validate_tpr_tp_cp_runtime(
            tp, SimpleNamespace(size=cp, backend_name=backend),
            SimpleNamespace(sequence_parallel=sp),
            ppo=ppo,
        )
