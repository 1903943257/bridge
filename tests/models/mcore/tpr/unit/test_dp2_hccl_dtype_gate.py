"""CPU guard against invalid 910B FP64 HCCL all-reduce benchmark artifacts."""
from __future__ import annotations

import json

import pytest

from ..parallel.compare_tpr_tp2cp2dp2_one_step import compare


def test_stale_float64_full_baseline_is_not_compared_to_current_fp32_tpr(tmp_path):
    full = tmp_path / "full.json"
    tpr = tmp_path / "tpr.json"
    full.write_text(json.dumps({
        "mode": "full",
        "hccl_stats_reduce_dtype": "float64",
    }))
    tpr.write_text(json.dumps({
        "mode": "tpr",
        "hccl_stats_reduce_dtype": "float32",
        "topology": {"tp": 2, "cp": 2, "dp": 2, "world": 8},
    }))
    with pytest.raises(AssertionError, match="Stale Full/TPR benchmark"):
        compare(full, tpr)


def test_collective_benchmark_paths_explicitly_use_fp32_for_world_reductions():
    # Guard the root cause found on Ascend 910B (not an NPU runtime test).
    from pathlib import Path
    parallel = Path(__file__).resolve().parents[1] / "parallel"
    for name in (
        "test_tpr_tp2cp2dp2_one_step_npu.py",
        "test_tpr_fixed_shape_40k_benchmark_npu.py",
    ):
        source = (parallel / name).read_text()
        assert '"hccl_stats_reduce_dtype": "float32"' in source
        assert "dtype=torch.float32" in source
        assert "dist.ReduceOp.MAX" in source
    dp2 = (parallel / "test_tpr_tp2cp2dp2_one_step_npu.py").read_text()
    assert "TPR_DP2_WORLD_FP32_REDUCE_PASS" in dp2
    assert "global_loss = torch.tensor(" in dp2
    assert "local_loss, dtype=torch.float32" in dp2
