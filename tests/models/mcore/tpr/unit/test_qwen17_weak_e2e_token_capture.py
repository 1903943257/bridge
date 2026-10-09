"""CPU-only PPO logical-token comparison and clip branch checks."""
from __future__ import annotations

import pytest
import torch

from ..correctness._qwen17_weak_e2e_token_capture import compare_ppo_tokens


def _snapshot(new, old, advantages):
    return {
        "logical_row_offset": torch.tensor([[0, 0], [0, 1], [1, 0]]),
        "new": torch.tensor(new, dtype=torch.float32),
        "old": torch.tensor(old, dtype=torch.float32),
        "advantage": torch.tensor(advantages, dtype=torch.float32),
    }


def test_ppo_clip_branch_disagreement_with_real_log_ratio():
    old = [0.0, 0.0, 0.0]
    advantage = [1.0, -1.0, 1.0]
    # Positive A + log(1.4) => clipped; negative A + log(.6)
    # => clipped; third token remains unclipped.
    n = _snapshot([0., 0., 0.], old, advantage)
    t = _snapshot([0.3364722366, -0.5108256237, 0.], old, advantage)
    result = compare_ppo_tokens(n, t)
    assert result["logical_tokens"] == 3
    assert result["clip_native"] == 0
    assert result["clip_tpr"] == 2
    assert result["clip_disagreement"] == 2
    assert result["worst_row_offset"] == (0, 1)


def test_ppo_diagnostics_refuse_mismatched_old_policy():
    n = _snapshot([0., 0., 0.], [0., 0., 0.], [1., -1., 1.])
    t = _snapshot([0., 0., 0.], [0., 0.01, 0.], [1., -1., 1.])
    with pytest.raises(ValueError, match="old policy"):
        compare_ppo_tokens(n, t)


def test_ppo_diagnostics_refuse_mismatched_logical_keys():
    n = _snapshot([0., 0., 0.], [0., 0., 0.], [1., -1., 1.])
    t = _snapshot([0., 0., 0.], [0., 0., 0.], [1., -1., 1.])
    t["logical_row_offset"][0] = torch.tensor([7, 7])
    with pytest.raises(ValueError, match="identities"):
        compare_ppo_tokens(n, t)


def test_threeway_cutoff_oracle_identifies_native_shape_drift():
    from ..correctness._qwen17_weak_e2e_token_capture import (
        compare_ppo_threeway,
    )

    n = _snapshot([0., 0., 0.], [0., 0., 0.], [1., -1., 1.])
    c = _snapshot([0.1, 0., -0.1], [0., 0., 0.], [1., -1., 1.])
    t = _snapshot([0.1, 0.5, -0.1], [0., 0., 0.], [1., -1., 1.])
    owners = {
        "logical_row_offset": n["logical_row_offset"].clone(),
        "segment_id_start_end": torch.tensor(
            [[3, 1, 20], [3, 1, 20], [5, 20, 40]]
        ),
    }
    result = "\n".join(compare_ppo_threeway(n, c, t, owners))
    assert "NATIVE_FULL_TO_NATIVE_CUTOFF" in result
    assert "NATIVE_CUTOFF_TO_TPR" in result
    assert "NATIVE_FULL_TO_TPR" in result
    assert "segment=3[1:20]" in result
    assert "segment=5[20:40]" in result
    assert "row=0 response=1" in result


def test_threeway_refuses_mismatched_owners():
    from ..correctness._qwen17_weak_e2e_token_capture import (
        compare_ppo_threeway,
    )

    n = _snapshot([0., 0., 0.], [0., 0., 0.], [1., -1., 1.])
    owners = {
        "logical_row_offset": n["logical_row_offset"] + 1,
        "segment_id_start_end": torch.tensor(
            [[3, 1, 20], [3, 1, 20], [5, 20, 40]]
        ),
    }
    with pytest.raises(ValueError, match="identities"):
        compare_ppo_threeway(n, n, n, owners)
