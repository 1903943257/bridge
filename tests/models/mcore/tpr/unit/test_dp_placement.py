# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU placement-contract tests; these do not enable distributed training."""

from itertools import combinations

import pytest

from verl.models.mcore.tpr.dp_placement import (
    _tree_token_cost,
    plan_dta_dfs,
    plan_verl_uid,
)


def _assert_exact_coverage(plan, n, k):
    assert len(plan.partitions) == k
    assert sorted(i for group in plan.partitions for i in group) == list(range(n))
    assert all(plan.partitions)


def test_dta_balances_shared_prefix_tree_cost_not_raw_token_sum():
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [9, 8, 7],
        [9, 8, 6],
    ]
    plan = plan_dta_dfs(sequences, 2)
    _assert_exact_coverage(plan, 4, 2)
    assert plan.equal_rows_per_rank
    assert plan.global_tree_tokens == 9
    assert plan.max_tree_tokens == 5
    assert plan.duplicated_tree_tokens == 0


def test_dta_uneven_partitions_are_offline_only():
    # The 20-token sample forces DTA's optimal 1-vs-3 assignment.
    sequences = [list(range(20)), [40, 41], [50, 51], [60, 61]]
    offline = plan_dta_dfs(sequences, 2, enforce_equal_rows=False)
    _assert_exact_coverage(offline, 4, 2)
    assert sorted(map(len, offline.partitions)) == [1, 3]
    assert not offline.equal_rows_per_rank
    assert offline.max_tree_tokens == 20
    with pytest.raises(ValueError, match="unequal per-DP row counts"):
        plan_dta_dfs(sequences, 2)


def test_dta_duplicate_trajectories_remain_distinct_rows():
    sequences = [[1, 2, 3]] * 4
    plan = plan_dta_dfs(sequences, 2)
    _assert_exact_coverage(plan, 4, 2)
    assert plan.global_tree_tokens == 3
    assert plan.tree_tokens_by_rank == (3, 3)
    assert plan.duplicated_tree_tokens == 3


def test_dta_minimax_matches_exhaustive_small_case():
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 9],
        [1, 2, 4],
        [6, 7],
        [6, 8, 9],
        [9, 0],
    ]
    plan = plan_dta_dfs(sequences, 3, enforce_equal_rows=False)
    ordered = sorted(map(tuple, sequences))
    brute_force = min(
        max(_tree_token_cost(ordered[a:b]) for a, b in zip((0, *cuts), (*cuts, len(ordered))))
        for cuts in combinations(range(1, len(ordered)), 2)
    )
    assert plan.max_tree_tokens == brute_force


def test_dta_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dp_size"):
        plan_dta_dfs([[1]], 0)
    with pytest.raises(ValueError, match=">="):
        plan_dta_dfs([[1]], 2)
    with pytest.raises(ValueError, match="empty"):
        plan_dta_dfs([[]], 1)
    with pytest.raises(TypeError, match="nonnegative"):
        plan_dta_dfs([[-1]], 1)


def test_native_uid_checks_contiguity_even_on_dp1():
    with pytest.raises(ValueError, match="contiguous"):
        plan_verl_uid([[1], [2], [3]], ["a", "b", "a"], 1)


def test_native_uid_dp1_preserves_every_row_without_importing_verl_balancer():
    plan = plan_verl_uid([[1, 2], [1, 3], [9]], ["a", "a", "b"], 1)
    assert plan.partitions == ((0, 1, 2),)
    assert plan.equal_rows_per_rank
    assert plan.policy == "verl_uid"
