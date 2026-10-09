# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""CPU placement-contract tests; these do not enable distributed training."""

from itertools import combinations

import pytest

from verl.models.mcore.tpr.dp_placement import (
    _tree_token_cost,
    plan_dta_dfs,
    plan_areal_dta,
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
    assert tuple(map(len, plan.partitions)) == (2, 2)
    assert plan.equal_rows_per_rank
    assert plan.global_tree_tokens == 3
    assert plan.tree_tokens_by_rank == (3, 3)
    assert plan.duplicated_tree_tokens == 3



def test_dta_duplicate_ties_choose_equal_rows_for_three_replicas():
    sequences = [[11, 22, 33]] * 6
    plan = plan_dta_dfs(sequences, 3)
    _assert_exact_coverage(plan, 6, 3)
    assert tuple(map(len, plan.partitions)) == (2, 2, 2)
    assert plan.tree_tokens_by_rank == (3, 3, 3)
    assert plan.duplicated_tree_tokens == 6

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


def test_native_uid_delegates_to_verl_balancer(monkeypatch):
    from verl.utils import seqlen_balancing

    captured = {}

    def fake_partition(*, seqlen_list, uid_list, k_partitions):
        captured.update(seqlen_list=seqlen_list, uid_list=uid_list, k=k_partitions)
        return [[0, 1], [2, 3]]

    monkeypatch.setattr(seqlen_balancing, "get_group_balanced_partitions", fake_partition)
    plan = plan_verl_uid([[1, 2], [1, 3], [9, 8], [9, 7]], ["a", "a", "b", "b"], 2)
    assert plan.partitions == ((0, 1), (2, 3))
    assert captured == {
        "seqlen_list": [2, 2, 2, 2],
        "uid_list": ["a", "a", "b", "b"],
        "k": 2,
    }


def test_native_uid_rejects_unequal_rows_from_upstream(monkeypatch):
    from verl.utils import seqlen_balancing

    monkeypatch.setattr(
        seqlen_balancing, "get_group_balanced_partitions",
        lambda **kw: [[0], [1, 2, 3]],
    )
    with pytest.raises(ValueError, match="unequal per-DP row counts"):
        plan_verl_uid([[1], [2], [3], [4]], ["a", "a", "b", "b"], 2)


def test_areal_dp1_keeps_all_original_rows():
    plan = plan_areal_dta(
        [[1, 2, 3], [1, 2, 4], [9, 8], [9, 7, 6]],
        dp_size=1,
    )
    _assert_exact_coverage(plan, 4, 1)
    assert plan.partitions == ((0, 1, 2, 3),)
    assert plan.policy == "areal_dta"
    assert plan.equal_rows_per_rank


def test_areal_direct_upstream_parity_on_distinct_leaves():
    from types import SimpleNamespace

    import torch

    from verl.models.mcore.tpr._vendor.areal_dta.dp import LB_by_DFS_and_TM
    from verl.models.mcore.tpr.dp_placement import _TreeTokenTimeModel

    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [9, 8, 7],
        [9, 8, 6],
    ]
    original = LB_by_DFS_and_TM(
        [torch.tensor(row, dtype=torch.long) for row in sequences],
        _TreeTokenTimeModel(),
        SimpleNamespace(K=2, mode="backward", block_size=None),
    )
    plan = plan_areal_dta(sequences, 2, enforce_equal_rows=False)
    _assert_exact_coverage(plan, 4, 2)
    assert plan.partitions == tuple(tuple(part) for part in original)
    assert plan.policy == "areal_dta"


def test_areal_rejects_fewer_unique_leaves_than_dp_replicas():
    with pytest.raises(ValueError, match="fewer independent trie leaves"):
        plan_areal_dta([[1, 2, 3]] * 4, dp_size=2)
    # Upstream merges a prefix that ends before another sequence, too.
    with pytest.raises(ValueError, match="fewer independent trie leaves"):
        plan_areal_dta([[1, 2], [1, 2, 3], [1, 2, 3]], dp_size=2)


def test_areal_strict_equal_cardinality_guard(monkeypatch):
    from verl.models.mcore.tpr._vendor.areal_dta import dp

    monkeypatch.setattr(dp, "LB_by_DFS_and_TM", lambda *a: [[0], [1, 2, 3]])
    sequences = [[1], [2], [3], [4]]
    with pytest.raises(ValueError, match="unequal per-DP row counts"):
        plan_areal_dta(sequences, dp_size=2)
    offline = plan_areal_dta(sequences, dp_size=2, enforce_equal_rows=False)
    assert offline.partitions == ((0,), (1, 2, 3))
    assert not offline.equal_rows_per_rank


def test_areal_rejects_invalid_time_model():
    with pytest.raises(TypeError, match="time_model"):
        plan_areal_dta([[1], [2]], 2, time_model=object())
