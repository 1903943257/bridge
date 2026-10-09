# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Larger CPU-only TPR DP placement workloads; no distributed/NPU training.

Run explicitly:
    pytest -vv -s tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py

Also run 512 trajectories with a 16K-token shared prefix:
    TPR_DTA_SCALE_STRESS=1 pytest -vv -s \
        tests/models/mcore/tpr/profiling/test_dp_placement_scale_cpu.py

These are deterministic synthetic *planning* workloads. The medium cases
also run our existing executable TPR tree builder to check estimated costs,
not a DTA runtime. No communication/optimizer claims.
"""

from __future__ import annotations

import os
import time

import pytest

from verl.models.mcore.tpr.dp_placement import (
    _uid_scoped_tree_cost,
    plan_tpr_dta_dp,
    plan_verl_uid,
)


def make_agentic_forest(
    *,
    n_groups: int,
    siblings_per_group: int,
    prefix_tokens: int,
    suffix_tokens: int,
) -> tuple[list[tuple[int, ...]], list[str]]:
    """Generate multi-UID, multi-level branching trajectories with long prefixes.

    Hierarchy: common stem -> group prompt -> 4-way branch -> 2-way branch ->
    per-leaf suffix.  Adjacent groups share an initial stem.  All rows are
    distinct, even when their prefix and suffix lengths match.
    """
    if min(n_groups, siblings_per_group, suffix_tokens) < 1 or prefix_tokens < 32:
        raise ValueError("invalid synthetic tree dimensions")
    common_stem = tuple(range(1, 17))
    seqs: list[tuple[int, ...]] = []
    uids: list[str] = []
    for group in range(n_groups):
        # Include group identity *after* the common stem; shared prompt length
        # is identical for every sibling within a group.
        prompt = (
            common_stem
            + (10_000 + group,)
            + tuple(100 + ((i * 7 + group * 11) % 8191) for i in range(prefix_tokens - 17))
        )
        for sibling in range(siblings_per_group):
            total_suffix = suffix_tokens + (sibling % 4) * (suffix_tokens // 8)
            # The split identities produce intermediate shared subtrees.
            first = (40_000 + group * 128 + sibling // 4,) * min(32, total_suffix // 4)
            second = (60_000 + group * 128 + sibling // 2,) * min(32, total_suffix // 4)
            remaining = total_suffix - len(first) - len(second)
            unique = (80_000 + group * 128 + sibling,) + tuple(
                100 + ((group * 19 + sibling * 17 + index * 3) % 8191)
                for index in range(remaining - 1)
            )
            seqs.append(prompt + first + second + unique)
            uids.append(f"rollout_{group:04d}")
    return seqs, uids


def _check_plan(plan, *, n_rows: int, n_ranks: int, total_raw_tokens: int):
    assert len(plan.partitions) == n_ranks
    assert all(plan.partitions)
    assert sorted(row for part in plan.partitions for row in part) == list(range(n_rows))
    assert plan.global_tree_tokens > 0
    assert 0 <= plan.duplicated_tree_tokens
    assert plan.global_tree_tokens <= total_raw_tokens
    assert sum(plan.tree_tokens_by_rank) == (
        plan.global_tree_tokens + plan.duplicated_tree_tokens
    )
    assert all(cost > 0 for cost in plan.tree_tokens_by_rank)


def _plan_timed(title, function):
    start = time.perf_counter()
    plan = function()
    seconds = time.perf_counter() - start
    print(
        f"TPR_DP_SCALE {title}: secs={seconds:.3f} "
        f"rows={[len(part) for part in plan.partitions]} "
        f"tree_costs={list(plan.tree_tokens_by_rank)} "
        f"max_tree_cost={plan.max_tree_tokens} "
        f"global_tree_tokens={plan.global_tree_tokens} "
        f"duplicated_tree_tokens={plan.duplicated_tree_tokens} "
        f"equal_rows={plan.equal_rows_per_rank}",
        flush=True,
    )
    return plan


@pytest.mark.parametrize(
    ("n_groups", "siblings", "prefix", "suffix", "dp"),
    [
        (16, 8, 2048, 256, 4),     # 128 rows, ~2K prefix, variable suffix
        (32, 8, 4096, 512, 8),     # 256 rows, ~4K prefix, variable suffix
    ],
)
def test_midscale_dta_vs_native_verl_placement(n_groups, siblings, prefix, suffix, dp):
    seqs, uids = make_agentic_forest(
        n_groups=n_groups,
        siblings_per_group=siblings,
        prefix_tokens=prefix,
        suffix_tokens=suffix,
    )
    n_rows = len(seqs)
    assert n_rows == n_groups * siblings
    raw_tokens = sum(map(len, seqs))

    native = _plan_timed(
        "verl_uid",
        lambda: plan_verl_uid(seqs, uids, dp),
    )
    keys = [f"{uid}_trace_{row}" for row, uid in enumerate(uids)]
    dta = _plan_timed(
        "tpr_dta_dp",
        lambda: plan_tpr_dta_dp(seqs, keys, dp, enforce_equal_rows=False),
    )
    for plan in (native, dta):
        _check_plan(plan, n_rows=n_rows, n_ranks=dp, total_raw_tokens=raw_tokens)
        assert plan.global_tree_tokens == _uid_scoped_tree_cost(seqs, uids)

    # No speed assertions: DTA/VERL planners optimize different objectives
    # and the fitted time model may value backward differently than tree cost.
    assert native.equal_rows_per_rank
    assert native.policy == "verl_uid"
    assert dta.policy == "tpr_dta"

    # Our execution builder currently constructs a per-token Python trie and
    # then compresses it.  Measure this independently: it can dominate CPU
    # planning memory/time for long shared prompts even when DTA DP is fast.
    import torch

    from verl.models.mcore.tpr.trajectory_tree import build_trajectory_trees

    input_rows = [torch.tensor(row, dtype=torch.long) for row in seqs]
    start = time.perf_counter()
    trees = build_trajectory_trees(keys, {"input_ids": input_rows})
    build_seconds = time.perf_counter() - start
    execution_tree_tokens = sum(
        node.segment.length for tree in trees for node in tree.nodes.values()
    )
    execution_nodes = sum(len(tree.nodes) for tree in trees)
    covered_rows = sorted(row for tree in trees for row in tree.member_rows)
    assert covered_rows == list(range(n_rows))
    assert execution_tree_tokens == dta.global_tree_tokens
    print(
        f"TPR_DP_SCALE executable_trie: secs={build_seconds:.3f} "
        f"trees={len(trees)} compact_nodes={execution_nodes} "
        f"execution_tree_tokens={execution_tree_tokens} "
        f"predicted_TPR_tree_tokens={dta.global_tree_tokens}",
        flush=True,
    )


def test_many_duplicates_and_prefix_contained_trajectories_survive_tpr_dp():
    base, base_uids = make_agentic_forest(
        n_groups=16,
        siblings_per_group=4,
        prefix_tokens=1024,
        suffix_tokens=256,
    )
    # 64 distinct leaves, 4 logical training samples per leaf, plus 16
    # terminal rows that are strict prefixes of existing trajectories.
    seqs = [row for row in base for _ in range(4)]
    uids = [uid for uid in base_uids for _ in range(4)]
    seqs.extend(row[:1024] for row in base[::4])
    uids.extend(base_uids[::4])
    assert len(seqs) == 272
    total_raw = sum(map(len, seqs))
    keys = [f"{uid}_trace_{row}" for row, uid in enumerate(uids)]
    dta = _plan_timed(
        "tpr_duplicates_and_contained",
        lambda: plan_tpr_dta_dp(seqs, keys, dp_size=4, enforce_equal_rows=False),
    )
    _check_plan(dta, n_rows=len(seqs), n_ranks=4, total_raw_tokens=total_raw)
    assert dta.global_tree_tokens == _uid_scoped_tree_cost(seqs, uids)


@pytest.mark.skipif(
    os.environ.get("TPR_DTA_SCALE_STRESS") != "1",
    reason="explicit opt-in: TPR_DTA_SCALE_STRESS=1 (may need >1 GiB CPU RAM)",
)
def test_long_prefix_16k_512_trajectories_8dp():
    seqs, uids = make_agentic_forest(
        n_groups=32,
        siblings_per_group=16,
        prefix_tokens=16384,
        suffix_tokens=1024,
    )
    n_rows = len(seqs)
    assert n_rows == 512
    total_raw = sum(map(len, seqs))
    keys = [f"{uid}_trace_{row}" for row, uid in enumerate(uids)]
    for name, fn in (
        ("verl_uid", lambda: plan_verl_uid(seqs, uids, 8)),
        ("tpr_dta_dp", lambda: plan_tpr_dta_dp(seqs, keys, 8, enforce_equal_rows=False)),
    ):
        plan = _plan_timed(f"stress_{name}", fn)
        _check_plan(plan, n_rows=n_rows, n_ranks=8, total_raw_tokens=total_raw)
        assert plan.global_tree_tokens == _uid_scoped_tree_cost(seqs, uids)
