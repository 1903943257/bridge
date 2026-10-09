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
def test_midscale_dta_vs_native_verl_placement(n_groups, siblings, prefix, suffix, dp, monkeypatch):
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

    from verl.models.mcore.tpr import trajectory_tree as old_tree_builder

    # Profile production implementation without changing its source. Timers
    # measure whole-row insertion, then final compression; the remainder is
    # validation / source-row tensor handling and builder overhead.
    subtimes = {"insert": 0.0, "compress": 0.0}
    for attribute, bucket in (
        ("_insert_sequence", "insert"),
        ("_compress_real_root", "compress"),
    ):
        original = getattr(old_tree_builder, attribute)

        def measured(*args, __fn=original, __bucket=bucket, **kwargs):
            started = time.perf_counter()
            try:
                return __fn(*args, **kwargs)
            finally:
                subtimes[__bucket] += time.perf_counter() - started

        monkeypatch.setattr(old_tree_builder, attribute, measured)

    build_trajectory_trees = old_tree_builder.build_trajectory_trees
    input_rows = [torch.tensor(row, dtype=torch.long) for row in seqs]
    start = time.perf_counter()
    trees = build_trajectory_trees(keys, {"input_ids": input_rows})
    build_seconds = time.perf_counter() - start
    print(
        f"TPR_DP_SCALE legacy_tree_breakdown: total_secs={build_seconds:.3f} "
        f"per_token_insert_secs={subtimes['insert']:.3f} "
        f"compress_secs={subtimes['compress']:.3f} "
        f"other_secs={max(0.0, build_seconds - sum(subtimes.values())):.3f}",
        flush=True,
    )

    from verl.models.mcore.tpr.trajectory_tree_radix import (
        build_trajectory_trees_radix,
    )

    start = time.perf_counter()
    radix_trees = build_trajectory_trees_radix(keys, {"input_ids": input_rows})
    radix_seconds = time.perf_counter() - start
    # Strong correctness gate: exact TPR public tree objects, node ids,
    # segment rows and spans, children and terminal ownership, not just counts.
    assert radix_trees == trees
    print(
        f"TPR_DP_SCALE tree_builder_compare: old_secs={build_seconds:.3f} "
        f"radix_secs={radix_seconds:.3f} "
        f"speedup={build_seconds / max(radix_seconds, 1e-9):.2f}x "
        f"rows={n_rows} prefix={prefix} compact_nodes="
        f"{sum(len(tree.nodes) for tree in trees)}",
        flush=True,
    )
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

    # Real DP2/DP8 executes the tree builder *after* sharding: each replica
    # builds only its local rows, not the entire controller batch. Report
    # max local-rank time, the relevant straggler estimate, separately from
    # the controller-sized global builder benchmark above.
    old_rank_secs, radix_rank_secs = [], []
    for rank, original_indices in enumerate(dta.partitions):
        local_batch = {
            "input_ids": [input_rows[i] for i in original_indices]
        }
        local_keys = [keys[i] for i in original_indices]

        t0 = time.perf_counter()
        old_rank_trees = build_trajectory_trees(local_keys, local_batch)
        old_rank_secs.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        radix_rank_trees = build_trajectory_trees_radix(local_keys, local_batch)
        radix_rank_secs.append(time.perf_counter() - t0)
        assert old_rank_trees == radix_rank_trees
        rank_tokens = sum(
            node.segment.length
            for tree in radix_rank_trees
            for node in tree.nodes.values()
        )
        assert rank_tokens == dta.tree_tokens_by_rank[rank]
    print(
        f"TPR_DP_SCALE rank_local_tree_compare: dp={dp} "
        f"max_old_secs={max(old_rank_secs):.3f} "
        f"max_radix_secs={max(radix_rank_secs):.3f} "
        f"max_time_speedup="
        f"{max(old_rank_secs) / max(max(radix_rank_secs), 1e-9):.2f}x "
        f"sum_old_secs={sum(old_rank_secs):.3f} "
        f"sum_radix_secs={sum(radix_rank_secs):.3f}",
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

    # Keep the legacy one-token-node constructor OFF in the 16K stress gate:
    # it may instantiate millions of temporary objects. Validate only the
    # candidate compressed builder against DP's estimated physical forest cost.
    import torch

    from verl.models.mcore.tpr.trajectory_tree_radix import (
        build_trajectory_trees_radix,
    )

    input_rows = [torch.tensor(row, dtype=torch.long) for row in seqs]
    started = time.perf_counter()
    fast_trees = build_trajectory_trees_radix(
        keys, {"input_ids": input_rows}
    )
    secs = time.perf_counter() - started
    token_cost = sum(
        node.segment.length
        for tree in fast_trees
        for node in tree.nodes.values()
    )
    assert token_cost == _uid_scoped_tree_cost(seqs, uids)
    assert sorted(row for tree in fast_trees for row in tree.member_rows) == list(range(n_rows))
    print(
        f"TPR_DP_SCALE stress_radix_builder: secs={secs:.3f} "
        f"trees={len(fast_trees)} compact_nodes="
        f"{sum(len(tree.nodes) for tree in fast_trees)} "
        f"execution_tree_tokens={token_cost}",
        flush=True,
    )
