# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prefix-aware DP placement *planning*, not distributed data dispatch.

DFS-contiguous minimax partitioning follows the AReaL-DTA algorithm:
https://github.com/areal-project/AReaL/blob/feat/dta/areal/experimental/dta/dp.py

This module intentionally does not touch Megatron process groups, split a
TensorDict, or override VERL's gradient/optimizer synchronization.  In
particular, VERL's current trainer / worker contract assumes an equal number
of examples on every DP replica.  DTA is an offline planning alternative until
a compatible variable-cardinality dispatch + mini-batch contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Policy = Literal["verl_uid", "dta_dfs", "areal_dta"]


@dataclass(frozen=True)
class DPPlacementPlan:
    """Original example indices assigned to each *logical DP replica*."""

    partitions: tuple[tuple[int, ...], ...]
    tree_tokens_by_rank: tuple[int, ...]
    global_tree_tokens: int
    duplicated_tree_tokens: int
    equal_rows_per_rank: bool
    policy: Policy

    @property
    def max_tree_tokens(self) -> int:
        return max(self.tree_tokens_by_rank, default=0)


def _sequences(token_sequences: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    result = []
    for row, sequence in enumerate(token_sequences):
        tokens = tuple(sequence)
        if not tokens:
            raise ValueError(f"sequence {row} is empty; pass unpadded real tokens")
        # Convert ordinary CPU tensors with .tolist() at the caller boundary.
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in tokens):
            raise TypeError(f"sequence {row} must contain nonnegative Python int token IDs")
        result.append(tokens)
    return tuple(result)


def _lcp(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def _tree_token_cost(sequences: Sequence[tuple[int, ...]]) -> int:
    if not sequences:
        return 0
    ordered = sorted(sequences)
    cost = len(ordered[0])
    for previous, current in zip(ordered, ordered[1:]):
        cost += len(current) - _lcp(previous, current)
    return cost


def _check_partitions(partitions: Sequence[Sequence[int]], n: int, dp_size: int) -> None:
    if len(partitions) != dp_size:
        raise ValueError(f"expected {dp_size} DP partitions, got {len(partitions)}")
    flat = [index for part in partitions for index in part]
    if any(not part for part in partitions):
        raise ValueError("each DP replica must receive at least one trajectory")
    if sorted(flat) != list(range(n)):
        raise ValueError("DP partitions must cover each input row exactly once")


def _result(
    sequences: tuple[tuple[int, ...], ...],
    partitions: Sequence[Sequence[int]],
    policy: Policy,
    dp_size: int,
    enforce_equal_rows: bool,
) -> DPPlacementPlan:
    _check_partitions(partitions, len(sequences), dp_size)
    equal = len({len(part) for part in partitions}) == 1
    if enforce_equal_rows and not equal:
        raise ValueError(
            "DTA produced unequal per-DP row counts; VERL's current batch "
            "dispatcher and TrainingWorker mini-batch contract require equal "
            "counts. Keep this plan offline; do not pass it to batch.reorder()."
        )
    costs = tuple(_tree_token_cost([sequences[i] for i in part]) for part in partitions)
    global_cost = _tree_token_cost(sequences)
    duplication = sum(costs) - global_cost
    if duplication < 0:
        raise AssertionError("DP partition cannot use fewer total unique tokens than the global trie")
    return DPPlacementPlan(
        partitions=tuple(tuple(part) for part in partitions),
        tree_tokens_by_rank=costs,
        global_tree_tokens=global_cost,
        duplicated_tree_tokens=duplication,
        equal_rows_per_rank=equal,
        policy=policy,
    )


def plan_dta_dfs(
    token_sequences: Sequence[Sequence[int]],
    dp_size: int,
    *,
    enforce_equal_rows: bool = True,
) -> DPPlacementPlan:
    """DTA DFS-contiguous minimax partition using unique trie-token cost.

    Duplicate trajectories remain distinct *logical rows*.  Global exact-token
    ordering is independent of prompt UID; each DP rebuilds its own local trie.
    No KV or dKV crosses DP ranks.  Set enforce_equal_rows=False ONLY to
    inspect/offline-evaluate DTA's variable-cardinality partition.
    """
    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    sequences = _sequences(token_sequences)
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")

    order = sorted(range(len(sequences)), key=lambda index: (sequences[index], index))
    lengths = [len(sequences[i]) for i in order]
    increment = [0] * len(order)
    for i in range(1, len(order)):
        increment[i] = lengths[i] - _lcp(sequences[order[i - 1]], sequences[order[i]])
    cumulative = [0]
    for value in increment:
        cumulative.append(cumulative[-1] + value)

    def cost(start: int, stop: int) -> int:
        """Cost of [start, stop) in sorted DFS order."""
        return lengths[start] + cumulative[stop] - cumulative[start + 1]

    def divide(max_cost: int) -> list[tuple[int, int]]:
        groups: list[tuple[int, int]] = []
        start = 0
        n = len(order)
        while start < n:
            left, right = start + 1, n
            while left < right:
                mid = (left + right + 1) // 2
                if cost(start, mid) <= max_cost:
                    left = mid
                else:
                    right = mid - 1
            groups.append((start, left))
            start = left
        return groups

    low, high = max(lengths), _tree_token_cost(sequences)
    while low < high:
        mid = (low + high) // 2
        if len(divide(mid)) <= dp_size:
            high = mid
        else:
            low = mid + 1
    groups = divide(low)
    # DTA's greedy feasibility scan deliberately fills each rank as far as it
    # can.  When many rows have identical token sequences, adding another row
    # costs zero tree tokens: a valid minimax solution may then contain 3/1
    # rows even though a 2/2 split has exactly the SAME optimal tree cost.
    #
    # For VERL's equal-cardinality dispatch, prefer equally sized contiguous
    # intervals *only if* they preserve the DTA minimax objective.  Do not
    # silently sacrifice load balance to force equal rows: when that is
    # impossible, keep the original unequal solution and let _result reject
    # it until variable-cardinality worker dispatch is supported.
    if enforce_equal_rows and len(order) % dp_size == 0:
        rows_per_rank = len(order) // dp_size
        equal_groups = [
            (rank * rows_per_rank, (rank + 1) * rows_per_rank)
            for rank in range(dp_size)
        ]
        if all(cost(start, end) <= low for start, end in equal_groups):
            groups = equal_groups

    # Monotone partition cost means splitting an interval cannot increase its
    # tree-token cost.  Split until all DP ranks have a non-empty interval.
    while len(groups) < dp_size:
        i = next((j for j in range(len(groups) - 1, -1, -1)
                  if groups[j][1] - groups[j][0] > 1), None)
        if i is None:
            raise AssertionError("could not construct non-empty DP partitions")
        start, end = groups[i]
        groups[i:i + 1] = [(start, end - 1), (end - 1, end)]
    partitions = [[order[i] for i in range(start, end)] for start, end in groups]
    return _result(sequences, partitions, "dta_dfs", dp_size, enforce_equal_rows)



class _TreeTokenTimeModel:
    """AReaL TimeModel protocol: use original tree-token metric, no scipy fit."""

    def pred(self, stats: dict) -> float:
        return float(stats["n_tree_tokens"])


def plan_areal_dta(
    token_sequences: Sequence[Sequence[int]],
    dp_size: int,
    *,
    mode: Literal["forward", "backward"] = "backward",
    block_size: int | None = None,
    time_model: object | None = None,
    enforce_equal_rows: bool = True,
) -> DPPlacementPlan:
    """Run **upstream AReaL-DTA's** LB_by_DFS_and_TM, with VERL guards.

    The upstream solver and trie are vendored with namespace-only changes in
    _vendor/areal_dta; no local rewrite of its DFS partition algorithm.
    Unlike plan_dta_dfs, upstream leafization merges duplicate or contained
    trajectories and can return uneven or empty DP bins.  Reject such cases
    safely rather than silently breaking VERL's equal-shard data dispatcher.

    For a fidelity/performance comparison, set enforce_equal_rows=False to
    inspect a non-empty *uneven* offline partition; do not dispatch that plan.
    Custom AReaL-compatible time models only need a .pred(stats) method.
    """
    from types import SimpleNamespace

    import torch

    from ._vendor.areal_dta.dp import LB_by_DFS_and_TM
    from ._vendor.areal_dta.token_trie import TokenTrie

    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    if mode not in ("forward", "backward"):
        raise ValueError("mode must be forward or backward")
    if block_size is not None and (type(block_size) is not int or block_size <= 0):
        raise ValueError("block_size must be a positive integer or None")
    sequences = _sequences(token_sequences)
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")
    predictor = _TreeTokenTimeModel() if time_model is None else time_model
    if not callable(getattr(predictor, "pred", None)):
        raise TypeError("time_model must expose a callable .pred(stats) method")
    tokens = [torch.tensor(seq, dtype=torch.long, device="cpu") for seq in sequences]
    if len(TokenTrie(tokens).inputs) < dp_size:
        raise ValueError(
            "AReaL-DTA leafization yields fewer independent trie leaves than "
            "DP replicas; use plan_dta_dfs for duplicate/contained trajectories "
            "or reduce dp_size"
        )
    bins = LB_by_DFS_and_TM(
        tokens,
        predictor,
        SimpleNamespace(K=dp_size, mode=mode, block_size=block_size),
    )
    return _result(sequences, bins, "areal_dta", dp_size, enforce_equal_rows)

def plan_verl_uid(
    token_sequences: Sequence[Sequence[int]],
    uid_list: Sequence[str],
    dp_size: int,
) -> DPPlacementPlan:
    """Delegate initial equal-cardinality UID placement to upstream VERL.

    This is the *safe* integration baseline.  It does not use DTA's cost model;
    it preserves existing VERL Karmarkar-Karp group balancing instead.
    """
    if type(dp_size) is not int or dp_size <= 0:
        raise ValueError("dp_size must be a positive integer")
    sequences = _sequences(token_sequences)
    if len(uid_list) != len(sequences):
        raise ValueError("uid_list and trajectories must have the same row count")
    if len(sequences) < dp_size:
        raise ValueError("number of trajectories must be >= dp_size")

    # Upstream get_group_balanced_partitions assumes each UID is one contiguous
    # group.  Do not silently turn repeated non-contiguous UIDs into new groups.
    seen = set()
    last_uid = object()
    for uid in uid_list:
        if uid != last_uid:
            if uid in seen:
                raise ValueError("VERL UID balancing requires contiguous rows for each UID")
            seen.add(uid)
            last_uid = uid

    if dp_size == 1:
        partitions = [list(range(len(sequences)))]
    else:
        from verl.utils.seqlen_balancing import get_group_balanced_partitions

        partitions = get_group_balanced_partitions(
            seqlen_list=[len(seq) for seq in sequences],
            uid_list=list(uid_list),
            k_partitions=dp_size,
        )
    return _result(sequences, partitions, "verl_uid", dp_size, True)
