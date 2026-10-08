# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Mini-batch/forest scope for sequential TPR tree schedules.

This module deliberately owns neither the PPO objective nor the optimizer.
The supplied per-tree callable can use the existing FixedTopologyScheduler
and SegmentExecutor; no_sync spans the entire forest, and gradient
finalization happens ONCE after all tree schedules finish successfully.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .tree_plan_builder import ForestExecutionPlan, TreeExecutionPlan


@dataclass(frozen=True, slots=True)
class TPRBatchResult:
    tree_results: tuple[Any, ...]
    normalized_loss: Tensor
    tree_count: int
    segment_count: int
    logical_loss_tokens: int


class TPRBatchRunner:
    """Execute a complete PPO mini-batch as a forest under one grad-sync scope.

    run_tree(plan) must execute a full single-tree F/B schedule and return an
    object with a scalar normalized_loss attribute. Its loss values must
    already use the SAME original mini-batch global denominator. The runner
    never normalizes again, and must not finalize per tree.
    """

    def __init__(
        self,
        *,
        run_tree: Callable[[TreeExecutionPlan], Any],
        finalize_gradients: Callable[[], None],
        no_sync_context: Callable[[], Any] | None = None,
    ) -> None:
        if not callable(run_tree) or not callable(finalize_gradients):
            raise TypeError("run_tree and finalize_gradients must be callable")
        self.run_tree = run_tree
        self.finalize_gradients = finalize_gradients
        self.no_sync_context = no_sync_context
        self._ran = False

    def run(self, forest: ForestExecutionPlan) -> TPRBatchResult:
        if self._ran:
            raise RuntimeError("TPRBatchRunner is single-use for one optimizer mini-batch")
        self._ran = True
        if not isinstance(forest, ForestExecutionPlan) or not forest.trees:
            raise ValueError("forest must contain at least one executable tree")

        results = []
        accumulated_loss: Tensor | None = None
        context = nullcontext() if self.no_sync_context is None else self.no_sync_context()
        with context:
            for tree_plan in forest.trees:
                output = self.run_tree(tree_plan)
                loss = getattr(output, "normalized_loss", None)
                if not isinstance(loss, Tensor) or loss.numel() != 1:
                    raise TypeError("run_tree must return an object with scalar Tensor normalized_loss")
                results.append(output)
                accumulated_loss = loss.detach() if accumulated_loss is None else accumulated_loss + loss.detach()

        # No finalization if any tree failed or produced an invalid loss.
        self.finalize_gradients()
        assert accumulated_loss is not None
        return TPRBatchResult(
            tree_results=tuple(results),
            normalized_loss=accumulated_loss,
            tree_count=len(forest.trees),
            segment_count=forest.segment_count,
            logical_loss_tokens=forest.logical_loss_tokens,
        )
