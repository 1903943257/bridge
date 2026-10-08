# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Consume TPR physical logits using VERL's existing PPO loss function.

The adapter only maps (segment query, logical sample) to token log-probabilities
and builds a compact *logical* TensorDict. It does not implement PPO, KL,
clipping, or objective normalization. This path currently supports additive
vanilla PPO with token-mean aggregation, not sequence-level objectives.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from .segment_plan import SegmentSpec
from .tree_plan_builder import SegmentObjectiveRef, TreeExecutionPlan


def _rows(batch: Any, key: str) -> list[Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing PPO field {key!r}") from exc
    values = list(value) if isinstance(value, (tuple, list)) else list(value.unbind())
    if not all(isinstance(row, Tensor) and row.ndim == 1 for row in values):
        raise ValueError(f"PPO field {key!r} must have 1-D rows")
    return values


def _native_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    return vocab_parallel_log_probs_from_logits(logits, labels)


def _native_entropy(logits: Tensor) -> Tensor:
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy

    return vocab_parallel_entropy(logits)


def _metadata(batch: Any, key: str, default=None):
    from verl.utils import tensordict_utils as tu

    return tu.get_non_tensor_data(batch, key=key, default=default)


class SegmentPPOObjectiveAdapter:
    """Bridge SegmentExecutor logits to an unmodified VERL PPO loss callable.

    Important: the fake one-token 'prompts' below are bookkeeping only for
    VERL's no_padding_2_padding response-shifting function. They are never
    fed to a model. This ensures the packed new_log_probs are shifted into
    the compact response layout exactly as in native forward_step.
    """

    def __init__(
        self,
        batch: Any,
        loss_function: Callable,
        *,
        log_prob_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        entropy_fn: Callable[[Tensor], Tensor] | None = None,
        calculate_entropy: bool | None = None,
        dp_group: Any = None,
    ) -> None:
        if loss_function is None:
            raise ValueError("PPO objective requires a VERL loss_function")
        cfg = getattr(loss_function, "keywords", {}).get("config")
        if cfg is not None:
            mode = cfg.policy_loss.get("loss_mode", "vanilla")
            if mode != "vanilla" or cfg.loss_agg_mode != "token-mean":
                raise NotImplementedError(
                    "segment-local PPO currently requires vanilla policy loss and token-mean "
                    f"aggregation, got {mode!r} and {cfg.loss_agg_mode!r}"
                )
        self.batch = batch
        self.loss_function = loss_function
        self.log_prob_fn = log_prob_fn or _native_log_probs
        self.entropy_fn = entropy_fn or _native_entropy
        self.calculate_entropy = (
            bool(_metadata(batch, "calculate_entropy", default=False))
            if calculate_entropy is None else bool(calculate_entropy)
        )
        self.dp_group = dp_group

    def bind_tree(self, tree_plan: TreeExecutionPlan):
        """Bind an executable tree to SegmentExecutor's loss hook.

        Return (loss_callback, per_segment_ref_counts). A single adapter can
        serve every tree of the original PPO mini-batch; each call retains the
        original batch-level normalization metadata.
        """
        if not isinstance(tree_plan, TreeExecutionPlan):
            raise TypeError("tree_plan must be a TreeExecutionPlan")
        refs_by_segment: dict[int, list[SegmentObjectiveRef]] = {
            segment_id: [] for segment_id in tree_plan.segment_plan.segments
        }
        for ref in tree_plan.objective_refs:
            refs_by_segment[ref.segment_id].append(ref)

        counts = {segment_id: len(refs) for segment_id, refs in refs_by_segment.items()}
        if not sum(counts.values()):
            raise ValueError("tree has no supervised logical response tokens")

        def segment_loss_fn(segment: SegmentSpec, logits: Tensor):
            return self.compute_loss(segment, logits, refs_by_segment[segment.segment_id])

        return segment_loss_fn, counts

    def _temperatures(self, refs: Sequence[SegmentObjectiveRef]) -> list[float]:
        temperature = self.batch.get("temperature", 1.0)
        if isinstance(temperature, Tensor):
            if temperature.numel() == 1:
                values = [float(temperature.item())] * len(refs)
            else:
                flat = temperature.flatten()
                values = [float(flat[ref.sample_row].item()) for ref in refs]
        elif isinstance(temperature, (float, int)):
            values = [float(temperature)] * len(refs)
        else:
            raise TypeError(f"unsupported per-sample temperature type: {type(temperature)!r}")
        if any(not 0 < value < float("inf") for value in values):
            raise ValueError("temperature must be finite and positive")
        return values

    def compute_loss(
        self,
        segment: SegmentSpec,
        logits: Tensor,
        refs: Sequence[SegmentObjectiveRef],
    ) -> tuple[Tensor, dict]:
        """Return a native-normalized scalar and its segment-local metrics.

        Caller performs backward, prefix-state gradient relay and finalization.
        Empty refs are intentionally rejected: no-loss segments should use the
        executor's connected zero/relay path instead of invoking PPO.
        """
        refs = tuple(refs)
        if not refs:
            raise ValueError("no objective refs: use a zero-gradient relay path for this segment")
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != segment.length:
            raise ValueError("expected unsharded physical logits [1, segment_length, local_vocab]")
        if _metadata(self.batch, "batch_num_tokens", default=None) is None:
            raise ValueError("PPO global batch_num_tokens is missing; compute it before TPR routing")
        temperatures = self._temperatures(refs)

        # Reuse the exact physical (query,target,temperature) log-prob compute.
        # Crucially, repeated *logical* refs are not discarded from the loss.
        unique: dict[tuple[int, int, float], int] = {}
        keys: list[tuple[int, int, float]] = []
        gather_ids: list[int] = []
        for ref, temperature in zip(refs, temperatures, strict=True):
            if ref.segment_id != segment.segment_id:
                raise ValueError("objective ref belongs to another segment")
            if not 0 <= ref.query_offset < segment.length:
                raise ValueError("objective ref query_offset is outside segment")
            key = (ref.query_offset, ref.target_token_id, temperature)
            if key not in unique:
                unique[key] = len(keys)
                keys.append(key)
            gather_ids.append(unique[key])

        positions = torch.tensor([key[0] for key in keys], device=logits.device, dtype=torch.long)
        targets = torch.tensor([key[1] for key in keys], device=logits.device, dtype=torch.long)
        temps = torch.tensor([key[2] for key in keys], device=logits.device, dtype=torch.float32)
        selected_logits = logits[0].index_select(0, positions)
        scaled_logits = selected_logits / temps.to(selected_logits.dtype)[:, None]
        unique_log_probs = self.log_prob_fn(scaled_logits, targets)
        if unique_log_probs.ndim != 1 or unique_log_probs.numel() != len(keys):
            raise ValueError("log_prob_fn must return one scalar per unique query/target/temperature")
        gather = torch.tensor(gather_ids, device=logits.device, dtype=torch.long)
        log_probs = unique_log_probs.index_select(0, gather)

        entropy = None
        if self.calculate_entropy:
            unique_entropy = self.entropy_fn(scaled_logits)
            if unique_entropy.ndim != 1 or unique_entropy.numel() != len(keys):
                raise ValueError("entropy_fn must return one value per unique query")
            entropy = unique_entropy.index_select(0, gather)

        # Construct minimal logical microbatch in original sample order.
        # Each local row gets a 1-token pseudo prompt, followed by its refs.
        # Packed model output is [lp0, ..., lp(R-1), unused_dummy] per row;
        # no_padding_2_padding extracts the first R entries for prompt_len=1.
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu

        by_row: dict[int, list[int]] = defaultdict(list)
        for i, ref in enumerate(refs):
            by_row[ref.sample_row].append(i)
        rows = sorted(by_row)
        for row in rows:
            by_row[row].sort(key=lambda i: refs[i].response_offset)
            offsets = [refs[i].response_offset for i in by_row[row]]
            if len(set(offsets)) != len(offsets):
                raise ValueError(f"duplicate logical token ownership on row {row}")

        old_rows = _rows(self.batch, "old_log_probs")
        advantage_rows = _rows(self.batch, "advantages")
        response_rows = _rows(self.batch, "response_mask")
        optional_fields = ("ref_log_prob", "rollout_is_weights")
        optional_rows = {key: _rows(self.batch, key) for key in optional_fields if key in self.batch}
        n = len(rows)
        max_r = max(len(by_row[row]) for row in rows)
        dev = logits.device
        output_fields: dict[str, Tensor] = {
            "prompts": torch.zeros((n, 1), dtype=torch.long, device=dev),
            "responses": torch.zeros((n, max_r), dtype=torch.long, device=dev),
            "attention_mask": torch.zeros((n, 1 + max_r), dtype=torch.long, device=dev),
            "response_mask": torch.zeros((n, max_r), dtype=torch.bool, device=dev),
            "old_log_probs": torch.zeros((n, max_r), dtype=log_probs.dtype, device=dev),
            "advantages": torch.zeros((n, max_r), dtype=log_probs.dtype, device=dev),
        }
        for name in optional_rows:
            output_fields[name] = torch.zeros((n, max_r), dtype=log_probs.dtype, device=dev)

        packed_lp: list[Tensor] = []
        packed_entropy: list[Tensor] = []
        for local_row, row in enumerate(rows):
            selected = by_row[row]
            count = len(selected)
            offsets = [refs[i].response_offset for i in selected]
            if any(i < 0 or i >= response_rows[row].numel() for i in offsets):
                raise ValueError(f"response_offset out of bounds for row {row}")
            if not bool(response_rows[row][offsets].to(bool).all()):
                raise ValueError(f"objective ref points to a masked token on row {row}")
            idx = torch.tensor(selected, dtype=torch.long, device=dev)
            lp_row = log_probs.index_select(0, idx)
            output_fields["attention_mask"][local_row, :count+1] = 1
            output_fields["response_mask"][local_row, :count] = True
            output_fields["responses"][local_row, :count] = torch.tensor(
                [refs[i].target_token_id for i in selected], device=dev
            )
            indices = torch.tensor(offsets, dtype=torch.long, device=old_rows[row].device)
            output_fields["old_log_probs"][local_row, :count] = old_rows[row].index_select(0, indices).to(dev)
            output_fields["advantages"][local_row, :count] = advantage_rows[row].index_select(0, indices).to(dev)
            for key, field_rows in optional_rows.items():
                output_fields[key][local_row, :count] = field_rows[row].index_select(0, indices).to(dev)
            packed_lp.extend((lp_row, lp_row.new_zeros(1)))
            if entropy is not None:
                e_row = entropy.index_select(0, idx)
                packed_entropy.extend((e_row, e_row.new_zeros(1)))

        compact = TensorDict(output_fields, batch_size=[n], device=dev)
        tu.assign_non_tensor(
            compact,
            dp_size=_metadata(self.batch, "dp_size", 1),
            batch_num_tokens=_metadata(self.batch, "batch_num_tokens"),
            global_batch_size=_metadata(self.batch, "global_batch_size", None),
        )
        model_output = {"log_probs": torch.cat(packed_lp, dim=0)}
        if entropy is not None:
            model_output["entropy"] = torch.cat(packed_entropy, dim=0)
        loss, metrics = self.loss_function(
            model_output=model_output,
            data=compact,
            dp_group=self.dp_group,
        )
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise TypeError("native VERL loss_function must return a scalar Tensor")
        return loss, metrics
