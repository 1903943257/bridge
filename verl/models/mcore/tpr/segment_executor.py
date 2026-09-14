# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Push/pop execution of one TPR segment against a Megatron GPT model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .context import KVPair, TPRAttentionContext, use_tpr_attention_context
from .kv_stack import KVStack
from .parallel.backend import TPRCPBackend, resolve_tpr_cp_backend
from .parallel.execution_context import (
    ShardedPastKVAnchors,
    accumulate_sharded_past_anchor_gradients,
    build_sharded_past_anchors,
    resolve_cp_group,
)
from .prefix_state import GDNLayerState, GDNPrefixAnchors, GDNPrefixState
from .rope import (
    build_sharded_rotary_pos_emb,
    disable_bound_context_parallel_sharding,
)
from .segment_plan import SegmentId, SegmentLossTerm, SegmentPlan, SegmentSpec
from .shard import PrefixShard, SequenceShard


@dataclass(frozen=True, slots=True)
class SegmentForwardResult:
    segment_id: SegmentId
    prefix_length: int
    suffix_length: int
    layer_count: int


@dataclass(frozen=True, slots=True)
class SegmentBackwardResult:
    segment_id: SegmentId
    loss_sum: Tensor
    normalized_loss: Tensor
    loss_term_count: int
    relayed_layer_count: int


@dataclass(frozen=True, slots=True)
class LeafVisitResult:
    """One physical leaf execution replacing an adjacent logical Push/Pop pair."""

    forward: SegmentForwardResult
    backward: SegmentBackwardResult


class SegmentExecutor:
    """Execute graph-free Push and gradient-carrying Pop for a SegmentPlan.

    The executor deliberately does not choose traversal order, synchronize
    gradients, or step an optimizer. Those responsibilities belong to the
    fixed-topology scheduler and the later Megatron entry adapter.
    """

    def __init__(
        self,
        model: nn.Module,
        plan: SegmentPlan,
        *,
        expected_layer_numbers: tuple[int, ...] | None = None,
        kv_stack: KVStack | None = None,
        loss_scale_func: Callable[[Tensor], Tensor] | None = None,
        cp_group: Any | None = None,
        cp_backend: TPRCPBackend | str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        if not hasattr(model, "rotary_pos_emb") or not callable(model.rotary_pos_emb):
            raise TypeError("model must expose a callable rotary_pos_emb")
        if expected_layer_numbers is None:
            expected_layer_numbers = _infer_layer_numbers(model)
        expected_layer_numbers = tuple(expected_layer_numbers)
        if not expected_layer_numbers:
            raise ValueError("expected_layer_numbers must not be empty")
        if any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0
            for layer in expected_layer_numbers
        ):
            raise ValueError(f"expected_layer_numbers must contain positive integers, got {expected_layer_numbers}")
        if len(set(expected_layer_numbers)) != len(expected_layer_numbers):
            raise ValueError(f"expected_layer_numbers contains duplicates: {expected_layer_numbers}")

        self.model = model
        self.plan = plan
        self.expected_layer_numbers = tuple(sorted(expected_layer_numbers))
        self.kv_stack = KVStack() if kv_stack is None else kv_stack
        self.loss_scale_func = loss_scale_func
        self.cp_group = cp_group
        self.gdn_layer_numbers = tuple(
            sorted(module.layer_number for module in model.modules()
                   if getattr(module, "tpr_state_kind", None) == "gdn")
        )
        if len(set(self.gdn_layer_numbers)) != len(self.gdn_layer_numbers):
            raise ValueError("GDN layer numbers must be unique")
        if set(self.gdn_layer_numbers).intersection(self.expected_layer_numbers):
            raise ValueError("expected_layer_numbers describes FA layers only, not GDN layers")
        if self.gdn_layer_numbers and cp_group is not None:
            backend_name = cp_backend if isinstance(cp_backend, str) else getattr(cp_backend, "backend_name", None)
            if backend_name != "ring":
                raise NotImplementedError("Hybrid TPR requires explicit Ring CP2 (or CP=1)")
        self.gdn_states: dict[SegmentId, GDNPrefixState] = {}
        if cp_group is None:
            if cp_backend is not None:
                raise ValueError("cp_backend requires a cp_group")
            self.cp_size, self.cp_rank = 1, 0
            self.cp_backend = None
        else:
            self.cp_size, self.cp_rank = resolve_cp_group(cp_group)
            if self.cp_size <= 1:
                raise ValueError(f"cp_group must contain more than one rank, got {self.cp_size}")
            self.cp_backend = resolve_tpr_cp_backend(
                cp_backend,
                cp_group=cp_group,
                parallel_size=self.cp_size,
                parallel_rank=self.cp_rank,
            )
            if self.gdn_layer_numbers and (
                self.cp_size != 2 or any(s.length % 4 for s in plan.segments.values())
            ):
                raise NotImplementedError("Hybrid GDN CP requires CP2 and unpadded lengths divisible by 4")
            for segment in plan.segments.values():
                try:
                    self.cp_backend.validate_segment_length(segment.length)
                except ValueError as exc:
                    raise ValueError(
                        f"segment {segment.segment_id} {exc}"
                    ) from exc
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def cp_enabled(self) -> bool:
        return self.cp_group is not None

    def push(self, segment_id: SegmentId) -> SegmentForwardResult:
        """Save graph-free FA KV and GDN final states; do not compute owned loss."""

        self._ensure_healthy()
        try:
            segment = self.plan.get(segment_id)
            context, logits = self._forward(
                segment,
                past_key_values={} if self.cp_enabled else self.kv_stack.build_past_key_values(),
                no_grad=True,
            )
            del logits
            self._assert_collected_layers(context)
            cached_key_values = _compact_kv_cache(context.new_key_values)
            layer_count = len(cached_key_values) + len(context.new_gdn_states)
            if self.gdn_layer_numbers:
                self.gdn_states[segment_id] = GDNPrefixState.save(
                    segment_id, segment.length, context.new_gdn_states
                )
            del context
            self.kv_stack.push(segment, cached_key_values, shard=self._segment_shard(segment))
            return SegmentForwardResult(
                segment_id=segment_id,
                prefix_length=segment.prefix_length,
                suffix_length=segment.length,
                layer_count=layer_count,
            )
        except Exception:
            self._failed = True
            raise

    def pop(self, segment_id: SegmentId) -> SegmentBackwardResult:
        """Recompute owned loss once and relay FA KV / direct-parent GDN gradients."""

        self._ensure_healthy()
        try:
            entry = self.kv_stack.top()
            if entry.segment.segment_id != segment_id:
                raise RuntimeError(f"cannot pop segment {segment_id}: stack top is {entry.segment.segment_id}")

            relayed_gradients = dict(entry.gradients)
            if relayed_gradients and tuple(relayed_gradients) != self.expected_layer_numbers:
                raise RuntimeError(
                    f"segment {segment_id} relayed KV gradient layers must be {self.expected_layer_numbers}, "
                    f"got {tuple(relayed_gradients)}"
                )
            popped_entry = self.kv_stack.pop(segment_id)
            segment = popped_entry.segment
            gdn_state = self.gdn_states.pop(segment_id) if self.gdn_layer_numbers else None
            relayed_gdn_gradients = {} if gdn_state is None else gdn_state.consume_gradients()
            del entry
            if self.cp_enabled:
                popped_entry.kv.release()
                anchors = build_sharded_past_anchors(self.kv_stack)
                past_key_values = {}
            else:
                anchors = self.kv_stack.build_past_anchors()
                past_key_values = anchors.key_values
            gdn_parent, gdn_anchors = self._gdn_parent_anchors()
            context, logits = self._forward(
                segment,
                past_key_values=past_key_values,
                no_grad=False,
                sharded_past_anchors=anchors if self.cp_enabled else None,
                initial_gdn_states={} if gdn_anchors is None else gdn_anchors.layer_states,
            )
            self._assert_collected_layers(context)

            loss_sum, normalized_loss = self._compute_loss(segment, logits)
            owned_loss_term_count = len(self._owned_loss_terms(segment))
            roots: list[Tensor] = []
            root_gradients: list[Tensor | None] = []
            if owned_loss_term_count or self.cp_enabled:
                backward_loss = self._prepare_backward_loss(
                    normalized_loss,
                    logits=logits,
                    has_owned_loss=owned_loss_term_count > 0,
                )
                if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
                    raise TypeError("loss_scale_func must return a scalar tensor")
                roots.append(backward_loss)
                root_gradients.append(None)
            for layer_number in self.expected_layer_numbers:
                if layer_number not in relayed_gradients:
                    continue
                new_key, new_value = context.new_key_values[layer_number]
                key_grad, value_grad = relayed_gradients[layer_number]
                roots.extend((new_key, new_value))
                root_gradients.extend((key_grad, value_grad))
            for layer_number, gradient in relayed_gdn_gradients.items():
                new_state = context.new_gdn_states[layer_number]
                roots.extend((new_state.conv_state, new_state.recurrent_state))
                root_gradients.extend((gradient.conv_state, gradient.recurrent_state))

            if roots:
                torch.autograd.backward(roots, grad_tensors=root_gradients)
                self._accumulate_past_anchor_gradients(anchors)
                if gdn_anchors is not None:
                    gdn_parent.accumulate_anchor_gradients(gdn_anchors)
            if gdn_state is not None:
                gdn_state.release()

            return SegmentBackwardResult(
                segment_id=segment_id,
                loss_sum=loss_sum.detach(),
                normalized_loss=normalized_loss.detach(),
                loss_term_count=owned_loss_term_count,
                relayed_layer_count=len(relayed_gradients) + len(relayed_gdn_gradients),
            )
        except Exception:
            self._failed = True
            raise

    def visit_leaf(self, segment_id: SegmentId) -> LeafVisitResult:
        """Forward/backward one leaf without writing its new KV to the path stack."""

        self._ensure_healthy()
        try:
            segment = self.plan.get(segment_id)
            if self.plan.children_of(segment_id):
                raise ValueError(f"segment {segment_id} is not a leaf")

            expected_parent = self.kv_stack.top().segment.segment_id if len(self.kv_stack) else None
            if segment.parent_id != expected_parent:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: expected parent on stack {segment.parent_id}, "
                    f"got {expected_parent}"
                )
            if segment.prefix_length != self.kv_stack.prefix_length:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: stack prefix length is {self.kv_stack.prefix_length}, "
                    f"got {segment.prefix_length}"
                )

            if self.cp_enabled:
                anchors = build_sharded_past_anchors(self.kv_stack)
                past_key_values = {}
            else:
                anchors = self.kv_stack.build_past_anchors()
                past_key_values = anchors.key_values
            gdn_parent, gdn_anchors = self._gdn_parent_anchors()
            context, logits = self._forward(
                segment,
                past_key_values=past_key_values,
                no_grad=False,
                sharded_past_anchors=anchors if self.cp_enabled else None,
                initial_gdn_states={} if gdn_anchors is None else gdn_anchors.layer_states,
            )
            self._assert_collected_layers(context)
            loss_sum, normalized_loss = self._compute_loss(segment, logits)
            owned_loss_term_count = len(self._owned_loss_terms(segment))

            # Every CP rank must traverse the same backward graph even when
            # this rank owns no loss term, otherwise Attention collectives hang.
            backward_loss = self._prepare_backward_loss(
                normalized_loss,
                logits=logits,
                has_owned_loss=owned_loss_term_count > 0,
            )
            if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
                raise TypeError("loss_scale_func must return a scalar tensor")

            torch.autograd.backward(backward_loss)
            self._accumulate_past_anchor_gradients(anchors)
            if gdn_anchors is not None:
                gdn_parent.accumulate_anchor_gradients(gdn_anchors)

            return LeafVisitResult(
                forward=SegmentForwardResult(
                    segment_id=segment_id,
                    prefix_length=segment.prefix_length,
                    suffix_length=segment.length,
                    layer_count=len(context.new_key_values) + len(context.new_gdn_states),
                ),
                backward=SegmentBackwardResult(
                    segment_id=segment_id,
                    loss_sum=loss_sum.detach(),
                    normalized_loss=normalized_loss.detach(),
                    loss_term_count=owned_loss_term_count,
                    relayed_layer_count=0,
                ),
            )
        except Exception:
            self._failed = True
            raise

    def _forward(
        self,
        segment: SegmentSpec,
        *,
        past_key_values,
        no_grad: bool,
        sharded_past_anchors: ShardedPastKVAnchors | None = None,
        initial_gdn_states: Mapping[int, GDNLayerState] | None = None,
    ) -> tuple[TPRAttentionContext, Tensor]:
        device = _model_device(self.model)
        shard = self._segment_shard(segment)
        local_token_ids = shard.select(segment.token_ids)
        input_ids = local_token_ids.to(device=device).unsqueeze(0)
        position_ids = shard.global_indices(device=device).add(segment.position_start).unsqueeze(0)
        attention_backend = None
        if self.cp_enabled:
            if self.cp_backend is None:
                raise RuntimeError("CP execution is missing its backend")
            attention_backend = self.cp_backend.make_attention_backend(
                self.kv_stack,
                expected_layer_numbers=self.expected_layer_numbers,
                current_shard=shard,
                past_anchors=sharded_past_anchors,
            )
        context = TPRAttentionContext(
            prefix_length=segment.prefix_length,
            suffix_length=segment.length,
            past_key_values=past_key_values,
            suffix_rotary_pos_emb=build_sharded_rotary_pos_emb(
                self.model.rotary_pos_emb,
                position_start=segment.position_start,
                shard=shard,
                disable_context_parallel_sharding=self.cp_enabled,
            ),
            attention_backend=attention_backend,
            initial_gdn_states=(
                self._gdn_parent_state().layer_states
                if initial_gdn_states is None and self.gdn_layer_numbers and len(self.kv_stack)
                else (initial_gdn_states or {})
            ),
        )
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context:
            rope_context = (
                disable_bound_context_parallel_sharding(self.model.rotary_pos_emb)
                if self.cp_enabled
                else nullcontext()
            )
            with rope_context, use_tpr_attention_context(context):
                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                )
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(f"model must return logits [batch, sequence, vocab], got {type(logits).__name__}")
        expected_shape = (1, shard.local_length)
        if tuple(logits.shape[:2]) != expected_shape:
            raise ValueError(f"model logits must start with shape {expected_shape}, got {tuple(logits.shape)}")
        return context, logits

    def _compute_loss(self, segment: SegmentSpec, logits: Tensor) -> tuple[Tensor, Tensor]:
        owned_terms = self._owned_loss_terms(segment)
        if not owned_terms:
            zero = logits.new_zeros((), dtype=torch.float32)
            return zero, zero
        shard = self._segment_shard(segment)
        local_offsets = tuple(shard.global_to_local(term.query_offset) for term in owned_terms)
        if any(offset is None for offset in local_offsets):
            raise RuntimeError(f"segment {segment.segment_id} has a loss term outside its local shard")
        query_offsets = torch.tensor(
            local_offsets,
            dtype=torch.long,
            device=logits.device,
        )
        targets = torch.tensor(
            [term.target_token_id for term in owned_terms],
            dtype=torch.long,
            device=logits.device,
        )
        max_target = max(term.target_token_id for term in owned_terms)
        if max_target >= logits.shape[-1]:
            raise ValueError(
                f"segment {segment.segment_id} target token {max_target} "
                f"is outside vocabulary size {logits.shape[-1]}"
            )
        weights = torch.tensor(
            [term.weight for term in owned_terms],
            dtype=torch.float32,
            device=logits.device,
        )
        selected_logits = logits[0].index_select(0, query_offsets).float()
        per_term_loss = F.cross_entropy(selected_logits, targets, reduction="none")
        loss_sum = torch.sum(per_term_loss * weights)
        return loss_sum, loss_sum / self.plan.total_loss_weight

    def _owned_loss_terms(self, segment: SegmentSpec) -> tuple[SegmentLossTerm, ...]:
        if not self.cp_enabled:
            return segment.loss_terms
        shard = self._segment_shard(segment)
        return tuple(
            term
            for term in segment.loss_terms
            if shard.owns(term.query_offset)
        )

    def _prepare_backward_loss(
        self,
        normalized_loss: Tensor,
        *,
        logits: Tensor,
        has_owned_loss: bool,
    ) -> Tensor:
        if has_owned_loss:
            backward_loss = normalized_loss * self.cp_size
        else:
            backward_loss = logits.float().sum() * 0.0
        return backward_loss if self.loss_scale_func is None else self.loss_scale_func(backward_loss)

    def _segment_shard(self, segment: SegmentSpec) -> SequenceShard:
        if not self.cp_enabled:
            return PrefixShard.full(segment.length)
        if self.cp_backend is None:
            raise RuntimeError("CP execution is missing its backend")
        return self.cp_backend.make_sequence_shard(segment.length)

    def _accumulate_past_anchor_gradients(self, anchors) -> None:
        if self.cp_enabled:
            accumulate_sharded_past_anchor_gradients(self.kv_stack, anchors)
        elif anchors.key_values:
            self.kv_stack.accumulate_anchor_gradients(anchors)

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("SegmentExecutor is failed and cannot continue")

    def _assert_collected_layers(self, context: TPRAttentionContext) -> None:
        context.assert_new_kv_layers(self.expected_layer_numbers)
        context.assert_new_gdn_layers(self.gdn_layer_numbers)

    def _gdn_parent_state(self) -> GDNPrefixState:
        return self.gdn_states[self.kv_stack.top().segment.segment_id]

    def _gdn_parent_anchors(self) -> tuple[GDNPrefixState | None, GDNPrefixAnchors | None]:
        if not self.gdn_layer_numbers or not len(self.kv_stack):
            return None, None
        parent = self._gdn_parent_state()
        return parent, parent.make_anchors()


def _compact_kv_cache(key_values: Mapping[int, KVPair]) -> dict[int, KVPair]:
    """Copy no-grad KV views into compact, independently owned storage."""

    return {
        layer_number: (
            key.detach().clone(memory_format=torch.contiguous_format),
            value.detach().clone(memory_format=torch.contiguous_format),
        )
        for layer_number, (key, value) in key_values.items()
    }


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc


def _infer_layer_numbers(model: nn.Module) -> tuple[int, ...]:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("cannot infer TPR layer numbers; pass expected_layer_numbers explicitly")
    result = []
    for layer in layers:
        attention = getattr(layer, "self_attention", None)
        if getattr(attention, "tpr_state_kind", None) == "gdn":
            continue
        layer_number = getattr(attention, "layer_number", None)
        if layer_number is None:
            raise ValueError("cannot infer layer_number from model.decoder.layers")
        result.append(layer_number)
    return tuple(result)
