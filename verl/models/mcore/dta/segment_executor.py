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

"""Push/pop execution of one DTA segment against a Megatron GPT model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .context import KVPair, TreeAttentionContext, use_tree_attention_context
from .kv_stack import KVStack
from .rope import build_suffix_rotary_pos_emb
from .segment_plan import SegmentId, SegmentPlan, SegmentSpec


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
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def push(self, segment_id: SegmentId) -> SegmentForwardResult:
        """Run a graph-free segment forward and push its new KV cache."""

        self._ensure_healthy()
        try:
            segment = self.plan.get(segment_id)
            context, logits = self._forward(
                segment,
                past_key_values=self.kv_stack.build_past_key_values(),
                no_grad=True,
            )
            del logits
            context.assert_new_kv_layers(self.expected_layer_numbers)
            cached_key_values = _compact_kv_cache(context.new_key_values)
            layer_count = len(cached_key_values)
            del context
            self.kv_stack.push(segment, cached_key_values)
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
        """Recompute/backward the stack top and relay past-KV gradients."""

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
            segment = self.kv_stack.pop(segment_id).segment
            del entry
            anchors = self.kv_stack.build_past_anchors()
            context, logits = self._forward(segment, past_key_values=anchors.key_values, no_grad=False)
            context.assert_new_kv_layers(self.expected_layer_numbers)

            loss_sum, normalized_loss = self._compute_loss(segment, logits)
            roots: list[Tensor] = []
            root_gradients: list[Tensor | None] = []
            if segment.loss_terms:
                backward_loss = (
                    normalized_loss if self.loss_scale_func is None else self.loss_scale_func(normalized_loss)
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

            if roots:
                torch.autograd.backward(roots, grad_tensors=root_gradients)
                if anchors.key_values:
                    self.kv_stack.accumulate_anchor_gradients(anchors)

            return SegmentBackwardResult(
                segment_id=segment_id,
                loss_sum=loss_sum.detach(),
                normalized_loss=normalized_loss.detach(),
                loss_term_count=len(segment.loss_terms),
                relayed_layer_count=len(relayed_gradients),
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

            anchors = self.kv_stack.build_past_anchors()
            context, logits = self._forward(segment, past_key_values=anchors.key_values, no_grad=False)
            context.assert_new_kv_layers(self.expected_layer_numbers)
            loss_sum, normalized_loss = self._compute_loss(segment, logits)

            if segment.loss_terms:
                backward_loss = (
                    normalized_loss if self.loss_scale_func is None else self.loss_scale_func(normalized_loss)
                )
                if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
                    raise TypeError("loss_scale_func must return a scalar tensor")
            else:
                # First-version leaf-direct deliberately still performs one
                # grad-enabled FWD+BWD for an empty-loss leaf. Keep the zero
                # connected to the graph; a no-op fast path is a later concern.
                backward_loss = logits.float().sum() * 0.0

            torch.autograd.backward(backward_loss)
            if anchors.key_values:
                self.kv_stack.accumulate_anchor_gradients(anchors)

            return LeafVisitResult(
                forward=SegmentForwardResult(
                    segment_id=segment_id,
                    prefix_length=segment.prefix_length,
                    suffix_length=segment.length,
                    layer_count=len(context.new_key_values),
                ),
                backward=SegmentBackwardResult(
                    segment_id=segment_id,
                    loss_sum=loss_sum.detach(),
                    normalized_loss=normalized_loss.detach(),
                    loss_term_count=len(segment.loss_terms),
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
    ) -> tuple[TreeAttentionContext, Tensor]:
        device = _model_device(self.model)
        input_ids = segment.token_ids.to(device=device).unsqueeze(0)
        position_ids = torch.arange(
            segment.position_start,
            segment.position_end,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        context = TreeAttentionContext(
            prefix_length=segment.prefix_length,
            suffix_length=segment.length,
            past_key_values=past_key_values,
            suffix_rotary_pos_emb=build_suffix_rotary_pos_emb(
                self.model.rotary_pos_emb,
                prefix_length=segment.prefix_length,
                suffix_length=segment.length,
            ),
        )
        grad_context = torch.no_grad() if no_grad else torch.enable_grad()
        with grad_context:
            with use_tree_attention_context(context):
                logits = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                )
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(f"model must return logits [batch, sequence, vocab], got {type(logits).__name__}")
        expected_shape = (1, segment.length)
        if tuple(logits.shape[:2]) != expected_shape:
            raise ValueError(f"model logits must start with shape {expected_shape}, got {tuple(logits.shape)}")
        return context, logits

    def _compute_loss(self, segment: SegmentSpec, logits: Tensor) -> tuple[Tensor, Tensor]:
        if not segment.loss_terms:
            zero = logits.new_zeros((), dtype=torch.float32)
            return zero, zero
        query_offsets = torch.tensor(
            [term.query_offset for term in segment.loss_terms],
            dtype=torch.long,
            device=logits.device,
        )
        targets = torch.tensor(
            [term.target_token_id for term in segment.loss_terms],
            dtype=torch.long,
            device=logits.device,
        )
        max_target = max(term.target_token_id for term in segment.loss_terms)
        if max_target >= logits.shape[-1]:
            raise ValueError(
                f"segment {segment.segment_id} target token {max_target} "
                f"is outside vocabulary size {logits.shape[-1]}"
            )
        weights = torch.tensor(
            [term.weight for term in segment.loss_terms],
            dtype=torch.float32,
            device=logits.device,
        )
        selected_logits = logits[0].index_select(0, query_offsets).float()
        per_term_loss = F.cross_entropy(selected_logits, targets, reduction="none")
        loss_sum = torch.sum(per_term_loss * weights)
        return loss_sum, loss_sum / self.plan.total_loss_weight

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("SegmentExecutor is failed and cannot continue")


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
        raise ValueError("cannot infer DTA layer numbers; pass expected_layer_numbers explicitly")
    result = []
    for layer in layers:
        attention = getattr(layer, "self_attention", None)
        layer_number = getattr(attention, "layer_number", None)
        if layer_number is None:
            raise ValueError("cannot infer layer_number from model.decoder.layers")
        result.append(layer_number)
    return tuple(result)
