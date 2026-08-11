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

"""Per-forward context for model-side external KV attention."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import Tensor

KVPair = tuple[Tensor, Tensor]
RotaryPosEmb = Tensor | tuple[Tensor, Tensor]


def _validate_layer_number(layer_number: int) -> None:
    if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer_number!r}")


def _validate_kv_pair(
    layer_number: int,
    key: Tensor,
    value: Tensor,
    *,
    expected_sequence_length: int,
    kind: str,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"layer {layer_number} {kind} K/V must be torch tensors")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"layer {layer_number} {kind} K/V must have shape [sequence, batch, heads, head_dim], "
            f"got K={key.shape}, V={value.shape}"
        )
    if key.shape != value.shape:
        raise ValueError(
            f"layer {layer_number} {kind} K/V shapes must match, got K={key.shape}, V={value.shape}"
        )
    if key.shape[0] != expected_sequence_length:
        raise ValueError(
            f"layer {layer_number} {kind} KV sequence length must be {expected_sequence_length}, "
            f"got {key.shape[0]}"
        )
    if key.shape[1] != 1:
        raise ValueError(f"layer {layer_number} {kind} KV batch size must be 1, got {key.shape[1]}")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError(
            f"layer {layer_number} {kind} K/V device and dtype must match, "
            f"got K=({key.device}, {key.dtype}), V=({value.device}, {value.dtype})"
        )


@dataclass(slots=True)
class TreeAttentionContext:
    """External KV state scoped to one prefix or suffix model forward.

    Tensor references are stored unchanged. In particular, this class never
    detaches or clones KV tensors, so an external prefix remains connected to
    the autograd graph used by a suffix backward pass.
    """

    prefix_length: int
    suffix_length: int
    past_key_values: Mapping[int, KVPair] = field(default_factory=dict)
    suffix_rotary_pos_emb: RotaryPosEmb | None = None
    _new_key_values: dict[int, KVPair] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prefix_length, int) or isinstance(self.prefix_length, bool) or self.prefix_length < 0:
            raise ValueError(f"prefix_length must be a non-negative integer, got {self.prefix_length!r}")
        if not isinstance(self.suffix_length, int) or isinstance(self.suffix_length, bool) or self.suffix_length <= 0:
            raise ValueError(f"suffix_length must be a positive integer, got {self.suffix_length!r}")
        if self.prefix_length == 0 and self.past_key_values:
            raise ValueError("past_key_values must be empty when prefix_length is 0")

        normalized_past: dict[int, KVPair] = {}
        for layer_number, pair in self.past_key_values.items():
            _validate_layer_number(layer_number)
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"layer {layer_number} past KV must be a (key, value) tuple")
            key, value = pair
            _validate_kv_pair(
                layer_number,
                key,
                value,
                expected_sequence_length=self.prefix_length,
                kind="past",
            )
            normalized_past[layer_number] = pair

        self.past_key_values = MappingProxyType(normalized_past)

    @property
    def new_key_values(self) -> Mapping[int, KVPair]:
        """A read-only view of KV tensors produced by the current segment."""

        return MappingProxyType(self._new_key_values)

    def get_past_kv(self, layer_number: int) -> KVPair | None:
        """Return this layer's prefix KV, or ``None`` for a zero-length prefix."""

        _validate_layer_number(layer_number)
        if self.prefix_length == 0:
            return None
        try:
            return self.past_key_values[layer_number]
        except KeyError as exc:
            raise KeyError(f"missing past KV for layer {layer_number}") from exc

    def set_new_kv(self, layer_number: int, key: Tensor, value: Tensor) -> None:
        """Record this layer's post-RoPE new key and raw new value."""

        _validate_layer_number(layer_number)
        if layer_number in self._new_key_values:
            raise RuntimeError(f"new KV for layer {layer_number} was already recorded")
        _validate_kv_pair(
            layer_number,
            key,
            value,
            expected_sequence_length=self.suffix_length,
            kind="new",
        )
        self._new_key_values[layer_number] = (key, value)

    def assert_new_kv_layers(self, expected_layer_numbers: Iterable[int]) -> None:
        """Fail if the collector does not contain exactly the expected layers."""

        expected = set(expected_layer_numbers)
        for layer_number in expected:
            _validate_layer_number(layer_number)
        actual = set(self._new_key_values)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(f"new KV layer mismatch: missing={missing}, unexpected={unexpected}")


_TREE_ATTENTION_CONTEXT: ContextVar[TreeAttentionContext | None] = ContextVar(
    "tree_attention_context",
    default=None,
)


def get_tree_attention_context() -> TreeAttentionContext | None:
    """Return the context for the current model forward, if one is active."""

    return _TREE_ATTENTION_CONTEXT.get()


@contextmanager
def use_tree_attention_context(context: TreeAttentionContext) -> Iterator[TreeAttentionContext]:
    """Activate ``context`` and reliably restore the previous context."""

    if not isinstance(context, TreeAttentionContext):
        raise TypeError(f"context must be a TreeAttentionContext, got {type(context).__name__}")
    token = _TREE_ATTENTION_CONTEXT.set(context)
    try:
        yield context
    finally:
        _TREE_ATTENTION_CONTEXT.reset(token)
