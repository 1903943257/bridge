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

"""Build the first real-agent TPR topology: one shared prompt root plus response leaves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """A zero-copy reference into one row of the original training batch."""

    row: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or isinstance(self.row, bool) or self.row < 0:
            raise ValueError(f"row must be a non-negative integer, got {self.row!r}")
        if not isinstance(self.start, int) or isinstance(self.start, bool) or self.start < 0:
            raise ValueError(f"start must be a non-negative integer, got {self.start!r}")
        if not isinstance(self.end, int) or isinstance(self.end, bool) or self.end < self.start:
            raise ValueError(f"end must be an integer >= start, got {self.end!r}")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class TrajectoryLeaf:
    """One response leaf in a prompt-root sibling tree."""

    key: str
    uid: str
    session_id: str
    trajectory_index: int
    row: int
    segment: SegmentRef

    @property
    def response_length(self) -> int:
        return self.segment.length


@dataclass(frozen=True, slots=True)
class PromptSiblingTree:
    """V0 topology: one exact shared prompt root and independent response leaves."""

    uid: str
    root: SegmentRef
    member_rows: tuple[int, ...]
    leaves: tuple[TrajectoryLeaf, ...]

    @property
    def prompt_length(self) -> int:
        return self.root.length


_REQUIRED_FIELDS = (
    "prompts",
    "responses",
    "input_ids",
    "response_mask",
    "loss_mask",
    "rollout_log_probs",
    "rm_scores",
)


def _rows(batch: Any, name: str) -> list[Any]:
    try:
        value = batch[name]
    except Exception as exc:
        raise ValueError(f"training batch is missing required field {name!r}") from exc

    if isinstance(value, (list, tuple)):
        return list(value)

    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        return list(unbind())

    raise TypeError(f"field {name!r} does not expose row-wise values: {type(value)!r}")


def _as_1d_tensor(value: Any, *, field: str, row: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field}[{row}] must be a torch.Tensor, got {type(value)!r}")
    if value.ndim != 1:
        raise ValueError(f"{field}[{row}] must be 1-D, got shape={tuple(value.shape)}")
    return value


def _parse_key(key: str) -> tuple[str, str, int]:
    if not isinstance(key, str):
        raise TypeError(f"trajectory key must be str, got {type(key)!r}")
    fields = key.rsplit("_", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise ValueError(f"unexpected trajectory key format: {key!r}")
    try:
        trajectory_index = int(fields[2])
    except ValueError as exc:
        raise ValueError(f"trajectory key has non-integer index: {key!r}") from exc
    return fields[0], fields[1], trajectory_index


def build_prompt_sibling_trees(
    keys: list[str] | tuple[str, ...],
    batch: Any,
) -> tuple[PromptSiblingTree, ...]:
    """Build prompt-root sibling trees from a ReplayBuffer/TQ training batch.

    V0 deliberately reuses only the exact prompt. Generated response prefixes
    remain independent leaves so GRPO/PPO loss semantics stay unchanged even
    when different sessions happen to emit identical trainable tokens.

    Segment coordinates are in ``input_ids`` space. The topology stores only
    row/range references and never copies token tensors. Response-level fields
    (advantages, old log-probs, masks, rollout log-probs, rewards) remain owned
    by the original batch row and can be consumed later by the TPR adapter.
    """

    keys = list(keys)
    if not keys:
        return ()

    rows_by_field = {name: _rows(batch, name) for name in _REQUIRED_FIELDS}
    row_count = len(rows_by_field["input_ids"])
    if len(keys) != row_count:
        raise ValueError(f"keys/batch row mismatch: keys={len(keys)} rows={row_count}")
    for name, values in rows_by_field.items():
        if len(values) != row_count:
            raise ValueError(f"field {name!r} has {len(values)} rows; expected {row_count}")

    seen_keys: set[str] = set()
    grouped_rows: dict[str, list[tuple[int, str, str, int]]] = {}
    for row, key in enumerate(keys):
        if key in seen_keys:
            raise ValueError(f"duplicate trajectory key: {key!r}")
        seen_keys.add(key)
        uid, session_id, trajectory_index = _parse_key(key)
        grouped_rows.setdefault(uid, []).append((row, key, session_id, trajectory_index))

    trees: list[PromptSiblingTree] = []
    for uid, members in grouped_rows.items():
        first_row = members[0][0]
        base_prompt = _as_1d_tensor(rows_by_field["prompts"][first_row], field="prompts", row=first_row)
        prompt_len = len(base_prompt)
        if prompt_len <= 0:
            raise ValueError(f"uid {uid!r} has an empty prompt")

        leaves: list[TrajectoryLeaf] = []
        member_rows: list[int] = []
        for row, key, session_id, trajectory_index in members:
            prompt = _as_1d_tensor(rows_by_field["prompts"][row], field="prompts", row=row)
            response = _as_1d_tensor(rows_by_field["responses"][row], field="responses", row=row)
            input_ids = _as_1d_tensor(rows_by_field["input_ids"][row], field="input_ids", row=row)
            response_mask = _as_1d_tensor(rows_by_field["response_mask"][row], field="response_mask", row=row)
            loss_mask = _as_1d_tensor(rows_by_field["loss_mask"][row], field="loss_mask", row=row)
            rollout_log_probs = _as_1d_tensor(
                rows_by_field["rollout_log_probs"][row], field="rollout_log_probs", row=row
            )
            rm_scores = _as_1d_tensor(rows_by_field["rm_scores"][row], field="rm_scores", row=row)

            if len(prompt) != prompt_len or not torch.equal(prompt, base_prompt):
                raise ValueError(f"uid {uid!r} row {row} does not share the exact prompt root")

            response_len = len(response)
            aligned_lengths = {
                "response_mask": len(response_mask),
                "loss_mask": len(loss_mask),
                "rollout_log_probs": len(rollout_log_probs),
                "rm_scores": len(rm_scores),
            }
            bad = {name: length for name, length in aligned_lengths.items() if length != response_len}
            if bad:
                raise ValueError(
                    f"uid {uid!r} row {row} response-field length mismatch: responses={response_len}, {bad}"
                )
            if not torch.equal(response_mask, loss_mask):
                raise ValueError(f"uid {uid!r} row {row} requires response_mask == loss_mask for V0")

            if len(input_ids) != prompt_len + response_len:
                raise ValueError(
                    f"uid {uid!r} row {row} input length mismatch: "
                    f"input={len(input_ids)} prompt={prompt_len} response={response_len}"
                )
            if not torch.equal(input_ids[:prompt_len], prompt):
                raise ValueError(f"uid {uid!r} row {row} input_ids prompt prefix mismatch")
            if not torch.equal(input_ids[prompt_len:], response):
                raise ValueError(f"uid {uid!r} row {row} input_ids response suffix mismatch")

            member_rows.append(row)
            leaves.append(
                TrajectoryLeaf(
                    key=key,
                    uid=uid,
                    session_id=session_id,
                    trajectory_index=trajectory_index,
                    row=row,
                    segment=SegmentRef(row=row, start=prompt_len, end=len(input_ids)),
                )
            )

        trees.append(
            PromptSiblingTree(
                uid=uid,
                root=SegmentRef(row=first_row, start=0, end=prompt_len),
                member_rows=tuple(member_rows),
                leaves=tuple(leaves),
            )
        )

    return tuple(trees)
