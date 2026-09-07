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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import verl.models.mcore.tpr.parallel.execution_context as cp_execution
import verl.models.mcore.tpr.segment_executor as segment_execution
from verl.models.mcore.tpr import (
    RangeSequenceShard,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
    TPRAttentionContext,
    get_tpr_attention_context,
)


class _FakeGroup:
    def __init__(self, rank):
        self.rank = rank


class _RecordingRotaryEmbedding:
    def __init__(self):
        self.calls = []

    def __call__(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        self.calls.append((max_seq_len, offset, packed_seq, cp_group))
        return torch.zeros(max_seq_len, 1, 1, 2)


class _FakeCPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.rotary_pos_emb = _RecordingRotaryEmbedding()
        self.calls = []

    def forward(self, *, input_ids, position_ids, attention_mask):
        del attention_mask
        context = get_tpr_attention_context()
        assert context is not None
        self.calls.append(
            SimpleNamespace(
                input_ids=input_ids.detach().clone(),
                position_ids=position_ids.detach().clone(),
                context=context,
            )
        )
        token_signal = input_ids.transpose(0, 1).float().view(-1, 1, 1, 1) * self.scale
        context.set_new_kv(1, token_signal, token_signal * 2)
        classes = torch.arange(32, dtype=torch.float32, device=input_ids.device)
        return token_signal.view(1, -1, 1) * classes.view(1, 1, -1)


class _RangeCPBackend:
    backend_name = "test-range"
    parallel_size = 2
    parallel_rank = 0

    def validate_segment_length(self, global_length):
        if global_length != 8:
            raise ValueError("test backend only accepts length 8")

    def make_sequence_shard(self, global_length):
        self.validate_segment_length(global_length)
        return RangeSequenceShard(global_length, ((0, 2), (6, 8)), cp_rank=0, cp_size=2)

    def make_attention_backend(
        self,
        kv_stack,
        *,
        expected_layer_numbers,
        current_shard,
        past_anchors,
    ):
        del expected_layer_numbers, past_anchors
        return SimpleNamespace(
            global_prefix_length=kv_stack.prefix_length,
            global_suffix_length=current_shard.global_length,
            local_suffix_length=current_shard.local_length,
            parallel_size=self.parallel_size,
            attention=lambda *args, **kwargs: None,
        )


def _plan(*, root_length=8, first_length=6, second_length=4):
    root = SegmentSpec(
        0,
        None,
        torch.arange(root_length, dtype=torch.long),
        position_start=0,
        prefix_length=0,
        loss_terms=(
            SegmentLossTerm(root_length // 2 - 1, 7),
            SegmentLossTerm(root_length // 2, 11),
            SegmentLossTerm(root_length - 1, 13),
            SegmentLossTerm(root_length - 1, 17),
        ),
    )
    first = SegmentSpec(
        1,
        0,
        torch.arange(100, 100 + first_length, dtype=torch.long),
        position_start=root_length,
        prefix_length=root_length,
        loss_terms=(SegmentLossTerm(0, 19), SegmentLossTerm(first_length - 1, 23)),
    )
    second = SegmentSpec(
        2,
        0,
        torch.arange(200, 200 + second_length, dtype=torch.long),
        position_start=root_length,
        prefix_length=root_length,
        loss_terms=(SegmentLossTerm(0, 29),),
    )
    return SegmentPlan((root, first, second), root_id=0)


def _single_segment_plan():
    root = SegmentSpec(
        0,
        None,
        torch.arange(8, dtype=torch.long),
        position_start=0,
        prefix_length=0,
        loss_terms=(SegmentLossTerm(1, 7), SegmentLossTerm(6, 11)),
    )
    return SegmentPlan((root,), root_id=0)


@pytest.fixture
def fake_cp_runtime(monkeypatch):
    def install(rank):
        resolver = lambda group: (2, group.rank)
        monkeypatch.setattr(segment_execution, "resolve_cp_group", resolver)
        monkeypatch.setattr(cp_execution, "_group_world_size_and_rank", resolver)
        return _FakeGroup(rank)

    return install


@pytest.mark.parametrize(
    ("rank", "expected_tokens", "expected_positions", "expected_rope"),
    [
        (0, [0, 1, 2, 3], [0, 1, 2, 3], (4, 0, False)),
        (1, [4, 5, 6, 7], [4, 5, 6, 7], (4, 4, False)),
    ],
)
def test_push_executes_only_the_local_contiguous_shard(
    fake_cp_runtime,
    rank,
    expected_tokens,
    expected_positions,
    expected_rope,
):
    group = fake_cp_runtime(rank)
    model = _FakeCPModel()
    executor = SegmentExecutor(model, _plan(), expected_layer_numbers=(1,), cp_group=group)

    executor.push(0)

    call = model.calls[0]
    assert call.input_ids.tolist() == [expected_tokens]
    assert call.position_ids.tolist() == [expected_positions]
    rope_call = model.rotary_pos_emb.calls[0]
    assert rope_call[:3] == expected_rope
    assert rope_call[3] is not None and rope_call[3].size() == 1
    assert call.context.prefix_length == 0
    assert call.context.suffix_length == 8
    assert call.context.local_suffix_length == 4
    state = executor.kv_stack.top().kv
    assert state.global_length == 8
    assert state.local_length == 4
    assert state.shard.cp_rank == rank
    assert state.shard.cp_size == 2
    assert state.key_values[1][0].shape[0] == 4


def test_executor_uses_backend_owned_noncontiguous_shard(fake_cp_runtime):
    group = fake_cp_runtime(0)
    model = _FakeCPModel()
    executor = SegmentExecutor(
        model,
        _single_segment_plan(),
        expected_layer_numbers=(1,),
        cp_group=group,
        cp_backend=_RangeCPBackend(),
    )

    executor.push(0)

    call = model.calls[0]
    assert call.input_ids.tolist() == [[0, 1, 6, 7]]
    assert call.position_ids.tolist() == [[0, 1, 6, 7]]
    assert [(length, offset) for length, offset, _, _ in model.rotary_pos_emb.calls] == [
        (2, 0),
        (2, 6),
    ]
    assert tuple(term.query_offset for term in executor._owned_loss_terms(executor.plan.get(0))) == (
        1,
        6,
    )
    assert isinstance(executor.kv_stack.top().kv.shard, RangeSequenceShard)


@pytest.mark.parametrize(
    ("rank", "expected_root_offsets", "expected_first_offsets", "expected_second_offsets"),
    [
        (0, (3,), (0,), (0,)),
        (1, (4, 7, 7), (5,), ()),
    ],
)
def test_loss_terms_have_exactly_one_query_owner(
    fake_cp_runtime,
    rank,
    expected_root_offsets,
    expected_first_offsets,
    expected_second_offsets,
):
    executor = SegmentExecutor(
        _FakeCPModel(),
        _plan(),
        expected_layer_numbers=(1,),
        cp_group=fake_cp_runtime(rank),
    )

    actual_root = tuple(term.query_offset for term in executor._owned_loss_terms(executor.plan.get(0)))
    actual_first = tuple(term.query_offset for term in executor._owned_loss_terms(executor.plan.get(1)))
    actual_second = tuple(term.query_offset for term in executor._owned_loss_terms(executor.plan.get(2)))
    assert actual_root == expected_root_offsets
    assert actual_first == expected_first_offsets
    assert actual_second == expected_second_offsets


def test_cp_backward_loss_compensates_for_megatron_dp_cp_average(fake_cp_runtime):
    executor = SegmentExecutor(
        _FakeCPModel(),
        _plan(),
        expected_layer_numbers=(1,),
        cp_group=fake_cp_runtime(0),
    )
    normalized_loss = torch.tensor(3.0, requires_grad=True)
    backward_loss = executor._prepare_backward_loss(
        normalized_loss,
        logits=torch.zeros(1, 1, 1, requires_grad=True),
        has_owned_loss=True,
    )

    backward_loss.backward()

    torch.testing.assert_close(normalized_loss.grad, torch.tensor(2.0))


def test_rank_without_a_loss_term_keeps_a_graph_connected_zero(fake_cp_runtime):
    executor = SegmentExecutor(
        _FakeCPModel(),
        _plan(),
        expected_layer_numbers=(1,),
        cp_group=fake_cp_runtime(1),
    )
    logits = torch.randn(1, 2, 8, requires_grad=True)
    backward_loss = executor._prepare_backward_loss(
        logits.new_zeros(()),
        logits=logits,
        has_owned_loss=False,
    )

    backward_loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


def test_non_divisible_segment_is_rejected_before_execution(fake_cp_runtime):
    with pytest.raises(ValueError, match="segment 1 length 5 must be divisible"):
        SegmentExecutor(
            _FakeCPModel(),
            _plan(first_length=5),
            expected_layer_numbers=(1,),
            cp_group=fake_cp_runtime(0),
        )


def test_tpr_context_rejects_mismatched_sharded_backend_lengths():
    backend = SimpleNamespace(
        global_prefix_length=4,
        global_suffix_length=8,
        local_suffix_length=4,
        parallel_size=2,
        attention=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="prefix length"):
        TPRAttentionContext(
            prefix_length=6,
            suffix_length=8,
            suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 2),
            attention_backend=backend,
        )
