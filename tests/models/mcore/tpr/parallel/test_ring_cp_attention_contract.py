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

import sys
from types import SimpleNamespace

import pytest
import torch

import verl.models.mcore.tpr.parallel.ring_attention as ring
from verl.models.mcore.tpr import (
    RingBlockKind,
    classify_ring_block,
    make_ring_sequence_shard,
)


def test_mindspeed_causal_ring_shard_uses_symmetric_ranges():
    rank_zero = make_ring_sequence_shard(16, cp_rank=0, cp_size=2)
    rank_one = make_ring_sequence_shard(16, cp_rank=1, cp_size=2)
    source = torch.arange(16)

    assert rank_zero.global_ranges == ((0, 4), (12, 16))
    assert rank_one.global_ranges == ((4, 8), (8, 12))
    assert rank_zero.select(source).tolist() == [0, 1, 2, 3, 12, 13, 14, 15]
    assert rank_one.select(source).tolist() == [4, 5, 6, 7, 8, 9, 10, 11]


def test_non_divisible_ring_slice_keeps_the_physical_tail_block():
    shard = make_ring_sequence_shard(63, cp_rank=0, cp_size=2)
    local = shard.select(torch.arange(63))

    slices = ring._iter_range_slices(local, shard)

    assert [item.logical_range for item in slices] == [(0, 16), (48, 63)]
    assert [item.physical_range for item in slices] == [(0, 16), (48, 64)]
    assert [item.tensor.shape[0] for item in slices] == [16, 16]
    assert slices[1].tensor[-1].item() == 0


def test_non_divisible_causal_block_masks_physical_padding():
    shard = make_ring_sequence_shard(63, cp_rank=0, cp_size=2)
    local = shard.select(torch.arange(63))
    tail = ring._iter_range_slices(local, shard)[1]

    attention_mask, valid_query, valid_kv = ring._physical_block_attention_mask(
        tail,
        tail,
        block_kind=RingBlockKind.CAUSAL,
    )

    assert attention_mask.shape == (16, 16)
    assert valid_query.tolist() == [True] * 15 + [False]
    assert valid_kv.tolist() == [True] * 15 + [False]
    assert torch.all(attention_mask[:, -1]).item()
    torch.testing.assert_close(
        attention_mask[:15, :15],
        torch.triu(torch.ones(15, 15, dtype=torch.bool), diagonal=1),
    )


def test_prefix_padding_mask_does_not_hide_later_current_kv():
    query_shard = make_ring_sequence_shard(63, cp_rank=1, cp_size=2)
    prefix_shard = make_ring_sequence_shard(127, cp_rank=0, cp_size=2)
    query = ring._iter_range_slices(query_shard.select(torch.arange(63)), query_shard)[0]
    prefix_tail = ring._iter_range_slices(
        prefix_shard.select(torch.arange(127)),
        prefix_shard,
    )[1]
    current = ring._iter_range_slices(query_shard.select(torch.arange(63)), query_shard)[0]

    prefix_mask, _, prefix_validity = ring._physical_block_attention_mask(
        query,
        prefix_tail,
        block_kind=RingBlockKind.FULL,
    )
    current_mask, _, current_validity = ring._physical_block_attention_mask(
        query,
        current,
        block_kind=RingBlockKind.CAUSAL,
    )

    assert prefix_mask.shape == (16, 32)
    assert torch.all(prefix_mask[:, -1]).item()
    assert not torch.any(prefix_mask[:, :-1]).item()
    assert prefix_validity.tolist() == [True] * 31 + [False]
    assert current_mask is None
    assert current_validity is None


def test_divisible_ring_blocks_keep_the_sparse_fast_paths():
    shard = make_ring_sequence_shard(64, cp_rank=0, cp_size=2)
    local = shard.select(torch.arange(64))
    first, second = ring._iter_range_slices(local, shard)

    full_mask, full_query_validity, full_kv_validity = (
        ring._physical_block_attention_mask(
            second,
            first,
            block_kind=RingBlockKind.FULL,
        )
    )
    causal_mask, causal_query_validity, causal_kv_validity = (
        ring._physical_block_attention_mask(
            second,
            second,
            block_kind=RingBlockKind.CAUSAL,
        )
    )

    assert (full_mask, full_query_validity, full_kv_validity) == (None, None, None)
    assert (causal_mask, causal_query_validity, causal_kv_validity) == (
        None,
        None,
        None,
    )


def test_physical_padding_reaches_fused_attention_with_custom_mask(monkeypatch):
    calls = []

    def npu_fusion_attention(query, key, value, head_num, input_layout, **kwargs):
        calls.append((query, key, value, head_num, input_layout, kwargs))
        statistics = query.new_zeros((1, head_num, query.shape[0], 8))
        return query.clone(), statistics, statistics

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_fusion_attention=npu_fusion_attention),
    )
    query = torch.zeros(16, 4, 8)
    key = torch.zeros(16, 2, 8)
    value = torch.zeros_like(key)
    attention_mask = torch.zeros(16, 16, dtype=torch.bool)
    attention_mask[:, -1] = True

    ring._block_attention_forward(
        query,
        key,
        value,
        query_heads=4,
        softmax_scale=8**-0.5,
        block_kind=RingBlockKind.CAUSAL,
        attention_mask=attention_mask,
    )

    assert len(calls) == 1
    called_query, called_key, _, _, layout, kwargs = calls[0]
    assert called_query.shape[0] == 16
    assert called_key.shape[0] == 16
    assert layout == "TND"
    assert torch.equal(kwargs["atten_mask"], attention_mask)
    assert kwargs["sparse_mode"] == 0
    assert kwargs["actual_seq_qlen"] == [16]
    assert kwargs["actual_seq_kvlen"] == [16]


@pytest.mark.parametrize(
    ("query_range", "kv_range", "is_prefix", "expected"),
    [
        ((8, 12), (0, 4), False, RingBlockKind.FULL),
        ((8, 12), (8, 12), False, RingBlockKind.CAUSAL),
        ((8, 12), (12, 16), False, RingBlockKind.SKIP),
        ((0, 4), (12, 16), True, RingBlockKind.FULL),
    ],
)
def test_ring_block_visibility(query_range, kv_range, is_prefix, expected):
    assert classify_ring_block(query_range, kv_range, is_prefix=is_prefix) is expected


def test_ring_block_visibility_rejects_partial_overlap():
    with pytest.raises(ValueError, match="partially overlapping"):
        classify_ring_block((4, 8), (6, 10), is_prefix=False)


def test_tnd_online_softmax_merge_handles_noncontiguous_layout_conversion():
    query_length, heads, head_dim = 4, 2, 8
    output_shape = (query_length, heads, head_dim)
    statistics_shape = (1, heads, query_length, 8)
    previous = (
        torch.zeros(output_shape),
        torch.zeros(statistics_shape),
        torch.ones(statistics_shape),
    )
    current = (
        torch.ones(output_shape),
        torch.zeros(statistics_shape),
        torch.ones(statistics_shape),
    )

    output, softmax_max, softmax_sum = ring._merge_attention(
        previous,
        current,
        query_length=query_length,
    )

    torch.testing.assert_close(output, torch.full(output_shape, 0.5))
    flattened_max = ring._flatten_tnd_softmax(softmax_max, (query_length,))
    flattened_sum = ring._flatten_tnd_softmax(softmax_sum, (query_length,))
    torch.testing.assert_close(flattened_max, torch.zeros_like(flattened_max))
    torch.testing.assert_close(flattened_sum, torch.full_like(flattened_sum, 2.0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: make_ring_sequence_shard(16, cp_rank=2, cp_size=2),
        lambda: make_ring_sequence_shard(16, cp_rank=0, cp_size=1),
        lambda: make_ring_sequence_shard(16, cp_rank=0, cp_size=2, padded_length=14),
    ],
)
def test_invalid_ring_shard_requests_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("blocks", [1, 5, 8])
def test_merge_keeps_fp32_until_final_output_for_backward(blocks):
    generator = torch.Generator().manual_seed(44)
    values = [torch.randn(4, 2, 8, generator=generator).bfloat16() for _ in range(blocks)]
    merged = None
    for value in values:
        merged = ring._merge_attention(merged, (value, torch.zeros(1, 2, 4, 8),
                                                 torch.ones(1, 2, 4, 8)), query_length=4)
        assert all(t.dtype == torch.float32 for t in merged)
    expected = torch.stack([v.float() for v in values]).mean(0)
    torch.testing.assert_close(merged[0], expected, atol=2e-7, rtol=2e-6)
    final = ring._finalize_attention_result(merged, dtype=torch.bfloat16)
    assert final[0].dtype == torch.bfloat16 and final[0].is_contiguous()
    assert final[1] is merged[1] and final[2] is merged[2]
    torch.testing.assert_close(final[0], merged[0].bfloat16(), atol=0, rtol=0)


def test_query_coalescing_slices_head_major_statistics_and_reassembles():
    length, heads, dim, lanes = 12, 4, 16, 8
    output = torch.arange(length * heads * dim).reshape(length, heads, dim).float()
    # Deliberately distinct values across heads and query rows catch slicing
    # [T,H,8] as if the physical storage were token-major.
    head_major = torch.arange(heads * length * lanes).reshape(heads, length, lanes).float()
    maximum = head_major.view(length, heads, lanes)
    total = (head_major + 10000).view(length, heads, lanes)
    restored_max, restored_sum = torch.empty_like(maximum), torch.empty_like(total)
    for rows in (slice(0, 6), slice(6, 12)):
        part = ring._slice_tnd_result((output, maximum, total), rows)
        assert part[0].untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
        torch.testing.assert_close(part[0], output[rows], atol=0, rtol=0)
        torch.testing.assert_close(part[1].view(heads, 6, lanes), head_major[:, rows], atol=0, rtol=0)
        torch.testing.assert_close(part[2].view(heads, 6, lanes), head_major[:, rows] + 10000, atol=0, rtol=0)
        for full, chunk in zip((restored_max, restored_sum), part[1:], strict=True):
            full.view(heads, length, lanes)[:, rows, :].copy_(chunk.view(heads, 6, lanes))
    torch.testing.assert_close(restored_max, maximum, atol=0, rtol=0)
    torch.testing.assert_close(restored_sum, total, atol=0, rtol=0)


def test_query_coalescing_requires_original_contiguous_unpadded_view():
    config = SimpleNamespace(
        coalesce_prefix_full=True, coalesce_prefix_query=True,
        segment_lengths=(128, 64), segment_padded_lengths=(128, 64),
    )
    q = torch.randn(16, 1, 4, 8)
    assert ring._can_coalesce_prefix_query(q, config)
    halves = q.squeeze(1).chunk(2, dim=0)
    assert all(item.untyped_storage().data_ptr() == q.untyped_storage().data_ptr() for item in halves)
    assert halves[1].storage_offset() - halves[0].storage_offset() == halves[0].numel()
    noncontiguous = q.transpose(0, 2).contiguous().transpose(0, 2)
    assert not ring._can_coalesce_prefix_query(noncontiguous, config)
    config.segment_padded_lengths = (128, 72)
    assert not ring._can_coalesce_prefix_query(q, config)
