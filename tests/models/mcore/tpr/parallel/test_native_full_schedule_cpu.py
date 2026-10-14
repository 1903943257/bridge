"""CPU-only geometry gates for the native THD/zigzag CP replay lane."""
import pytest

from .native_full_megatron_schedule import native_cp_query_indices


@pytest.mark.parametrize("length", [1, 7, 16, 17, 127, 36817, 44070, 49152])
def test_zigzag_cp_is_exhaustive_nonoverlapping(length):
    shards = [native_cp_query_indices(length, cp_rank=r) for r in range(4)]
    padded = ((length + 15) // 16) * 16
    assert all(len(x) == padded // 4 for x in shards)
    physical_positions = [position for shard in shards for position in shard]
    assert len(physical_positions) == len(set(physical_positions))
    assert sorted(physical_positions) == list(range(padded))
    assert sum(position < length for position in physical_positions) == length


def test_original_long_trajectory_query_has_one_cp_owner():
    prompt = 17483
    for length in (36817, 44070, 27546, 42080, 49152, 18583, 41041, 49152):
        shards = [set(native_cp_query_indices(length, cp_rank=r)) for r in range(4)]
        for query in (prompt - 1, min(length - 2, 43008), length - 2):
            assert sum(query in shard for shard in shards) == 1


def test_invalid_cp_is_rejected():
    with pytest.raises(ValueError):
        native_cp_query_indices(0, 0)
    with pytest.raises(ValueError):
        native_cp_query_indices(16, 4)
