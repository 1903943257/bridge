"""Test-only CP2 whole-FA reference with native zigzag ownership preserved."""

from collections import Counter

import torch


def restore_zigzag_gather(value):
    """AllGather rank order [chunk0,chunk3,chunk1,chunk2] -> chronological."""
    assert value.shape[0] % 4 == 0
    chunks = value.chunk(4, dim=0)
    return torch.cat((chunks[0], chunks[2], chunks[3], chunks[1]), dim=0).contiguous()


def install_whole_fa_allgather(monkeypatch):
    from verl.models.mcore.tpr.parallel import ring_attention as ring
    from verl.models.mcore.tpr.parallel import allgather_attention as ag
    from verl.models.mcore.tpr.rectangular_attention import rectangular_causal_attention
    counts = Counter()
    original_rs = ag._reduce_scatter_tensor

    def reduce_scatter(*args, **kwargs):
        counts["reduce_scatter"] += 1
        return original_rs(*args, **kwargs)

    def attention(query, current_key, current_value, *, prefix_blocks, current_shard,
                  cp_group, softmax_scale=None):
        blocks, config = ring._normalize_inputs(query, current_key, current_value,
            prefix_blocks=prefix_blocks, current_shard=current_shard,
            cp_group=cp_group, softmax_scale=softmax_scale)
        assert config.cp_size == 2
        assert all(a == b for a, b in zip(config.segment_lengths, config.segment_padded_lengths))
        assert all(length % 4 == 0 for length in config.segment_lengths)
        counts["fa_allgather"] += 1
        def gather(value):
            counts["all_gather"] += 1
            return restore_zigzag_gather(ag.all_gather_sequence(value, cp_group))
        full_q = gather(query)
        keys, values = [], []
        for block in blocks:
            keys.append(gather(block.key))
            values.append(gather(block.value))
        keys.append(gather(current_key))
        values.append(gather(current_value))
        # Every rank computes full attention, but only owns its query outputs.
        # AllGather's SUM ReduceScatter VJP combines disjoint query objectives;
        # no CP division or extra state-gradient SUM is needed.
        output = rectangular_causal_attention(full_q, torch.cat(keys), torch.cat(values),
                                              softmax_scale=config.softmax_scale, dropout_p=0.)
        return output.index_select(0, current_shard.global_indices(device=output.device)).contiguous()

    monkeypatch.setattr(ring, "ring_cp_attention", attention)
    monkeypatch.setattr(ag, "_reduce_scatter_tensor", reduce_scatter)
    return counts
