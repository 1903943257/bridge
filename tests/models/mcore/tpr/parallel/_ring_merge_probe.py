"""Forward-only shadow diagnostics; real Ring outputs/backward remain untouched."""

from contextlib import contextmanager
from unittest.mock import patch

import torch


def block_reference(query, key, value, *, scale, causal, mask):
    q, k, v = (x.detach().cpu().float() for x in (query, key, value))
    repeats = q.shape[1] // k.shape[1]
    assert repeats * k.shape[1] == q.shape[1]
    k, v = (x.repeat_interleave(repeats, dim=1) for x in (k, v))
    scores = (q.transpose(0, 1) @ k.permute(1, 2, 0)) * scale
    excluded = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool)
    if mask is not None:
        excluded |= mask.detach().cpu()
    elif causal:
        excluded |= torch.arange(k.shape[0])[None, :] > (
            k.shape[0] - q.shape[0] + torch.arange(q.shape[0])[:, None])
    # This unpadded L4 fixture should skip fully invisible blocks entirely.
    # Reject unexpected all-masked rows rather than propagate NaNs into metrics.
    assert not excluded.all(-1).any(), "unexpected fully-masked launched block in unpadded L4"
    scores = scores.masked_fill(excluded, -torch.inf)
    maximum = scores.max(-1).values
    weights = (scores - maximum[..., None]).exp()
    total = weights.sum(-1)
    output = (weights / total[..., None]) @ v.transpose(0, 1)
    return output.transpose(0, 1).contiguous(), maximum.T.contiguous(), total.T.contiguous()


@contextmanager
def ring_merge_probe():
    from verl.models.mcore.tpr.parallel import ring_attention as ring

    original_block, original_merge = ring._block_attention_forward, ring._merge_attention
    state = {"groups": [], "block_errors": [], "dtypes": set()}
    pending = []

    def block(q, k, v, **kwargs):
        actual = original_block(q, k, v, **kwargs)
        oracle = block_reference(q, k, v, scale=kwargs["softmax_scale"],
                                 causal=kwargs["block_kind"] is ring.RingBlockKind.CAUSAL,
                                 mask=kwargs.get("attention_mask"))
        pending.append(oracle)
        state["dtypes"].add(tuple(str(t.dtype) for t in actual))
        flat = [ring._flatten_tnd_softmax(t, (q.shape[0],)).detach().cpu().float()
                for t in actual[1:]]
        # Verify redundant statistic lanes, then compare log-normalizer (stable
        # across equivalent max/sum representations) as well as max and sum.
        for stat in flat:
            assert torch.equal(stat, stat[..., :1].expand_as(stat)), "non-replicated softmax lanes"
        left, maximum, total = oracle
        right = actual[0].detach().cpu().float()
        state["block_errors"].append({
            "output_rel": float(torch.linalg.vector_norm(right-left) / torch.linalg.vector_norm(left).clamp_min(1e-30)),
            "max_abs": float((flat[0][..., 0]-maximum).abs().max()),
            "sum_rel": float(torch.linalg.vector_norm(flat[1][..., 0]-total) / torch.linalg.vector_norm(total).clamp_min(1e-30)),
            "lse_abs": float((flat[0][..., 0]+flat[1][..., 0].log()-maximum-total.log()).abs().max()),
        })
        return actual

    def merge(previous, current, *, query_length):
        actual = original_merge(previous, current, query_length=query_length)
        assert len(pending) == 1
        output, maximum, total = pending.pop()
        if previous is None:
            state["groups"].append({"fp32": None, "oracle": None, "count": 0})
        group = state["groups"][-1]
        packed = [output.to(current[0].device)]
        for value, template in zip((maximum, total), current[1:]):
            # Independent inverse of NPU's head-major softmax storage.
            lanes = template.shape[-1]
            packed.append(value.T.contiguous()[..., None].expand(-1, -1, lanes)
                          .reshape(template.shape).contiguous().to(template.device))
        group["fp32"] = original_merge(group["fp32"], tuple(t.float() for t in current), query_length=query_length)
        group["oracle"] = original_merge(group["oracle"], tuple(packed), query_length=query_length)
        group["actual"] = actual[0].detach().cpu().float()
        group["count"] += 1
        return actual

    with patch.object(ring, "_block_attention_forward", block), patch.object(ring, "_merge_attention", merge):
        yield state
    assert not pending


def report_ring_merge(state, expected, report, rank, dtype):
    groups = state["groups"]
    assert len(groups) == 2, "expected two native zigzag query chunks"
    def joined(name):
        return torch.cat([g[name][0].detach().cpu() for g in groups]).reshape_as(expected)
    fp32, oracle = joined("fp32"), joined("oracle")
    actual = torch.cat([g["actual"] for g in groups]).reshape_as(expected)
    report("merge/FP32-reference-vs-native", expected, actual)
    report("merge/FP32-reference-vs-oracle-blocks-native-merge", expected, oracle)
    report("merge/FP32-reference-vs-actual-blocks-FP32-merge", expected, fp32)
    report("merge/FP32-reference-vs-actual-blocks-FP32-merge-final-cast", expected, fp32.to(dtype))
    errors = state["block_errors"]
    worst = {key: max(e[key] for e in errors) for key in errors[0]}
    print(f"STAGE-4.4 RING-MERGE r={rank} L04 blocks/query={[g['count'] for g in groups]} "
          f"actual(output,max,sum) dtypes={sorted(state['dtypes'])} "
          f"block worst={worst}; physical launched masks used; "
          "fully-masked rows=0; forward shadows only, no backward changes", flush=True)
