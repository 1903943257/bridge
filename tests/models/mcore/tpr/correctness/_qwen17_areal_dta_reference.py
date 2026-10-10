"""Forward-only numerical reproduction of AReaL-DTA's HF DynamicCache execution.

Algorithm adapted from AReaL feat/dta commit a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6
(Apache-2.0): experimental/dta/{trie,token_trie,dta_engine}.py.
Preserves leafization, optimized forward permutation, fixed KV buffers and
fork-position logprobs. NOT DTA pop/backward or a production implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from ._qwen17_dta_style_reference import (
    _cache_to_pairs, _logp_from_previous_logits, longest_common_prefix,
)


@dataclass
class _Node:
    depth: int
    seq_id: int = -1
    chain_tail_depth: int = 0
    child_ids: list[int] = field(default_factory=list)


def _areal_forward_order(lens: list[int], lcps: list[int]) -> list[int]:
    """Port of AReaL CompressedTrie.get_order_forward (not lexical order)."""
    if len(lcps) != max(len(lens) - 1, 0):
        raise ValueError("invalid LCP shape")
    nodes = [_Node(depth=0)]
    def new_node(depth, seq_id=-1):
        nodes.append(_Node(depth=depth, seq_id=seq_id))
        return len(nodes) - 1
    stack = [(0, 0)]
    for seq_id, size in enumerate(lens):
        lcp = lcps[seq_id - 1] if seq_id else 0
        if len(stack) >= 2:
            while stack[-2][1] > lcp:
                child = stack.pop()[0]
                nodes[stack[-1][0]].child_ids.append(child)
            child = stack.pop()[0]
            if stack[-1][1] < lcp:
                stack.append((new_node(lcp), lcp))
            nodes[stack[-1][0]].child_ids.append(child)
        elif stack[-1][1] < lcp:
            stack.append((new_node(lcp), lcp))
        stack.append((new_node(size, seq_id), size))
    while len(stack) >= 2:
        child = stack.pop()[0]
        nodes[stack[-1][0]].child_ids.append(child)

    def child_order(idx):
        return sorted(nodes[idx].child_ids, key=lambda j: nodes[j].chain_tail_depth)
    def chain(idx):
        node = nodes[idx]
        if node.seq_id != -1:
            node.chain_tail_depth = node.depth
            return
        for child in node.child_ids:
            chain(child)
        children = child_order(idx)
        node.chain_tail_depth = nodes[children[0]].chain_tail_depth if children else node.depth
    order = []
    def visit(idx):
        if nodes[idx].seq_id != -1:
            order.append(nodes[idx].seq_id)
        else:
            for child in child_order(idx):
                visit(child)
    chain(0)
    visit(0)
    if sorted(order) != list(range(len(lens))):
        raise AssertionError("CompressedTrie omitted or duplicated leaves")
    return order


@dataclass(frozen=True)
class ARealForwardPlan:
    sequences: tuple[torch.Tensor, ...]
    attachments: tuple[tuple[tuple[int, int], ...], ...]  # original row, length
    lcp_lens: tuple[int, ...]


def areal_forward_plan(rows, *, forward_permute: bool = True) -> ARealForwardPlan:
    """Port of AReaL TokenTrie leafization and forward_permute."""
    rows = tuple(rows)
    if not rows or any(x.ndim != 1 or x.numel() < 2 for x in rows):
        raise ValueError("need nonempty 1-D rows of at least two tokens")
    if len({x.device for x in rows}) != 1:
        raise ValueError("rows must be on the same device")
    indices = sorted(range(len(rows)), key=lambda i: rows[i].tolist())
    sorted_rows = [rows[i] for i in indices]
    lcps = [longest_common_prefix(a, b)
            for a, b in zip(sorted_rows, sorted_rows[1:])]
    leaves, attachments, leaf_lcps = [], [], []
    fork = -1
    for i, tokens in enumerate(sorted_rows):
        if i == len(sorted_rows) - 1 or lcps[i] < min(tokens.numel(), sorted_rows[i+1].numel()):
            leaves.append(tokens)
            attachments.append(tuple((indices[k], int(sorted_rows[k].numel()))
                                     for k in range(fork + 1, i + 1)))
            if i < len(sorted_rows) - 1:
                leaf_lcps.append(lcps[i])
            fork = i
    order = (_areal_forward_order([len(t) for t in leaves], leaf_lcps)
             if forward_permute else list(range(len(leaves))))
    sequences = [leaves[i] for i in order]
    return ARealForwardPlan(
        sequences=tuple(sequences),
        attachments=tuple(attachments[i] for i in order),
        lcp_lens=tuple(longest_common_prefix(a, b)
                       for a, b in zip(sequences, sequences[1:])),
    )


@dataclass(frozen=True)
class ARealForwardResult:
    logprobs: tuple[torch.Tensor, ...]
    physical_m: tuple[int, ...]
    physical_starts: tuple[int, ...]
    physical_rows: tuple[tuple[int, ...], ...]
    token_owner: tuple[tuple[int, ...], ...]
    total_processed_tokens: int
    dense_tokens: int
    n_leaves: int


@torch.no_grad()
def hf_areal_forward_only(
    model, token_rows, cache_factory, *, forward_permute: bool = True
) -> ARealForwardResult:
    """DTAEngine.forward + push_forward_only, including persistent KV views."""
    rows = tuple(token_rows)
    plan = areal_forward_plan(rows, forward_permute=forward_permute)
    config = model.config
    layers = int(config.num_hidden_layers)
    heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", None) or
                   config.hidden_size // config.num_attention_heads)
    longest = max(map(len, plan.sequences))
    dtype = next(model.parameters()).dtype
    device = rows[0].device
    keys = [torch.zeros((1, heads, longest, head_dim), device=device, dtype=dtype)
            for _ in range(layers)]
    values = [torch.zeros_like(k) for k in keys]
    logps = torch.zeros(longest - 1, dtype=torch.float32, device=device)
    owners = [-1] * (longest - 1)
    outputs = [None] * len(rows)
    output_owners = [None] * len(rows)
    fork_positions = {lcp - 1 for lcp in plan.lcp_lens if lcp > 0}
    fork_logits = {}
    physical_m, starts, physical_rows = [], [], []

    for visit, (tokens, attachments) in enumerate(
        zip(plan.sequences, plan.attachments, strict=True)
    ):
        start = 0 if visit == 0 else plan.lcp_lens[visit-1]
        end = int(tokens.numel())
        if not 0 <= start <= end:
            raise AssertionError("invalid DTA prefix")
        if start < end:
            cache = cache_factory()
            # AReaL initializes all cache layers even when start is zero.
            for layer in range(layers):
                cache.update(keys[layer][:, :, :start, :],
                             values[layer][:, :, :start, :], layer_idx=layer)
            out = model(input_ids=tokens[start:end].view(1, -1),
                        past_key_values=cache, use_cache=True)
            if out.logits.shape[:2] != (1, end - start):
                raise AssertionError("unexpected HF logits shape")
            pairs = _cache_to_pairs(out.past_key_values)
            if len(pairs) != layers:
                raise AssertionError("incorrect layer count")
            for layer, (k, v) in enumerate(pairs):
                if k.shape[-2] != end or v.shape[-2] != end:
                    raise AssertionError("incorrect HF KV length")
                keys[layer][:, :, start:end, :].copy_(k[:, :, start:end, :])
                values[layer][:, :, start:end, :].copy_(v[:, :, start:end, :])
            logits = out.logits[0]
            if end - start > 1:
                logps[start:end-1] = _logp_from_previous_logits(
                    logits[:-1], tokens[start+1:end])
                owners[start:end-1] = [visit] * (end-start-1)
            if start:
                if start - 1 not in fork_logits:
                    raise AssertionError(f"missing fork logit at {start-1}")
                logps[start-1] = _logp_from_previous_logits(
                    fork_logits[start-1].float().view(1, -1), tokens[start:start+1])[0]
                owners[start-1] = visit
            for pos in fork_positions:
                if start <= pos < end:
                    fork_logits[pos] = logits[pos-start].detach().clone()
            physical_m.append(end-start)
            starts.append(start)
            physical_rows.append(tuple(row_id for row_id, _ in attachments))
        for row_id, length in attachments:
            if outputs[row_id] is not None:
                raise AssertionError("duplicate row attachment")
            outputs[row_id] = logps[:length-1].clone()
            output_owners[row_id] = tuple(owners[:length-1])
    if any(v is None for v in outputs) or any(v is None for v in output_owners):
        raise AssertionError("missing logical trajectory")
    if any(not bool(torch.isfinite(v).all()) for v in outputs):
        raise AssertionError("nonfinite logprobs")
    return ARealForwardResult(
        logprobs=tuple(outputs),
        physical_m=tuple(physical_m),
        physical_starts=tuple(starts),
        physical_rows=tuple(physical_rows),
        token_owner=tuple(output_owners),
        total_processed_tokens=sum(physical_m),
        dense_tokens=sum(int(x.numel()) for x in rows),
        n_leaves=len(plan.sequences),
    )
