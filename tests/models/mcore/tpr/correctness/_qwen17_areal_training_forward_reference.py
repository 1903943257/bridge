"""AReaL-DTA training loss-side Forward reproduction (no actual backward).

Independent Apache-2.0 adaptation of AReaL feat/dta at
a5b0b4811a3ef7bf58f0270abcd81d7154f03ce6.
Mirrors TokenTrie.backward_permute and DTAEngine.backward's
push/cache_len/build_cache/pop_byblock/pop Forward scheduling.
Push model calls run under no_grad; Pop model calls run with
autograd enabled and detached, requires_grad prefix K/V.
No torch.autograd.backward, parameter grads, or gradient relay.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import torch

from ._qwen17_areal_dta_reference import (
    _areal_forward_order, ARealForwardPlan, areal_forward_plan,
)
from ._qwen17_dta_style_reference import (
    _cache_to_pairs, _logp_from_previous_logits, longest_common_prefix,
)


def areal_backward_plan(rows) -> ARealForwardPlan:
    """TokenTrie.backward_permute: leafize, priority DFS, reverse order."""
    lexical = areal_forward_plan(rows, forward_permute=False)
    lens = [len(x) for x in lexical.sequences]
    order = _areal_forward_order(lens, lexical.lcp_lens, backward=True)
    seqs = tuple(lexical.sequences[i] for i in order)
    return ARealForwardPlan(
        sequences=seqs,
        attachments=tuple(lexical.attachments[i] for i in order),
        lcp_lens=tuple(longest_common_prefix(a, b)
                       for a, b in zip(seqs, seqs[1:])),
    )


def _fork_positions(lens, lcps, block_size: int):
    """Exact AReaL _get_forkpos(lens, lcp_lens, block_size)."""
    pos = {lcp - 1 for lcp in lcps if lcp > 0}
    for i, end in enumerate(lens):
        start = 0 if i == len(lcps) else lcps[i]
        pop_len = end - start
        n_blocks = ceil(pop_len / block_size)
        actual = ceil(pop_len / n_blocks)
        for b in range(n_blocks):
            pop_start = max(end - (b + 1) * actual, start)
            if pop_start > 0:
                pos.add(pop_start - 1)
    return tuple(sorted(pos))


@dataclass(frozen=True)
class TrainingForwardResult:
    logprobs: tuple[torch.Tensor, ...]
    token_owner: tuple[tuple[int, ...], ...]
    events: tuple[tuple[str, int, int, int], ...]  # kind, start, end, leaf
    physical_m_cache: tuple[int, ...]
    physical_m_pop: tuple[int, ...]
    pop_starts: tuple[int, ...]
    n_leaves: int
    n_cache_forwards: int
    n_pop_forwards: int
    block_size: int
    cut_f1_tail: bool


def hf_areal_training_loss_forward(
    model, token_rows, cache_factory, *, block_size: int = 64,
    cut_f1_tail: bool = True, max_seq_len: int | None = None,
) -> TrainingForwardResult:
    """Reconstruct exactly the Forward logprobs passed to loss in each Pop.

    This is a numeric oracle, NOT an implementation of backward/gradients.
    """
    if block_size == -1:
        block_size = int(1e9)
    if block_size <= 0:
        raise ValueError("block_size must be positive or -1")
    rows = tuple(token_rows)
    plan = areal_backward_plan(rows)
    max_len = max(map(len, plan.sequences))
    max_seq_len = max_len if max_seq_len is None else max_seq_len
    if max_seq_len < max_len:
        raise ValueError("max_seq_len too short")
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or
                   cfg.hidden_size // cfg.num_attention_heads)
    dtype = next(model.parameters()).dtype
    device = rows[0].device
    kv_k = [torch.zeros((1, n_heads, max_seq_len, head_dim),
                        dtype=dtype, device=device) for _ in range(n_layers)]
    kv_v = [torch.zeros_like(k) for k in kv_k]
    tokens = torch.empty(max_seq_len, dtype=rows[0].dtype, device=device)
    logprobs = torch.zeros(max_seq_len, dtype=torch.float32, device=device)
    logp_owner = [-1] * max_seq_len
    fork_pos = _fork_positions(
        [len(x) for x in plan.sequences], plan.lcp_lens, block_size)
    fork_logits = {p: None for p in fork_pos}
    fork_owner = {p: -1 for p in fork_pos}
    outputs = [None] * len(rows)
    output_owner = [None] * len(rows)
    pending = []  # (original row, original sequence length)
    events = []
    pop_m, cache_m, pop_starts = [], [], []
    cur_len = 0
    current_leaf = -1

    def build_prefix_cache(start, with_grad):
        cache = cache_factory()
        prefix_kv = []
        for layer in range(n_layers):
            k = kv_k[layer][:, :, :start, :]
            v = kv_v[layer][:, :, :start, :]
            if with_grad:
                k = k.detach().requires_grad_(True)
                v = v.detach().requires_grad_(True)
            cache.update(k, v, layer_idx=layer)
            prefix_kv.append((k, v))
        return cache, prefix_kv

    def run_model(start, end, *, pop):
        if end <= start:
            raise AssertionError("empty model Forward")
        if pop:
            with torch.enable_grad():
                cache, prefix_kv = build_prefix_cache(start, True)
                out = model(input_ids=tokens[start:end].unsqueeze(0),
                            past_key_values=cache, use_cache=True)
        else:
            with torch.no_grad():
                cache, prefix_kv = build_prefix_cache(start, False)
                out = model(input_ids=tokens[start:end].unsqueeze(0),
                            past_key_values=cache, use_cache=True)
        pairs = _cache_to_pairs(out.past_key_values)
        if len(pairs) != n_layers or any(k.shape[-2] != end for k, v in pairs):
            raise AssertionError("model cache sequence length mismatch")
        if tuple(out.logits.shape[:2]) != (1, end - start):
            raise AssertionError("model logits have unexpected shape")
        events.append(("POP" if pop else "CACHE", start, end, current_leaf))
        return out.logits[0], pairs

    def build_cache(start, end):
        if not start < end <= max_seq_len:
            raise AssertionError("invalid cache interval")
        logits, pairs = run_model(start, end, pop=False)
        with torch.no_grad():
            for layer, (k, v) in enumerate(pairs):
                kv_k[layer][:, :, start:end, :].copy_(k[:, :, start:end, :])
                kv_v[layer][:, :, start:end, :].copy_(v[:, :, start:end, :])
            if end - start > 1:
                logprobs[start:end - 1] = _logp_from_previous_logits(
                    logits[:-1], tokens[start + 1:end])
                logp_owner[start:end - 1] = [len(events) - 1] * (end - start - 1)
            for pos in fork_pos:
                if start <= pos < end:
                    fork_logits[pos] = logits[pos - start].detach().clone()
                    fork_owner[pos] = len(events) - 1
        cache_m.append(end - start)

    def push(new_tokens, attachments, cache_len):
        nonlocal cur_len
        start, end = cur_len, cur_len + len(new_tokens)
        # The AReaL scheduler permits cache_len < start: no new KV is built.
        if not 0 <= cache_len <= end:
            raise AssertionError("invalid push cache_len")
        with torch.no_grad():
            tokens[start:end] = new_tokens
            pending.extend(attachments)
            if start < cache_len:
                build_cache(start, cache_len)
            if start > 0:
                mid_logits = fork_logits[start - 1]
                if mid_logits is None:
                    raise AssertionError(f"push missing fork logit {start-1}")
                logprobs[start - 1] = _logp_from_previous_logits(
                    mid_logits.float().view(1, -1), tokens[start:start + 1])[0]
                logp_owner[start - 1] = fork_owner[start - 1]
        cur_len = end

    def pop(start):
        nonlocal cur_len, pending
        end = cur_len
        if not 0 <= start < end:
            raise AssertionError("invalid pop interval")
        with torch.enable_grad():
            logits, _pairs = run_model(start, end, pop=True)
            event_id = len(events) - 1
            suffix = (_logp_from_previous_logits(
                logits[:-1], tokens[start + 1:end])
                if end - start > 1 else
                torch.empty(0, dtype=torch.float32, device=device))
            if start:
                previous_logits = fork_logits.get(start - 1)
                if previous_logits is None:
                    raise AssertionError(f"pop missing fork logit {start-1}")
                middle = previous_logits.float().detach().requires_grad_(True)
                mid_logprob = _logp_from_previous_logits(
                    middle.view(1, -1), tokens[start:start + 1])
                if start > 1:
                    prefix = logprobs[:start - 1].detach().requires_grad_(True)
                    merged = torch.cat((prefix, mid_logprob, suffix))
                    owners = (tuple(logp_owner[:start - 1]) +
                              (fork_owner[start - 1],) +
                              (event_id,) * len(suffix))
                else:
                    merged = torch.cat((mid_logprob, suffix))
                    owners = (fork_owner[start - 1],) + (event_id,) * len(suffix)
            else:
                merged = suffix
                owners = (event_id,) * len(suffix)
            # Upstream loss_fn receives logprobs[:length-1] for terminal
            # attachments in the current pop, not the cached push outputs.
            for row_id, length in pending:
                if start < length <= end:
                    if outputs[row_id] is not None:
                        raise AssertionError("duplicate terminal attachment")
                    outputs[row_id] = merged[:length - 1].detach().clone()
                    output_owner[row_id] = owners[:length - 1]
        with torch.no_grad():
            pending = [(r, length) for r, length in pending if length <= start]
            for pos in fork_pos:
                if start <= pos < end:
                    fork_logits[pos] = None
                    fork_owner[pos] = -1
        pop_m.append(end - start)
        pop_starts.append(start)
        cur_len = start

    def pop_byblock(start):
        if not 0 <= start < cur_len:
            raise AssertionError("invalid pop_byblock interval")
        end = cur_len
        length = end - start
        n_blocks = ceil(length / block_size)
        actual = ceil(length / n_blocks)
        for b in range(n_blocks):
            pop(max(end - (b + 1) * actual, start))

    for i, (ids, attachments) in enumerate(
        zip(plan.sequences, plan.attachments, strict=True)
    ):
        current_leaf = i
        if i:
            lcp = plan.lcp_lens[i - 1]
            if lcp < cur_len:
                pop_byblock(lcp)
            if cur_len != lcp:
                raise AssertionError("pop did not reach LCP")
        new_tokens = ids[cur_len:]
        lcp_next = plan.lcp_lens[i] if i < len(plan.sequences) - 1 else 0
        new_len = len(new_tokens)
        next_pop_len = cur_len + new_len - lcp_next
        if next_pop_len > block_size:
            n_blocks = ceil(next_pop_len / block_size)
            actual = ceil(next_pop_len / n_blocks)
            cache_len = max(cur_len + new_len - actual, lcp_next)
        else:
            cache_len = lcp_next
        if not cut_f1_tail:
            cache_len = cur_len + new_len
        push(new_tokens, attachments, cache_len)
    if cur_len > 0:
        pop_byblock(0)
    if pending or cur_len != 0 or any(x is None for x in outputs):
        raise AssertionError("incomplete DTA training stack/attachment")
    for result, ids in zip(outputs, rows, strict=True):
        if result.shape != (len(ids) - 1,) or not torch.isfinite(result).all():
            raise AssertionError("invalid final loss logprobs")
    if any(any(v < 0 for v in owners) for owners in output_owner):
        raise AssertionError("unowned logprob: expected an actual CACHE/POP Forward")
    return TrainingForwardResult(
        logprobs=tuple(outputs),
        token_owner=tuple(output_owner),
        events=tuple(events),
        physical_m_cache=tuple(cache_m),
        physical_m_pop=tuple(pop_m),
        pop_starts=tuple(pop_starts),
        n_leaves=len(plan.sequences),
        n_cache_forwards=len(cache_m),
        n_pop_forwards=len(pop_m),
        block_size=block_size,
        cut_f1_tail=cut_f1_tail,
    )
