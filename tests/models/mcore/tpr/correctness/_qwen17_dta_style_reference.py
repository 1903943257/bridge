"""Independent, HF DynamicCache-based DTA *forward* control.

A small experimental port of the *forward-only* part of
areal/experimental/dta/dta_engine.py at AReaL feat/dta.  Not a copy of our
TPR SegmentExecutor, KVStack or Megatron TPRSelfAttention.  This uses the
real HF model/past_key_values protocol and explicitly retains LCP KV across
consecutive DFS leaves, including the preceding-token logit at each fork.

This does NOT implement DTA's Pop/backward KV-gradient relay; it MUST NOT be
used to claim DTA training equivalence. Never use in production TPR code.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DTAForwardResult:
    # One 1-D [num_tokens-1] tensor per ORIGINAL input trajectory.
    logprobs: tuple[torch.Tensor, ...]
    physical_m: tuple[int, ...]
    physical_starts: tuple[int, ...]
    total_processed_tokens: int
    dense_tokens: int


def longest_common_prefix(a: torch.Tensor, b: torch.Tensor) -> int:
    """Same adjacent-token LCP semantics as AReaL-DTA TokenTrie."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("DTA token rows must be 1-D")
    length = min(a.numel(), b.numel())
    if length == 0:
        return 0
    equal = a[:length].cpu() == b[:length].cpu()
    bad = torch.nonzero(~equal).flatten()
    return int(bad[0]) if bad.numel() else length


def _cache_to_pairs(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Support modern HF DynamicCache.layers and older key_cache variants."""
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache, strict=True))
    raise TypeError(
        "Unsupported transformers DynamicCache layout; inspect installed "
        "transformers version instead of changing numerical semantics"
    )


def _make_prefix_cache(cache_factory, pairs, keep: int):
    cache = cache_factory()
    if keep < 0:
        raise ValueError("negative prefix length")
    if not keep:
        return cache
    for idx, (key, value) in enumerate(pairs):
        if key.shape[-2] < keep or value.shape[-2] < keep:
            raise AssertionError("Prefix KV is shorter than the requested LCP")
        # Detached because this first-stage experiment measures forward drift.
        # DTA proper reconnects these tensors via KV grad injections in pop().
        cache.update(
            key[..., :keep, :].detach(),
            value[..., :keep, :].detach(),
            idx,
        )
    return cache


def _run_chunk(model, tokens, start: int, end: int, cache):
    if not (0 <= start < end <= tokens.numel()):
        raise ValueError(f"Invalid physical chunk [{start}:{end}]")
    out = model(
        input_ids=tokens[start:end].view(1, -1),
        past_key_values=cache,
        use_cache=True,
    )
    if out.logits.ndim != 3 or out.logits.shape[:2] != (1, end - start):
        raise AssertionError("HF DTA forward returned unexpected logits shape")
    if out.past_key_values is None:
        raise AssertionError("HF DTA requires use_cache=True with returned KV")
    pairs = _cache_to_pairs(out.past_key_values)
    if not pairs or any(pair[0].shape[-2] != end for pair in pairs):
        raise AssertionError("HF DTA returned KV with unexpected total length")
    return out.logits[0], pairs


def _logp_from_previous_logits(logits: torch.Tensor, next_tokens: torch.Tensor):
    if logits.ndim != 2 or next_tokens.ndim != 1:
        raise AssertionError("Expected [M,vocab] logits and [M] labels")
    if logits.shape[0] != next_tokens.numel():
        raise AssertionError("shifted token target alignment is invalid")
    return (
        logits.float()
        .log_softmax(dim=-1)
        .gather(dim=-1, index=next_tokens.long().unsqueeze(-1))
        .squeeze(-1)
    )


@torch.no_grad()
def hf_full_logprobs(model, tokens, cache_factory):
    """Full-row control using the exact HF cache-enabled call site."""
    if tokens.ndim != 1 or tokens.numel() < 2:
        raise ValueError("Need >=2 token IDs")
    logits, _ = _run_chunk(model, tokens, 0, tokens.numel(), cache_factory())
    return _logp_from_previous_logits(logits[:-1], tokens[1:])


@torch.no_grad()
def hf_cached_chunk_logprobs(model, tokens, boundaries, cache_factory):
    """Independent HF KV-cache forward with *fixed* physical chunk boundaries.

    Unlike lexicographic DFS, this probes the exact [0:134]...[189:192]
    path of a chosen TPR leaf without importing any Megatron components.
    """
    boundaries = tuple(int(x) for x in boundaries)
    if len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != tokens.numel():
        raise ValueError("Chunk boundaries must cover [0, len(tokens)]")
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Chunk boundaries must strictly increase")
    cache = cache_factory()
    result = torch.empty(tokens.numel() - 1, dtype=torch.float32, device=tokens.device)
    previous_tail = None
    for start, end in zip(boundaries, boundaries[1:]):
        logits, _ = _run_chunk(model, tokens, start, end, cache)
        if start:
            if previous_tail is None:
                raise AssertionError("Missing fork token logit")
            result[start - 1] = _logp_from_previous_logits(
                previous_tail.view(1, -1), tokens[start:start + 1]
            )[0]
        if end - start > 1:
            result[start:end - 1] = _logp_from_previous_logits(
                logits[:-1], tokens[start + 1:end]
            )
        previous_tail = logits[-1].detach().clone()
    if not bool(torch.isfinite(result).all()):
        raise AssertionError("HF cached logprobs contain NaN/Inf")
    return result


@torch.no_grad()
def hf_dta_lcp_forward(model, token_rows, cache_factory) -> DTAForwardResult:
    """DTA-style shared-prefix DFS over lexicographically sorted real rows.

    This follows AReaL's forward-only LCP/Prefix DynamicCache/Push semantics.
    It intentionally does NOT import that library or our TPR, and does not
    implement its optimized forward_permute (ordering impacts shapes/throughput).
    A fully duplicate row is returned with its cached logprobs, no extra call.
    """
    rows = tuple(x.contiguous() for x in token_rows)
    if not rows or any(x.ndim != 1 or x.numel() < 2 for x in rows):
        raise ValueError("Need nonempty 1-D token rows of >=2 tokens")
    if len({x.device for x in rows}) != 1:
        raise ValueError("DTA rows must share the same device")
    order = sorted(range(len(rows)), key=lambda i: (rows[i].tolist(), i))
    ordered = [rows[i] for i in order]
    lcps = [
        longest_common_prefix(a, b)
        for a, b in zip(ordered, ordered[1:])
    ]
    # AReaL caches the last logit at each branch (fork) position rather than
    # keeping [S,vocab] logits of all previous prefixes.
    fork_positions = {lcp - 1 for lcp in lcps if lcp > 0}
    fork_logits: dict[int, torch.Tensor] = {}
    outputs: list[torch.Tensor | None] = [None] * len(rows)
    cache_pairs = ()
    last_tokens = None
    last_logp = None
    lengths, starts = [], []
    for idx, original_idx in enumerate(order):
        tokens = rows[original_idx]
        start = 0 if idx == 0 else longest_common_prefix(last_tokens, tokens)
        # AReaL can permute visits to reduce recomputation; this minimal
        # lexicographic DFS still has the same PrefixCache/Pop semantics.
        for pos in list(fork_logits):
            if pos >= start:
                del fork_logits[pos]
        if start == tokens.numel():
            if last_logp is None:
                raise AssertionError("Duplicate sequence without cached logprobs")
            out_logp = last_logp.clone()
            new_pairs = cache_pairs
        else:
            cache = _make_prefix_cache(cache_factory, cache_pairs, start)
            logits, new_pairs = _run_chunk(
                model, tokens, start, tokens.numel(), cache
            )
            out_logp = torch.empty(
                tokens.numel() - 1, dtype=torch.float32, device=tokens.device
            )
            if start > 1:
                out_logp[:start - 1] = last_logp[:start - 1]
            if start:
                if start - 1 not in fork_logits:
                    raise AssertionError(
                        f"Missing fork logit at {start-1}; LCP cache traversal invalid"
                    )
                out_logp[start - 1] = _logp_from_previous_logits(
                    fork_logits[start - 1].view(1, -1),
                    tokens[start:start + 1],
                )[0]
            if tokens.numel() - start > 1:
                out_logp[start:tokens.numel() - 1] = _logp_from_previous_logits(
                    logits[:-1], tokens[start + 1:]
                )
            for position in fork_positions:
                if start <= position < tokens.numel():
                    fork_logits[position] = logits[position - start].detach().clone()
            lengths.append(tokens.numel() - start)
            starts.append(start)
        if not bool(torch.isfinite(out_logp).all()):
            raise AssertionError(f"DTA output nonfinite for original row {original_idx}")
        outputs[original_idx] = out_logp.clone()
        cache_pairs = new_pairs
        last_logp, last_tokens = out_logp, tokens
    if any(result is None for result in outputs):
        raise AssertionError("DTA DFS omitted a logical sequence")
    return DTAForwardResult(
        logprobs=tuple(outputs),
        physical_m=tuple(lengths),
        physical_starts=tuple(starts),
        total_processed_tokens=sum(lengths),
        dense_tokens=sum(x.numel() for x in rows),
    )


def error_summary(reference, observed):
    """Report distribution, not just mean loss, for the whole logical tree."""
    if len(reference) != len(observed):
        raise AssertionError("Reference/DTA logical row count differs")
    rows = []
    for i, (ref, actual) in enumerate(zip(reference, observed, strict=True)):
        if ref.shape != actual.shape:
            raise AssertionError(f"row{i}: output logprob shape mismatch")
        diff = (ref.float() - actual.float()).abs()
        if not bool(torch.isfinite(diff).all()):
            raise AssertionError("nonfinite logprob delta")
        worst_idx = int(diff.argmax())
        rows.append({
            "row": i,
            "count": int(diff.numel()),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "worst_query": worst_idx,
        })
    all_diff = torch.cat([
        (r.float() - a.float()).abs().detach().cpu()
        for r, a in zip(reference, observed, strict=True)
    ])
    return rows, {
        "max_abs": float(all_diff.max()),
        "mean_abs": float(all_diff.mean()),
        "p95_abs": float(torch.quantile(all_diff, 0.95)),
        "num_tokens": int(all_diff.numel()),
        "num_gt_0p2": int((all_diff > 0.2).sum()),
    }
