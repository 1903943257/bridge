"""Select the earliest available trajectory in each Uni-Agent Session.

Capture rows may be re-ordered by VERL actor minibatch shuffling. Prefer a
preserved TQ trajectory ordinal if present; otherwise select first encountered
row by chronological capture_index/batch_idx then row offset and annotate the
limitation. This is NOT guaranteed to be Gateway chronological "first" when
the original TQ trajectory suffix was lost during PPO data processing.

Per-session identities are (uid, session_id), *not* capture row / uid alone.
The selector is stateful across EVERY actor mini-batch in a global step.
No reward/advantage mutation is performed on historical captured PPO data.
"""
from __future__ import annotations

from collections.abc import Sequence

from verl.utils import tensordict_utils as tu

ORDINAL_FIELDS = ("trajectory_index", "trajectory_idx", "tq_trajectory_index")


def _column(batch, key: str, n: int, *, required: bool = False):
    raw = tu.get(batch, key, default=None)
    if raw is None:
        if required:
            raise ValueError(
                f"offline session-first selection requires original '{key}' "
                "per row, refusing to confuse several sessions with one UID"
            )
        return None
    if hasattr(raw, "tolist"):
        v = raw.tolist()
    elif hasattr(raw, "unbind"):
        v = list(raw.unbind(0))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str,bytes)):
        v = list(raw)
    else:
        v = tu.unwrap_non_tensor_data(raw)
    if not isinstance(v, list):
        v = [v] * n
    if len(v) != n:
        raise ValueError(f"{key} row count {len(v)} differs from {n}")
    result = []
    for entry in v:
        value = tu.unwrap_non_tensor_data(entry)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(value)
    return result


def session_rows(batch):
    n = len(batch["input_ids"])
    uids = _column(batch,"uid",n,required=True)
    sessions = _column(batch,"session_id",n,required=True)
    if any(u is None or str(u)=="" for u in uids):
        raise ValueError("missing uid row for session-first selection")
    if any(s is None or str(s)=="" for s in sessions):
        raise ValueError("missing session_id row for session-first selection")
    ordinals = None
    field = None
    for name in ORDINAL_FIELDS:
        candidate = _column(batch,name,n)
        if candidate is not None:
            if ordinals is not None:
                raise ValueError("multiple trajectory ordinal columns in capture")
            ordinals = candidate
            field = name
    if ordinals is not None:
        if any(type(x) is not int or x < 0 for x in ordinals):
            raise ValueError(f"{field} trajectory indices must be nonnegative int")
    identities = [(str(u),str(s)) for u,s in zip(uids,sessions)]
    return identities, ordinals, field


def first_session_indices(ordered_batches):
    """Return per-capture row indices to keep plus selection evidence.

    Input iterable contains (path_string, batch) in capture chronological order.
    If an explicit trajectory ordinal exists, select smallest original ordinal
    globally; otherwise the first encountered row. All subsequent occurrences
    of the same Session are dropped, even across actor mini-batches.
    """
    candidates = {}
    counts = {}
    descriptions = []
    padding_rows = 0
    for capture_order, (path,batch) in enumerate(ordered_batches):
        ids, ordinals, field = session_rows(batch)
        descriptions.append({
            "capture":str(path),
            "row_count":len(ids),
            "trajectory_ordinal_field":field,
        })
        for row,identity in enumerate(ids):
            # VERL adds pad<random> synthetic rows to complete Actor mini-batches.
            # These are not real Uni-Agent Sessions and must not be selected.
            if identity[0].startswith("pad"):
                padding_rows += 1
                continue
            counts[identity] = counts.get(identity, 0) + 1
            if ordinals is None:
                order=(1,capture_order,row)
            else:
                order=(0,int(ordinals[row]),capture_order,row)
            previous=candidates.get(identity)
            if previous is None or order < previous[0]:
                candidates[identity]=(order,capture_order,row,field)
    kept_by_capture=[[] for _ in ordered_batches]
    exact=0
    inferred=0
    for identity,(order,ci,row,field) in candidates.items():
        kept_by_capture[ci].append(row)
        if field is None:
            inferred+=1
        else:
            exact+=1
    for indices in kept_by_capture:
        indices.sort()
    return kept_by_capture,{
        "sessions_seen":len(candidates),
        "excluded_padding_rows":padding_rows,
        "sessions_with_multiple_captured_trajectories":sum(
            1 for count in counts.values() if count > 1
        ),
        "discarded_trajectory_rows":sum(counts.values())-len(candidates),
        "original_trajectory_ordinal_present":exact,
        "capture_first_fallback_sessions":inferred,
        "input_capture_descriptions":descriptions,
        "selection_semantics":"first_available_capture_order_when_original_ordinal_missing",
    }
