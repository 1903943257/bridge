"""CPU-only compare exact Trie vs Qwen3 Turn-constrained tree on captured PPO.

No NPU, no checkpoint loading. Fails if either tree loses any original PPO
(query,target,response_offset) supervision or exact trajectory reconstruction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from verl.utils import tensordict_utils as tu
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans


def _supervision(forest):
    refs = []
    for entry in forest.trees:
        for ref in entry.objective_refs:
            segment = entry.segment_plan.get(ref.segment_id)
            refs.append((
                ref.sample_row, ref.response_offset, ref.target_token_id,
                segment.position_start + ref.query_offset,
            ))
    return sorted(refs)


def _reconstruct(forest, batch):
    seen = set()
    for entry in forest.trees:
        tree = entry.tree

        def walk(nid, prefix):
            node = tree.get(nid)
            span = node.segment
            part = batch["input_ids"][span.row][span.start:span.end].tolist()
            current = prefix + part
            for row in node.terminal_rows:
                if row in seen or current != batch["input_ids"][row].tolist():
                    raise AssertionError(f"row={row} Turn tree reconstruction fails")
                seen.add(row)
            for kid in node.children:
                walk(kid, current)

        walk(tree.root_id, [])
    if seen != set(range(len(batch["input_ids"]))):
        raise AssertionError(f"Turn tree missing terminal rows {set(range(len(batch['input_ids'])))-seen}")


def analyze(path: Path, rows: tuple[int,...], minimum: int):
    original = torch.load(path, map_location="cpu", weights_only=False)
    if original.get("capture_stage") != "actor_update_mini_batch":
        raise AssertionError("Expected actor update capture")
    if not rows:
        rows = tuple(range(len(original["keys"])))
    batch = tu.index_select_tensor_dict(original["tensordict"], list(rows))
    keys = tuple(original["keys"][i] for i in rows)
    exact = build_tree_execution_plans(keys, batch, tree_policy="exact")
    turned = build_tree_execution_plans(
        keys, batch, tree_policy="qwen_turn",
        turn_min_segment_tokens=minimum
    )
    _reconstruct(exact, batch)
    _reconstruct(turned, batch)
    if _supervision(exact) != _supervision(turned):
        raise AssertionError("Turn and exact PPO logical supervision differ")
    logical_tokens = sum(row.numel() for row in batch["input_ids"].unbind(0))
    tokens = sum(row.bool().sum().item() for row in batch["response_mask"].unbind(0))
    print(f"TPR_TURN_SOURCE rows={rows} logical_tokens={logical_tokens} PPO_tokens={tokens}")
    for i, tensor in enumerate(batch["input_ids"].unbind(0)):
        print(f"TPR_TURN_MARKERS row={rows[i]} length={tensor.numel()} "
              f"im_end_count={int((tensor == 151645).sum())} "
              f"prompt_length={int(batch['prompts'][i].numel())}")
    for policy, forest in (("exact", exact), ("qwen_turn", turned)):
        unique = 0
        forward = backward = calls = 0
        print(f"TPR_TURN_POLICY policy={policy} trees={len(forest.trees)} segments={forest.segment_count}")
        for entry in forest.trees:
            plan = entry.segment_plan
            for sid, segment in sorted(plan.segments.items()):
                push_pop = bool(plan.children_of(sid))
                unique += segment.length
                backward += segment.length
                forward += segment.length * (2 if push_pop else 1)
                calls += 2 if push_pop else 1
                print(f"  node={sid} parent={segment.parent_id} "
                      f"depth={plan.depth_of(sid)} "
                      f"length={segment.length} prefix={segment.prefix_length} "
                      f"kind={'PUSH_POP' if push_pop else 'LEAF'}")
        print(f"TPR_TURN_SUMMARY policy={policy} unique_tokens={unique} "
              f"forward_tokens={forward} backward_tokens={backward} "
              f"forward_calls={calls} backward_calls={forest.segment_count}")
    print("TPR_TURN_PPO_OWNERSHIP PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capture", type=Path)
    p.add_argument("--rows", default="0,1,2,4,6",
                   help="comma-separated original row ids; empty => all")
    p.add_argument("--min-segment-tokens", type=int, default=128)
    args = p.parse_args()
    analyze(args.capture,
            tuple(int(x) for x in args.rows.split(",") if x),
            args.min_segment_tokens)
