import os
from pathlib import Path

import pytest
import torch

from verl.models.mcore.tpr.trajectory_tree import build_trajectory_trees


_DEFAULT_TQ_DUMP = Path(
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt"
)


def _dump_path() -> Path:
    return Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ_DUMP)))


def _rows(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        return list(unbind())
    raise TypeError(f"value does not expose row-wise tensors: {type(value)!r}")


@pytest.fixture(scope="module")
def real_tq_batch():
    path = _dump_path()
    if not path.is_file():
        pytest.skip(
            f"real TQ dump not found: {path}; "
            "set TPR_REAL_TQ_BATCH=/path/to/tq_batch.pt to override"
        )

    dump = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dump, dict):
        raise TypeError(f"expected torch.load(...) to return dict, got {type(dump)!r}")
    if "keys" not in dump or "tensordict" not in dump:
        raise KeyError(f"TQ dump must contain 'keys' and 'tensordict', got keys={tuple(dump.keys())}")

    keys = list(dump["keys"])
    batch = dump["tensordict"]
    return path, keys, batch


def _tree_unique_token_count(trees) -> int:
    return sum(node.segment.length for tree in trees for node in tree.nodes.values())


def _walk_and_validate(tree, input_rows, *, verbose: bool = False):
    terminal_rows: set[int] = set()

    def walk(node_id: int, prefix_parts: list[torch.Tensor], depth: int = 0):
        node = tree.get(node_id)
        ref = node.segment
        tokens = input_rows[ref.row][ref.start : ref.end]
        current_parts = prefix_parts + [tokens]

        if verbose:
            print(
                "  " * depth
                + (
                    f"node={node.node_id} "
                    f"range=[{ref.start}:{ref.end}) "
                    f"len={ref.length} "
                    f"members={node.member_rows} "
                    f"terminal={node.terminal_rows}"
                )
            )

        for row in node.terminal_rows:
            reconstructed = torch.cat(current_parts)
            expected = input_rows[row]
            assert torch.equal(reconstructed, expected), (
                f"row {row} reconstruction mismatch: "
                f"tree_tokens={reconstructed.numel()} input_tokens={expected.numel()}"
            )
            terminal_rows.add(row)

        for child_id in node.children:
            child = tree.get(child_id)
            assert child.parent_id == node.node_id
            assert child.segment.start == node.segment.end
            assert set(child.member_rows).issubset(node.member_rows)
            walk(child_id, current_parts, depth + 1)

    walk(tree.root_id, [])
    return terminal_rows


def test_real_tq_dump_builds_lossless_compressed_forest(real_tq_batch):
    path, keys, batch = real_tq_batch
    input_rows = _rows(batch["input_ids"])

    assert len(keys) == len(input_rows)
    assert len(keys) > 1

    trees = build_trajectory_trees(keys, batch)
    assert trees

    seen_terminal_rows: set[int] = set()
    for tree in trees:
        root = tree.get(tree.root_id)
        assert root.segment.start == 0
        seen_terminal_rows.update(_walk_and_validate(tree, input_rows))

    assert seen_terminal_rows == set(range(len(input_rows)))

    logical_tokens = sum(row.numel() for row in input_rows)
    unique_tokens = _tree_unique_token_count(trees)
    assert unique_tokens <= logical_tokens

    print()
    print(f"TQ dump: {path}")
    print(f"trajectories: {len(input_rows)}")
    print(f"trees: {len(trees)}")
    print(f"logical tokens: {logical_tokens}")
    print(f"unique tree tokens: {unique_tokens}")
    print(f"reuse ratio: {logical_tokens / unique_tokens:.6f}x")

    for tree in trees:
        root = tree.get(tree.root_id)
        print(
            f"tree={tree.key} members={tree.member_rows} "
            f"nodes={len(tree.nodes)} root=[{root.segment.start}:{root.segment.end}) "
            f"root_len={root.segment.length}"
        )

    print("REAL TQ TREE CHECK: PASS")


def test_real_tq_dump_shared_prefix_crosses_prompt_boundary(real_tq_batch):
    _, keys, batch = real_tq_batch

    if "prompts" not in batch.keys():
        pytest.skip("TQ dump has no prompts field; cannot compare prompt boundary")

    input_rows = _rows(batch["input_ids"])
    prompt_rows = _rows(batch["prompts"])
    trees = build_trajectory_trees(keys, batch)

    # Regression for the django11163 N=8 dump: all trajectories share six
    # generated tokens beyond the original 17,483-token prompt.
    assert len(input_rows) == 8
    assert len(trees) == 1

    tree = trees[0]
    root = tree.get(tree.root_id)
    first_row = tree.member_rows[0]
    prompt_len = prompt_rows[first_row].numel()

    print()
    print(
        f"prompt_len={prompt_len} root_len={root.segment.length} "
        f"shared_response_tokens={root.segment.length - prompt_len}"
    )

    assert prompt_len == 17483
    assert root.segment.length == 17489
    assert root.segment.length > prompt_len


def test_real_tq_dump_prints_full_tree(real_tq_batch):
    """Diagnostic-only structural dump kept as a regression aid for real trajectories."""

    _, keys, batch = real_tq_batch
    input_rows = _rows(batch["input_ids"])
    trees = build_trajectory_trees(keys, batch)

    print()
    for tree in trees:
        root = tree.get(tree.root_id)
        print(
            f"Tree {tree.key}: members={tree.member_rows}, "
            f"nodes={len(tree.nodes)}, root=[{root.segment.start}:{root.segment.end})"
        )
        _walk_and_validate(tree, input_rows, verbose=True)
