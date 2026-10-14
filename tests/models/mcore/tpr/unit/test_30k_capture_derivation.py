"""CPU checks for immutable real-PPO one-step capture truncation.

The derived 30K experiment must not regenerate labels or change the original
prompt, and all response-aligned fields must be cropped identically.
"""
from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("tensordict")
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from ..parallel.prepare_tpr_30k_one_step import derive_capture


def _source_capture(path, *, overlong_prompt=False):
    prompts = [
        torch.tensor(list(range(1, 13 if (overlong_prompt and i == 3) else 4)),
                     dtype=torch.int64)
        for i in range(8)
    ]
    responses = [torch.tensor(list(range(20, 20 + 7 + (i % 3))),
                              dtype=torch.int64) for i in range(8)]
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
    fields = {
        "input_ids": [torch.cat((p, r)) for p, r in zip(prompts, responses)],
        "prompts": prompts,
        "responses": responses,
        "attention_mask": [torch.ones(n, dtype=torch.int64) for n in lengths],
        "response_mask": [torch.ones(len(x), dtype=torch.bool) for x in responses],
        "loss_mask": [torch.ones(len(x), dtype=torch.int64) for x in responses],
        "old_log_probs": [torch.arange(len(x), dtype=torch.float32) / 10
                          for x in responses],
        "advantages": [torch.arange(len(x), dtype=torch.float32)
                       for x in responses],
    }
    nested = {
        key: tu.nested_tensor_from_tensor_list(rows, ragged_idx=1)
        for key, rows in fields.items()
    }
    nested["temperature"] = torch.ones(8, dtype=torch.float32)
    batch = TensorDict(nested, batch_size=[8])
    # Real captured TensorDicts contain batch-level scalar metadata, not
    # only row-aligned tensors. These values must survive shallow cloning
    # without being revalidated as shape-[8] arrays.
    tu.assign_non_tensor(batch, scalar_metadata=17,
                         scalar_tensor_metadata=torch.tensor(1.25))
    keys = [f"uid1_tpr_{i}" for i in range(8)]
    torch.save({
        "capture_stage": "actor_update_mini_batch",
        "keys": keys, "tensordict": batch,
    }, path)


def test_derived_capture_truncates_all_response_labels_without_mutating_source(tmp_path):
    src, target = tmp_path / "source.pt", tmp_path / "truncated.pt"
    _source_capture(src)
    before = torch.load(src, weights_only=False)
    stats = derive_capture(src, target, cap=8)
    raw = torch.load(target, weights_only=False)
    assert raw["keys"] == before["keys"]
    assert raw["offline_sequence_cap"] == 8
    assert stats["max_sequence_length"] == 8
    cropped, original = raw["tensordict"], before["tensordict"]
    assert set(cropped.keys()) == set(original.keys())
    assert tu.get_non_tensor_data(cropped, key="scalar_metadata", default=None) == 17
    scalar_value = tu.get_non_tensor_data(cropped, key="scalar_tensor_metadata", default=None)
    assert torch.equal(torch.as_tensor(scalar_value), torch.tensor(1.25))
    for row in range(8):
        assert torch.equal(cropped["prompts"][row], original["prompts"][row])
        assert torch.equal(cropped["input_ids"][row], original["input_ids"][row][:8])
        assert torch.equal(
            cropped["responses"][row], original["responses"][row][:5]
        )
        for key in ("response_mask", "loss_mask", "old_log_probs", "advantages"):
            assert torch.equal(cropped[key][row], original[key][row][:5])
    assert stats["ppo_tokens"] == 40
    assert len(before["input_ids"][0]) == 10


def test_derived_capture_rejects_prompt_longer_than_cap(tmp_path):
    src, target = tmp_path / "source.pt", tmp_path / "truncated.pt"
    _source_capture(src, overlong_prompt=True)
    with pytest.raises(ValueError, match="prompt alone"):
        derive_capture(src, target, cap=8)
    assert not target.exists()
