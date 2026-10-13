"""CPU-only guards for safe nightly multi-capture PPO replay selection."""
import pytest
import torch

from ..parallel.prepare_real_actor_20step_manifest import prepare


class _Rows:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def unbind(self, dim=0):
        if dim != 0:
            raise ValueError(dim)
        return tuple(self.values)


def _capture(path, step, sizes, signs):
    inputs, mask, old, adv, prompts, responses = [], [], [], [], [], []
    for size, sign in zip(sizes, signs):
        ids = torch.arange(1,size+1, dtype=torch.long)
        inputs.append(ids)
        prompts.append(ids[:2])
        responses.append(ids[2:])
        mask.append(torch.ones(size-2, dtype=torch.bool))
        old.append(torch.zeros(size-2))
        adv.append(torch.full((size-2,), float(sign)))
    payload = {
        "capture_stage":"actor_update_mini_batch",
        "capture_global_step":step,
        "keys":tuple(f"uid_tpr_{i}" for i in range(len(sizes))),
        "tensordict":{
            "input_ids":_Rows(inputs),
            "prompts":_Rows(prompts),
            "responses":_Rows(responses),
            "response_mask":_Rows(mask),
            "loss_mask":_Rows(mask),
            "old_log_probs":_Rows(old),
            "advantages":_Rows(adv),
            "temperature":torch.tensor([0.6]*len(sizes)),
        },
    }
    torch.save(payload, path)


def test_selection_retains_one_positive_and_one_negative(tmp_path):
    _capture(
        tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt",
        1, [30,31,50,60], [-1,-1,+1,-1],
    )
    report = prepare(tmp_path, tmp_path/"manifest.json",steps=1,
                     max_seq_len=60,max_rows=3,max_total_tokens=150)
    item = report["items"][0]
    assert item["rows"] == [0,1,2]
    assert item["positive_advantage_present"]
    assert item["negative_advantage_present"]
    assert item["ppo_tokens"] == (30-2)+(31-2)+(50-2)


def test_zero_advantage_is_preserved_and_marked(tmp_path):
    _capture(
        tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt",
        1,[20,21],[0,0],
    )
    result = prepare(tmp_path,tmp_path/"manifest.json",steps=1,
                     max_seq_len=60,max_rows=3,max_total_tokens=120)
    assert result["items"][0]["rows"] == [0,1]
    assert not result["items"][0]["positive_advantage_present"]
    assert not result["items"][0]["negative_advantage_present"]


def test_missing_online_step_not_silently_reused(tmp_path):
    _capture(
        tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt",
        1,[20,21],[-1,1],
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        prepare(tmp_path,tmp_path/"manifest.json",steps=2)


def test_multiple_actor_minibatches_at_one_step_fail_closed(tmp_path):
    for capture_index in (0,1):
        _capture(
            tmp_path/f"actor_update_step000001_rank0_capture{capture_index:06d}_batch0.pt",
            1,[20,21],[-1,1],
        )
    with pytest.raises(RuntimeError, match="multiple rank0"):
        prepare(tmp_path,tmp_path/"manifest.json",steps=1)


def test_refuses_truncated_or_single_row_comparison(tmp_path):
    _capture(
        tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt",
        1,[20,55],[-1,1],
    )
    with pytest.raises(RuntimeError,match="only 1 eligible"):
        prepare(tmp_path,tmp_path/"manifest.json",steps=1,max_seq_len=50)
