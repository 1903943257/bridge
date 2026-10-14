"""CPU-only guards for safe nightly multi-capture PPO replay selection."""
import pytest
import torch

from ..parallel.prepare_real_actor_20step_manifest import prepare


class _Rows:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

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
        "capture_index":int(path.name.split("capture")[1].split("_")[0]),
        "capture_mini_batch_idx":0,
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
            "uid":["example"]*len(sizes),
            "session_id":torch.arange(len(sizes),dtype=torch.long),
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


def test_multiple_actor_minibatches_at_one_step_are_preserved(tmp_path):
    for capture_index in (0,1):
        _capture(
            tmp_path/f"actor_update_step000001_rank0_capture{capture_index:06d}_batch0.pt",
            1,[20,21],[-1,1],
        )
    result=prepare(tmp_path,tmp_path/"manifest.json",steps=1)
    # Same uid/session IDs across the two snapshots: later duplicates must
    # NOT be trained as a second original chain.
    assert result["online_steps"]==1
    assert result["steps"]==1
    assert result["session_audits"][0]["original_capture_count"]==2
    assert result["items"][0]["rows"]==[0,1]


def test_single_valid_session_row_does_not_force_fake_sibling(tmp_path):
    _capture(
        tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt",
        1,[20,55],[-1,1],
    )
    result=prepare(tmp_path,tmp_path/"manifest.json",steps=1,max_seq_len=50)
    assert result["items"][0]["rows"]==[0]


def test_distinct_sessions_in_repeated_actor_minibatches_replayed_in_order(tmp_path):
    f1=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    f2=tmp_path/"actor_update_step000001_rank0_capture000001_batch0.pt"
    _capture(f1,1,[20,21],[-1,1])
    _capture(f2,1,[40,41],[-1,1])
    # A separate pair of session IDs in capture1, so both physical actor
    # updates must be replayed in sequence from the original captures.
    raw=torch.load(f2,weights_only=False)
    raw["tensordict"]["session_id"]=torch.tensor([2,3])
    torch.save(raw,f2)
    out=prepare(tmp_path,tmp_path/"manifest.json",steps=1)
    assert out["online_steps"]==1
    assert out["steps"]==2
    assert [x["source_capture_index"] for x in out["items"]]==[0,1]
    assert [x["rows"] for x in out["items"]]==[[0,1],[0,1]]


def test_actual_session_first_ordinal_overrides_row_order(tmp_path):
    from ..parallel.offline_session_selection import first_session_indices
    batch={
        "input_ids":[torch.tensor([1,2]),torch.tensor([1,3])],
        "uid":["u","u"],"session_id":torch.tensor([7,7]),
        "trajectory_index":torch.tensor([1,0]),
    }
    rows,meta=first_session_indices([("capture.pt",batch)])
    assert rows==[[1]]
    assert meta["original_trajectory_ordinal_present"]==1
    assert meta["capture_first_fallback_sessions"]==0


def test_missing_session_id_is_rejected_not_grouped_by_uid(tmp_path):
    from ..parallel.offline_session_selection import first_session_indices
    batch={"input_ids":[torch.tensor([1])], "uid":["u"]}
    with pytest.raises(ValueError,match="session_id"):
        first_session_indices([("capture.pt",batch)])


def test_unordered_capture_without_original_ordinal_is_reported_as_approximation(tmp_path):
    from ..parallel.offline_session_selection import first_session_indices
    batch={
        "input_ids":[torch.tensor([1,2]),torch.tensor([1,3])],
        "uid":["u","u"],"session_id":torch.tensor([7,7]),
    }
    rows,meta=first_session_indices([("capture.pt",batch)])
    assert rows==[[0]]
    assert meta["capture_first_fallback_sessions"]==1
    assert "first_available_capture_order" in meta["selection_semantics"]
