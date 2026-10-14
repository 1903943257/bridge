"""CPU contracts: all eight first Session chains, one merged replay batch per step."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from ..parallel.prepare_real_actor_20step_manifest import prepare
from ..parallel.offline_session_selection import first_session_indices


def _capture(path, global_step, sessions, *, ordinals=None, oversize=None):
    """Build a genuine ragged TensorDict-like Actor capture for CPU tests.

    sessions are (session_id, chain_id, reward_sign). If two entries share the
    same session_id they represent two original chains; no values are faked
    after capture, only selected rows in manifest preprocessing.
    """
    lengths = []
    for i, (sid, chain, sign) in enumerate(sessions):
        lengths.append(oversize if oversize and sid == 0 else 14 + sid + chain)
    def ragged(seq, dtype=None):
        return tu.nested_tensor_from_tensor_list([
            torch.tensor(x, dtype=dtype) for x in seq
        ])
    input_ids = [[sid + 11] * length for length, (sid,_,_) in zip(lengths,sessions)]
    prompt = [row[:4] for row in input_ids]
    response = [row[4:] for row in input_ids]
    masks = [[1] * len(row) for row in response]
    values = {
        "input_ids":ragged(input_ids, torch.long),
        "prompts":ragged(prompt, torch.long),
        "responses":ragged(response, torch.long),
        "response_mask":ragged(masks, torch.bool),
        "loss_mask":ragged(masks, torch.bool),
        "old_log_probs":ragged([
            [-0.1 - chain/10] * len(row)
            for row, (_,chain,_) in zip(response,sessions)
        ],torch.float32),
        "advantages":ragged([
            [float(sign)] * len(row)
            for row,(_,_,sign) in zip(response,sessions)
        ],torch.float32),
        "temperature":torch.ones(len(sessions),dtype=torch.float32),
        "session_id":torch.tensor([s for s,_,_ in sessions],dtype=torch.long),
        "global_steps":torch.full((len(sessions),),global_step,dtype=torch.long),
    }
    if ordinals is not None:
        values["trajectory_index"]=torch.tensor(ordinals,dtype=torch.long)
    td=TensorDict(values,batch_size=[len(sessions)])
    tu.assign_non_tensor(td,uid=["same_prompt"]*len(sessions))
    ordinal=int(path.name.split("capture")[1].split("_")[0])
    torch.save({
        "capture_stage":"actor_update_mini_batch",
        "capture_global_step":global_step,
        "capture_index":ordinal,
        "capture_mini_batch_idx":0,
        "keys":tuple(f"same_prompt_tpr_{i}" for i in range(len(sessions))),
        "tensordict":td,
    },path)


def test_complete_eight_sessions_keep_all_without_token_budget(tmp_path):
    first=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    second=tmp_path/"actor_update_step000001_rank0_capture000001_batch0.pt"
    _capture(first,1,[(i,0,-1 if i%2 else 1) for i in range(4)])
    # Session 1's second chain is longer; it must be discarded, not chosen.
    _capture(second,1,[(i,0,1) for i in range(4,8)]+[(1,1,1)])
    target=tmp_path/"out"/"manifest.json"
    report=prepare(tmp_path,target,steps=1,max_seq_len=100,expected_sessions=8)
    assert report["protocol"]=="real_actor_session_first_8merged_v3"
    assert report["steps"]==report["online_steps"]==1
    assert report["max_total_tokens"] is None
    assert report["items"][0]["rows"]==list(range(8))
    assert report["items"][0]["session_count"]==8
    assert report["session_audits"][0]["discarded_trajectory_rows"]==1
    combined=torch.load(report["items"][0]["capture"],weights_only=False)
    td=combined["tensordict"]
    assert td.batch_size[0]==8
    assert len(set(combined["keys"]))==8
    assert sorted(int(x) for x in td["session_id"].tolist())==list(range(8))
    for r in range(8):
        assert len(td["input_ids"][r])==14+r
        assert torch.allclose(
            td["old_log_probs"][r],
            torch.full((10+r,),-0.1),
        )
    assert sum(int(x.numel()) for x in td["input_ids"].unbind(0))==sum(14+i for i in range(8))


def test_padding_uid_excluded_before_eight_session_count(tmp_path):
    a=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    b=tmp_path/"actor_update_step000001_rank0_capture000001_batch0.pt"
    _capture(a,1,[(i,0,1) for i in range(5)])
    _capture(b,1,[(i,0,1) for i in range(5,8)]+[(7,1,1),(7,2,1)])
    second=torch.load(b,weights_only=False)
    # Keep two synthetic padding rows but DO NOT count them as Sessions.
    from verl.utils import tensordict_utils as tu
    tu.assign_non_tensor(second["tensordict"],uid=[
        "same_prompt","same_prompt","same_prompt",
        "pad754d4dd491864b9db6d9d560c0afd5ef",
        "pad754d4dd491864b9db6d9d560c0afd5ef",
    ])
    torch.save(second,b)
    result=prepare(tmp_path,tmp_path/"out"/"manifest.json",steps=1,max_seq_len=100)
    assert result["session_audits"][0]["excluded_padding_rows"]==2
    assert result["items"][0]["session_count"]==8


def test_reject_missing_one_of_eight_sessions(tmp_path):
    f=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    _capture(f,1,[(i,0,1) for i in range(7)])
    with pytest.raises(AssertionError,match="expected exactly 8"):
        prepare(tmp_path,tmp_path/"out"/"manifest.json",steps=1)


def test_first_over_limit_is_not_replaced_by_shorter_second(tmp_path):
    f=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    _capture(f,1,[(i,0,1) for i in range(8)]+[(0,1,1)],
             oversize=105)
    with pytest.raises(AssertionError,match="first trajectory length"):
        prepare(tmp_path,tmp_path/"out"/"manifest.json",
                steps=1,max_seq_len=100)


def test_original_ordinal_overrides_actor_shuffle(tmp_path):
    f=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    rows=[(0,1,1),(0,0,-1)]+[(i,0,1) for i in range(1,8)]
    ordinals=[1,0]+[0]*7
    _capture(f,1,rows,ordinals=ordinals)
    report=prepare(tmp_path,tmp_path/"out"/"manifest.json",
                   steps=1,max_seq_len=100)
    merged=torch.load(report["items"][0]["capture"],weights_only=False)
    assert report["session_audits"][0]["original_trajectory_ordinal_present"]==8
    assert torch.all(merged["tensordict"]["advantages"][0]==-1)


def test_twenty_online_steps_produce_twenty_complete_replay_batches(tmp_path):
    for step in range(1,21):
        f=tmp_path/f"actor_update_step{step:06d}_rank0_capture{step:06d}_batch0.pt"
        _capture(f,step,[(i,0,1) for i in range(8)])
    report=prepare(tmp_path,tmp_path/"out"/"manifest.json",
                   steps=20,max_seq_len=100)
    assert len(report["items"])==20
    assert [item["source_global_step"] for item in report["items"]]==list(range(1,21))
    assert all(item["session_count"]==8 for item in report["items"])


def test_missing_online_step_rejected(tmp_path):
    f=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    _capture(f,1,[(i,0,1) for i in range(8)])
    with pytest.raises(FileNotFoundError,match="missing"):
        prepare(tmp_path,tmp_path/"out"/"manifest.json",steps=2)


def test_order_fallback_explicitly_flagged(tmp_path):
    f=tmp_path/"actor_update_step000001_rank0_capture000000_batch0.pt"
    _capture(f,1,[(i,0,1) for i in range(8)])
    result=prepare(tmp_path,tmp_path/"out"/"manifest.json",steps=1)
    assert result["session_audits"][0]["capture_first_fallback_sessions"]==8
    assert result["historical_ppo_labels_modified"] is False


def test_identity_missing_session_field_rejected(tmp_path):
    data={"input_ids":[torch.tensor([1,2])],"uid":["same_prompt"]}
    with pytest.raises(ValueError,match="session_id"):
        first_session_indices([("missing.pt",data)])
