"""CPU-only correctness of canonical TP witness and PPO parity comparator."""
import math

import pytest

from ..parallel.tpr_10k_canonical_witness import (
    canonical_shape,owner_and_local_flat,shard_kind,
)
from ..parallel.compare_tpr_10k_parallel_parity import (
    _effective_ppo,_vector_metrics,compare_pair,
)


@pytest.mark.parametrize("name,shape,tp,expected", [
    ("embedding.word_embeddings.weight",(4,3),2,(8,3)),
    ("decoder.layers.0.self_attention.linear_qkv.weight",(4,3),2,(8,3)),
    ("decoder.layers.0.self_attention.linear_proj.weight",(4,3),2,(4,6)),
    ("decoder.layers.0.mlp.linear_fc1.weight",(8,3),2,(16,3)),
    ("decoder.layers.0.mlp.linear_fc2.weight",(4,3),2,(4,6)),
    ("decoder.layers.0.input_layernorm.weight",(3,),2,(3,)),
    ("decoder.layers.0.self_attention.q_layernorm.weight",(3,),2,(3,)),
])
def test_canonical_qwen_tp_layouts(name,shape,tp,expected):
    assert canonical_shape(name,shape,tp)==expected
    visited=[]
    local_size=math.prod(shape)
    global_size=math.prod(expected)
    for index in range(global_size):
        owner,pos=owner_and_local_flat(name,index,shape,tp)
        assert 0<=owner<tp
        assert 0<=pos<local_size
        visited.append((owner,pos))
    if shard_kind(name) in ("replicated","tp_sum"):
        assert all(owner==0 for owner,_ in visited)
        assert len(set(visited))==global_size
    else:
        assert len(set(visited))==global_size
        assert len(set(visited))==local_size*tp


def test_qk_layernorm_requires_tp_sum_for_gradients():
    assert shard_kind("decoder.layers.7.self_attention.q_layernorm.weight")=="tp_sum"
    assert shard_kind("decoder.layers.7.self_attention.k_layernorm.weight")=="tp_sum"
    assert shard_kind("decoder.layers.7.input_layernorm.weight")=="replicated"


def test_gate_up_interleaved_tp2_shard_mapping():
    name="decoder.layers.0.mlp.linear_fc1.weight"
    shape=(8,3)
    tp=2
    # Global gate rows 0..7, up rows 8..15:
    # TP0 gate 0..3 / up 8..11; TP1 gate 4..7 / up 12..15.
    assert owner_and_local_flat(name,4*3,shape,tp)==(1,0)
    assert owner_and_local_flat(name,8*3,shape,tp)==(0,4*3)
    assert owner_and_local_flat(name,12*3,shape,tp)==(1,4*3)


def _result(vals,logprobs):
    return {
        "gradient_witness":{
            "names":["decoder.layers.0.input_layernorm.weight"],
            "canonical_shapes":[[2]],
            "samples_per_name":[2],
            "samples_per_parameter":64,
            "values":vals,
            "replica_max_abs":0.0,
            "complete_norm_gradients":{
                "names":["decoder.layers.0.input_layernorm.weight"],
                "shapes":[[2]],
                "values":vals,
                "elements":2,
            },
        },
        "global_loss":0.1,
        "token_logprobs":logprobs,
    }


def test_clip_branch_change_with_small_logprob_difference():
    # positive advantage, old=0: PPO clip becomes active for ratio>1.2.
    active,eff,ratio=_effective_ppo(math.log(1.21),0.0,1.0)
    assert active and eff==0.0 and ratio>1.2
    active,eff,ratio=_effective_ppo(math.log(1.19),0.0,1.0)
    assert not active and eff==pytest.approx(-1.19)
    active,eff,ratio=_effective_ppo(math.log(0.79),0.0,-1.0)
    assert active and eff==0.0


def test_compare_pair_exact_and_clip_changes():
    labels={(0,0):(0.0,1.0),(0,1):(0.0,-1.0)}
    left=_result([1.0,2.0],[[0,0,math.log(1.19)],[0,1,math.log(0.79)]])
    right=_result([1.0,2.0],[[0,0,math.log(1.21)],[0,1,math.log(0.81)]])
    r=compare_pair(left,right,labels)
    assert r["canonical_sampled_parameter_gradient"]["relative_l2"]==0
    assert r["clip_state_changes"]==2
    assert r["effective_dloss_dlogprob"]["relative_l2"]>0


def test_compare_pair_different_canonical_coordinate_layout_fails():
    a=_result([1.0,2.0],[[0,0,0.0]])
    b=_result([1.0,2.0],[[0,0,0.0]])
    b["gradient_witness"]["canonical_shapes"]=[[3]]
    with pytest.raises(AssertionError,match="coordinates"):
        compare_pair(a,b,{(0,0):(0.0,1.0)})


def test_vector_metrics_detect_opposite_direction():
    m=_vector_metrics([1.0,-1.0],[-1.0,1.0])
    assert m["cosine"]==pytest.approx(-1.0)
    assert m["relative_l2"]==pytest.approx(2.0)



def test_10k_dp2_isolation_gate_does_not_open_production_dp(monkeypatch):
    from verl.models.mcore.tpr.megatron_adapter import _validate_tpr_parallel_sizes
    sizes = {"TP":2, "CP":1, "DP":2, "PP":1, "EP":1}
    for key in (
        "TPR_EXPERIMENTAL_OFFLINE_DP2",
        "TPR_RUN_10K_PARALLEL_PARITY",
        "TPR_10K_PARITY_ONLY",
        "TPR_RUN_TP2CP2DP2_ONE_STEP",
    ):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(sizes,ppo=True)
    monkeypatch.setenv("TPR_EXPERIMENTAL_OFFLINE_DP2","1")
    monkeypatch.setenv("TPR_RUN_10K_PARALLEL_PARITY","1")
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(sizes,ppo=True)
    monkeypatch.setenv("TPR_10K_PARITY_ONLY","1")
    assert _validate_tpr_parallel_sizes(sizes,ppo=True)==2
    with pytest.raises(NotImplementedError):
        _validate_tpr_parallel_sizes(sizes,ppo=False)


def test_10k_cp_dp_matrix_has_no_tp1_case_or_pair():
    from ..parallel.compare_tpr_10k_parallel_parity import CASES, PAIRS
    assert len(CASES) == 5
    assert len(PAIRS) == 5
    assert "tpr_tp1cp1dp1" not in CASES
    assert not any("tpr_tp1cp1dp1" in pair for pair in PAIRS)
    assert all(case.startswith("tpr_tp2") or case == "unshared_tp2cp1dp1"
               for case in CASES)
