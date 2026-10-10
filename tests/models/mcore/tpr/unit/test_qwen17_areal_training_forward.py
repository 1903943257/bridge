"""AReaL-DTA backward-schedule Forward CPU gates (no backward gradients)."""
import pytest
import torch
from .test_qwen17_areal_exact_reference import FakeDynamicCache, FakeModel, row
from ..correctness._qwen17_dta_style_reference import hf_full_logprobs
from ..correctness._qwen17_areal_training_forward_reference import (
    areal_backward_plan, _fork_positions, hf_areal_training_loss_forward,
)


def rows_for(case):
    data = {
        "single": [[1,2,3,4,5,6]],
        "shared": [
            [1,2,3,4,5,6], [1,2,3,4,7,8],
            [1,2,3,9,8,7], [1,2,4,5,6,7]],
        "nested": [
            [1,2,3], [1,2,3,4,5,6], [1,2,3,4,5,6],
            [1,2,3,4,8], [1,2,9,1]],
        "disjoint": [[1,2,3], [4,5,6], [7,8,9]],
        "prefix": [
            [1,2,3,4,5,6,7,8,9], [1,2,3,4,5,7,8,9],
            [1,2,3,4,5,6,8,9], [1,2,3,4,5,6]],
    }
    return tuple(row(x) for x in data[case])


@pytest.mark.parametrize("case", ("single","shared","nested","disjoint","prefix"))
@pytest.mark.parametrize("block_size", (1,2,3,4,7,-1))
@pytest.mark.parametrize("cut_f1_tail", (True,False))
def test_loss_pop_forward_matches_full_toy(case,block_size,cut_f1_tail):
    rows=rows_for(case)
    model=FakeModel().eval()
    full=tuple(hf_full_logprobs(model,x,FakeDynamicCache) for x in rows)
    model.calls.clear()
    got=hf_areal_training_loss_forward(
        model,rows,FakeDynamicCache,block_size=block_size,
        cut_f1_tail=cut_f1_tail,
    )
    for a,b in zip(full,got.logprobs,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert got.n_pop_forwards>=got.n_leaves
    assert got.n_cache_forwards==len(got.physical_m_cache)
    assert tuple(m for _,m in model.calls)==tuple(e-s for _,s,e,_ in got.events)
    assert all(0 <= owner < len(got.events)
               for seq in got.token_owner for owner in seq)


def test_backward_permute_and_fork_boundary():
    rows=rows_for("shared")
    plan=areal_backward_plan(rows)
    assert sorted(i for group in plan.attachments for i,_ in group)==list(range(4))
    assert _fork_positions([7], [], 3)==(0,3)
    from ..correctness._qwen17_areal_dta_reference import _areal_forward_order
    assert _areal_forward_order([5,4,3,2], [3,2,1], backward=True)==[0,1,2,3]
    assert _areal_forward_order([5,5,4], [3,2], backward=True)==[1,0,2]


def test_reject_invalid_block_and_short_buffer():
    model=FakeModel()
    with pytest.raises(ValueError):
        hf_areal_training_loss_forward(
            model,rows_for("single"),FakeDynamicCache,block_size=0)
    with pytest.raises(ValueError):
        hf_areal_training_loss_forward(
            model,rows_for("single"),FakeDynamicCache,max_seq_len=3)


def test_grad_enabled_only_in_pop_and_no_gradients_accumulate():
    class Trace(FakeModel):
        def __init__(self):
            super().__init__()
            self.flags=[]
        def forward(self,*,input_ids,past_key_values,use_cache):
            self.flags.append((
                torch.is_grad_enabled(),
                any(layer.keys.requires_grad for layer in past_key_values.layers)))
            return super().forward(input_ids=input_ids,past_key_values=past_key_values,
                                   use_cache=use_cache)
    model=Trace()
    result=hf_areal_training_loss_forward(
        model,rows_for("single"),FakeDynamicCache,block_size=2)
    assert tuple(x[0] for x in model.flags)==tuple(
        kind=="POP" for kind,*_ in result.events)
    assert tuple(x[1] for x in model.flags)==tuple(
        kind=="POP" for kind,*_ in result.events)
    assert all(param.grad is None for param in model.parameters())


def test_exact_pop_boundaries_one_trajectory():
    model=FakeModel()
    result=hf_areal_training_loss_forward(
        model,rows_for("single"),FakeDynamicCache,block_size=2)
    assert result.events==(
        ("CACHE",0,4,0),("POP",4,6,0),("POP",2,4,0),("POP",0,2,0))
    no_limit=hf_areal_training_loss_forward(
        FakeModel(),rows_for("single"),FakeDynamicCache,block_size=-1)
    assert no_limit.events==(("POP",0,6,0),)


def test_shape_sensitive_toy_exposes_forward_difference():
    class ShapeSensitiveFake(FakeModel):
        def forward(self,*,input_ids,past_key_values,use_cache):
            result=super().forward(
                input_ids=input_ids,past_key_values=past_key_values,use_cache=use_cache)
            result.logits[...,0]+=input_ids.shape[1]*0.01
            return result
    model=ShapeSensitiveFake()
    rows=rows_for("shared")
    full=tuple(hf_full_logprobs(model,x,FakeDynamicCache) for x in rows)
    train=hf_areal_training_loss_forward(model,rows,FakeDynamicCache,block_size=2)
    assert any(not torch.equal(a,b) for a,b in zip(full,train.logprobs))
