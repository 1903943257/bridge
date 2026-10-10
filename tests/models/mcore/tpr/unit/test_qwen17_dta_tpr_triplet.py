"""CPU-only contract for the real TPR-vs-DTA PPO comparison loss adapter."""
import math

import pytest
import torch
from tensordict import TensorDict

from ..correctness.test_qwen3_1_7b_tpr_dta_triplet_npu import (
    _loss_adapter,_compare_snapshots,
)


@pytest.mark.parametrize("objective",("ppo","ppo_unclipped","fixed_logprob"))
def test_compact_two_rows_shifted_ppo(monkeypatch,objective):
    monkeypatch.setenv("TPR_DTA_BWD_OBJECTIVE",objective)
    monkeypatch.setenv("TPR_TRIPLET_TOTAL_VALID_TOKENS","4")
    # For each compact row: fake one-token prompt, 2 supervised responses.
    data=TensorDict({
        "prompts":torch.zeros(2,1,dtype=torch.long),
        "responses":torch.zeros(2,2,dtype=torch.long),
        "attention_mask":torch.ones(2,3,dtype=torch.long),
        "response_mask":torch.ones(2,2,dtype=torch.bool),
        "old_log_probs":torch.zeros(2,2),
        "advantages":torch.tensor([[1.,-0.5],[-1.,2.]]),
    },batch_size=[2])
    lp=torch.tensor([0.2,-0.4,123.0,-0.1,0.3,456.0],requires_grad=True)
    loss,_=_loss_adapter(model_output={"log_probs":lp},data=data)
    actual=lp[[0,1,3,4]]
    old=torch.zeros_like(actual)
    adv=data["advantages"].flatten()
    if objective=="ppo":
        ratio=(actual-old).exp()
        expected=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).sum()/4
    elif objective=="ppo_unclipped":
        expected=-((actual-old).exp()*adv).sum()/4
    else:
        expected=-(actual*adv).sum()/4
    # The production adapter accumulates separate physical-segment
    # reductions, while this independent oracle reduces all four tokens
    # at once. The FP32 summation order differs by one ULP (5.96e-8).
    # This tolerance applies ONLY to the scalar CPU test; it does not
    # relax any NPU logprob/gradient/optimizer parity diagnostics.
    torch.testing.assert_close(
        loss, expected, rtol=0, atol=torch.finfo(torch.float32).eps
    )
    loss.backward()
    assert lp.grad is not None
    # Independently verify the complete token-level derivative, including
    # active PPO clipping; dummy/padded logits must receive exactly zero.
    expected_grad = torch.zeros_like(lp)
    ratios = (actual.detach() - old).exp()
    if objective == "fixed_logprob":
        token_grad = -adv / 4
    elif objective == "ppo_unclipped":
        token_grad = -adv * ratios / 4
    else:
        active_clip = ((adv > 0) & (ratios > 1.2)) | (
            (adv < 0) & (ratios < 0.8)
        )
        token_grad = torch.where(
            active_clip, torch.zeros_like(adv), -adv * ratios / 4
        )
    expected_grad[[0, 1, 3, 4]] = token_grad
    torch.testing.assert_close(
        lp.grad, expected_grad, rtol=2e-7, atol=1e-7
    )
    assert lp.grad[2].item() == 0 and lp.grad[5].item() == 0


def test_native_prompt_query_offset(monkeypatch):
    monkeypatch.setenv("TPR_DTA_BWD_OBJECTIVE","fixed_logprob")
    monkeypatch.setenv("TPR_TRIPLET_TOTAL_VALID_TOKENS","2")
    data=TensorDict({
        "prompts":torch.zeros(1,3,dtype=torch.long),
        "responses":torch.zeros(1,2,dtype=torch.long),
        "attention_mask":torch.ones(1,5,dtype=torch.long),
        "response_mask":torch.ones(1,2,dtype=torch.bool),
        "old_log_probs":torch.zeros(1,2),
        "advantages":torch.tensor([[2.,-3.]]),
    },batch_size=[1])
    lp=torch.arange(5,dtype=torch.float32,requires_grad=True)
    loss,_=_loss_adapter(model_output={"log_probs":lp},data=data)
    torch.testing.assert_close(loss,-(2*lp[2]-3*lp[3])/2)
    loss.backward()
    torch.testing.assert_close(lp.grad,torch.tensor([0.,0.,-1.,1.5,0.]))


def test_sampled_gradient_metric_identity():
    grad={
        "layer0":torch.tensor([1.,2.,3.]),
        "layer1":torch.tensor([-4.,5.]),
    }
    m=_compare_snapshots(grad,grad)
    assert m["rel_l2"]==0
    assert abs(m["cosine"]-1)<1e-12
    assert m["count"]==2
    assert m["sample_values"]==5



def test_clip_branch_flip_is_distinct_from_outside_clip():
    from ..correctness.test_qwen3_1_7b_tpr_dta_triplet_npu import _clip_branch_disagreement
    old=torch.zeros(1,4)
    adv=torch.tensor([[1.,-1.,1.,-1.]])
    native=torch.tensor([[0.,0.3,-0.3,0.]])
    # positive A, large positive log-ratio => clipped
    # negative A, large negative log-ratio => clipped
    tpr=torch.tensor([[0.3,-0.3,-0.3,0.]])
    found=_clip_branch_disagreement(native,tpr,old,adv,0.2)
    assert found["tokens"]==4
    assert found["full_clipped"]==0
    assert found["tpr_clipped"]==2
    assert found["branch_flips"]==2
    assert found["full_outside"]==2
    assert found["tpr_outside"]==3


def test_nonzero_full_clipping_not_attributed_to_tpr():
    from ..correctness.test_qwen3_1_7b_tpr_dta_triplet_npu import _clip_branch_disagreement
    old=torch.zeros(1,2)
    adv=torch.ones(1,2)
    native=torch.tensor([[0.3,0.]])
    tpr=torch.tensor([[0.3,0.]])
    result=_clip_branch_disagreement(native,tpr,old,adv,0.2)
    assert result["full_clipped"]==result["tpr_clipped"]==1
    assert result["branch_flips"]==0



def test_megatron_same_checkpoint_old_source_diagnostic():
    from ..correctness.test_qwen3_1_7b_tpr_dta_triplet_npu import (
        _choose_megatron_old,_clip_branch_disagreement)
    hf=torch.tensor([[-3.,-3.,-3.,-3.]])
    meg=torch.tensor([[-2.70,-3.00,-3.25,-3.00]])
    old_hf=_choose_megatron_old(hf,meg,"hf_full")
    old_meg=_choose_megatron_old(hf,meg,"megatron_native")
    torch.testing.assert_close(old_hf,hf,rtol=0,atol=0)
    torch.testing.assert_close(old_meg,meg,rtol=0,atol=0)
    adv=torch.tensor([[1.,1.,-1.,1.]])
    baseline=_clip_branch_disagreement(meg,meg,old_hf,adv,0.2)
    matched=_clip_branch_disagreement(meg,meg,old_meg,adv,0.2)
    assert baseline["full_clipped"]==1
    assert matched["full_clipped"]==0
    assert baseline["branch_flips"]==matched["branch_flips"]==0
    with pytest.raises(ValueError):
        _choose_megatron_old(hf,meg,"unknown")
    with pytest.raises(ValueError):
        _choose_megatron_old(hf,meg[:,:2],"hf_full")
    with pytest.raises(ValueError):
        bad=meg.clone()
        bad[0,0]=float("nan")
        _choose_megatron_old(hf,bad,"megatron_native")


def test_changing_old_policy_not_equivalent_to_fixing_forward():
    from ..correctness.test_qwen3_1_7b_tpr_dta_triplet_npu import (
        _choose_megatron_old,)
    hf=torch.tensor([[-2.5,-2.0]])
    meg=torch.tensor([[-2.2,-2.1]])
    tpr=torch.tensor([[-2.0,-2.3]])
    for source in ("hf_full","megatron_native"):
        _=_choose_megatron_old(hf,meg,source)
        # Neither the native nor the TPR computed logits change because
        # no model is rerun as a result of selecting the old-policy tensor.
        torch.testing.assert_close((meg-tpr).abs(),
                                   torch.tensor([[0.2,0.2]]),
                                   atol=1e-6,rtol=0)
