"""CPU: actual AReaL-DTA gradient-relay fidelity and clipped PPO update.

Unlike the Forward-only fake, this model has nonzero trainable prefix K/V.
Checks parameters and optimizer after complete DTA pop/backward traversal.
"""
from types import SimpleNamespace
import copy
import math

import pytest
import torch
import torch.nn.functional as F

from ..correctness._qwen17_areal_native_dta_backward import DTAEngine
from ..correctness._qwen17_areal_training_forward_reference import areal_backward_plan


class FakeDynamicCache:
    def __init__(self):
        self.layers = []

    def update(self, keys, values, layer_idx):
        if layer_idx == len(self.layers):
            self.layers.append(SimpleNamespace(keys=keys, values=values))
        else:
            self.layers[layer_idx].keys = torch.cat(
                (self.layers[layer_idx].keys, keys), dim=-2)
            self.layers[layer_idx].values = torch.cat(
                (self.layers[layer_idx].values, values), dim=-2)


class TinyCausalKV(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2,
            num_key_value_heads=1,num_attention_heads=1,hidden_size=4,head_dim=4)
        self.embed = torch.nn.Embedding(20,4)
        self.q = torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)
                                      for _ in range(2)])
        self.k = torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)
                                      for _ in range(2)])
        self.v = torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)
                                      for _ in range(2)])
        self.o = torch.nn.ModuleList([torch.nn.Linear(4,4,bias=False)
                                      for _ in range(2)])
        self.head = torch.nn.Linear(4,20,bias=False)

    def forward(self, input_ids, past_key_values, use_cache=True):
        assert use_cache
        x = self.embed(input_ids)
        for l in range(2):
            start = (
                past_key_values.layers[l].keys.shape[-2]
                if len(past_key_values.layers)>l else 0
            )
            q = self.q[l](x)
            k = self.k[l](x)
            v = self.v[l](x)
            if start:
                all_k=torch.cat((past_key_values.layers[l].keys[:,0,:,:],k),dim=1)
                all_v=torch.cat((past_key_values.layers[l].values[:,0,:,:],v),dim=1)
            else:
                all_k,all_v=k,v
            logits=q@all_k.transpose(-2,-1) / math.sqrt(q.shape[-1])
            causal=(torch.arange(all_k.shape[1])[None,:]
                    <= torch.arange(x.shape[1])[:,None]+start)
            attn=logits.masked_fill(~causal,-1e5).softmax(dim=-1)
            x=torch.tanh(x + self.o[l](attn@all_v))
            past_key_values.update(k[:,None,:,:],v[:,None,:,:],layer_idx=l)
        return SimpleNamespace(logits=self.head(x),past_key_values=past_key_values)


def test_data():
    return [
        torch.tensor([1,2,3,4,5,6,7],dtype=torch.long),
        torch.tensor([1,2,3,4,5,8,9],dtype=torch.long),
        torch.tensor([1,2,3,4,5,8,9],dtype=torch.long),
        torch.tensor([1,2,3,4,10,11],dtype=torch.long),
        torch.tensor([1,2,3],dtype=torch.long),
    ]


def _loss(lp,entropy,reference,i,scale, *, clip=0.2):
    n=min(len(lp),len(reference[i]))
    assert n>=1
    lp=lp[:n]
    advantage=scale[i][:n]
    ratio=(lp-reference[i][:n]).exp()
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    return -surrogate.sum() / 20 - 0.01*entropy[:n].sum()/20


def _run_full(model,rows,refs,adv):
    model.zero_grad(set_to_none=True)
    out=[]
    for i,r in enumerate(rows):
        forward=model(r.unsqueeze(0),FakeDynamicCache(),True)
        logits=forward.logits[0]
        lp=F.log_softmax(logits.float(),-1)[:-1].gather(
            -1,r[1:].unsqueeze(-1)).squeeze(-1)
        v=F.log_softmax(logits.float(),-1)
        entropy=-(v.exp()*v).sum(-1)
        out.append(lp.detach().clone())
        _loss(lp,entropy,refs,i,adv).backward()
    return out


def _run_dta(model,rows,refs,adv,block_size, cut_f1_tail=True):
    model.zero_grad(set_to_none=True)
    plan=areal_backward_plan(rows)
    trie=SimpleNamespace(inputs=plan.sequences,lcp_lens=plan.lcp_lens,
        attach_lists=[[({"_sequence_batch_id":i},length)
                       for i,length in group] for group in plan.attachments],
        n_sequences=len(rows))
    engine=DTAEngine(model.config,torch.device("cpu"),torch.float32,
                     max(len(t) for t in rows))
    captured=[None]*len(rows)
    def f(logp, entropy, att):
        i=att["_sequence_batch_id"]
        captured[i]=logp.detach().clone()
        return _loss(logp,entropy,refs,i,adv)
    engine.backward(model,trie,f,block_size,cut_f1_tail=cut_f1_tail)
    assert all(t is not None for t in captured)
    return captured


def _grad_comparison(full,dta):
    for (an,a),(bn,b) in zip(full.named_parameters(),dta.named_parameters(),strict=True):
        assert an==bn
        assert a.grad is not None and b.grad is not None, an
        torch.testing.assert_close(a.grad,b.grad,atol=8e-5,rtol=8e-4)


@pytest.mark.parametrize("block", (1,2,3,4,8,-1))
@pytest.mark.parametrize("cut_f1_tail",(True,False))
def test_full_dta_backward_gradient_and_sgd_update(block,cut_f1_tail):
    torch.manual_seed(2026)
    base=TinyCausalKV()
    dta=copy.deepcopy(base)
    rows=test_data()
    with torch.no_grad():
        refs=[]
        for x in rows:
            out=base(x.unsqueeze(0),FakeDynamicCache(),True)
            lp=F.log_softmax(out.logits[0].float(),-1)[:-1].gather(
                -1,x[1:].unsqueeze(-1)).squeeze(-1)
            refs.append(lp.detach().clone())
    adv=[torch.sin(torch.arange(len(x)-1).float() * 0.6 + i) + 0.4
         for i,x in enumerate(rows)]
    full_lp=_run_full(base,rows,refs,adv)
    dta_lp=_run_dta(dta,rows,refs,adv,block,cut_f1_tail)
    for x,y in zip(full_lp,dta_lp,strict=True):
        torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
    _grad_comparison(base,dta)
    opt_full=torch.optim.SGD(base.parameters(),lr=0.001)
    opt_dta=torch.optim.SGD(dta.parameters(),lr=0.001)
    opt_full.step();opt_dta.step()
    for (an,a),(bn,b) in zip(base.named_parameters(),dta.named_parameters(),strict=True):
        assert an==bn
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)


def test_dta_gradients_nonzero_from_shared_prefix():
    torch.manual_seed(73)
    model=TinyCausalKV()
    rows=test_data()
    refs=[torch.zeros(len(r)-1) for r in rows]
    adv=[torch.ones(len(r)-1) for r in rows]
    _run_dta(model,rows,refs,adv,block_size=2)
    assert torch.linalg.vector_norm(model.embed.weight.grad[1:4]).item()>0
    assert torch.linalg.vector_norm(model.k[0].weight.grad).item()>0
    assert torch.linalg.vector_norm(model.v[0].weight.grad).item()>0


@pytest.mark.parametrize("mode", ("native","m_split","fp32_linear","m_split_fp32"))
def test_linear_ablation_is_autograd_safe_and_restored(mode):
    from ..correctness.test_qwen3_1_7b_areal_backward_ppo_npu import _linear_ablation
    class OneLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn=torch.nn.Module()
            self.self_attn.q_proj=torch.nn.Linear(4,4,bias=False)
        def forward(self,x):
            return self.self_attn.q_proj(x)
    net=OneLayer().bfloat16()
    x=torch.arange(32,dtype=torch.float32).reshape(1,8,4).bfloat16().requires_grad_(True)
    original=net.self_attn.q_proj.forward
    with _linear_ablation(net,mode,tile_m=3) as count:
        y=net(x)
        assert y.shape==(1,8,4)
        assert torch.isfinite(y.float()).all()
        y.float().square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad.float()).all()
        assert net.self_attn.q_proj.weight.grad is not None
        assert count == (0 if mode=="native" else 1)
    assert net.self_attn.q_proj.forward==original


def test_fixed_logprob_objective_has_identical_token_gradient_despite_drift():
    """Controlled logprob coefficient must not change due to PPO clipping."""
    from ..correctness.test_qwen3_1_7b_areal_backward_ppo_npu import _ppo_row_loss
    adv=torch.tensor([1.0,-2.0,0.75],dtype=torch.float32)
    old=torch.tensor([-3.,-4.,-5.])
    entropy=torch.zeros(4)
    grads=[]
    for logp_values in ([-3.,-4.,-5.],[-1.,-6.,-4.]):
        lp=torch.tensor(logp_values,requires_grad=True)
        loss=_ppo_row_loss(lp,entropy,old,adv,p=1,s=3,clip_eps=0.2,
                           entropy_coef=0.0,n_rows=1,objective="fixed_logprob")
        loss.backward()
        grads.append(lp.grad.detach().clone())
    torch.testing.assert_close(grads[0],grads[1],rtol=0,atol=0)
    torch.testing.assert_close(grads[0],-adv/3,rtol=0,atol=0)


def test_clipped_ppo_objective_differs_from_fixed_logprob():
    """Confirms objective switch tests a real upstream gradient mechanism."""
    from ..correctness.test_qwen3_1_7b_areal_backward_ppo_npu import _ppo_row_loss
    old=torch.tensor([0.,0.])
    adv=torch.tensor([1.,-1.])
    entropy=torch.zeros(3)
    lp=torch.tensor([0.4,-0.4],requires_grad=True)
    for mode in ("ppo","fixed_logprob"):
        lp.grad=None
        loss=_ppo_row_loss(lp,entropy,old,adv,p=1,s=2,clip_eps=0.2,
                           entropy_coef=0.0,n_rows=1,objective=mode)
        loss.backward()
        actual=lp.grad.detach().clone()
        if mode=="ppo":
            torch.testing.assert_close(actual,torch.zeros_like(actual),atol=0,rtol=0)
        else:
            torch.testing.assert_close(actual,-adv/2,atol=0,rtol=0)
