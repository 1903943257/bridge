"""Native-only SBH GQA control and CPU FP32 two-step output oracle.

Based on MindSpeed tests_extend/unit_tests/mindspeed/core/context_parallel/
test_ringattn_context_parallel.py. No VERL/TPR attention, runtime or metrics.
Official UT's model shape is changed to S128/B1/Hq8/Hkv2/D256; CP window=1,
overlap off, cache_policy=None (ordinary official test). No numeric gate waiver.
"""

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module")
def native_runtime():
    pytest.importorskip("torch_npu")
    assert int(os.environ.get("WORLD_SIZE", "1")) == 2
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("hccl")
    assert dist.get_world_size() == 2
    # Parse native defaults without pytest argv; no Megatron training driver,
    # model-parallel initialization or VERL patch/adapter is required here.
    argv = sys.argv[:]
    try:
        sys.argv[:] = [argv[0]]
        from mindspeed.args_utils import get_full_args
        args = get_full_args()
    finally:
        sys.argv[:] = argv
    module = importlib.import_module(
        "mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel")
    print(f"NATIVE-OFFICIAL source={module.__file__} "
          f"fused_update={getattr(args, 'use_fused_ring_attention_update', False)}", flush=True)
    yield SimpleNamespace(rank=dist.get_rank(), module=module)
    if owned:
        dist.destroy_process_group()


def indices(rank):
    return torch.cat((torch.arange(rank * 32, (rank + 1) * 32),
                      torch.arange((3 - rank) * 32, (4 - rank) * 32)))


def report(label, left, right, rank):
    left, right = left.detach().cpu().float(), right.detach().cpu().float()
    assert left.shape == right.shape, (label, left.shape, right.shape)
    assert torch.isfinite(left).all() and torch.isfinite(right).all(), label
    a, b = left.reshape(-1), right.reshape(-1)
    na, nb = a.norm(), b.norm()
    rel = (b-a).norm() / na.clamp_min(1e-30)
    cosine = (a*b).sum() / (na*nb).clamp_min(1e-30)
    print(f"NATIVE-ORACLE r={rank} {label} rel={rel.item():.9e} cos={cosine.item():.9f} "
          f"max_abs={(b-a).abs().max().item():.9e} exact={torch.equal(left,right)}", flush=True)


def fp32_attention(q, k, v, causal):
    """CPU FP32 GQA, return [T,H,D] context and [T,H] LSE."""
    q = q.cpu().float().reshape(-1, 8, 256).transpose(0, 1)
    k = k.cpu().float().reshape(-1, 2, 256).repeat_interleave(4, dim=1).transpose(0, 1)
    v = v.cpu().float().reshape(-1, 2, 256).repeat_interleave(4, dim=1).transpose(0, 1)
    logits = (q @ k.transpose(-1, -2)) * 0.0625
    if causal:
        assert q.shape[1] == k.shape[1]
        mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).triu(1)
        logits = logits.masked_fill(mask, -torch.inf)
    return (logits.softmax(-1) @ v).transpose(0, 1), logits.logsumexp(-1).T


def fp32_merge(parts):
    """Global-LSE correction, no intermediate BF16 rounding."""
    ls = torch.full((len(parts), 64, 8), -torch.inf, dtype=torch.float32)
    for i, (ids, _, lse) in enumerate(parts):
        ls[i, ids] = lse
    final_lse = ls.logsumexp(0)
    out = torch.zeros(64, 8, 256, dtype=torch.float32)
    for i, (ids, block_out, _) in enumerate(parts):
        out[ids] += (ls[i, ids] - final_lse[ids]).exp().unsqueeze(-1) * block_out
    return out, final_lse


def run_native(runtime, monkeypatch):
    import torch_npu
    gen = torch.Generator().manual_seed(445501)
    base = [torch.randn(128, 1, h * 256, generator=gen).bfloat16() for h in (8, 2, 2)]
    dy = torch.randn(128, 1, 2048, generator=gen).bfloat16() * 1e-3
    whole = [x.npu().requires_grad_() for x in base]
    mask = torch.ones(2048, 2048, dtype=torch.bool, device=whole[0].device).triu(1)
    result = torch_npu.npu_fusion_attention(*whole, 8, "SBH", pse=None,
        padding_mask=None, atten_mask=mask, scale=0.0625, pre_tockens=128,
        next_tockens=0, keep_prob=1.0, inner_precise=0, sparse_mode=3)
    result[0].backward(dy.npu())
    idx = indices(runtime.rank)
    local = [x[idx].npu().clone().requires_grad_() for x in base]
    steps = []
    original = torch_npu.npu_fusion_attention

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        assert args[4] == "SBH"
        steps.append(dict(q=args[0].detach().cpu().clone(), k=args[1].detach().cpu().clone(),
            v=args[2].detach().cpu().clone(), output=out[0].detach().cpu().clone(),
            maximum=out[1].detach().cpu().clone(), total=out[2].detach().cpu().clone(),
            causal=kwargs["sparse_mode"] == 3))
        return out

    # Ordinary native official-style config: singleton inner window, outer CP.
    cp_para = dict(causal=True, cp_group=dist.group.WORLD, cp_size=2, rank=runtime.rank,
        cp_global_ranks=[0, 1], cp_inner_ranks=[runtime.rank], cp_outer_ranks=[0, 1],
        cp_dkv_outer_ranks=[0, 1], cp_group_for_intra_window=None,
        cp_group_for_send_recv_overlap=None, cp_group_for_intra_window_send_recv_overlap=None,
        pse=None, pse_type=1, cache_policy=None, megatron_cp_in_bnsd=False)
    with monkeypatch.context() as patch:
        patch.setattr(torch_npu, "npu_fusion_attention", capture)
        out = runtime.module.ringattn_context_parallel(*local, 8, cp_para,
                                                      softmax_scale=0.0625, attn_mask=None)
        out.backward(dy[idx].npu())
    expected = [(64, 64), (32, 64) if runtime.rank == 0 else (64, 32)]
    assert [(s['q'].shape[0], s['k'].shape[0]) for s in steps] == expected
    assert [s['causal'] for s in steps] == [True, False]
    return dict(base=base, idx=idx, steps=steps, native=out.detach().cpu(),
        whole=result[0].detach().cpu(), whole_max=result[1].detach().cpu(),
        whole_sum=result[2].detach().cpu(),
        whole_grads=[x.grad.detach().cpu() for x in whole],
        native_grads=[x.grad.detach().cpu() for x in local])


def test_official_style_gqa(native_runtime, monkeypatch):
    data = run_native(native_runtime, monkeypatch)
    report("official-SBH-whole-vs-native/output", data['whole'][data['idx']],
           data['native'], native_runtime.rank)
    for name, ref, actual in zip(('dQ', 'dK', 'dV'), data['whole_grads'], data['native_grads']):
        report(f"official-SBH-whole-vs-native/{name}", ref[data['idx']], actual, native_runtime.rank)
    print("NATIVE-OFFICIAL COMPLETE: direct MindSpeed, no TPR/VERL wrapper; diagnostics only", flush=True)


def test_native_output_oracle(native_runtime, monkeypatch):
    data = run_native(native_runtime, monkeypatch)
    rank = native_runtime.rank
    kernel_parts, reference_parts = [], []
    for step, record in enumerate(data['steps']):
        ids = torch.arange(32, 64) if step == 1 and rank == 0 else torch.arange(64)
        maximum = record['maximum'].reshape(1, 8, -1, 8)[0, :, :, 0].T.float()
        total = record['total'].reshape(1, 8, -1, 8)[0, :, :, 0].T.float()
        context = record['output'].float().reshape(-1, 8, 256)
        kernel_parts.append((ids, context, maximum + total.log()))
        context_ref, lse_ref = fp32_attention(record['q'], record['k'], record['v'], record['causal'])
        reference_parts.append((ids, context_ref, lse_ref))
        report(f"step{step}/kernel-context-vs-FP32", context_ref, context, rank)
    kernel_merge, kernel_lse = fp32_merge(kernel_parts)
    ref_merge, ref_lse = fp32_merge(reference_parts)
    whole_ref, whole_lse = fp32_attention(*data['base'], causal=True)
    native = data['native'].reshape(64, 8, 256)
    report("1-vs-2/whole-FA-vs-native", data['whole'][data['idx']].reshape_as(native), native, rank)
    report("2-vs-3/native-vs-kernel-FP32-merge", native, kernel_merge, rank)
    report("2-vs-3-rounded/native-vs-kernel-merge-final-BF16", native, kernel_merge.bfloat16(), rank)
    report("3-vs-4/kernel-merge-vs-FP32-block-merge", ref_merge, kernel_merge, rank)
    report("4-vs-whole-FP32/decomposition", whole_ref[data['idx']], ref_merge, rank)
    report("kernel-merge-vs-whole-FP32/LSE", whole_lse[data['idx']], kernel_lse, rank)
    report("FP32-block-merge-vs-whole-FP32/LSE", whole_lse[data['idx']], ref_lse, rank)
    # Saved CPU tensors enable rechecking correction without rerunning kernels.
    path = Path('tests/models/mcore/tpr/logs') / f'native_output_oracle_rank{rank}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(**data, kernel_fp32_merge=kernel_merge, reference_fp32_merge=ref_merge,
                    whole_fp32=whole_ref), path)
    print(f"NATIVE-ORACLE saved={path}; CPU FP32 only, no training PASS claim", flush=True)
