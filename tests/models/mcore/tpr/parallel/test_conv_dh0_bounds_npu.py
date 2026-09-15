"""Compact dh0 buffer canaries + CPU oracle, no model/TPR/HCCL required."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from mindspeed_ops.utils import is_arch35

pytestmark = pytest.mark.skipif(is_arch35(), reason="arch32 causal-conv regression")


def _run(monkeypatch, t, train_state, final_mode, forced_bt=None):
    import mindspeed_ops.arch32.triton.convolution as conv
    from mindspeed_ops.api.triton.convolution import causal_conv1d

    if forced_bt is not None:
        # Test-only tile control, never applied to production or Stage 4.5.
        monkeypatch.setattr(conv, "get_vector_num", lambda: t // forced_bt)
    generator = torch.Generator().manual_seed(4519)
    device = torch.device("npu", torch.npu.current_device())

    def sample(shape):
        return (torch.randn(shape, generator=generator) * 0.1).to(torch.bfloat16)

    cpu = [sample((1, t, 3072)), sample((4, 3072)), sample((3072,)), sample((1, 3072, 4))]
    x, w, bias, h0 = [v.to(device).requires_grad_(True) for v in cpu]
    h0.requires_grad_(train_state)
    dy = sample(x.shape)
    dht = sample(h0.shape)
    original = conv.causal_conv1d_bwd_kernel
    launches = []

    class CanaryKernel:
        def __getattr__(self, name):
            return getattr(original, name)

        def __getitem__(self, grid):
            launch = original[grid]

            def call(*args, **kwargs):
                compact = kwargs["dh0"]
                nt = (kwargs["T"] + kwargs["BT"] - 1) // kwargs["BT"]
                assert compact.shape[0] == min(nt, (kwargs["W"] + kwargs["BT"] - 1) // kwargs["BT"])
                # Reserve all addresses reached by the OLD unguarded store,
                # so that the old bug corrupts a canary, not unrelated memory.
                extent = nt * compact[0].numel()
                guard = 32
                storage = torch.full((guard + extent + guard,), 37., dtype=compact.dtype, device=device)
                payload = storage[guard:guard + compact.numel()].view_as(compact)
                payload.copy_(compact)
                kwargs["dh0"] = payload
                launch(*args, **kwargs)
                torch.npu.synchronize()
                assert torch.all(storage[:guard] == 37).item(), "dh0 head canary corrupted"
                assert torch.all(storage[guard + compact.numel():] == 37).item(), "dh0 tail canary corrupted"
                compact.copy_(payload)
                launches.append((kwargs["BT"], kwargs["BD"], tuple(compact.shape)))
            return call

    monkeypatch.setattr(conv, "causal_conv1d_bwd_kernel", CanaryKernel())
    y, ht = causal_conv1d(x, w, bias=bias, initial_state=h0, activation="silu", output_final_state=final_mode != "off")
    torch.npu.synchronize()
    roots, grads = [y], [dy.to(device)]
    if final_mode == "nonzero":
        roots.append(ht)
        grads.append(dht.to(device))
    torch.autograd.backward(roots, grads)
    torch.npu.synchronize()
    assert len(launches) == 1
    if forced_bt is not None:
        assert launches[0][0] == forced_bt
    # CPU FP32 oracle using the actual BF16 input values. h0 slot 0 is unused.
    rx, rw, rb, rh = [v.float().requires_grad_(True) for v in cpu]
    padded = torch.cat((rh.transpose(1, 2)[:, 1:], rx), dim=1)
    z = sum(padded[:, j:j+t] * rw[j] for j in range(4)) + rb
    ry = torch.nn.functional.silu(z)
    rr, rg = [ry], [dy.float()]
    if final_mode == "nonzero":
        rr.append(rx[:, -4:].transpose(1, 2))
        rg.append(dht.float())
    torch.autograd.backward(rr, rg)
    pairs = [("dx", x, rx), ("dw", w, rw), ("db", bias, rb)]
    if train_state:
        pairs.append(("dh0", h0, rh))
    else:
        assert h0.grad is None
    for name, actual, ref in pairs:
        assert actual.grad is not None
        observed = actual.grad.detach().cpu().float()
        assert torch.isfinite(observed).all(), name
        # Existing OPS BF16 backward UT tolerances, unchanged.
        torch.testing.assert_close(observed, ref.grad.to(torch.bfloat16).float(), atol=0.05, rtol=0.05, msg=name)
    if train_state:
        assert torch.count_nonzero(h0.grad[..., 0]).item() == 0
    print(f"DH0-BOUNDS T={t} train_state={train_state} final={final_mode} launch={launches} canaries=exact", flush=True)


@pytest.mark.parametrize("t", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("train_state", [False, True])
@pytest.mark.parametrize("final_mode", ["off", "unused", "nonzero"])
def test_dh0_compact_buffer(monkeypatch, t, train_state, final_mode):
    _run(monkeypatch, t, train_state, final_mode)


@pytest.mark.parametrize("bt", [1, 2, 4, 8])
def test_dh0_compact_buffer_tile_edges(monkeypatch, bt):
    _run(monkeypatch, 64, True, "nonzero", forced_bt=bt)
