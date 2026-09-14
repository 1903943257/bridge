"""Independent CPU FP32 oracle: explicit prefix + causal current attention.

No fused attention, Ring helpers, autocast, or NPU matmul are used. Inputs are
the captured (already BF16-rounded) post-RoPE values, not pre-rounding values.
"""

import math

import torch


def prefix_attention(q, k, v, pk, pv, scale):
    """[S,B,H,D] inputs -> [S,B,Hq*D]; contiguous query-head GQA groups."""
    assert q.ndim == 4 and q.shape[1] == 1
    assert k.shape == v.shape and pk.shape == pv.shape
    assert k.shape[0] == q.shape[0] and k.shape[1:] == pk.shape[1:]
    assert q.shape[-1] == k.shape[-1] and q.shape[2] % k.shape[2] == 0
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    assert math.isfinite(scale) and scale > 0
    key = torch.cat((pk, k), dim=0).repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    value = torch.cat((pv, v), dim=0).repeat_interleave(q.shape[2] // k.shape[2], dim=2)
    scores = (q.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)) * scale
    # Query i sees all prefix tokens and current tokens j <= i, including self.
    allowed = torch.arange(key.shape[0], device=q.device)[None, :] <= (
        pk.shape[0] + torch.arange(q.shape[0], device=q.device)[:, None])
    probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
    output = probabilities @ value.permute(1, 2, 0, 3)
    return output.permute(2, 0, 1, 3).contiguous().flatten(2)


def fp32_reference(entry, upstream):
    names = ("q", "k", "v", "pk", "pv")
    with torch.enable_grad(), torch.autocast(device_type="cpu", enabled=False):
        inputs = {name: entry[name].detach().to(device="cpu", dtype=torch.float32)
                  .clone().requires_grad_(True) for name in names}
        output = prefix_attention(**inputs, scale=entry["scale"])
        assert output.dtype == torch.float32
        grads = torch.autograd.grad(output, tuple(inputs.values()),
                                    grad_outputs=upstream.detach().cpu().float())
    return {"output": output.detach(), **{n: g.detach() for n, g in zip(names, grads)}}
