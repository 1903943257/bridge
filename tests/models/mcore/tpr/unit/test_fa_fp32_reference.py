"""CPU tests for the independent L4 diagnostic reference, no NPU required."""

import pytest
import torch

from ..parallel._fa_fp32_reference import fp32_reference, prefix_attention


@pytest.mark.parametrize("heads", [2, 4])
@pytest.mark.parametrize("scale", [None, 0.7])
def test_fp32_reference_matches_per_query_float64(heads, scale):
    generator = torch.Generator().manual_seed(44)
    shapes = {"q": (3, 1, heads, 4), "k": (3, 1, 2, 4), "v": (3, 1, 2, 4),
              "pk": (2, 1, 2, 4), "pv": (2, 1, 2, 4)}
    entry = {n: torch.randn(shape, generator=generator) for n, shape in shapes.items()}
    entry["scale"] = scale
    upstream = torch.randn(3, 1, heads * 4, generator=generator)
    actual = fp32_reference(entry, upstream)
    inputs = {n: entry[n].double().requires_grad_() for n in shapes}
    # Independent scalar head/query traversal: never forms a causal mask or
    # repeats heads, and includes only visible prefix/current values.
    rows = []
    for time in range(3):
        columns = []
        for head in range(heads):
            kv_head = head // (heads // 2)
            keys = torch.cat((inputs["pk"][:, 0, kv_head], inputs["k"][:time+1, 0, kv_head]))
            values = torch.cat((inputs["pv"][:, 0, kv_head], inputs["v"][:time+1, 0, kv_head]))
            scores = keys @ inputs["q"][time, 0, head] * (0.5 if scale is None else scale)
            columns.append(scores.softmax(0) @ values)
        rows.append(torch.cat(columns))
    output = torch.stack(rows).unsqueeze(1)
    grads = torch.autograd.grad(output, tuple(inputs.values()), grad_outputs=upstream.double())
    expected = {"output": output, **dict(zip(shapes, grads))}
    for name in expected:
        assert actual[name].dtype == torch.float32 and actual[name].device.type == "cpu"
        torch.testing.assert_close(actual[name].double(), expected[name], atol=2e-6, rtol=2e-5)


def test_reference_masks_future_current_values():
    q = torch.ones(3, 1, 1, 1)
    k = torch.zeros_like(q)
    pk = torch.zeros(2, 1, 1, 1)
    v = torch.tensor([3., 6., 100.]).reshape_as(q).requires_grad_()
    pv = torch.tensor([0., 0.]).reshape_as(pk).requires_grad_()
    output = prefix_attention(q, k, v, pk, pv, None)
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([1., 2.25, 21.8]))
    dv, dpv = torch.autograd.grad(output[0].sum(), (v, pv))
    torch.testing.assert_close(dv.flatten(), torch.tensor([1/3, 0., 0.]))
    torch.testing.assert_close(dpv.flatten(), torch.tensor([1/3, 1/3]))
