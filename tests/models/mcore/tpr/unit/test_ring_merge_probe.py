"""CPU checks for the test-only block oracle, without loading Ring/NPU."""

import pytest
import torch

from ..parallel._ring_merge_probe import block_reference


@pytest.mark.parametrize("causal", [False, True])
def test_block_reference_gqa_and_statistics(causal):
    q = torch.zeros(3, 4, 2)
    k = torch.zeros(3, 2, 2)
    v = torch.arange(12.).reshape(3, 2, 2)
    output, maximum, total = block_reference(q, k, v, scale=0.5, causal=causal, mask=None)
    for t in range(3):
        visible = t + 1 if causal else 3
        expected = v[:visible].mean(0).repeat_interleave(2, dim=0)
        torch.testing.assert_close(output[t], expected)
        torch.testing.assert_close(total[t], torch.full((4,), float(visible)))
    assert torch.count_nonzero(maximum) == 0


def test_explicit_mask_and_fully_masked_guard():
    q = torch.zeros(2, 1, 1)
    k = torch.zeros(3, 1, 1)
    v = torch.tensor([2., 4., 100.]).reshape(3, 1, 1)
    mask = torch.tensor([[False, True, True], [False, False, True]])
    output, _, total = block_reference(q, k, v, scale=1., causal=False, mask=mask)
    torch.testing.assert_close(output.flatten(), torch.tensor([2., 3.]))
    torch.testing.assert_close(total.flatten(), torch.tensor([1., 2.]))
    with pytest.raises(AssertionError, match="fully-masked"):
        block_reference(q, k, v, scale=1., causal=False, mask=torch.ones_like(mask))
