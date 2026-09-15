"""Opt-in native-checkpoint compatibility gates; no replacement checkpoint code.

Run CP=1 first, then CP=2 with torchrun. These tests intentionally require
contracts that the current TPR/full-recompute combination does not satisfy.
They must pass before enabling full recomputation in long-context profiles.
"""

import os

import pytest
import torch

from megatron.core import tensor_parallel
from verl.models.mcore.tpr.context import (
    TPRAttentionContext,
    get_tpr_attention_context,
    use_tpr_attention_context,
)
from ..profiling.test_tpr_qwen3_ring_cp_profile_npu import profile_runtime

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_NATIVE_RECOMPUTE_CONTRACT") != "1",
    reason="Set TPR_RUN_NATIVE_RECOMPUTE_CONTRACT=1 for native checkpoint contract gates",
)


def _context(device):
    return TPRAttentionContext(
        prefix_length=0,
        suffix_length=4,
        past_key_values={},
        suffix_rotary_pos_emb=torch.zeros(4, 1, 1, 8, device=device),
    )


def test_native_full_checkpoint_preserves_pop_kv_roots(profile_runtime):
    """Pop requires layer-internal KV to be differentiable backward roots."""
    device = profile_runtime.device
    context = _context(device)
    hidden = torch.randn(4, 1, 8, device=device, requires_grad=True)

    def layer_forward(inputs):
        key = (inputs * 2).unsqueeze(2)
        value = (inputs * 3).unsqueeze(2)
        get_tpr_attention_context().set_new_kv(1, key, value)
        return inputs * 4

    with use_tpr_attention_context(context):
        output = tensor_parallel.checkpoint(layer_forward, False, hidden)
    assert output.requires_grad
    key, value = context.new_key_values[1]
    assert key.requires_grad and value.requires_grad, (
        "Native full checkpoint collected KV under no_grad; TPR Pop cannot "
        "inject prefix gradients into these non-output tensors. Configuration "
        "passthrough alone cannot establish the required autograd roots."
    )


def test_native_full_checkpoint_restores_tpr_context_on_replay(profile_runtime):
    """Native replay runs after SegmentExecutor._forward exits its context."""
    device = profile_runtime.device
    context = _context(device)
    hidden = torch.randn(4, 1, 8, device=device, requires_grad=True)
    observed_contexts = []

    def layer_forward(inputs):
        observed_contexts.append(get_tpr_attention_context())
        return inputs * inputs

    with use_tpr_attention_context(context):
        output = tensor_parallel.checkpoint(layer_forward, False, hidden)
    output.sum().backward()
    assert len(observed_contexts) == 2
    assert all(item is context for item in observed_contexts), (
        "Native checkpoint replay does not restore the external TPR attention "
        "context. Keeping the context active alone also does not solve Pop KV "
        "roots or duplicate set_new_kv collection."
    )
