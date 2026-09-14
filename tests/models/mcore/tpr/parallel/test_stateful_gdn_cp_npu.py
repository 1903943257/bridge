"""Stage 4.1: single real Qwen GDN, non-packed CP1 vs CP2, state VJPs.

No tree, FA, THD, projection chunk controls or threshold relaxation.
"""

import pytest
import torch
import torch.distributed as dist

from verl.utils.device import is_torch_npu_available
from baseline._qwen35_baseline_utils import (
    AllToAllProbe, DTYPE, HIDDEN_SIZE, allreduce_parameter_gradients,
    assert_gradient_maps_close, broadcast_module_state, destroy_npu_runtime,
    gather_native_zigzag, gradient_map_diagnostics, initialize_npu_runtime,
    process_groups, qwen35_config, zigzag_indices,
)

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires two Ascend NPUs", allow_module_level=True)


@pytest.fixture(scope="module")
def runtime():
    value = initialize_npu_runtime(world_size=2)
    yield value
    destroy_npu_runtime(value)


def _model(runtime, cp, layer_number=1):
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.ssm.gated_delta_net import GatedDeltaNetSubmodules
    import mindspeed.core.ssm.gated_delta_net as native
    from verl.models.mcore.tpr.gated_delta_net import TPRGatedDeltaNet

    backend = LocalSpecProvider()
    submodules = GatedDeltaNetSubmodules(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )
    cls = TPRGatedDeltaNet if cp == 1 else native.GatedDeltaNet
    return cls(
        qwen35_config(cp_size=cp), submodules=submodules, layer_number=layer_number,
        bias=False, conv_bias=False, conv_init=0.1, use_qk_l2norm=True,
        A_init_range=(1, 16), pg_collection=process_groups(runtime, cp_size=cp),
    ).to(device=runtime.device, dtype=DTYPE).train()


def _random(shape, device, seed, dtype=DTYPE):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 0.02).to(device=device, dtype=dtype)


def _state_shard(state, rank):
    from verl.models.mcore.tpr.prefix_state import GDNLayerState

    # Qwen0.8B: Q/K/V each have 2048 channels; slice each independently.
    conv = torch.cat([part[:, rank*1024:(rank+1)*1024]
                      for part in state.conv_state.split(2048, dim=1)], dim=1)
    recurrent = state.recurrent_state[:, rank*8:(rank+1)*8]
    return GDNLayerState(conv.contiguous(), recurrent.contiguous())


def _state_map(state):
    return {"conv": state.conv_state, "recurrent": state.recurrent_state}


def _grads(model):
    result = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        result[name] = parameter.grad.detach().clone()
    return result


def _compare(label, left, right):
    for name in left:
        assert torch.isfinite(left[name]).all() and torch.isfinite(right[name]).all(), (label, name)
    print(f"STAGE-4.1 {label}: {gradient_map_diagnostics(left, right)}", flush=True)
    assert_gradient_maps_close(left, right, rtol=0.02, cosine_min=0.999)


@pytest.mark.parametrize("with_initial", [False, True], ids=["zero-state", "initial-state-vjp"])
def test_single_stateful_gdn_cp2_matches_cp1(runtime, with_initial):
    import mindspeed.core.ssm.gated_delta_net as native
    from verl.models.mcore.tpr.parallel.gdn_state import forward_gdn_cp_with_state, gdn_cp_state_shapes
    from verl.models.mcore.tpr.prefix_state import GDNLayerState
    from verl.models.mcore.tpr.context import TPRAttentionContext, use_tpr_attention_context

    torch.manual_seed(410001)
    reference, target = _model(runtime, 1), _model(runtime, 2)
    broadcast_module_state(reference)
    target.load_state_dict(reference.state_dict(), strict=True)
    assert gdn_cp_state_shapes(reference) == ((1, 6144, 4), (1, 16, 128, 128))
    assert gdn_cp_state_shapes(target) == ((1, 3072, 4), (1, 8, 128, 128))
    x = _random((128, 1, HIDDEN_SIZE), runtime.device, 410002)
    desired = _random(x.shape, runtime.device, 410003)
    indices = zigzag_indices(128, cp_rank=runtime.cp_group.rank(), cp_size=2, device=runtime.device)
    ref_x = x.clone().requires_grad_(True)
    cp_x = x.index_select(0, indices).contiguous().requires_grad_(True)
    ref_initial = cp_initial = None
    if with_initial:
        ref_initial = GDNLayerState(
            _random((1, 6144, 4), runtime.device, 410004).requires_grad_(True),
            _random((1, 16, 128, 128), runtime.device, 410005, torch.float32).requires_grad_(True),
        )
        local = _state_shard(ref_initial, runtime.cp_group.rank())
        cp_initial = GDNLayerState(local.conv_state.detach().clone().requires_grad_(True),
                                   local.recurrent_state.detach().clone().requires_grad_(True))

    def objective(output, expected, state):
        # Every CP rank contributes disjoint output tokens and disjoint state
        # channels/heads. Use GLOBAL denominators; never multiply by CP twice.
        return ((output.float() - expected.float()).square().sum() / x.numel()
                + state.conv_state.float().square().sum() / (6144 * 4)
                + state.recurrent_state.float().square().sum() / (16 * 128 * 128))

    with AllToAllProbe(native) as ref_probe:
        # Independent reference: the existing Stage 3.2 CP1 implementation,
        # not the new CP adapter run with size=1.
        context = TPRAttentionContext(
            prefix_length=128 if with_initial else 0, suffix_length=128,
            initial_gdn_states={} if ref_initial is None else {1: ref_initial},
        )
        with use_tpr_attention_context(context):
            ref_output, ref_bias = reference(ref_x, attention_mask=None)
        context.assert_new_gdn_layers((1,))
        ref_state = context.new_gdn_states[1]
        assert ref_bias is None
        ref_loss = objective(ref_output, desired, ref_state)
        ref_loss.backward()
    with AllToAllProbe(native) as cp_probe:
        (cp_output, cp_bias), cp_state = forward_gdn_cp_with_state(target, cp_x, cp_initial)
        assert cp_bias is None
        cp_loss = objective(cp_output, desired.index_select(0, indices), cp_state)
        cp_loss.backward()
    assert not ref_probe.calls, ref_probe.calls
    assert cp_probe.count("cp2hp") == 6 and cp_probe.count("hp2cp") == 1, cp_probe.calls
    allreduce_parameter_gradients(target, runtime.cp_group)
    total_loss = cp_loss.detach().clone()
    dist.all_reduce(total_loss, group=runtime.cp_group)
    output = gather_native_zigzag(cp_output.detach(), runtime.cp_group)
    dx = gather_native_zigzag(cp_x.grad, runtime.cp_group)
    local_reference = _state_shard(ref_state, runtime.cp_group.rank())
    print(f"STAGE-4.1 rank={runtime.rank} state shapes conv/recurrent="
          f"{tuple(cp_state.conv_state.shape)}/{tuple(cp_state.recurrent_state.shape)} "
          f"dtypes={cp_state.conv_state.dtype}/{cp_state.recurrent_state.dtype}; "
          f"loss CP1/CP2={ref_loss.item():.9e}/{total_loss.item():.9e}; "
          "CP1 A2A=0; CP2 cp2hp=6/hp2cp=1", flush=True)
    torch.testing.assert_close(output, ref_output.detach(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(total_loss, ref_loss.detach(), atol=1e-7, rtol=5e-4)
    _compare("input-gradient", {"x": ref_x.grad}, {"x": dx})
    _compare("parameter-gradient", _grads(reference), _grads(target))
    for name in ("conv", "recurrent"):
        _compare(f"final-state/{name}", {name: _state_map(local_reference)[name]},
                 {name: _state_map(cp_state)[name]})
    if with_initial:
        reference_dstate = GDNLayerState(ref_initial.conv_state.grad, ref_initial.recurrent_state.grad)
        local_dstate = _state_shard(reference_dstate, runtime.cp_group.rank())
        for name, value in _state_map(cp_initial).items():
            assert value.grad is not None and torch.count_nonzero(value.grad).item() > 0
            _compare(f"initial-state-gradient/{name}", {name: _state_map(local_dstate)[name]},
                     {name: value.grad})
    print(f"STAGE-4.1 PASS rank={runtime.rank} with_initial={with_initial}", flush=True)
