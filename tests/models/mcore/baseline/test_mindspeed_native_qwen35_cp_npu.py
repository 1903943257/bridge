"""Whole native Qwen3.5 CP1/CP2, no TPR, AG or projection control.

Reuses P3 random-init GPTModel/native ModelSpec and unchanged numerical gates.
All 24 layer inputs/outputs and their gradients are compared in logical order.
"""

import gc
from copy import deepcopy
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from . import test_qwen35_hybrid_cp_npu as p3
from ._qwen35_baseline_utils import (
    SEQUENCE_LENGTH, AllToAllProbe, make_qwen35_model, assert_hybrid_architecture,
    broadcast_module_state, full_tokens, zigzag_indices, selected_output_and_loss,
    gather_native_zigzag, allreduce_parameter_gradients, gradient_map_diagnostics,
    assert_tensor_close_by_norm, assert_tensor_gradient_close, assert_gradient_maps_close,
)

runtime = p3.runtime
pytestmark = p3.pytestmark


@contextmanager
def layer_trace(model):
    records, handles = {}, []

    def record(name, tensor):
        assert isinstance(tensor, torch.Tensor) and tensor.requires_grad
        records[name] = tensor.detach().cpu().clone()
        def gradient(grad):
            records[name + '.grad'] = grad.detach().cpu().clone()
        tensor.register_hook(gradient)

    for layer in model.decoder.layers:
        number = layer.layer_number
        def pre(module, args, kwargs, number=number):
            value = kwargs.get('hidden_states', args[0] if args else None)
            record(f'L{number:02d}.input', value)
        def post(module, args, output, number=number):
            record(f'L{number:02d}.output', output[0] if isinstance(output, tuple) else output)
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def test_mindspeed_native_qwen35_cp(runtime, monkeypatch):
    import mindspeed.core.ssm.gated_delta_net as gdn
    import mindspeed.core.context_parallel.dot_product_attention as dpa
    from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel

    assert SEQUENCE_LENGTH % 4 == 0
    # Verify the actual native DPA binding before instrumenting it.
    assert dpa.ringattn_context_parallel is ringattn_context_parallel
    results, initial = {}, None
    for cp in (1, 2):
        torch.manual_seed(353001)
        model = make_qwen35_model(runtime, cp_size=cp, tpr=False)
        gdn_layers, fa_layers = assert_hybrid_architecture(model)
        assert all(type(layer.self_attention) is gdn.GatedDeltaNet for layer in gdn_layers)
        assert not any('.tpr.' in type(module).__module__ for module in model.modules())
        assert model.config.attention_dropout == model.config.hidden_dropout == 0
        if initial is None:
            broadcast_module_state(model, src=0)
            initial = {
                n: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
                for n, v in model.state_dict().items()
            }
        else:
            model.load_state_dict(initial, strict=True)
        current = model.state_dict()
        assert current.keys() == initial.keys(), 'initial state keys differ'
        for name, value in current.items():
            expected = initial[name]
            if isinstance(value, torch.Tensor):
                assert isinstance(expected, torch.Tensor), name
                assert torch.equal(value.detach().cpu(), expected), name
            else:
                # TE extra-state entries can legitimately be None.
                assert type(value) is type(expected) and value == expected, name
        tokens, positions, labels, valid = full_tokens(runtime.device)
        if cp == 2:
            idx = zigzag_indices(SEQUENCE_LENGTH, cp_rank=runtime.cp_group.rank(),
                                 cp_size=2, device=runtime.device)
            tokens, positions = tokens[:, idx], positions[:, idx]
            labels, valid = labels[idx], valid[idx]
        native_calls = []
        def native(*args, **kwargs):
            params = args[4]
            assert params['causal'] and params['cp_size'] == 2
            native_calls.append((tuple(args[0].shape), tuple(args[1].shape)))
            return ringattn_context_parallel(*args, **kwargs)

        with monkeypatch.context() as patch, AllToAllProbe(gdn) as a2a, layer_trace(model) as trace:
            patch.setattr(dpa, 'ringattn_context_parallel', native)
            logits = model(input_ids=tokens, position_ids=positions, attention_mask=None)
            probe, loss = selected_output_and_loss(logits, labels, valid)
            loss.backward()
        assert len(trace) == 24 * 4, list(trace)
        if cp == 1:
            assert not a2a.calls and not native_calls
        else:
            assert len(native_calls) == 6, native_calls
            assert a2a.count('cp2hp') == 108 and a2a.count('hp2cp') == 18
            allreduce_parameter_gradients(model, runtime.cp_group)  # exactly once
        scalar = loss.detach().clone()
        if cp == 2:
            dist.all_reduce(scalar, group=runtime.cp_group)
            probe = gather_native_zigzag(probe.detach(), runtime.cp_group)
            # Small boundary tensors only; canonical gather outside autograd.
            for name in sorted(trace):
                trace[name] = gather_native_zigzag(trace[name].to(runtime.device), runtime.cp_group).cpu()
        gradients = {n: v.detach().cpu().clone() for n, v in p3._parameter_gradients(model).items()}
        results[cp] = dict(loss=scalar.cpu(), probe=probe.detach().cpu(), trace=trace, gradients=gradients)
        print(f'NATIVE-QWEN CP={cp} layers=18GDN+6FA S={SEQUENCE_LENGTH} '
              f'loss={scalar.item():.9f} A2A={a2a.count("cp2hp")}/{a2a.count("hp2cp")} '
              f'native_Ring={len(native_calls)} GDR={gdn_layers[0].self_attention.gated_delta_rule} '
              f'conv={gdn.causal_conv1d}; no TPR/AG/shape-control', flush=True)
        del logits, loss, probe, model, gradients, gdn_layers, fa_layers
        gc.collect()
        torch.npu.empty_cache()

    ref, actual = results[1], results[2]
    first = None
    for layer in range(1, 25):
        fields = []
        for boundary in ('input', 'output', 'input.grad', 'output.grad'):
            name = f'L{layer:02d}.{boundary}'
            left, right = ref['trace'][name], actual['trace'][name]
            assert torch.isfinite(left).all() and torch.isfinite(right).all(), name
            m = gradient_map_diagnostics({name: left}, {name: right}).aggregate
            exact = torch.equal(left, right)
            if first is None and not boundary.endswith('.grad') and not exact:
                first = name
            fields.append(f'{boundary}={m.relative_l2:.6e}/{m.cosine:.9f}/exact:{exact}')
        if runtime.rank == 0:
            print(f'NATIVE-QWEN L{layer:02d} rel/cos ' + ' '.join(fields), flush=True)
    diagnostics = gradient_map_diagnostics(ref['gradients'], actual['gradients'])
    print(f'NATIVE-QWEN r={runtime.rank} first-forward={first} parameters={diagnostics}', flush=True)
    # Preserve P3 thresholds, evaluate all gates after diagnostics.
    failures = []
    gates = {
        'output': lambda: assert_tensor_close_by_norm(ref['probe'], actual['probe'],
            rtol=8e-2, cosine_min=0.995, max_abs=2e-1, label='native output'),
        'loss': lambda: torch.testing.assert_close(actual['loss'], ref['loss'], atol=2e-2, rtol=2e-2),
        'input-grad': lambda: assert_tensor_gradient_close(ref['trace']['L01.input.grad'],
            actual['trace']['L01.input.grad'], rtol=1e-1, cosine_min=0.99),
        'parameter-grad': lambda: assert_gradient_maps_close(ref['gradients'], actual['gradients'],
            rtol=1e-1, cosine_min=0.995),
    }
    for name, gate in gates.items():
        try:
            gate()
        except AssertionError as exc:
            failures.append(f'{name}: {exc}')
    failed = torch.tensor(bool(failures), device=runtime.device, dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=runtime.cp_group)
    if failed.item():
        pytest.fail('\n'.join(failures) or 'native numerical gate failed on peer rank')
    print('NATIVE-QWEN PASS: unchanged P3 gates; not a multi-step training claim', flush=True)
