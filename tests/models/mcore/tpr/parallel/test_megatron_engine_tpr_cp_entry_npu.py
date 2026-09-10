# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Engine-level TPR CP correctness using real Megatron/MindSpeed groups.

CP=2::

    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29509 \
        -m pytest -s -v -k cp2 \
        tests/models/mcore/tpr/parallel/test_megatron_engine_tpr_cp_entry_npu.py

Hybrid CP=4::

    torchrun --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29510 \
        -m pytest -s -v -k cp4 \
        tests/models/mcore/tpr/parallel/test_megatron_engine_tpr_cp_entry_npu.py
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tensordict import TensorDict

from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest
from verl.utils import tensordict_utils as tu

from ._cp_gdn_test_utils import clone_parameter_gradients, named_tensor_comparison
from ._tpr_cp_test_utils import (
    _GRAD_ATOL,
    _GRAD_COSINE_MIN,
    _GRAD_RELATIVE_L2_TOL,
    _GRAD_RTOL,
    _LOSS_RELATIVE_TOL,
    _equivalence_tpr_plan,
    _logical_logprob_indices,
    _make_model,
    _run_independent_cp_reference,
    _tokens,
    cp_runtime,
    hybrid_cp_runtime,
)


_CP2_ENGINE_CASES = ((128, 64, 32), (127, 63, 31))
_CP4_ENGINE_CASES = ((128, 64, 32), (127, 63, 31))
_MINDSPEED_ALGORITHM = {
    "allgather": "kvallgather_cp_algo",
    "ulysses": "ulysses_cp_algo",
    "ring": "megatron_cp_algo",
    "hybrid": "hybrid_cp_algo",
}


def _run_engine_tpr(model, plan, runtime, *, expected_backend):
    # Import after the fixture has installed the target MindSpeed patch set.
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine

    calls = {"no_sync_enter": 0, "no_sync_exit": 0, "finalize": 0}

    @contextmanager
    def no_sync():
        calls["no_sync_enter"] += 1
        try:
            yield
        finally:
            calls["no_sync_exit"] += 1

    def finalize(model_chunks, num_tokens, *, force_all_reduce=False, **kwargs):
        del kwargs
        calls["finalize"] += 1
        assert model_chunks == [model]
        assert num_tokens is None
        assert force_all_reduce
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                raise AssertionError(f"missing parameter gradient before CP finalize: {name}")
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=runtime.cp_group)
            parameter.grad.div_(runtime.cp_size)

    model.zero_grad(set_to_none=True)
    model.config.context_parallel_algo = _MINDSPEED_ALGORITHM[expected_backend]
    model.config.no_sync_func = no_sync
    model.config.grad_scale_func = lambda loss: loss
    model.config.finalize_model_grads_func = finalize
    model.config.calculate_per_token_loss = False

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.module = [model]
    engine.tf_config = model.config
    engine.engine_config = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=runtime.cp_size,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        dynamic_context_parallel=False,
        tpr_cp_backend=None,
        override_transformer_config={},
        tpr_enabled=True,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1

    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})
    assert list(data.keys()) == [TPR_REQUEST_KEY]
    output = engine.forward_backward_batch(data, loss_function=None, forward_only=False)

    assert calls == {"no_sync_enter": 1, "no_sync_exit": 1, "finalize": 1}
    assert output["metrics"]["tpr_cp_size"] == runtime.cp_size
    assert output["metrics"]["tpr_cp_backend"] == expected_backend
    expected_peak_path = max(
        segment.prefix_length + segment.length for segment in plan.segments.values()
    )
    assert output["metrics"]["tpr_peak_path_tokens"] == expected_peak_path
    assert output["loss"] == pytest.approx(
        output["metrics"]["tpr_loss_sum"] / plan.total_loss_weight,
        rel=1e-6,
        abs=1e-6,
    )

    rank_losses = [torch.empty((), dtype=torch.float32, device=runtime.device) for _ in range(runtime.cp_size)]
    local_loss = torch.tensor(output["loss"], dtype=torch.float32, device=runtime.device)
    dist.all_gather(rank_losses, local_loss, group=runtime.cp_group)
    assert all(torch.equal(value, rank_losses[0]) for value in rank_losses[1:])
    return output, clone_parameter_gradients(model)


def _run_engine_equivalence(
    runtime,
    prefix_length,
    first_suffix_length,
    second_suffix_length,
    backend,
):
    dist.barrier(group=runtime.cp_group)
    torch.manual_seed(290000 + prefix_length + first_suffix_length + second_suffix_length)
    max_sequence_length = prefix_length + max(first_suffix_length, second_suffix_length)
    reference_model = _make_model(runtime, max_sequence_length=max_sequence_length)
    tpr_model = _make_model(runtime, max_sequence_length=max_sequence_length)
    tpr_model.load_state_dict(reference_model.state_dict(), strict=True)

    prefix = _tokens(17, prefix_length)
    first = _tokens(701, first_suffix_length)
    second = _tokens(1301, second_suffix_length)
    first_trajectory = torch.cat((prefix, first))
    second_trajectory = torch.cat((prefix, second))
    plan = _equivalence_tpr_plan(prefix, first, second)
    logical_indices, logical_count = _logical_logprob_indices(first_trajectory, second_trajectory)

    reference = _run_independent_cp_reference(
        reference_model,
        first_trajectory,
        second_trajectory,
        runtime,
        logical_indices,
        logical_count,
    )
    output, actual_gradients = _run_engine_tpr(
        tpr_model,
        plan,
        runtime,
        expected_backend=backend,
    )

    reference_loss = float(reference.normalized_loss.item())
    actual_loss = float(output["loss"])
    loss_relative = abs(actual_loss - reference_loss) / max(abs(reference_loss), 1e-12)
    if loss_relative > _LOSS_RELATIVE_TOL:
        raise AssertionError(
            f"loss relative difference {loss_relative:.6e} exceeds {_LOSS_RELATIVE_TOL:.6e}"
        )

    gradient_metrics, worst_gradient = named_tensor_comparison(
        reference.parameter_gradients,
        actual_gradients,
    )
    for name in sorted(reference.parameter_gradients):
        torch.testing.assert_close(
            actual_gradients[name],
            reference.parameter_gradients[name],
            atol=_GRAD_ATOL,
            rtol=_GRAD_RTOL,
            msg=lambda message, name=name: f"parameter gradient mismatch for {name}: {message}",
        )
    if gradient_metrics.relative_l2 > _GRAD_RELATIVE_L2_TOL:
        raise AssertionError(
            f"parameter-gradient relative L2 {gradient_metrics.relative_l2:.6e} "
            f"exceeds {_GRAD_RELATIVE_L2_TOL:.6e}"
        )
    if gradient_metrics.cosine < _GRAD_COSINE_MIN:
        raise AssertionError(
            f"parameter-gradient cosine {gradient_metrics.cosine:.9f} "
            f"is below {_GRAD_COSINE_MIN:.9f}"
        )

    if runtime.rank == 0:
        print(
            f"\nMegatronEngine TPR CP={runtime.cp_size} {backend} correctness passed\n"
            f"  topology: P={prefix_length}, S1={first_suffix_length}, S2={second_suffix_length}\n"
            f"  loss reference/TPR: {reference_loss:.9f} / {actual_loss:.9f}\n"
            f"  loss relative diff: {loss_relative:.6e}\n"
            f"  gradient tensors: {len(actual_gradients)}\n"
            f"  gradient relative L2: {gradient_metrics.relative_l2:.6e}\n"
            f"  gradient cosine: {gradient_metrics.cosine:.9f}\n"
            f"  worst gradient: {worst_gradient[0]} ({worst_gradient[1].relative_l2:.6e})"
        )
    dist.barrier(group=runtime.cp_group)


@pytest.mark.parametrize(
    ("prefix_length", "first_suffix_length", "second_suffix_length"),
    _CP2_ENGINE_CASES,
)
@pytest.mark.parametrize("backend", ("allgather", "ulysses", "ring"))
def test_cp2_megatron_engine_tpr_matches_independent_cp(
    cp_runtime,
    prefix_length,
    first_suffix_length,
    second_suffix_length,
    backend,
):
    _run_engine_equivalence(
        cp_runtime,
        prefix_length,
        first_suffix_length,
        second_suffix_length,
        backend,
    )


@pytest.mark.parametrize(
    ("prefix_length", "first_suffix_length", "second_suffix_length"),
    _CP4_ENGINE_CASES,
)
def test_cp4_megatron_engine_hybrid_tpr_matches_independent_cp(
    hybrid_cp_runtime,
    prefix_length,
    first_suffix_length,
    second_suffix_length,
):
    _run_engine_equivalence(
        hybrid_cp_runtime,
        prefix_length,
        first_suffix_length,
        second_suffix_length,
        "hybrid",
    )
