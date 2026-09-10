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

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from .test_fixed_topology_scheduler_npu import (
    _PREFIX_LENGTH,
    _SUFFIX_1_LENGTH,
    _SUFFIX_2_LENGTH,
    _plan,
    _tokens,
)
from .test_segment_push_pop_npu import _install_single_rank_runtime, _make_model, _parameter_grads
from verl.models.mcore.tpr import TPR_REQUEST_KEY, TPRForwardBackwardRequest
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


def test_megatron_engine_tpr_thin_entry_runs_hooks_once(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _make_model(device)
    # Importing the MindSpeed-backed engine patches the global norm spec to
    # PTNorm. Build this tiny Apex-free fixture first, as the model-side tests do.
    from verl.workers.engine.megatron.transformer_impl import MegatronEngine

    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffix_1 = _tokens(1100, _SUFFIX_1_LENGTH, device)
    suffix_2 = _tokens(1700, _SUFFIX_2_LENGTH, device)
    plan = _plan(prefix, suffix_1, suffix_2)

    calls = {"no_sync_enter": 0, "no_sync_exit": 0, "loss_scale": 0, "finalize": 0}

    @contextmanager
    def no_sync():
        calls["no_sync_enter"] += 1
        try:
            yield
        finally:
            calls["no_sync_exit"] += 1

    def loss_scale(loss):
        calls["loss_scale"] += 1
        return loss

    def finalize(model_chunks, num_tokens, *, force_all_reduce=False, **kwargs):
        del kwargs
        calls["finalize"] += 1
        assert model_chunks == [model]
        assert num_tokens is None
        assert force_all_reduce

    model.config.no_sync_func = no_sync
    model.config.grad_scale_func = loss_scale
    model.config.finalize_model_grads_func = finalize
    model.config.calculate_per_token_loss = False

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.module = [model]
    engine.engine_config = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        tpr_enabled=True,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1

    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, **{TPR_REQUEST_KEY: TPRForwardBackwardRequest(plan)})
    output = engine.forward_backward_batch(data, loss_function=None, forward_only=False)

    assert calls == {"no_sync_enter": 1, "no_sync_exit": 1, "loss_scale": 3, "finalize": 1}
    assert output["loss"] == output["metrics"]["tpr_loss"]
    assert output["metrics"]["tpr_cp_size"] == 1
    assert output["metrics"]["tpr_cp_backend"] == "none"
    assert output["metrics"]["tpr_peak_path_tokens"] == _PREFIX_LENGTH + _SUFFIX_1_LENGTH
    assert output["metrics"]["tpr_segment_count"] == 3
    assert output["metrics"]["tpr_direct_leaf_count"] == 2
    assert torch.isfinite(torch.tensor(output["loss"]))
    _parameter_grads(model)
