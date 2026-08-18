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

import pytest
import torch
import torch.nn.functional as F

from test_segment_push_pop_npu import (
    _VOCAB_SIZE,
    _assert_gradients_close,
    _install_single_rank_runtime,
    _make_model,
    _parameter_grads,
)
from verl.models.mcore.dta import (
    FixedTopologyScheduler,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
)
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

_PREFIX_LENGTH = 1024
_SUFFIX_1_LENGTH = 512
_SUFFIX_2_LENGTH = 256
_TOTAL_LOSS_WEIGHT = 2814


class _RecordingExecutor(SegmentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root_relayed_gradients = None

    def pop(self, segment_id):
        if segment_id == self.plan.root_id:
            self.root_relayed_gradients = {
                layer: (key.detach().clone(), value.detach().clone())
                for layer, (key, value) in self.kv_stack.get_new_kv_gradients(segment_id).items()
            }
        return super().pop(segment_id)


def _tokens(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device) % _VOCAB_SIZE


def _internal_terms(tokens, *, weight=1.0):
    tokens = tokens.cpu()
    return tuple(
        SegmentLossTerm(index, int(tokens[index + 1]), weight=weight)
        for index in range(tokens.numel() - 1)
    )


def _plan(prefix, suffix_1, suffix_2):
    prefix_cpu = prefix.cpu()
    suffix_1_cpu = suffix_1.cpu()
    suffix_2_cpu = suffix_2.cpu()
    prefix_terms = _internal_terms(prefix_cpu, weight=2.0) + (
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(suffix_1_cpu[0])),
        SegmentLossTerm(_PREFIX_LENGTH - 1, int(suffix_2_cpu[0])),
    )
    return SegmentPlan(
        [
            SegmentSpec(0, None, prefix_cpu, 0, 0, prefix_terms),
            SegmentSpec(
                1,
                0,
                suffix_1_cpu,
                _PREFIX_LENGTH,
                _PREFIX_LENGTH,
                _internal_terms(suffix_1_cpu),
            ),
            SegmentSpec(
                2,
                0,
                suffix_2_cpu,
                _PREFIX_LENGTH,
                _PREFIX_LENGTH,
                _internal_terms(suffix_2_cpu),
            ),
        ],
        root_id=0,
    )


def _trajectory_loss(model, tokens):
    length = tokens.numel()
    position_ids = torch.arange(length, device=tokens.device).unsqueeze(0)
    causal_mask = torch.triu(
        torch.ones((1, 1, length, length), dtype=torch.bool, device=tokens.device),
        diagonal=1,
    )
    logits = model(
        input_ids=tokens.unsqueeze(0),
        position_ids=position_ids,
        attention_mask=causal_mask,
    )
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, _VOCAB_SIZE),
        tokens[1:].reshape(-1),
        reduction="sum",
    ) / _TOTAL_LOSS_WEIGHT


def test_fixed_branching_schedule_matches_two_full_trajectories(monkeypatch):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)
    model = _make_model(device)
    prefix = _tokens(17, _PREFIX_LENGTH, device)
    suffix_1 = _tokens(1100, _SUFFIX_1_LENGTH, device)
    suffix_2 = _tokens(1700, _SUFFIX_2_LENGTH, device)

    reference_loss_1 = _trajectory_loss(model, torch.cat((prefix, suffix_1)))
    reference_loss_1.backward()
    reference_loss_2 = _trajectory_loss(model, torch.cat((prefix, suffix_2)))
    reference_loss_2.backward()
    reference_loss = reference_loss_1.detach() + reference_loss_2.detach()
    reference_gradients = _parameter_grads(model)

    model.zero_grad(set_to_none=True)
    plan = _plan(prefix, suffix_1, suffix_2)
    assert plan.total_loss_weight == _TOTAL_LOSS_WEIGHT
    executor = _RecordingExecutor(model, plan, expected_layer_numbers=(1, 2))
    result = FixedTopologyScheduler(plan, executor).run()
    dta_gradients = _parameter_grads(model)

    assert result.pushed_segment_count == result.popped_segment_count == 1
    assert result.direct_leaf_count == 2
    assert result.executed_segment_count == 3
    assert result.peak_path_tokens == _PREFIX_LENGTH + _SUFFIX_1_LENGTH
    assert executor.root_relayed_gradients is not None
    assert set(executor.root_relayed_gradients) == {1, 2}
    assert all(
        torch.count_nonzero(gradient).item() > 0
        for pair in executor.root_relayed_gradients.values()
        for gradient in pair
    )
    executor.kv_stack.assert_empty()
    torch.testing.assert_close(
        result.normalized_loss.float(),
        reference_loss.float(),
        atol=2e-2,
        rtol=2e-2,
    )
    _assert_gradients_close(dta_gradients, reference_gradients)
