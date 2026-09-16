"""Opt-in CP=2/4 OFF/ON Push/Visit/Pop gate using existing numerical tolerances."""

import os

import pytest
import torch

from verl.models.mcore.tpr import SegmentExecutor
from ..profiling.test_tpr_qwen3_ring_cp_profile_npu import profile_runtime
from . import _tpr_cp_test_utils as utils

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_RING_COALESCING_TREE") != "1",
    reason="Set TPR_RUN_RING_COALESCING_TREE=1",
)


def _compare(actual, expected):
    loss = float(actual.normalized_loss)
    reference_loss = float(expected.normalized_loss)
    assert abs(loss - reference_loss) / max(abs(reference_loss), 1e-12) <= utils._LOSS_RELATIVE_TOL
    torch.testing.assert_close(actual.target_logprobs, expected.target_logprobs,
                               atol=utils._LOGPROB_ATOL, rtol=utils._LOGPROB_RTOL)
    metrics = utils.tensor_comparison(expected.target_logprobs, actual.target_logprobs)
    assert metrics.relative_l2 <= utils._LOGPROB_RELATIVE_L2_TOL
    assert metrics.cosine >= utils._LOGPROB_COSINE_MIN
    for field in ("parameter_gradients", "prefix_gradients"):
        left, right = getattr(actual, field), getattr(expected, field)
        assert left and left.keys() == right.keys()
        if field == "prefix_gradients":
            left = {f"{layer}.{index}": tensor for layer, pair in left.items() for index, tensor in enumerate(pair)}
            right = {f"{layer}.{index}": tensor for layer, pair in right.items() for index, tensor in enumerate(pair)}
            assert all(torch.count_nonzero(tensor).item() for tensor in left.values())
        for name in left:
            torch.testing.assert_close(left[name], right[name], atol=utils._GRAD_ATOL, rtol=utils._GRAD_RTOL)
        metrics, _ = utils.named_tensor_comparison(right, left)
        assert metrics.relative_l2 <= utils._GRAD_RELATIVE_L2_TOL
        assert metrics.cosine >= utils._GRAD_COSINE_MIN


@pytest.mark.parametrize("merge_mode", ["kv", "query"])
@pytest.mark.parametrize("prefix_length,suffix_length", [(128, 64), (127, 63)])
def test_ring_coalescing_tree_off_on(profile_runtime, monkeypatch, prefix_length, suffix_length, merge_mode):
    runtime = profile_runtime
    assert runtime.cp_size in (2, 4)
    torch.manual_seed(261000)
    model = utils._make_model(runtime, max_sequence_length=prefix_length + suffix_length)
    prefix = utils._tokens(17, prefix_length)
    first = utils._tokens(701, suffix_length)
    second = utils._tokens(1301, suffix_length)
    plan = utils._equivalence_tpr_plan(prefix, first, second)
    indices, count = utils._logical_logprob_indices(torch.cat((prefix, first)), torch.cat((prefix, second)))
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_QUERY", "0")
    monkeypatch.setenv("TPR_RING_COALESCE_PREFIX_FULL", "1" if merge_mode == "query" else "0")
    reference = utils._run_independent_segmented_cp_reference(
        model, prefix, first, second, runtime, indices, count, cp_backend="ring",
    )
    captured = {}
    original_pop = SegmentExecutor.pop

    def pop(executor, segment_id):
        if segment_id == plan.root_id:
            captured.update(utils._clone_prefix_gradients(executor, segment_id=segment_id))
        return original_pop(executor, segment_id)

    monkeypatch.setattr(SegmentExecutor, "pop", pop)
    baseline = None
    for enabled in ("0", "1", "1"):
        flag = "TPR_RING_COALESCE_PREFIX_QUERY" if merge_mode == "query" else "TPR_RING_COALESCE_PREFIX_FULL"
        monkeypatch.setenv(flag, enabled)
        captured.clear()
        actual = utils._run_tpr_cp(model, plan, runtime, indices, count, cp_backend="ring")
        # _EquivalenceRun is frozen; construct a result including the captured KV gradients.
        from dataclasses import replace
        actual = replace(actual, prefix_gradients=dict(captured))
        _compare(actual, reference)
        if baseline is None:
            baseline = actual
        else:
            _compare(actual, baseline)
            assert actual.execution_trace == baseline.execution_trace
