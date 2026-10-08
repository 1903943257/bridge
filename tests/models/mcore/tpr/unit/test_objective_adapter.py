from functools import partial

import pytest
import torch

pytest.importorskip("tensordict")
from tensordict import TensorDict

from verl.models.mcore.tpr.objective_adapter import SegmentPPOObjectiveAdapter
from verl.models.mcore.tpr.segment_plan import SegmentSpec
from verl.models.mcore.tpr.tree_plan_builder import SegmentObjectiveRef
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss


class _ActorConfig:
    loss_agg_mode = "token-mean"
    policy_loss = {"loss_mode": "vanilla"}
    clip_ratio = 0.2
    clip_ratio_low = None
    clip_ratio_high = None
    entropy_coeff = 0.0
    use_kl_loss = False
    loss_scale_factor = None

    def __init__(self):
        self.global_batch_info = {}

    def get(self, name, default=None):
        return getattr(self, name, default)


def _batch(*, temperatures=(1.0, 1.0), with_denominator=True):
    data = TensorDict(
        {
            "input_ids": torch.tensor([[10, 11, 12, 13], [10, 11, 12, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "old_log_probs": torch.tensor([[-1.4, -1.1], [-1.5, 0.0]]),
            "advantages": torch.tensor([[1.0, 2.0], [0.5, 0.0]]),
            "temperature": torch.tensor(temperatures),
        },
        batch_size=[2],
    )
    if with_denominator:
        tu.assign_non_tensor(data, batch_num_tokens=3, dp_size=1, global_batch_size=2)
    return data


def _segment():
    return SegmentSpec(
        segment_id=0,
        parent_id=None,
        token_ids=torch.tensor([10, 11], dtype=torch.long),
        position_start=0,
        prefix_length=0,
    )


def _refs():
    return (
        SegmentObjectiveRef(0, 0, 1, 0, 0),
        SegmentObjectiveRef(0, 1, 2, 0, 1),
        SegmentObjectiveRef(0, 0, 1, 1, 0),
    )


def _cpu_log_probs(logits, labels):
    return torch.log_softmax(logits, dim=-1).gather(-1, labels[:, None]).squeeze(-1)


def _baseline(logits, temps):
    selected = torch.stack(
        [
            logits[0, 0] / temps[0],
            logits[0, 1] / temps[0],
            logits[0, 0] / temps[1],
        ]
    )
    new_lp = _cpu_log_probs(selected, torch.tensor([1, 2, 1]))
    old_lp = torch.tensor([-1.4, -1.1, -1.5])
    advantages = torch.tensor([1.0, 2.0, 0.5])
    ratios = torch.exp((new_lp - old_lp).clamp(-20, 20))
    pg1 = -advantages * ratios
    pg2 = -advantages * ratios.clamp(0.8, 1.2)
    return torch.maximum(pg1, pg2).sum() / 3


@pytest.mark.parametrize("temps", [(1.0, 1.0), (1.0, 2.0)])
def test_compact_adapter_reuses_native_ppo_loss_and_matches_full_gradients(temps):
    torch.manual_seed(3)
    data = _batch(temperatures=temps)
    segment = _segment()
    logits = torch.randn(1, 2, 5, requires_grad=True)
    baseline_logits = logits.detach().clone().requires_grad_(True)
    native_loss_fn = partial(ppo_loss, config=_ActorConfig())

    adapter = SegmentPPOObjectiveAdapter(
        data, native_loss_fn, log_prob_fn=_cpu_log_probs
    )
    loss, metrics = adapter.compute_loss(segment, logits, _refs())
    expected = _baseline(baseline_logits, temps)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, baseline_logits.grad, rtol=1e-5, atol=1e-6)
    assert "actor/pg_loss" in metrics


def test_duplicate_logical_refs_share_physical_logprob_but_not_objective():
    logits = torch.randn(1, 2, 5, requires_grad=True)
    seen = []

    def logprob_fn(logits, labels):
        seen.append(len(labels))
        return _cpu_log_probs(logits, labels)

    adapter = SegmentPPOObjectiveAdapter(
        _batch(),
        partial(ppo_loss, config=_ActorConfig()),
        log_prob_fn=logprob_fn,
    )
    loss, _ = adapter.compute_loss(_segment(), logits, _refs())
    assert seen == [2]  # (q=0,target=1) reused for both rows
    assert loss.requires_grad


def test_reject_missing_global_normalization_and_wrong_segment_ref():
    logits = torch.randn(1, 2, 5, requires_grad=True)
    loss_fn = partial(ppo_loss, config=_ActorConfig())
    missing = SegmentPPOObjectiveAdapter(
        _batch(with_denominator=False), loss_fn, log_prob_fn=_cpu_log_probs
    )
    with pytest.raises(ValueError, match="batch_num_tokens"):
        missing.compute_loss(_segment(), logits, _refs())

    adapter = SegmentPPOObjectiveAdapter(_batch(), loss_fn, log_prob_fn=_cpu_log_probs)
    invalid = (SegmentObjectiveRef(7, 0, 1, 0, 0),)
    with pytest.raises(ValueError, match="another segment"):
        adapter.compute_loss(_segment(), logits, invalid)


def test_unsupported_sequence_objectives_fail_before_forward():
    cfg = _ActorConfig()
    cfg.policy_loss = {"loss_mode": "gspo"}
    with pytest.raises(NotImplementedError, match="vanilla"):
        SegmentPPOObjectiveAdapter(_batch(), partial(ppo_loss, config=cfg))
