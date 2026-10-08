"""CPU integration: original VERL PPO objective versus tree/segment accumulation.

Exercises TreePlanBuilder -> SegmentPPOObjectiveAdapter -> TPRBatchRunner
without an Ascend device or Megatron model. The physical token-logits are
trainable tensors; the same tensors are used in the dense logical oracle.
This intentionally does not test FA/GDN PrefixState gradient relay.
"""

from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tensordict")
from tensordict import TensorDict

from verl.models.mcore.tpr.objective_adapter import SegmentPPOObjectiveAdapter
from verl.models.mcore.tpr.tpr_batch_runner import TPRBatchRunner
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import ppo_loss


class _ActorConfig:
    policy_loss = {"loss_mode": "vanilla"}
    loss_agg_mode = "token-mean"
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


def _log_prob_fn(logits, labels):
    return torch.log_softmax(logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)


def _inputs():
    tokens = [
        torch.tensor([10, 11, 12, 20, 30]),
        torch.tensor([10, 11, 12, 21, 31]),
        torch.tensor([10, 11, 12, 20, 32]),
        torch.tensor([90, 91, 92, 93, 94]),
    ]
    topology = {
        "input_ids": tokens,
        "responses": [row[-3:] for row in tokens],
        "response_mask": [torch.ones(3, dtype=torch.bool) for _ in tokens],
        "loss_mask": [torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool) for _ in tokens],
    }
    keys = ["u_s_0", "u_s_1", "u_s_2", "v_s_0"]

    # Native ppo_loss uses no_padding_2_padding, which expects an ordinary
    # full-length packed model_output and the original row-wise metadata.
    batch = TensorDict(
        {
            "input_ids": torch.stack(tokens),
            "prompts": torch.stack([row[:2] for row in tokens]),
            "responses": torch.stack([row[-3:] for row in tokens]),
            "attention_mask": torch.ones(4, 5, dtype=torch.long),
            "response_mask": torch.ones(4, 3, dtype=torch.bool),
            "old_log_probs": torch.tensor(
                [[-1.5, -2.2, -3.0],
                 [-1.6, -1.9, -2.3],
                 [-1.4, -2.1, -2.7],
                 [-1.2, -1.8, -2.6]]
            ),
            "advantages": torch.tensor(
                [[1.0, -1.0, 0.2],
                 [0.7, -0.3, 0.9],
                 [-0.5, 1.1, -0.8],
                 [0.4, -1.2, 0.3]]
            ),
            "temperature": torch.tensor([1.0, 1.2, 1.0, 0.8]),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(batch, dp_size=1, batch_num_tokens=12, global_batch_size=4)
    return topology, keys, batch


def test_tree_local_native_ppo_sums_to_dense_logical_ppo_with_equal_gradients():
    torch.manual_seed(19)
    topology, keys, batch = _inputs()
    forest = build_tree_execution_plans(keys, topology, require_loss_mask_alignment=True)
    assert len(forest.trees) == 2
    assert forest.logical_loss_tokens == 12

    # Give each physical segment its own independent logits tensor. These
    # represent the *one* model compute for each unique node, not per-row copies.
    logits_by_node = {
        (tree_index, segment.segment_id): torch.randn(
            1, segment.length, 128, requires_grad=True
        )
        for tree_index, tree in enumerate(forest.trees)
        for segment in tree.segment_plan.segments.values()
    }

    native_loss_fn = partial(ppo_loss, config=_ActorConfig())
    adapter = SegmentPPOObjectiveAdapter(
        batch, native_loss_fn, log_prob_fn=_log_prob_fn
    )

    # Dense native oracle: expand physical query logits back into 4 independent
    # logical full sequences. The old-logprob / advantage values are *not*
    # deduplicated, including the two rows that share target token 20.
    logical: dict[tuple[int, int], torch.Tensor] = {}
    for tree_index, tree in enumerate(forest.trees):
        for ref in tree.objective_refs:
            logits = logits_by_node[(tree_index, ref.segment_id)]
            temp = batch["temperature"][ref.sample_row]
            lp = torch.log_softmax(
                (logits[0, ref.query_offset] / temp).float(), dim=-1
            )[ref.target_token_id]
            logical[(ref.sample_row, ref.response_offset)] = lp
    assert len(logical) == 12

    flat_logits = []
    for row in range(4):
        unused = logical[(row, 0)].new_zeros(())
        # Native no_padding_2_padding selects positions 1,2,3 for prompt_len=2.
        flat_logits.extend((
            unused,
            logical[(row, 0)],
            logical[(row, 1)],
            logical[(row, 2)],
            unused,
        ))
    full_output = {"log_probs": torch.stack(flat_logits)}
    oracle_loss, _ = native_loss_fn(
        model_output=full_output, data=batch, dp_group=None
    )
    params = tuple(logits_by_node.values())
    oracle_grads = torch.autograd.grad(
        oracle_loss, params, allow_unused=True
    )

    called = []
    def run_tree(tree_plan):
        tree_index = next(
            i for i, candidate in enumerate(forest.trees)
            if candidate is tree_plan
        )
        callback, counts = adapter.bind_tree(tree_plan)
        local_losses = []
        for segment in tree_plan.segment_plan.segments.values():
            if not counts[segment.segment_id]:
                continue
            loss, _ = callback(
                segment, logits_by_node[(tree_index, segment.segment_id)]
            )
            loss.backward()
            local_losses.append(loss.detach())
        called.append(tree_plan.tree.key)
        return SimpleNamespace(
            normalized_loss=torch.stack(local_losses).sum()
        )

    sync = []
    result = TPRBatchRunner(
        run_tree=run_tree,
        finalize_gradients=lambda: sync.append("finalize"),
    ).run(forest)

    torch.testing.assert_close(
        result.normalized_loss, oracle_loss.detach(), rtol=1e-5, atol=1e-6
    )
    for tensor, expected_grad in zip(params, oracle_grads, strict=True):
        if expected_grad is None:
            assert tensor.grad is None
        else:
            assert tensor.grad is not None
            torch.testing.assert_close(
                tensor.grad, expected_grad, rtol=2e-5, atol=1e-6
            )
    assert len(called) == 2
    assert sync == ["finalize"]
