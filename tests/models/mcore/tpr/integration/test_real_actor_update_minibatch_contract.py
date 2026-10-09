"""P0 contract for a captured *actor-update* mini_batch_td (not a TQ rollout).

This gate deliberately refuses reconstructed old logprobs or synthetic
advantages. It validates that the snapshot can feed the real TPR Forest PPO
route. It does NOT execute an optimizer step or claim full E2E correctness.

Capture at the VERL actor update boundary after PPO preprocessing and before
the model update, not from the earlier inference/rollout TQ dump. Store:
    torch.save({
        "capture_stage": "actor_update_mini_batch",
        "tensordict": mini_batch_td.cpu(),
        "keys": tuple(exact_trajectory_keys),
    }, path)

Opt-in: TPR_RUN_REAL_ACTOR_MINIBATCH_CONTRACT=1
        TPR_REAL_ACTOR_MINIBATCH=/path/to/captured_actor_update.pt
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
from verl.utils import tensordict_utils as tu


pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_REAL_ACTOR_MINIBATCH_CONTRACT") != "1",
    reason="Set TPR_RUN_REAL_ACTOR_MINIBATCH_CONTRACT=1",
)


def _rows(batch: TensorDict, key: str):
    value = batch[key]
    return list(value) if isinstance(value, (list, tuple)) else list(value.unbind(0))


def test_real_actor_update_minibatch_preserves_ppo_inputs_and_forest():
    value = os.getenv("TPR_REAL_ACTOR_MINIBATCH")
    if not value:
        pytest.fail("Set TPR_REAL_ACTOR_MINIBATCH to a captured actor-update mini_batch_td")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"actor mini-batch capture does not exist: {path}")
    dump = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(dump, dict) or dump.get("capture_stage") != "actor_update_mini_batch":
        pytest.fail(
            "Capture must be a dict with capture_stage=actor_update_mini_batch; "
            "an earlier TQ rollout dump is NOT a valid PPO actor-update capture"
        )
    batch = dump.get("tensordict")
    if not isinstance(batch, TensorDict):
        pytest.fail("capture must contain a TensorDict named tensordict")
    required = (
        "input_ids", "prompts", "responses", "attention_mask",
        "response_mask", "loss_mask", "advantages", "old_log_probs",
        "temperature",
    )
    missing = [key for key in required if key not in batch.keys()]
    if missing:
        pytest.fail(
            f"actor mini-batch missing ORIGINAL PPO fields {missing}; "
            "do not synthesize advantages or recompute old_log_probs"
        )
    keys = dump.get("keys")
    if not isinstance(keys, (tuple, list)) or not keys:
        pytest.fail("capture needs exact trajectory keys in rollout order")
    if any(not isinstance(k, str) or not k for k in keys):
        pytest.fail("trajectory keys must be nonempty strings")
    if len(set(keys)) != len(keys):
        pytest.fail("trajectory keys must be unique per trajectory")
    if len(keys) != batch.batch_size[0]:
        pytest.fail("trajectory key count disagrees with actor mini-batch rows")

    response_masks = _rows(batch, "response_mask")
    old_probs = _rows(batch, "old_log_probs")
    advantages = _rows(batch, "advantages")
    prompts = _rows(batch, "prompts")
    responses = _rows(batch, "responses")
    ids = _rows(batch, "input_ids")
    loss_masks = _rows(batch, "loss_mask")
    valid_tokens = 0
    for row, (ids_row, p, r, mask, lm, old, advantage) in enumerate(
        zip(ids, prompts, responses, response_masks, loss_masks,
            old_probs, advantages, strict=True)
    ):
        if ids_row.numel() != p.numel() + r.numel():
            pytest.fail(f"row {row}: input_ids does not match prompt/response lengths")
        if not torch.equal(ids_row[:p.numel()].long(), p.long()) or not torch.equal(
            ids_row[p.numel():].long(), r.long()
        ):
            pytest.fail(f"row {row}: input_ids differs from the actor prompt+response")
        if not (r.numel() == mask.numel() == old.numel() == advantage.numel()):
            pytest.fail(f"row {row}: response/mask/old_log_probs/advantages length mismatch")
        if lm.numel() == ids_row.numel():
            lm = lm[-r.numel():]
        if lm.numel() != r.numel() or not torch.equal(lm.bool(), mask.bool()):
            pytest.fail(f"row {row}: loss_mask and response_mask disagree")
        if not bool(torch.isfinite(old.float()).all()) or not bool(
            torch.isfinite(advantage.float()).all()
        ):
            pytest.fail(f"row {row}: nonfinite old_log_probs or advantages")
        valid_tokens += int(mask.bool().sum())
    if valid_tokens == 0:
        pytest.fail("actor mini-batch has no PPO training tokens")
    temperature = batch["temperature"].float()
    if not bool(torch.isfinite(temperature).all()) or not bool((temperature > 0).all()):
        pytest.fail("actor mini-batch temperature is nonfinite or nonpositive")
    # Tree planning sees exactly the identities and tensors used by PPO,
    # instead of a synthetic real-token counterfactual.
    tu.assign_non_tensor(
        batch,
        tpr_trajectory_keys=tuple(keys),
        batch_num_tokens=valid_tokens,
        global_batch_size=len(keys),
        dp_size=1,
    )
    if _trajectory_keys_from_minibatch(batch) != tuple(keys):
        pytest.fail("actor mini-batch trajectory keys were lost before Engine routing")
    plan = build_tree_execution_plans(tuple(keys), batch)
    if plan.logical_loss_tokens != valid_tokens:
        pytest.fail(
            f"Forest PPO denominator mismatch: {plan.logical_loss_tokens} vs {valid_tokens}"
        )
    print(
        "P0 REAL_ACTOR_MINIBATCH CONTRACT PASS "
        f"rows={len(keys)} trees={len(plan.trees)} "
        f"segments={plan.segment_count} ppo_tokens={valid_tokens} "
        "original_old_log_probs=True original_advantages=True "
        "optimizer_step=UNVERIFIED e2e=UNVERIFIED",
        flush=True,
    )
