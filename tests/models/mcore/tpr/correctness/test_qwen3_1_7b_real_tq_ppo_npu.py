"""Phase-5 real-checkpoint numerical gate: Qwen3-1.7B + real SWE TQ tokens.

NO synthetic model. The original 8 recorded trajectories are loaded from the
TQ dump. Long records are cropped to a REAL prompt suffix and REAL generated
response prefix to make a single-NPU correctness gate affordable. This is not
the full-length performance experiment.

When the TQ dump has no actor advantages, deterministic *probe coefficients*
are used ONLY for a numerical gradient equivalence gate (never reported as RL
advantages). For true end-to-end PPO, capture and load the actor-update batch.
"""

from __future__ import annotations

import gc
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Qwen3-1.7B TPR PPO correctness requires an Ascend NPU", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_PPO") != "1",
    reason="Set TPR_RUN_QWEN17_PPO=1 to run real Qwen3-1.7B PPO correctness",
)

_DEFAULT_TQ = Path(
    "/workspace/tq_dump/django11163/"
    "swe-django-11163-qwen3-8b-n8-train-tq_uniagent-tq-smoke/"
    "GBS1_N8_in16384_out114688/1/0/tq_batch.pt"
)


class _VanillaPPOConfig:
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

    def get(self, key, default=None):
        return getattr(self, key, default)


def _rows(batch, field):
    value = batch[field]
    return list(value) if isinstance(value, (list, tuple)) else list(value.unbind())


def _load_real_tq_probe(*, prompt_length: int, response_length: int):
    path = Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ)))
    if not path.is_file():
        pytest.fail(f"real TQ dump is required (no synthetic fallback): {path}")

    dump = torch.load(path, map_location="cpu", weights_only=False)
    original = dump["tensordict"]
    original_keys = tuple(dump["keys"])
    full_rows = _rows(original, "input_ids")
    prompt_rows = _rows(original, "prompts")
    response_masks = _rows(original, "response_mask")
    if len(full_rows) != 8 or len(original_keys) != 8:
        pytest.fail(f"expected the recorded 8 SWE trajectories, found {len(full_rows)}")

    ids, mask, real_advantages = [], [], []
    source_advantages = _rows(original, "advantages") if "advantages" in original else None
    for row, (full, prompt, response_mask) in enumerate(
        zip(full_rows, prompt_rows, response_masks, strict=True)
    ):
        p = prompt.numel()
        if p < prompt_length or full.numel() < p + response_length:
            pytest.fail(f"TQ row {row} is too short for the requested REAL token window")
        segment = torch.cat((
            full[p-prompt_length:p],
            full[p:p+response_length],
        )).long().contiguous()
        ids.append(segment)
        mask.append(response_mask[:response_length].bool())
        if source_advantages is not None:
            real_advantages.append(source_advantages[row][:response_length].float())

    if not any(bool(x.any()) for x in mask):
        pytest.fail("no supervised real response tokens in cropped TQ fixture")

    if source_advantages is None:
        # Probe-only nonzero coefficients; actual rollout advantages require
        # capture at actor train_mini_batch, which the TQ dump may predate.
        advantages = torch.stack([
            torch.tensor(
                [(-1.0 if (i + row) % 2 else 1.0) for i in range(response_length)],
                dtype=torch.float32,
            ) for row in range(8)
        ])
        print("PHASE5 ADVANTAGES: numerical probe coefficients, NOT RL actor-update advantages")
    else:
        advantages = torch.stack(real_advantages)
        if not bool((advantages * torch.stack(mask)).abs().sum()):
            pytest.fail("actual advantages are all zero; this cannot validate PPO gradients")

    sequences = torch.stack(ids)
    masks = torch.stack(mask)
    batch = TensorDict({
        "input_ids": sequences,
        "prompts": sequences[:, :prompt_length].clone(),
        "responses": sequences[:, prompt_length:].clone(),
        "attention_mask": torch.ones_like(sequences, dtype=torch.long),
        "response_mask": masks,
        "loss_mask": masks.clone(),
        "advantages": advantages,
        "old_log_probs": torch.zeros((8, response_length)),
        "temperature": torch.ones(8, dtype=torch.float32),
    }, batch_size=[8])
    tu.assign_non_tensor(
        batch,
        tpr_trajectory_keys=original_keys,
        batch_num_tokens=int(masks.sum().item()),
        global_batch_size=8,
        dp_size=1,
    )
    return batch


def _native_response_logprobs(model, token_row, *, prompt_length, temperature):
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    dev = token_row.device
    positions = torch.arange(token_row.numel(), device=dev).unsqueeze(0)
    logits = model(
        input_ids=token_row.unsqueeze(0),
        position_ids=positions,
        attention_mask=None,
    )
    assert logits.ndim == 3 and tuple(logits.shape[:2]) == (1, token_row.numel())
    # VERL's shift convention: query at position t-1 predicts input_ids[t].
    shifted_labels = torch.cat((token_row[1:], token_row[-1:]))
    logits = logits[0] / torch.as_tensor(temperature, dtype=logits.dtype, device=dev)
    log_probs = vocab_parallel_log_probs_from_logits(logits, shifted_labels)
    return log_probs, log_probs[prompt_length-1:-1]


def _selected_gradient_snapshot(model):
    output = {}
    for name, parameter in model.named_parameters():
        # Sample first/middle/last attention, normalization and MLP matrices.
        sample = any(f"decoder.layers.{layer}." in name for layer in (0, 13, 27))
        if not sample and not any(term in name for term in (
            "embedding.word_embeddings", "final_layernorm", "output_layer",
        )):
            continue
        grad = parameter.grad
        if grad is None:
            raise AssertionError(f"missing gradient for real model parameter {name}")
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"non-finite gradient in {name}")
        output[name] = grad.detach().float().cpu().clone()
    if not output:
        raise AssertionError("could not select any Qwen3-1.7B parameter gradients")
    return output


def _compare_grads(reference, actual):
    assert reference.keys() == actual.keys()
    square_diff = square_ref = dot = square_actual = 0.0
    for name, ref in reference.items():
        got = actual[name]
        delta = got - ref
        square_diff += torch.sum(delta.square()).item()
        square_ref += torch.sum(ref.square()).item()
        square_actual += torch.sum(got.square()).item()
        dot += torch.sum(got * ref).item()
        rel_l2 = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref).clamp_min(1e-12))
        print(f"grad {name}: relative_l2={rel_l2:.6f}")
    relative_l2 = (square_diff / max(square_ref, 1e-24)) ** .5
    cosine = dot / max((square_ref * square_actual) ** .5, 1e-24)
    print(f"Qwen3-1.7B PPO grad: global_relative_l2={relative_l2:.6f}, cosine={cosine:.8f}")
    assert relative_l2 < .08 and cosine > .997


def test_real_qwen3_1_7b_tq_ppo_loss_and_gradients():
    """Real-weight native row-wise PPO vs TPR Forest (CP=TP=PP=DP=1)."""
    os.environ.setdefault(
        "TPR_QWEN_MODEL_PATH",
        os.environ.get("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"),
    )
    from .test_tpr_qwen3_compatibility_npu import (
        _initialize_single_rank_megatron,
        _make_qwen_model,
    )
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead
    from verl.workers.utils.losses import ppo_loss

    prompt_len = int(os.getenv("TPR_QWEN17_PPO_PROMPT", "64"))
    response_len = int(os.getenv("TPR_QWEN17_PPO_RESPONSE", "64"))
    batch = _load_real_tq_probe(prompt_length=prompt_len, response_length=response_len)
    device = torch.device("npu")
    _initialize_single_rank_megatron()

    ref = _make_qwen_model(device, tpr=False, max_sequence_length=prompt_len+response_len)
    ref_params = sum(p.numel() for p in ref.parameters())
    assert 1_500_000_000 <= ref_params <= 1_900_000_000, f"wrong model size {ref_params}"

    # The OLD policy is the real frozen Qwen checkpoint prior to this update,
    # not invented log-probabilities. Recompute once on the actual TQ tokens.
    old_probs = []
    with torch.no_grad():
        for row in range(8):
            _, response_lp = _native_response_logprobs(
                ref, batch["input_ids"][row].to(device),
                prompt_length=prompt_len, temperature=1.0
            )
            old_probs.append(response_lp.detach().float().cpu())
    batch["old_log_probs"] = torch.stack(old_probs)

    loss_fn = partial(ppo_loss, config=_VanillaPPOConfig())
    native_loss = torch.zeros((), device=device, dtype=torch.float32)
    ref.zero_grad(set_to_none=True)
    for row in range(8):
        token_row = batch["input_ids"][row].to(device)
        lp, _ = _native_response_logprobs(
            ref, token_row, prompt_length=prompt_len, temperature=1.0
        )
        # Rebuild a one-row native VERL loss view, rather than slicing the
        # full TQ metadata (whose NonTensorData keys may not be row-indexable).
        mini = TensorDict({
            key: batch[key][row:row+1].to(device)
            for key in (
                "prompts", "responses", "attention_mask", "response_mask",
                "old_log_probs", "advantages",
            )
        }, batch_size=[1])
        tu.assign_non_tensor(
            mini, batch_num_tokens=int(batch["response_mask"].sum()),
            global_batch_size=8, dp_size=1,
        )
        # Native PPO uses the ORIGINAL global denominator for each row.
        mini_model_output = {"log_probs": lp}
        row_loss, _ = loss_fn(model_output=mini_model_output, data=mini, dp_group=None)
        row_loss.backward()
        native_loss += row_loss.detach().float()
    native_grads = _selected_gradient_snapshot(ref)

    del ref
    gc.collect()
    torch.npu.empty_cache()

    tpr = _make_qwen_model(device, tpr=True, max_sequence_length=prompt_len+response_len)
    assert sum(p.numel() for p in tpr.parameters()) == ref_params
    tpr.config.no_sync_func = None
    tpr.config.grad_scale_func = lambda value: value
    tpr.config.finalize_model_grads_func = lambda *args, **kwargs: None
    tpr.config.calculate_per_token_loss = False
    tpr.zero_grad(set_to_none=True)
    # Test the ACTUAL patched Engine entry, not just direct adapter invocation.
    # This confirms global batch normalization happens before Tree building.
    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [tpr]
    engine.engine_config = SimpleNamespace(
        tpr_enabled=True, tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
        use_fused_kernels=False, dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = tpr.config
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None

    output = engine.forward_backward_batch(
        batch, loss_function=partial(ppo_loss, config=_VanillaPPOConfig()),
        forward_only=False,
    )
    tpr_grads = _selected_gradient_snapshot(tpr)
    actual_loss = float(sum(output["loss"]))
    expected_loss = float(native_loss.item())
    print(f"Qwen3-1.7B real TQ PPO native_loss={expected_loss:.8f}, tpr_loss={actual_loss:.8f}")
    torch.testing.assert_close(
        torch.tensor(actual_loss), torch.tensor(expected_loss),
        rtol=2e-2, atol=1e-3,
    )
    _compare_grads(native_grads, tpr_grads)
    assert output["metrics"]["tpr/forest_trees"] == [1]
    assert output["metrics"]["tpr/logical_loss_tokens"] == [int(batch["response_mask"].sum())]
    print("QWEN3-1.7B REAL TQ PPO NUMERICAL GATE: PASS")
