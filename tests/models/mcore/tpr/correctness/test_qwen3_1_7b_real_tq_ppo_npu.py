"""Phase-5 real-checkpoint numerical gate: Qwen3-1.7B + real SWE TQ tokens.

NO synthetic model. The original 8 recorded trajectories are loaded from the
TQ dump. The default fixture keeps ALL 8 real unpadded trajectories, including
multi-level branches. Shortened windows are explicit debug-only mode.

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


def _as_jagged(rows):
    """Keep *all* unpadded real tokens from every variable-length TQ row."""
    return torch.nested.as_nested_tensor(
        [row.detach().cpu().contiguous() for row in rows], layout=torch.jagged
    )


def _load_real_tq_probe(
    *, prompt_length: int | None = None, response_length: int | None = None
):
    """Full 8-row real TQ by default; optional opt-in cropped-real-data smoke.

    Cropped mode is enabled ONLY when both lengths are explicitly specified.
    In full mode the original token sequence, prompt boundary, response mask,
    and compressed trie are preserved exactly.
    """
    path = Path(os.environ.get("TPR_REAL_TQ_BATCH", str(_DEFAULT_TQ)))
    if not path.is_file():
        pytest.fail(f"real TQ dump is required (no synthetic fallback): {path}")
    if (prompt_length is None) != (response_length is None):
        pytest.fail("Set BOTH TPR_QWEN17_PPO_PROMPT and _RESPONSE, or neither")
    if prompt_length is not None and (prompt_length <= 0 or response_length <= 0):
        pytest.fail("Explicit cropped-real-data window lengths must be positive")

    dump = torch.load(path, map_location="cpu", weights_only=False)
    original = dump["tensordict"]
    original_keys = tuple(dump["keys"])
    full_rows = _rows(original, "input_ids")
    prompt_rows = _rows(original, "prompts")
    response_rows = _rows(original, "responses")
    response_masks = _rows(original, "response_mask")
    if len(full_rows) != 8 or len(original_keys) != 8:
        pytest.fail(f"expected recorded 8 SWE trajectories; found {len(full_rows)}")

    token_rows, prompt_out, response_out, masks, advantage_rows = [], [], [], [], []
    source_advantages = _rows(original, "advantages") if "advantages" in original else None
    if source_advantages is None:
        print("PHASE5: TQ has no actor advantages. Using diagnostic nonzero "
              "coefficients ONLY for the native-vs-TPR mathematical gradient gate.")

    for row, (full, prompt, resp, mask) in enumerate(zip(
        full_rows, prompt_rows, response_rows, response_masks, strict=True
    )):
        if not torch.equal(full[:prompt.numel()], prompt):
            pytest.fail(f"real TQ row {row} has an inconsistent prompt boundary")
        if not torch.equal(full[prompt.numel():], resp):
            pytest.fail(f"real TQ row {row} has an inconsistent response suffix")
        if mask.numel() != resp.numel():
            pytest.fail(f"real TQ row {row} response/mask lengths disagree")

        if prompt_length is None:
            p, r = prompt.clone().long(), resp.clone().long()
            selected_mask = mask.clone().bool()
        else:
            if prompt.numel() < prompt_length or resp.numel() < response_length:
                pytest.fail(f"TQ row {row} too short for cropped REAL token window")
            p, r = prompt[-prompt_length:].clone().long(), resp[:response_length].clone().long()
            selected_mask = mask[:response_length].clone().bool()

        token_rows.append(torch.cat((p, r)))
        prompt_out.append(p)
        response_out.append(r)
        masks.append(selected_mask)

        if source_advantages is None:
            # These are NOT rollout advantages. They ensure nonzero, signed
            # gradients in a mathematical equivalence test on REAL inputs.
            arange = torch.arange(len(r), dtype=torch.long)
            a = torch.where((arange + row) % 2 == 0, 1.0, -1.0).float()
        else:
            a = source_advantages[row][:len(r)].float().clone()
            if a.numel() != len(r):
                pytest.fail(f"TQ row {row} advantage length does not match response")
        advantage_rows.append(a)

    count = sum(int(mask.sum()) for mask in masks)
    if count == 0:
        pytest.fail("no supervised response tokens in real TQ fixture")

    # A real actor-update dump should supply a loss_mask, whose response
    # portion must agree with this vanilla token-mean PPO test.
    if "loss_mask" in original and prompt_length is None:
        for row, (original_loss_mask, mask, full) in enumerate(zip(
            _rows(original, "loss_mask"), masks, token_rows, strict=True
        )):
            if original_loss_mask.numel() == full.numel():
                original_loss_mask = original_loss_mask[-mask.numel():]
            if not torch.equal(original_loss_mask.bool(), mask.bool()):
                pytest.fail(
                    f"real TQ row {row}: loss_mask does not match response_mask; "
                    "cannot safely reuse a PPO token-mean denominator"
                )

    batch = TensorDict({
        "input_ids": _as_jagged(token_rows),
        "prompts": _as_jagged(prompt_out),
        "responses": _as_jagged(response_out),
        "attention_mask": _as_jagged([
            torch.ones_like(row) for row in token_rows
        ]),
        "response_mask": _as_jagged(masks),
        "loss_mask": _as_jagged([mask.clone() for mask in masks]),
        "advantages": _as_jagged(advantage_rows),
        "old_log_probs": _as_jagged([
            torch.zeros_like(mask, dtype=torch.float32) for mask in masks
        ]),
        "temperature": torch.ones(8, dtype=torch.float32),
    }, batch_size=[8])
    tu.assign_non_tensor(
        batch,
        tpr_trajectory_keys=original_keys,
        batch_num_tokens=count,
        global_batch_size=8,
        dp_size=1,
    )
    mode = "FULL_REAL_TQ" if prompt_length is None else "CROPPED_REAL_TQ_DEBUG"
    print(
        f"PHASE5 DATA MODE={mode}; rows=8; logical_tokens="
        f"{sum(len(row) for row in token_rows)}; supervised_tokens={count}; "
        f"lengths={[len(row) for row in token_rows]}"
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

    # Default is the UNMODIFIED TQ forest. Both env vars are opt-in CROPPED
    # smoke controls only, and must be absent to validate all real tokens.
    raw_p = os.getenv("TPR_QWEN17_PPO_PROMPT")
    raw_r = os.getenv("TPR_QWEN17_PPO_RESPONSE")
    prompt_len = int(raw_p) if raw_p is not None else None
    response_len = int(raw_r) if raw_r is not None else None
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
                prompt_length=len(batch["prompts"][row]), temperature=1.0
            )
            old_probs.append(response_lp.detach().float().cpu())
    batch["old_log_probs"] = _as_jagged(old_probs)
    tu.assign_non_tensor(batch, tpr_capture_log_probs=True)

    loss_fn = partial(ppo_loss, config=_VanillaPPOConfig())
    native_loss = torch.zeros((), device=device, dtype=torch.float32)
    ref.zero_grad(set_to_none=True)
    for row in range(8):
        token_row = batch["input_ids"][row].to(device)
        lp, _ = _native_response_logprobs(
            ref, token_row, prompt_length=len(batch["prompts"][row]), temperature=1.0
        )
        # Rebuild a one-row native VERL loss view, rather than slicing the
        # full TQ metadata (whose NonTensorData keys may not be row-indexable).
        mini = TensorDict({
            key: batch[key][row].unsqueeze(0).to(device)
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
    captured = getattr(engine, "_tpr_captured_log_probs", None)
    assert captured is not None, "TPR logical new-logprob capture was not enabled"
    expected_keys = {
        (row, offset)
        for row in range(8)
        for offset in torch.nonzero(batch["response_mask"][row]).flatten().tolist()
    }
    assert set(captured) == expected_keys
    baseline_lp = torch.tensor(
        [float(old_probs[row][offset]) for row, offset in sorted(expected_keys)]
    )
    tpr_lp = torch.tensor([captured[key] for key in sorted(expected_keys)])
    torch.testing.assert_close(tpr_lp, baseline_lp, rtol=2e-2, atol=2e-1)
    print(f"Qwen3-1.7B real TQ new_log_probs: {len(expected_keys)} logical tokens aligned")
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
