"""P0 weak offline training step: real TQ tokens -> default BF16 TPR -> AdamW.

This is a training-side E2E smoke, NOT an actor rollout/production-VERL
optimizer acceptance test. It reuses recorded TQ tokens and real
Qwen3-1.7B weights; if TQ has no actor advantages the existing Phase-5
fixture uses signed *diagnostic* coefficients. Old logprobs are explicitly
recomputed at the initial checkpoint. No FP32 Linear forward, no fixed-M
tile, no grouped GEMM / custom dW path is installed.

Actual training execution:
  1) TPR Forest Forward / VERL PPO loss / autograd backward
  2) torch.optim.AdamW.step() on FP32 master parameters (standard
     mixed-precision optimizer semantics, NOT VERL/Megatron optimizer)
  3) copy updated masters back to the real BF16 Qwen checkpoint model
  4) forward again; verify finite output and actual nonzero updates.

Optional baseline: TPR_QWEN17_WEAK_E2E_EXECUTION=native runs a controlled
Native row-wise PPO+AdamW step using identical data/config; execute Native and
TPR in separate processes, then compare measured wall time and NPU peaks.
This is not a production MindSpeed reference or all-parameter parity test.

Optional lower-HBM mode: TPR_QWEN17_WEAK_E2E_MASTER_DEVICE=cpu. In this
mode optimizer wall time includes host-side copies; not a throughput
comparison. Always report NPU memory and times separately.

The user must opt in: TPR_RUN_QWEN17_WEAK_E2E=1.
"""
from __future__ import annotations

import gc
import os
import time
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Ascend NPU required", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    os.getenv("TPR_RUN_QWEN17_WEAK_E2E") != "1",
    reason="Set TPR_RUN_QWEN17_WEAK_E2E=1",
)


def _sync():
    torch.npu.synchronize()


def _make_training_engine(model):
    """Use the already-working genuine TPR Engine forward/backward *entry*.

    This deliberately does not claim to construct a VERL production Engine
    or a production optimizer. The gradient finalizer is a no-op on this
    single-rank, unwrapped test model whose grads live in .grad.
    """
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    model.config.no_sync_func = None
    model.config.grad_scale_func = lambda value: value
    model.config.finalize_model_grads_func = lambda *args, **kwargs: None
    model.config.calculate_per_token_loss = False
    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [model]
    engine.engine_config = SimpleNamespace(
        tpr_enabled=True, tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
        use_fused_kernels=False, dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = model.config
    engine.enable_routing_replay = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None
    return engine


def _run_real_adamw_master_step(model, *, lr: float, master_device: str):
    """One *real* torch AdamW optimizer step using the computed PPO grads.

    The FP32 masters are optimizer storage, NOT FP32 GEMM inputs and NOT
    any replacement for Megatron's main_grad or grad finalization.
    """
    if master_device not in ("npu", "cpu"):
        raise AssertionError("master_device must be 'npu' or 'cpu'")
    if not 0 < lr < 1:
        raise AssertionError("weak E2E LR must be in (0,1)")
    _sync()
    total_optimizer_start = time.perf_counter()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise AssertionError("real model exposes no trainable parameters")
    masters = []
    mapped = []
    nonzero_grad_params = 0
    missing_grad = []
    for name, param in named:
        grad = param.grad
        if grad is None:
            missing_grad.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            raise AssertionError(f"nonfinite TPR PPO gradient: {name}")
        master = torch.nn.Parameter(
            param.detach().to(device=master_device, dtype=torch.float32).clone()
        )
        master.grad = grad.detach().to(
            device=master_device, dtype=torch.float32
        ).clone()
        if bool((master.grad != 0).any()):
            nonzero_grad_params += 1
        masters.append(master)
        mapped.append((name, param, master))
    if missing_grad:
        raise AssertionError(
            f"missing gradients for {len(missing_grad)} trainable params: "
            f"{missing_grad[:12]}"
        )
    if nonzero_grad_params == 0:
        raise AssertionError("all real PPO parameter gradients are zero")

    opt = torch.optim.AdamW(
        masters, lr=lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, foreach=False, fused=False,
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(masters, max_norm=1.0)
    if not bool(torch.isfinite(grad_norm)):
        raise AssertionError("nonfinite FP32 master gradient norm")

    # A bounded deterministic, *strided* sample across EVERY trainable
    # tensor for cross-process Native-vs-TPR diagnostics. Sampling is not
    # an all-parameter exact equivalence claim.
    signature_dir = os.getenv("TPR_QWEN17_WEAK_E2E_SIGNATURE_DIR")
    signature = {} if signature_dir else None
    if signature is not None:
        from ._qwen17_weak_e2e_sampling import exact_sample_indices

        for name, _param, master in mapped:
            flat = master.grad.detach().reshape(-1)
            # NEVER use torch.linspace(...).long() for tensor indices:
            # FP32 rounds large Embedding's last valid index to numel,
            # triggering an out-of-bounds NPU index_select / ACL 507035.
            # Compute exact indices on CPU with Python integer arithmetic.
            cpu_indices = torch.tensor(
                exact_sample_indices(flat.numel()), dtype=torch.long
            )
            indices = cpu_indices.to(flat.device)
            assert int(cpu_indices[-1]) == flat.numel() - 1
            sampled = flat.index_select(0, indices)
            signature[name] = {
                "grad": sampled.detach().float().cpu().clone(),
                "indices": cpu_indices,
            }

    step_start = time.perf_counter()
    opt.step()
    if master_device == "npu":
        _sync()
    opt_seconds = time.perf_counter() - step_start

    changed_master = 0
    changed_bf16 = 0
    master_delta_sq = 0.0
    state_count = 0
    with torch.no_grad():
        for name, param, master in mapped:
            delta = master.detach() - param.detach().to(
                device=master_device, dtype=torch.float32
            )
            if not bool(torch.isfinite(master).all()):
                raise AssertionError(f"nonfinite AdamW master: {name}")
            if bool((delta != 0).any()):
                changed_master += 1
            master_delta_sq += float(delta.square().sum())
            if signature is not None:
                indices = signature[name]["indices"].to(delta.device)
                signature[name]["update"] = delta.reshape(-1).index_select(
                    0, indices
                ).float().cpu().clone()
            cast_update = master.detach().to(
                device=param.device, dtype=param.dtype
            )
            if not bool(torch.isfinite(cast_update).all()):
                raise AssertionError(f"nonfinite BF16 cast after AdamW: {name}")
            if bool((cast_update != param).any()):
                changed_bf16 += 1
            param.copy_(cast_update)
            state = opt.state[master]
            if not {"exp_avg", "exp_avg_sq", "step"}.issubset(state):
                raise AssertionError(f"AdamW moment buffers missing: {name}")
            for slot in ("exp_avg", "exp_avg_sq"):
                if not bool(torch.isfinite(state[slot]).all()):
                    raise AssertionError(f"nonfinite AdamW {slot}: {name}")
            state_count += 1
    _sync()
    if not changed_master or master_delta_sq == 0 or not changed_bf16:
        raise AssertionError(
            f"optimizer did not update usable weights: FP32={changed_master}, "
            f"BF16={changed_bf16}, delta_sq={master_delta_sq}"
        )
    total_optimizer_seconds = time.perf_counter() - total_optimizer_start
    if signature is not None:
        from pathlib import Path

        base = Path(signature_dir)
        base.mkdir(parents=True, exist_ok=True)
        backend = os.getenv("TPR_QWEN17_WEAK_E2E_EXECUTION", "tpr").lower()
        if backend not in ("tpr", "native"):
            raise AssertionError(f"unexpected signature backend {backend}")
        path = base / f"{backend}_optimizer_sample.pt"
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite prior optimizer sample: {path}"
            )
        torch.save(signature, path)
        print(
            "P0 WEAK_TQ OPTIMIZER_SAMPLE "
            f"path={path} tensors={len(signature)} entries_per_tensor<=512 "
            "all_parameters_sampled=True full_parameter_equivalence=False",
            flush=True,
        )
    print(
        "P0 WEAK_TQ OPTIMIZER_STEP status=PASS "
        f"optimizer=torch.optim.AdamW master_dtype=FP32 master_device={master_device} "
        f"grad_norm={float(grad_norm):.9g} "
        f"updated_master_tensors={changed_master}/{len(mapped)} "
        f"updated_model_tensors={changed_bf16}/{len(mapped)} "
        f"master_update_l2={master_delta_sq**0.5:.9g} "
        f"adamw_state_tensors={state_count} "
        f"optimizer_core_seconds={opt_seconds:.6f} "
        f"optimizer_total_seconds={total_optimizer_seconds:.6f} "
        "megatron_optimizer=UNVERIFIED",
        flush=True,
    )
    return total_optimizer_seconds


def test_real_tq_qwen17_default_bf16_weak_actor_optimizer_step():
    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)

    from ..profiling._qwen3_profile_target import resolve_qwen3_profile_target
    from . import test_tpr_qwen3_compatibility_npu as fixture
    from .test_qwen3_1_7b_real_tq_ppo_npu import (
        _VanillaPPOConfig, _as_jagged, _load_real_tq_probe,
        _native_response_logprobs, _rows,
    )
    from verl.workers.utils.losses import ppo_loss

    for forbidden in (
        "TPR_QWEN17_PPO_TILE_GEMM", "TPR_QWEN17_GPT_FP32_GEMM",
        "TPR_QWEN17_PPO_FC2_FIXED_M",
        "TPR_QWEN17_SPLIT_FP32_DW_BACKWARD",
    ):
        if os.getenv(forbidden):
            pytest.fail(
                f"P0 default BF16 weak E2E forbids diagnostic override {forbidden}; "
                "unset it before running"
            )
    target = resolve_qwen3_profile_target()
    if target.size != "1.7B":
        pytest.fail(f"requires real Qwen3-1.7B, got {target.label}")
    fixture.QWEN_MODEL_PATH = target.path
    token_capture_dir = os.getenv("TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE_DIR")
    raw_p = os.getenv("TPR_QWEN17_WEAK_E2E_PROMPT", "128")
    raw_s = os.getenv("TPR_QWEN17_WEAK_E2E_RESPONSE", "64")
    if not (raw_p.isdigit() and raw_s.isdigit()):
        pytest.fail("weak E2E prompt/response must be positive integer lengths")
    p, s = int(raw_p), int(raw_s)
    if p <= 0 or s <= 0:
        pytest.fail("weak E2E requires positive prompt/response lengths")
    batch = _load_real_tq_probe(prompt_length=p, response_length=s)
    lengths = [row.numel() for row in _rows(batch, "input_ids")]
    longest = max(lengths)
    fixture._initialize_single_rank_megatron()
    device = torch.device("npu")
    print(
        "P0 WEAK_TQ CONFIG "
        f"checkpoint={target.path} rows={len(lengths)} lengths={lengths} "
        "model_dtype=BF16 linear=DEFAULT attention=CONTROLLED_CANN "
        "optimizer=TORCH_ADAMW_FP32_MASTER old_logprobs=RECOMPUTED "
        "advantages=DIAGNOSTIC_WHEN_MISSING "
        "data=CROPPED_REAL_TQ not_production_actor_step=True",
        flush=True,
    )

    # Only the 'old policy' values are recomputed. This is a numerical
    # counterfactual, NOT historical rollout behavior-policy logprobs.
    reference = fixture._make_qwen_model(
        device, tpr=False, max_sequence_length=longest
    )
    target.assert_model_scale(reference)
    old_rows = []
    with torch.no_grad():
        for row in range(len(lengths)):
            _, old_lp = _native_response_logprobs(
                reference, batch["input_ids"][row].to(device),
                prompt_length=len(batch["prompts"][row]), temperature=1.0,
            )
            old_rows.append(old_lp.detach().float().cpu())
    batch["old_log_probs"] = _as_jagged(old_rows)
    execution = os.getenv("TPR_QWEN17_WEAK_E2E_EXECUTION", "tpr").lower()
    if execution not in ("native", "tpr", "cutoff", "core"):
        pytest.fail("weak E2E execution must be 'native', 'tpr', 'cutoff' or 'core'")

    # No-gradient native core-attention shape oracle. Hold the *actual*
    # post-RoPE Q/K/V fixed, only change square-vs-rectangular CANN
    # attention dimensions; this isolates the kernel from any GEMM, RoPE
    # projection or external Prefix KV cache.
    if execution == "core":
        from .test_qwen3_1_7b_real_tq_ppo_npu import (
            _native_core_square_vs_rectangular_oracle,
        )
        from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
        from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans

        core_row = int(os.getenv("TPR_QWEN17_WEAK_E2E_CORE_ROW", "5"))
        if not 0 <= core_row < len(lengths):
            pytest.fail(f"core-row={core_row} is out of range")
        forest = build_tree_execution_plans(
            _trajectory_keys_from_minibatch(batch), batch
        )
        # Test actual physical M used in this Forest for nodes whose KV
        # end matches the native S=192, including M=3 for [189:192].
        starts = sorted({
            seg.position_start
            for tree in forest.trees
            for seg in tree.segment_plan.segments.values()
            if seg.position_end == lengths[core_row]
            and 0 < seg.position_start < lengths[core_row]
        })
        if not starts:
            pytest.fail("core oracle requires a multi-node tree with leaves ending at S")
        # Tail query 189 is the worst Native-Full vs Forest PPO token, row 5.
        token_positions = tuple(
            range(max(starts), lengths[core_row])
        )
        print(
            "P0 WEAK_TQ CORE_GEOMETRY "
            f"row={core_row} total_S={lengths[core_row]} "
            f"query_starts={starts} tail_queries={token_positions} "
            "same_post_rope_QKV=True",
            flush=True,
        )
        start_core = time.perf_counter()
        shape_results = _native_core_square_vs_rectangular_oracle(
            reference, batch, row=core_row,
            segment_starts=tuple(starts),
            token_positions=token_positions,
            selected_layers=(1, 2, 3, 4, 14, 28),
            max_length=256,
            require_complete=True,
        )
        _sync()
        if not shape_results:
            raise AssertionError("CORE oracle produced no numerical comparisons")
        worst = max(shape_results, key=lambda item: item[3])
        query3 = [entry for entry in shape_results
                  if lengths[core_row] - entry[1] == 3]
        if len(query3) != 6:
            raise AssertionError(
                f"Expected Q=3 vs KV=192 across six layers; got {len(query3)}"
            )
        print(
            "P0 WEAK_TQ CORE_SUMMARY "
            f"compared_shapes={len(shape_results)} "
            f"worst_layer={worst[0]} worst_query={lengths[core_row]-worst[1]} "
            f"worst_max_abs={worst[3]:.9g} "
            f"q3_max_abs={max(entry[3] for entry in query3):.9g} "
            "numeric_parity=DIAGNOSTIC_ONLY",
            flush=True,
        )
        print(
            "P0 WEAK_TQ RESULT status=PASS execution=CORE "
            f"elapsed_seconds={time.perf_counter()-start_core:.6f} "
            "same_qkv=True optimizer_step=False "
            "interpretation=ISOLATED_ATTENTION_SHAPE numerical_parity=UNVERIFIED",
            flush=True,
        )
        return

    # Native per-physical-segment cutoff oracle isolates a crucial causal
    # shape confounder WITHOUT any TPR KV reuse or optimizer operation:
    # compare Native(full S=192) vs Native(prefix to owner segment end).
    # Reuse Phase-5's already-implemented, row-identity-validated oracle.
    # Run this mode alone, alongside SAVED prior Native/TPR PPO token traces.
    if execution == "cutoff":
        if not token_capture_dir:
            pytest.fail(
                "cutoff oracle requires TPR_QWEN17_WEAK_E2E_TOKEN_CAPTURE_DIR "
                "(run shell wrapper with TOKEN_CAPTURE=1)"
            )
        from pathlib import Path
        from ._qwen17_weak_e2e_token_capture import save_ppo_tokens
        from .test_qwen3_1_7b_real_tq_ppo_npu import (
            _native_cutoff_oracle_per_segment,
        )

        start_cutoff = time.perf_counter()
        cutoff, details = _native_cutoff_oracle_per_segment(
            reference, batch, max_length=256,
        )
        _sync()
        if cutoff is None or details is None:
            raise AssertionError("real TQ S=192 cutoff oracle unexpectedly skipped")
        save_ppo_tokens(
            batch, cutoff, Path(token_capture_dir) / "cutoff_ppo_tokens.pt"
        )
        # Keep ownership metadata aligned with the exact PPO token keys.
        # It enables reporting which physical Segment contributed a drift.
        ordered = sorted(cutoff)
        segment_metadata = {
            "logical_row_offset": torch.tensor(ordered, dtype=torch.int64),
            "segment_id_start_end": torch.tensor(
                [(details[key][1], details[key][2], details[key][3])
                 for key in ordered],
                dtype=torch.int64,
            ),
        }
        meta_path = Path(token_capture_dir) / "cutoff_segment_owners.pt"
        if meta_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite prior cutoff segment owners: {meta_path}"
            )
        torch.save(segment_metadata, meta_path)
        print(
            "P0 WEAK_TQ RESULT status=PASS execution=CUTOFF "
            f"ppo_tokens={len(cutoff)} "
            f"segments={len(set((v[1], v[2], v[3]) for v in details.values()))} "
            f"elapsed_seconds={time.perf_counter()-start_cutoff:.6f} "
            "optimizer_step=False native_per_segment_cutoff=True "
            "purpose=CAUSAL_SHAPE_ORACLE",
            flush=True,
        )
        return

    # A separate native control uses identical real tokens, same checkpoint,
    # diagnostic old-logprobs/advantages, PPO loss and the same FP32-master
    # optimizer. This is the *controlled CANN* reference, not production
    # MindSpeed/VERL. Run in another pytest process to isolate memory/time.
    if execution == "native":
        from tensordict import TensorDict

        reference.zero_grad(set_to_none=True)
        native_log_probs = [] if token_capture_dir else None
        total_valid = sum(
            int(mask.bool().sum()) for mask in _rows(batch, "response_mask")
        )
        torch.npu.reset_peak_memory_stats()
        _sync()
        start = time.perf_counter()
        native_loss = 0.0
        for row in range(len(lengths)):
            lp, _ = _native_response_logprobs(
                reference, batch["input_ids"][row].to(device),
                prompt_length=len(batch["prompts"][row]), temperature=1.0,
            )
            mini = TensorDict({
                key: batch[key][row].unsqueeze(0).to(device)
                for key in (
                    "prompts", "responses", "attention_mask",
                    "response_mask", "old_log_probs", "advantages",
                )
            }, batch_size=[1])
            tu.assign_non_tensor(
                mini, batch_num_tokens=total_valid,
                global_batch_size=len(lengths), dp_size=1,
            )
            value, _ = ppo_loss(
                model_output={"log_probs": lp}, data=mini,
                config=_VanillaPPOConfig(), dp_group=None,
            )
            value.backward()
            native_loss += float(value.detach())
            if native_log_probs is not None:
                prompt_length = len(batch["prompts"][row])
                native_log_probs.append(lp[prompt_length - 1:-1].detach())
        _sync()
        backward_seconds = time.perf_counter() - start
        if not (float("-inf") < native_loss < float("inf")):
            raise AssertionError("native PPO returned nonfinite loss")
        print(
            "P0 WEAK_TQ BACKWARD status=PASS execution=NATIVE "
            f"loss={native_loss:.9g} ppo_tokens={total_valid} "
            f"forward_backward_seconds={backward_seconds:.6f}",
            flush=True,
        )
        if native_log_probs is not None:
            from pathlib import Path
            from ._qwen17_weak_e2e_token_capture import save_ppo_tokens

            captured = {}
            masks = _rows(batch, "response_mask")
            for row, logical_lp in enumerate(native_log_probs):
                lp_cpu = logical_lp.float().cpu()
                if lp_cpu.numel() != masks[row].numel():
                    raise AssertionError("Native response logprobs/mask length mismatch")
                for i in torch.nonzero(
                    masks[row].bool(), as_tuple=False
                ).flatten().tolist():
                    captured[(row, i)] = float(lp_cpu[i])
            save_ppo_tokens(
                batch, captured, Path(token_capture_dir) / "native_ppo_tokens.pt"
            )
        optimizer_seconds = _run_real_adamw_master_step(
            reference,
            lr=float(os.getenv("TPR_QWEN17_WEAK_E2E_LR", "0.0001")),
            master_device=os.getenv(
                "TPR_QWEN17_WEAK_E2E_MASTER_DEVICE", "npu"
            ).lower(),
        )
        reference.zero_grad(set_to_none=True)
        with torch.no_grad():
            _, updated = _native_response_logprobs(
                reference, batch["input_ids"][0].to(device),
                prompt_length=len(batch["prompts"][0]), temperature=1.0,
            )
        if not bool(torch.isfinite(updated).all()):
            raise AssertionError("native post-update forward is nonfinite")
        _sync()
        print(
            "P0 WEAK_TQ RESULT status=PASS execution=NATIVE "
            f"updated_forward_tokens={updated.numel()} "
            f"npu_peak_alloc_mib={torch.npu.max_memory_allocated()/(1024**2):.3f} "
            f"npu_peak_reserved_mib={torch.npu.max_memory_reserved()/(1024**2):.3f} "
            f"forward_backward_seconds={backward_seconds:.6f} "
            f"optimizer_seconds={optimizer_seconds:.6f} "
            "real_tokens=True real_checkpoint=True "
            "real_ppo_loss=True backward=True real_adamw_step=True "
            "production_verl_optimizer=False real_behavior_policy=False "
            "full_tq=False native_update_parity=NOT_MEASURED",
            flush=True,
        )
        return

    del reference
    gc.collect()
    torch.npu.empty_cache()

    tpr = fixture._make_qwen_model(
        device, tpr=True, max_sequence_length=longest
    )
    target.assert_model_scale(tpr)
    tpr.zero_grad(set_to_none=True)
    engine = _make_training_engine(tpr)
    from verl.models.mcore.tpr.megatron_adapter import _trajectory_keys_from_minibatch
    from verl.models.mcore.tpr.tree_plan_builder import build_tree_execution_plans
    # Reuse EXACTLY the production TPR Engine's trajectory-key resolver,
    # including its required tu.get_non_tensor_data(..., default=None)
    # contract. The real TQ loader preserves these identities.
    keys = _trajectory_keys_from_minibatch(batch)
    forest = build_tree_execution_plans(
        keys, batch, require_loss_mask_alignment=True,
    )
    expected_tokens = sum(int(m.bool().sum()) for m in _rows(batch, "response_mask"))
    assert expected_tokens > 0 and forest.logical_loss_tokens == expected_tokens
    if token_capture_dir:
        # Optional per-logical-token probe; involves D2H transfers from each
        # physical Segment and invalidates performance timing comparability.
        tu.assign_non_tensor(batch, tpr_capture_log_probs=True)
        print("P0 WEAK_TQ TOKEN_CAPTURE enabled=True timings_are_diagnostic=True")
    torch.npu.reset_peak_memory_stats()
    _sync()
    start = time.perf_counter()
    out = engine.forward_backward_batch(
        batch, loss_function=partial(ppo_loss, config=_VanillaPPOConfig()),
        forward_only=False,
    )
    _sync()
    backward_seconds = time.perf_counter() - start
    loss = float(sum(out["loss"]))
    if not (float("-inf") < loss < float("inf")):
        raise AssertionError(f"nonfinite TPR PPO objective: {loss}")
    if out["metrics"]["tpr/logical_loss_tokens"] != [expected_tokens]:
        raise AssertionError("Forest lost logical PPO training tokens")
    print(
        "P0 WEAK_TQ BACKWARD status=PASS "
        f"loss={loss:.9g} forest_trees={len(forest.trees)} "
        f"segments={forest.segment_count} ppo_tokens={expected_tokens} "
        f"forward_backward_seconds={backward_seconds:.6f}",
        flush=True,
    )
    if token_capture_dir:
        from pathlib import Path
        from ._qwen17_weak_e2e_token_capture import save_ppo_tokens

        captured = getattr(engine, "_tpr_captured_log_probs", None)
        if captured is None:
            raise AssertionError("TPR engine did not capture PPO logprobs")
        save_ppo_tokens(
            batch, captured, Path(token_capture_dir) / "tpr_ppo_tokens.pt"
        )
    master_device = os.getenv("TPR_QWEN17_WEAK_E2E_MASTER_DEVICE", "npu").lower()
    lr = float(os.getenv("TPR_QWEN17_WEAK_E2E_LR", "0.0001"))
    optimizer_seconds = _run_real_adamw_master_step(
        tpr, lr=lr, master_device=master_device,
    )
    tpr.zero_grad(set_to_none=True)
    with torch.no_grad():
        probe_tokens = batch["input_ids"][0].to(device)
        # Non-tree forward on the now-updated TPR model.
        _, updated_lp = _native_response_logprobs(
            tpr, probe_tokens, prompt_length=len(batch["prompts"][0]),
            temperature=1.0,
        )
    if not bool(torch.isfinite(updated_lp).all()):
        raise AssertionError("updated real Qwen forward produced nonfinite logprobs")
    _sync()
    print(
        "P0 WEAK_TQ RESULT status=PASS execution=TPR "
        f"updated_forward_tokens={updated_lp.numel()} "
        f"npu_peak_alloc_mib={torch.npu.max_memory_allocated()/(1024**2):.3f} "
        f"npu_peak_reserved_mib={torch.npu.max_memory_reserved()/(1024**2):.3f} "
        f"forward_backward_seconds={backward_seconds:.6f} "
        f"optimizer_seconds={optimizer_seconds:.6f} "
        "real_tokens=True real_checkpoint=True "
        "real_ppo_loss=True backward=True real_adamw_step=True "
        "production_verl_optimizer=False real_behavior_policy=False "
        "full_tq=False native_update_parity=UNVERIFIED",
        flush=True,
    )
