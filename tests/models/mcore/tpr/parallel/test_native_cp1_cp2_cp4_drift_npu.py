"""P0-1: native MindSpeed/Megatron CP=1/2/4, WITHOUT any TPR operators.

Run ONLY from the synced /workspace/uni-agent/verl test runtime:
    bash tests/models/mcore/tpr/parallel/run_native_cp_drift.sh

On the HOST, git pull in /workspace/bridge. Then INSIDE DOCKER, rsync its
tests/models/mcore/tpr/ tree to /workspace/uni-agent/verl/tests/models/mcore/tpr/.
The launcher uses explicit
--master_addr=127.0.0.1 and --master_port=29561/29562/29564, then invokes
the comparator from /workspace/bridge/tools/compare_native_cp_drift.py.
Numerical diagnostic, NOT a production Megatron-DDP optimizer benchmark.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.utils.device import is_torch_npu_available

pytestmark = pytest.mark.skipif(
    os.environ.get("TPR_P0_NATIVE_CP") != "1" or not is_torch_npu_available(check_device=True),
    reason="Set TPR_P0_NATIVE_CP=1 on an Ascend NPU with torchrun",
)

SEQ_LEN = 1024
RESPONSE_LEN = 512
SAMPLE_COUNT = 512


def _expected_positions(cp: int, rank: int) -> torch.Tensor:
    """ONLY a validation oracle; the model's data comes from MindSpeed."""
    if cp == 1:
        return torch.arange(SEQ_LEN, dtype=torch.long)
    chunk = SEQ_LEN // (2 * cp)
    return torch.cat((
        torch.arange(rank * chunk, (rank + 1) * chunk),
        torch.arange((2 * cp - 1 - rank) * chunk, (2 * cp - rank) * chunk),
    ))


@contextmanager
def _runtime(cp: int):
    import torch_npu  # noqa: F401
    from megatron.core import parallel_state

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if int(os.environ["WORLD_SIZE"]) != cp:
        raise AssertionError("P0 native CP must use world_size == CP (TP=PP=DP=1)")
    if dist.is_initialized() or parallel_state.model_parallel_is_initialized():
        raise AssertionError("P0 native CP requires a fresh distributed process")

    dist.init_process_group(backend="hccl")
    # Same MindSpeed bootstrap as the bridge CP correctness fixtures.
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = original_argv

    from mindspeed.args_utils import get_full_args
    vars(get_full_args()).pop("", None)
    repatch({
        "context_parallel_size": cp,
        "context_parallel_algo": "megatron_cp_algo",
        "seq_length": SEQ_LEN,
        "use_flash_attn": True,
        "attention_mask_type": "causal",
        # Standard non-EOD, non-packed native CP path. The lightweight
        # MindSpeed args fallback (when megatron.training is absent) does not
        # always populate this flag, but get_batch_on_this_cp_rank requires it.
        "reset_attention_mask": False,
    })
    # Match the existing bridge CP fixtures: this runtime may provide only
    # Megatron-Core, without the optional megatron.training package.
    # MindSpeed's get_full_args() falls back to its own args singleton.
    args = get_full_args()
    if args.context_parallel_size != cp or args.context_parallel_algo != "megatron_cp_algo":
        raise AssertionError("MindSpeed did not apply the requested native Ring CP configuration")
    if not getattr(args, "use_flash_attn", False):
        raise AssertionError("Native MindSpeed CP requires FlashAttention")
    if args.reset_attention_mask is not False:
        raise AssertionError("P0 requires native non-EOD CP without attention-mask reset")

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=cp, expert_model_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(20261011)

    runtime = SimpleNamespace(
        cp=cp, rank=parallel_state.get_context_parallel_rank(),
        group=parallel_state.get_context_parallel_group(),
        tp=parallel_state.get_tensor_model_parallel_group(),
        pp=parallel_state.get_pipeline_model_parallel_group(),
        device=torch.device("npu", local_rank),
    )
    assert runtime.group.size() == cp and runtime.tp.size() == runtime.pp.size() == 1
    try:
        yield runtime
    finally:
        dist.barrier(group=runtime.group)
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _input_tokens(vocab_size: int) -> tuple[torch.Tensor, str]:
    """Optional real TQ 512+512 window; otherwise seeded fixed-length diagnostic."""
    path = os.getenv("TPR_REAL_TQ_BATCH")
    if path:
        from ..correctness.test_qwen3_1_7b_real_tq_ppo_npu import _rows
        dump = torch.load(path, map_location="cpu", weights_only=False)
        data = dump["tensordict"]
        row = int(os.getenv("TPR_P0_CP_TQ_ROW", "0"))
        prompts, responses = _rows(data, "prompts"), _rows(data, "responses")
        if row < 0 or row >= len(prompts):
            raise AssertionError(f"invalid TQ row {row}")
        if prompts[row].numel() < 512 or responses[row].numel() < 512:
            raise AssertionError("P0 real TQ requires >=512 prompt and >=512 response tokens")
        tokens = torch.cat((prompts[row][-512:], responses[row][:512])).long().cpu()
        source = f"real_tq:{Path(path).name}:row{row}:prompt_tail512+response_head512"
    else:
        gen = torch.Generator(device="cpu").manual_seed(20261011)
        tokens = torch.randint(0, vocab_size, (SEQ_LEN,), generator=gen, dtype=torch.long)
        source = "synthetic_seed20261011_NOT_REAL_TQ"
    assert tokens.shape == (SEQ_LEN,)
    assert int(tokens.min()) >= 0 and int(tokens.max()) < vocab_size
    return tokens.contiguous(), source


def _native_batch(tokens: torch.Tensor, runtime):
    """Use MindSpeed's OWN causal Ring CP zigzag slicing, not a homegrown layout."""
    try:
        from mindspeed.core.context_parallel.get_batch_utils import get_batch_on_this_cp_rank
        batch_module = "mindspeed.core.context_parallel.get_batch_utils"
    except ImportError:
        from mindspeed.utils import get_batch_on_this_cp_rank
        batch_module = "mindspeed.utils"

    full = {
        "tokens": tokens.unsqueeze(0).to(runtime.device),
        "position_ids": torch.arange(SEQ_LEN, device=runtime.device).unsqueeze(0),
        "attention_mask": None,
    }
    local = get_batch_on_this_cp_rank(full)
    positions = local["position_ids"].detach().cpu().flatten()
    expected = _expected_positions(runtime.cp, runtime.rank)
    if not torch.equal(positions, expected):
        raise AssertionError(f"official MindSpeed CP layout mismatch on rank {runtime.rank}: "
                             f"actual={positions.tolist()[:16]} expected={expected.tolist()[:16]}")
    if not torch.equal(local["tokens"].detach().cpu().flatten(), tokens[expected]):
        raise AssertionError("native CP token/position alignment is wrong")
    if local["tokens"].shape != (1, SEQ_LEN // runtime.cp):
        raise AssertionError("native CP local shape mismatch")
    return local, positions, batch_module


def _model(runtime, checkpoint):
    from transformers import AutoConfig
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from verl.models.mcore.config_converter import hf_to_mcore_config_dense, get_hf_rope_theta
    from ..correctness.test_tpr_qwen3_compatibility_npu import (
        _validate_checkpoint_files, _load_qwen3_torch_spec_weights,
    )

    _validate_checkpoint_files(checkpoint)
    hf = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
    if hf.model_type != "qwen3" or hf.num_hidden_layers != 28 or hf.hidden_size != 2048:
        raise AssertionError("P0 native CP expects the real dense Qwen3-1.7B checkpoint")
    config = hf_to_mcore_config_dense(
        hf, torch.bfloat16,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        context_parallel_size=runtime.cp, expert_model_parallel_size=1,
        sequence_parallel=False, attention_dropout=0.0, hidden_dropout=0.0,
        apply_rope_fusion=False, bias_dropout_fusion=False, use_cpu_initialization=True,
    )
    config.context_parallel_algo = "megatron_cp_algo"
    config.attention_mask_type = "causal"
    # The native core attention spec is NEVER replaced by TPR or a test FA wrapper.
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    model = GPTModel(
        config=config, transformer_layer_spec=spec, vocab_size=hf.vocab_size,
        max_sequence_length=hf.max_position_embeddings,
        pre_process=True, post_process=True, parallel_output=False,
        share_embeddings_and_output_weights=hf.tie_word_embeddings,
        position_embedding_type="rope", rotary_base=get_hf_rope_theta(hf),
        pg_collection=SimpleNamespace(tp=runtime.tp, cp=runtime.group, pp=runtime.pp, embd=None),
    ).to(device=runtime.device, dtype=torch.bfloat16)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for layer in model.decoder.layers:
        if "tpr" in type(layer.self_attention).__module__.lower():
            raise AssertionError("Native P0 inadvertently installed TPR attention")
    _load_qwen3_torch_spec_weights(model, hf, checkpoint)
    model.train()
    return model


@contextmanager
def _capture_shapes(model, runtime):
    seen = {}
    handles = []
    for layer_idx, layer in enumerate(model.decoder.layers):
        for name, module in (
            ("qkv", layer.self_attention.linear_qkv),
            ("proj", layer.self_attention.linear_proj),
            ("fc1", layer.mlp.linear_fc1),
            ("fc2", layer.mlp.linear_fc2),
        ):
            key = f"layer{layer_idx}.{name}"
            def capture(_module, args, kwargs, *, label=key):
                x = next((x for x in args if isinstance(x, torch.Tensor)), None)
                if x is None:
                    x = kwargs.get("hidden_states")
                if not isinstance(x, torch.Tensor):
                    raise AssertionError(f"{label}: Linear received no tensor")
                m = x.shape[0] * x.shape[1] if x.ndim == 3 else x.shape[0]
                seen.setdefault(label, []).append(int(m))
            handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()
        target = SEQ_LEN // runtime.cp
        assert len(seen) == 28 * 4, f"incomplete Linear shape capture: {len(seen)}"
        assert all(v == [target] for v in seen.values()), (
            f"unexpected Linear M (expected {target}): "
            f"{[(k, v) for k, v in seen.items() if v != [target]][:12]}"
        )


@contextmanager
def _native_ring_probe(cp):
    """Prove CP>1 actually enters MindSpeed native Ring, not a silent local FA."""
    if cp == 1:
        yield [0]
        return
    module = importlib.import_module("mindspeed.core.context_parallel.dot_product_attention")
    native = getattr(module, "ringattn_context_parallel")
    calls = [0]
    def counted(*args, **kwargs):
        calls[0] += 1
        return native(*args, **kwargs)
    with patch.object(module, "ringattn_context_parallel", counted):
        yield calls
    if calls[0] < 28:
        raise AssertionError(
            f"MindSpeed native Ring called {calls[0]} times for 28 layers; "
            "P0 refuses to report a CP baseline through an unverified attention backend"
        )


def _save_gradients(model, runtime, full):
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
    samples = {}
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            continue
        # Bare GPTModel (not DDP). SUM rank-local dL/dtheta exactly once:
        # loss already has GLOBAL token denominator, so DO NOT divide by CP.
        # Use parameter grad's native BF16 dtype, without FP32 or other kernel interventions.
        if runtime.cp > 1:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=runtime.group)
        grad = param.grad.detach()
        indices = torch.tensor(exact_sample_indices(grad.numel(), SAMPLE_COUNT),
                               device=runtime.device, dtype=torch.long)
        flat = grad.reshape(-1)
        samples[name] = {
            "shape": tuple(grad.shape),
            "numel": grad.numel(),
            "indices": indices.cpu(),
            "values": flat.index_select(0, indices).float().cpu(),
            "norm": float(torch.linalg.vector_norm(grad.float()).item()),
        }
        if full and runtime.rank == 0:
            samples[name]["full"] = grad.cpu().clone()
    if missing:
        raise AssertionError(f"Missing native parameter gradients: {missing[:12]}")
    if not samples:
        raise AssertionError("No native parameter gradients")
    return samples


def _weight_signature(model) -> str:
    """Bounded exact-BF16 sampled parameter fingerprint, on rank 0 only."""
    from ..correctness._qwen17_weak_e2e_sampling import exact_sample_indices
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(param.shape)).encode("ascii"))
        indices = torch.tensor(
            exact_sample_indices(param.numel(), 16), dtype=torch.long,
            device=param.device,
        )
        witness = param.detach().flatten().index_select(0, indices).contiguous().cpu()
        # NumPy cannot represent PyTorch BF16 directly. Reinterpret the CPU
        # BF16 bits as int16 (no arithmetic cast, no information loss).
        digest.update(witness.view(torch.int16).numpy().tobytes())
    return digest.hexdigest()


def test_native_cp_1_2_4_numerical_reference():
    cp = int(os.environ["WORLD_SIZE"])
    if cp not in (1, 2, 4):
        pytest.fail("P0 requires exactly CP=1, CP=2, or CP=4")
    out_dir = Path(os.getenv("TPR_P0_CP_OUTPUT_DIR", "/tmp/tpr_p0_native_cp"))
    checkpoint = Path(os.getenv("TPR_QWEN_1_7B_PATH", "/workspace/hf_models/Qwen3-1.7B"))
    with _runtime(cp) as runtime:
        model = _model(runtime, checkpoint)
        weight_sha = _weight_signature(model) if runtime.rank == 0 else None
        tokens, source = _input_tokens(model.vocab_size)
        token_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
        batch, positions, batch_module = _native_batch(tokens, runtime)
        device_positions = batch["position_ids"][0]
        target_positions = device_positions < SEQ_LEN - 1
        supervised = target_positions & (device_positions >= SEQ_LEN - RESPONSE_LEN - 1)
        local_supervised = int(supervised.sum().item())
        if local_supervised == 0:
            raise AssertionError("P0 objective requires at least one supervised token per rank")
        supervised_count = torch.tensor(local_supervised, device=runtime.device)
        dist.all_reduce(supervised_count, op=dist.ReduceOp.SUM, group=runtime.group)
        total_supervised = int(supervised_count.item())
        if total_supervised != RESPONSE_LEN:
            raise AssertionError(f"unexpected global NLL denominator {total_supervised}")

        model.zero_grad(set_to_none=True)
        with _capture_shapes(model, runtime) as shapes, _native_ring_probe(cp) as native_ring_calls:
            logits = model(batch["tokens"], batch["position_ids"], None)
            if logits.shape != (1, SEQ_LEN // cp, model.vocab_size):
                raise AssertionError(f"unexpected native GPT logits {tuple(logits.shape)}")
            # Use the same global targets on all ranks. No local-roll/shift across
            # noncontiguous Zigzag halves (which would be incorrect).
            selected = torch.nonzero(target_positions).flatten()
            query = positions.index_select(0, selected.cpu()).to(runtime.device)
            labels = tokens.to(runtime.device).index_select(0, query + 1)
            local_logprobs = F.log_softmax(logits[0].index_select(0, selected).float(), dim=-1)
            local_logprobs = local_logprobs.gather(1, labels[:, None]).squeeze(1)
            local_supervision = query >= SEQ_LEN - RESPONSE_LEN - 1
            loss = -local_logprobs[local_supervision].sum() / float(total_supervised)
            loss.backward()
        # Reduce only detached scalar and sparse logprob witness values.
        loss_global = loss.detach().clone()
        dist.all_reduce(loss_global, op=dist.ReduceOp.SUM, group=runtime.group)
        logprob = torch.zeros(SEQ_LEN - 1, device=runtime.device, dtype=torch.float32)
        count = torch.zeros_like(logprob, dtype=torch.int32)
        logprob.index_copy_(0, query, local_logprobs.detach())
        count.index_fill_(0, query, 1)
        dist.all_reduce(logprob, op=dist.ReduceOp.SUM, group=runtime.group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=runtime.group)
        if not bool((count == 1).all().item()):
            raise AssertionError("every global query must have exactly one CP logprob owner")
        full = os.getenv("TPR_P0_CP_SAVE_FULL_GRADS", "0") == "1"
        grads = _save_gradients(model, runtime, full)
        if runtime.rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            output = out_dir / f"native_cp{cp}.pt"
            versions = {}
            for package in ("mindspeed", "megatron-core", "torch-npu", "transformers"):
                try:
                    versions[package] = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    versions[package] = "unavailable"
            payload = {
                "tag": "P0_NATIVE_RING_CP_DIAGNOSTIC_NOT_PARITY_PASS",
                "cp": cp, "model": "Qwen3-1.7B", "checkpoint": str(checkpoint),
                "dtype": "bf16", "source": source, "token_sha256": token_sha,
                "sampled_weight_sha256_rank0": weight_sha,
                "sequence_length": SEQ_LEN, "response_length": RESPONSE_LEN,
                "objective": "512_response_targets_including_prompt_boundary_nll",
                "denominator": total_supervised, "batch_module": batch_module,
                "native_ring_calls_rank0": native_ring_calls[0],
                "local_m": SEQ_LEN // cp, "linear_m": shapes,
                # Serialize as a plain built-in str; torch.__version__ is a
                # TorchVersion subclass and breaks weights_only=True loads.
                "torch": str(torch.__version__), "versions": versions,
                "loss": float(loss_global.item()),
                "logprobs": logprob.cpu(), "gradients": grads,
                "full_gradients_saved": full,
            }
            torch.save(payload, output)
            print(f"P0_NATIVE_CP cp={cp} status=DIAGNOSTIC_ONLY "
                  f"source={source} m={SEQ_LEN // cp} ring_calls={native_ring_calls[0]} "
                  f"loss={payload['loss']:.9f} logprob_count={len(logprob)} "
                  f"grads={len(grads)} full_grads={full} file={output}", flush=True)
        dist.barrier(group=runtime.group)
