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

"""Final single-rank Engine-level reference comparison for the DTA MVP."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import gc
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from test_segment_push_pop_npu import (
    _SingleProcessGroup,
    _assert_gradients_close,
    _install_single_rank_runtime,
    _parameter_grads,
)
from verl.models.mcore.dta import (
    DTA_REQUEST_KEY,
    DTAForwardBackwardRequest,
    SegmentExecutor,
    SegmentLossTerm,
    SegmentPlan,
    SegmentSpec,
    replace_self_attention_with_dta,
)
from verl.utils import tensordict_utils as tu
from verl.utils.device import is_torch_npu_available

if not is_torch_npu_available(check_device=True):
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)

_VOCAB_SIZE = 2048

_LENGTH_CASES = (
    (1, 1),
    (1, 128),
    (128, 1),
    (128, 128),
    (1024, 128),
    (128, 1024),
    (8192, 128),
    (128, 8192),
    (16384, 1),
    (1, 16384),
    (16384, 512),
    (512, 16384),
    (8192, 8192),
)


def _make_model(
    device,
    *,
    dta: bool,
    max_sequence_length: int,
    core_attention_module=None,
    model_shape=None,
):
    dtype = torch.bfloat16
    shape = {
        "num_layers": 2,
        "hidden_size": 128,
        "ffn_hidden_size": 256,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "kv_channels": 32,
        "vocab_size": _VOCAB_SIZE,
    }
    if model_shape is not None:
        shape.update(model_shape)
    vocab_size = shape.pop("vocab_size")
    config = TransformerConfig(
        **shape,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=True,
        sequence_parallel=False,
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )
    process_group = _SingleProcessGroup()
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False, pp_rank=0)
    if dta:
        spec = replace_self_attention_with_dta(spec)
    if core_attention_module is not None:
        for layer_spec in spec.layer_specs:
            layer_spec.submodules.self_attention.submodules.core_attention = core_attention_module
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        pg_collection=SimpleNamespace(tp=process_group, cp=process_group, pp=process_group, embd=None),
    ).to(device=device, dtype=dtype)
    model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = process_group
    model.train()
    return model


def _tokens(start, length, device):
    return torch.arange(start, start + length, dtype=torch.long, device=device) % _VOCAB_SIZE


def _internal_terms(tokens, *, weight=1.0, sample_id=None):
    tokens = tokens.cpu()
    return tuple(
        SegmentLossTerm(index, int(tokens[index + 1]), weight=weight, sample_id=sample_id)
        for index in range(tokens.numel() - 1)
    )


def _make_plan(prefix, *suffixes):
    prefix = prefix.cpu()
    suffixes = tuple(suffix.cpu() for suffix in suffixes)
    if not suffixes:
        raise ValueError("at least one suffix is required")
    prefix_length = prefix.numel()
    suffix_length = suffixes[0].numel()
    if any(suffix.numel() != suffix_length for suffix in suffixes[1:]):
        raise ValueError("the focused Engine comparison requires equal suffix lengths")
    sibling_count = len(suffixes)
    total_loss_weight = sibling_count * (prefix_length + suffix_length - 1)
    branch_terms = tuple(
        SegmentLossTerm(prefix_length - 1, int(suffix[0]), sample_id=sample_id)
        for sample_id, suffix in enumerate(suffixes, start=1)
    )
    segments = [
        SegmentSpec(
            0,
            None,
            prefix,
            0,
            0,
            _internal_terms(prefix, weight=float(sibling_count)) + branch_terms,
        )
    ]
    segments.extend(
        SegmentSpec(
            sample_id,
            0,
            suffix,
            prefix_length,
            prefix_length,
            _internal_terms(suffix, sample_id=sample_id),
        )
        for sample_id, suffix in enumerate(suffixes, start=1)
    )
    plan = SegmentPlan(segments, root_id=0)
    # Pin branch-point label shift and loss ownership before comparing gradients.
    actual_branch_terms = plan.get(0).loss_terms[-sibling_count:]
    assert tuple(term.query_offset for term in actual_branch_terms) == (prefix_length - 1,) * sibling_count
    assert tuple(term.target_token_id for term in actual_branch_terms) == tuple(
        int(suffix[0]) for suffix in suffixes
    )
    assert tuple(term.sample_id for term in actual_branch_terms) == tuple(range(1, sibling_count + 1))
    assert plan.total_loss_weight == total_loss_weight
    return plan


def _is_device_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        "out of memory" in message
        or "memory allocation" in message
        or "acl_error_rt_memory_allocation" in message
    )


def _configure_model_runtime(model):
    @contextmanager
    def no_sync():
        yield

    model.config.no_sync_func = no_sync
    model.config.grad_scale_func = lambda loss: loss
    model.config.finalize_model_grads_func = lambda *args, **kwargs: None
    model.config.calculate_per_token_loss = False


def _make_engine(model, *, dta_enabled, monkeypatch):
    # Import after constructing the Apex-free tiny models: importing MindSpeed's
    # engine patches Megatron's global norm spec on the target server. The test
    # needs the Megatron engine only; suppress verl.engine's unrelated eager
    # MindSpeed-engine import, which cannot safely repatch already-built test
    # dataclasses in a fresh pytest process.
    monkeypatch.setitem(sys.modules, "verl.workers.engine.mindspeed", ModuleType("verl.workers.engine.mindspeed"))
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    engine = MegatronEngineWithLMHead.__new__(MegatronEngineWithLMHead)
    engine.module = [model]
    engine.engine_config = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        dta_enabled=dta_enabled,
        pad_bshd_to_minibatch_max=False,
        use_remove_padding=False,
        dynamic_context_parallel=False,
    )
    engine.model_config = SimpleNamespace(mtp=SimpleNamespace(enable=False))
    engine.tf_config = model.config
    engine.enable_routing_replay = False
    engine._distillation_use_topk_active = False
    engine.get_data_parallel_size = lambda: 1
    engine.get_data_parallel_group = lambda: None
    return engine


def _reference_loss_function(*, model_output, data, dp_group):
    del dp_group
    logits = model_output["logits"]
    tokens = data["input_ids"]
    loss_sum = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, _VOCAB_SIZE),
        tokens[:, 1:].reshape(-1),
        reduction="sum",
    )
    return loss_sum / data["batch_num_tokens"], {}


def _install_reference_forward(engine, observed_microbatches, monkeypatch):
    # Keep Megatron's real no-pipeline schedule and only provide its single-rank
    # process-group collection explicitly; the focused fixture does not create
    # torch.distributed global groups.
    from megatron.core.pipeline_parallel import schedules
    from verl.workers.engine.megatron import transformer_impl

    process_group = _SingleProcessGroup()
    # Megatron releases differ in how many fields the no-pipeline schedule
    # validates. Populate the complete single-rank collection; unused embedding
    # groups are explicitly None, matching Megatron's own default builder.
    pg_collection = SimpleNamespace(
        tp=process_group,
        cp=process_group,
        pp=process_group,
        dp=process_group,
        dp_cp=process_group,
        tp_dp_cp=process_group,
        embd=None,
        pos_embd=None,
        gtp_remat=None,
        expt_gtp_remat=None,
        dp_cp_gtp_remat=process_group,
    )
    monkeypatch.setattr(
        transformer_impl,
        "get_forward_backward_func",
        lambda: partial(schedules.forward_backward_no_pipelining, pg_collection=pg_collection),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, *args, **kwargs: tensor)
    # The tiny fixture supplies explicit process groups to the schedule rather
    # than initializing Megatron's global parallel state. Pin the PP queries
    # that verl performs after the schedule to their true single-rank values.
    monkeypatch.setattr(transformer_impl.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(transformer_impl.mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        transformer_impl.mpu,
        "is_pipeline_first_stage",
        lambda ignore_virtual=False: True,
    )
    monkeypatch.setattr(
        transformer_impl.mpu,
        "is_pipeline_last_stage",
        lambda ignore_virtual=False: True,
    )
    original_zeros = torch.zeros

    def npu_compatible_zeros(*args, **kwargs):
        # This Megatron revision hard-codes CUDA for the schedule's scalar
        # token counter. Production MindSpeed rewrites it through
        # transfer_to_npu; this focused Megatron-engine fixture does not load
        # that global patcher, so translate only the literal CUDA device here.
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "npu"
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", npu_compatible_zeros)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        del logits_processor_func
        batch = next(batch_iter).to(torch.device("npu"))
        observed_microbatches.append(int(batch.batch_size[0]))
        tokens = batch["input_ids"]
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones((1, 1, length, length), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        output = {"logits": model(tokens, positions, causal_mask)}
        return output, partial(postprocess_micro_batch_func, data=batch)

    def postprocess(self, output, data, forward_only, loss_function):
        assert not forward_only
        loss, metrics = loss_function(model_output=output, data=data, dp_group=None)
        return loss * data["num_micro_batch"], {
            "model_output": {},
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

    engine.forward_step = MethodType(forward_step, engine)
    engine.postprocess_micro_batch_func = MethodType(postprocess, engine)


def _reference_data(*trajectories):
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    trajectories = torch.stack(trajectories)
    batch_size = trajectories.shape[0]
    data = TensorDict(
        {
            "input_ids": trajectories,
            "loss_mask": torch.ones(
                (batch_size, trajectories.shape[1] - 1), dtype=torch.bool, device=trajectories.device
            ),
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor(
        data,
        micro_batch_size_per_gpu=1,
        use_dynamic_bsz=False,
        pad_mode="no_padding",
    )
    return data


@pytest.mark.parametrize(("prefix_length", "suffix_length"), _LENGTH_CASES)
def test_engine_dta_matches_native_megatron_microbatch_accumulation(
    monkeypatch,
    prefix_length,
    suffix_length,
):
    torch.manual_seed(2026)
    device = torch.device("npu")
    _install_single_rank_runtime(monkeypatch, device)

    full_length = prefix_length + suffix_length
    reference_model = _make_model(device, dta=False, max_sequence_length=full_length)
    dta_model = _make_model(device, dta=True, max_sequence_length=full_length)
    dta_model.load_state_dict(reference_model.state_dict(), strict=True)
    _configure_model_runtime(reference_model)
    _configure_model_runtime(dta_model)
    reference_engine = _make_engine(reference_model, dta_enabled=False, monkeypatch=monkeypatch)
    dta_engine = _make_engine(dta_model, dta_enabled=True, monkeypatch=monkeypatch)

    prefix = _tokens(17, prefix_length, device)
    suffix_1 = _tokens(700, suffix_length, device)
    suffix_2 = _tokens(1300, suffix_length, device)
    plan = _make_plan(prefix, suffix_1, suffix_2)

    observed_microbatches = []
    _install_reference_forward(reference_engine, observed_microbatches, monkeypatch)
    reference_oom = None
    try:
        reference_output = reference_engine.forward_backward_batch(
            _reference_data(torch.cat((prefix, suffix_1)), torch.cat((prefix, suffix_2))),
            loss_function=_reference_loss_function,
            forward_only=False,
        )
    except RuntimeError as error:
        if not _is_device_oom(error):
            raise
        reference_oom = str(error)
    if reference_oom is not None:
        reference_model.zero_grad(set_to_none=True)
        dta_model.zero_grad(set_to_none=True)
        del reference_engine, dta_engine, reference_model, dta_model
        gc.collect()
        torch.npu.empty_cache()
        pytest.skip(
            f"Reference Engine OOM for P={prefix_length}, S={suffix_length}; "
            "capacity is covered by the separate DTA long-context test"
        )
    assert observed_microbatches == [1, 1]
    assert len(reference_output["loss"]) == 2
    # Each microbatch loss is already normalized by the global token count.
    # The schedule's multiply/divide-by-n pair only preserves gradient scale.
    reference_loss = sum(reference_output["loss"])
    reference_gradients = _parameter_grads(reference_model)

    counters = {"prefix_push": 0, "prefix_pop": 0}
    original_push = SegmentExecutor.push
    original_pop = SegmentExecutor.pop

    def counted_push(self, segment_id):
        if segment_id == self.plan.root_id:
            counters["prefix_push"] += 1
        return original_push(self, segment_id)

    def counted_pop(self, segment_id):
        if segment_id == self.plan.root_id:
            counters["prefix_pop"] += 1
        return original_pop(self, segment_id)

    monkeypatch.setattr(SegmentExecutor, "push", counted_push)
    monkeypatch.setattr(SegmentExecutor, "pop", counted_pop)
    dta_data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(dta_data, **{DTA_REQUEST_KEY: DTAForwardBackwardRequest(plan)})
    dta_output = dta_engine.forward_backward_batch(dta_data, loss_function=None, forward_only=False)
    dta_gradients = _parameter_grads(dta_model)

    assert counters == {"prefix_push": 1, "prefix_pop": 1}
    torch.testing.assert_close(
        torch.tensor(dta_output["loss"]),
        torch.tensor(reference_loss),
        atol=2e-2,
        rtol=2e-2,
    )
    _assert_gradients_close(dta_gradients, reference_gradients)
