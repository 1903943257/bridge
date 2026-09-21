"""Shared NPU helpers for the Qwen3.5 Stage 3/4 baseline punctures."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mindspeed_ops.api.triton.chunk_gated_delta_rule import (
    chunk_gated_delta_rule as STAGE1_CHUNK_GATED_DELTA_RULE,
)
from mindspeed_ops.api.triton.convolution import causal_conv1d as STAGE1_CAUSAL_CONV1D
from mindspeed_ops.arch32.triton.convolution import (
    causal_conv1d_bwd_impl as STAGE1_CAUSAL_CONV1D_BWD_IMPL,
    causal_conv1d_fwd_impl as STAGE1_CAUSAL_CONV1D_FWD_IMPL,
)


DTYPE = torch.bfloat16
VOCAB_SIZE = 248320
HIDDEN_SIZE = 1024
SEQUENCE_LENGTH = int(os.getenv("STAGE34_SEQUENCE_LENGTH", "64"))
SERVER_MCORE_SHA = "55ac7082517c3878ae653c07c09c534b8aed49f6"
SERVER_MINDSPEED_SHA = "376e9cc302a2c00bfcc84e49a557b47fec941c87"
SERVER_VERL_SHA = "91462fad3b753a191ddf6ca1f3a2b761b90326d9"
SERVER_MINDSPEED_OPS_SHA = os.getenv(
    "STAGE34_MINDSPEED_OPS_SHA", "babee85cb00ade056b1c1627e64b10a524447025"
)
SERVER_MINDSPEED_OPS_ROOT = os.getenv("STAGE34_MINDSPEED_OPS_ROOT", "/workspace/MindSpeed-Ops")


def _assert_imported_git_head(
    module,
    *,
    component,
    expected,
    expected_root=None,
    require_clean=False,
):
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise AssertionError(f"cannot locate imported {component} package")
    module_directory = Path(module_file).resolve().parent
    root_result = subprocess.run(
        ["git", "-C", str(module_directory), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    checkout_root = Path(root_result.stdout.strip()).resolve()
    head_result = subprocess.run(
        ["git", "-C", str(checkout_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_head = head_result.stdout.strip()
    if actual_head != expected:
        raise AssertionError(
            f"imported {component} HEAD is {actual_head}, expected server baseline {expected}; "
            f"module={module_file}"
        )
    if expected_root is not None and checkout_root != Path(expected_root).resolve():
        raise AssertionError(
            f"imported {component} checkout is {checkout_root}, expected {Path(expected_root).resolve()}; "
            f"module={module_file}"
        )
    status_result = subprocess.run(
        ["git", "-C", str(checkout_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty_entries = tuple(line for line in status_result.stdout.splitlines() if line)
    if require_clean and dirty_entries:
        preview = "\n".join(dirty_entries[:20])
        raise AssertionError(f"imported {component} checkout is dirty:\n{preview}")
    return SimpleNamespace(
        root=str(checkout_root),
        module=str(Path(module_file).resolve()),
        head=actual_head,
        dirty=bool(dirty_entries),
        dirty_entries=dirty_entries,
    )


def initialize_npu_runtime(*, world_size: int):
    """Initialize MindSpeed before importing any Megatron model-spec module."""
    if int(os.getenv("WORLD_SIZE", "1")) != world_size:
        raise RuntimeError(f"expected WORLD_SIZE={world_size}")

    import torch_npu  # noqa: F401

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl")

    saved_argv = sys.argv[:]
    try:
        sys.argv[:] = [sys.argv[0]]
        from mindspeed.megatron_adaptor import repatch
    finally:
        sys.argv[:] = saved_argv

    from mindspeed.args_utils import get_full_args

    vars(get_full_args()).pop("", None)
    repatch(
        {
            "context_parallel_size": world_size,
            "context_parallel_algo": "megatron_cp_algo",
            "experimental_attention_variant": "gated_delta_net",
            "use_naive_l2norm": True,
            "use_flash_attn": True,
            "deterministic_mode": False,
        }
    )

    from megatron.core import parallel_state
    import megatron.core as megatron_core
    import mindspeed
    import mindspeed_ops
    from verl.utils import device as verl_device

    megatron_core_state = _assert_imported_git_head(
        megatron_core,
        component="Megatron-Core",
        expected=SERVER_MCORE_SHA,
    )
    mindspeed_state = _assert_imported_git_head(
        mindspeed,
        component="MindSpeed",
        expected=SERVER_MINDSPEED_SHA,
    )
    verl_state = _assert_imported_git_head(
        verl_device,
        component="verl",
        expected=SERVER_VERL_SHA,
    )
    mindspeed_ops_state = _assert_imported_git_head(
        mindspeed_ops,
        component="MindSpeed-Ops",
        expected=SERVER_MINDSPEED_OPS_SHA,
        expected_root=SERVER_MINDSPEED_OPS_ROOT,
        require_clean=False,
    )

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=world_size,
            expert_model_parallel_size=1,
        )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(350800)
    if dist.get_rank() == 0:
        print(
            "STAGE34 ENVIRONMENT"
            f"\n  Megatron-Core: HEAD={megatron_core_state.head}, dirty={megatron_core_state.dirty}"
            f"\n  MindSpeed: HEAD={mindspeed_state.head}, dirty={mindspeed_state.dirty}"
            f"\n  verl: HEAD={verl_state.head}, dirty={verl_state.dirty}"
            f"\n  MindSpeed-Ops module: {mindspeed_ops_state.module}"
            f"\n  MindSpeed-Ops checkout: {mindspeed_ops_state.root}"
            f"\n  MindSpeed-Ops: HEAD={mindspeed_ops_state.head}, dirty={mindspeed_ops_state.dirty}"
        )
    return SimpleNamespace(
        rank=dist.get_rank(),
        world_size=world_size,
        device=torch.device("npu", local_rank),
        cp_group=parallel_state.get_context_parallel_group(),
        tp_group=parallel_state.get_tensor_model_parallel_group(),
        pp_group=parallel_state.get_pipeline_model_parallel_group(),
        mindspeed_ops=mindspeed_ops_state,
    )


def destroy_npu_runtime(runtime):
    dist.barrier(group=runtime.cp_group)
    from megatron.core import parallel_state

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def process_groups(runtime, *, cp_size: int):
    from megatron.core.process_groups_config import ProcessGroupCollection

    groups = ProcessGroupCollection()
    groups.tp = runtime.tp_group
    groups.cp = runtime.tp_group if cp_size == 1 else runtime.cp_group
    groups.pp = runtime.pp_group
    groups.embd = None
    return groups


class AllToAllProbe:
    """Count the low-level MindSpeed GDN collectives without changing their math."""

    def __init__(self, gated_delta_net_module):
        self.module = gated_delta_net_module
        self.original_cp2hp = gated_delta_net_module._all_to_all_cp2hp
        self.original_hp2cp = gated_delta_net_module._all_to_all_hp2cp
        self.calls = []

    def __enter__(self):
        def cp2hp(input_, cp_group):
            output = self.original_cp2hp(input_, cp_group)
            self.calls.append("cp2hp")
            return output

        def hp2cp(input_, cp_group):
            output = self.original_hp2cp(input_, cp_group)
            self.calls.append("hp2cp")
            return output

        self.module._all_to_all_cp2hp = cp2hp
        self.module._all_to_all_hp2cp = hp2cp
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module._all_to_all_cp2hp = self.original_cp2hp
        self.module._all_to_all_hp2cp = self.original_hp2cp

    def count(self, direction):
        return self.calls.count(direction)


def broadcast_module_state(module, *, src=0):
    """Ensure all CP ranks start from the same parameters and buffers."""
    with torch.no_grad():
        for name, value in module.state_dict().items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"cannot broadcast non-Tensor state {name!r}: {type(value).__name__}")
            dist.broadcast(value, src=src)


def gather_native_zigzag(local_tensor, cp_group, *, seq_dim=0):
    """Gather Megatron's two-chunk CP shards back into global sequence order."""
    cp_size = cp_group.size()
    local_first = local_tensor.movedim(seq_dim, 0).contiguous()
    if local_first.shape[0] % 2:
        raise ValueError("local zigzag sequence length must be even")
    gathered = [torch.empty_like(local_first) for _ in range(cp_size)]
    dist.all_gather(gathered, local_first, group=cp_group)
    chunk_length = local_first.shape[0] // 2
    chunks = [None] * (2 * cp_size)
    for rank, rank_tensor in enumerate(gathered):
        chunks[rank], chunks[2 * cp_size - rank - 1] = rank_tensor.split(chunk_length, dim=0)
    return torch.cat(chunks, dim=0).movedim(0, seq_dim).contiguous()


@dataclass(frozen=True)
class PackedBoundaryMetadata:
    """Logical and padded boundaries that a future PrefixState must retain."""

    actual_cu_seqlens: tuple[int, ...]
    padded_cu_seqlens: tuple[int, ...]
    segment_ids: tuple[int, ...]

    def validate(self):
        expected_offsets = len(self.segment_ids) + 1
        if len(self.actual_cu_seqlens) != expected_offsets:
            raise AssertionError("actual cu_seqlens do not match segment identities")
        if len(self.padded_cu_seqlens) != expected_offsets:
            raise AssertionError("padded cu_seqlens do not match segment identities")
        if self.actual_cu_seqlens[0] != 0 or self.padded_cu_seqlens[0] != 0:
            raise AssertionError("packed sequence offsets must start at zero")
        for actual_start, actual_end, padded_start, padded_end in zip(
            self.actual_cu_seqlens,
            self.actual_cu_seqlens[1:],
            self.padded_cu_seqlens,
            self.padded_cu_seqlens[1:],
        ):
            if actual_end <= actual_start:
                raise AssertionError("logical packed segments must be non-empty")
            if padded_end - padded_start < actual_end - actual_start:
                raise AssertionError("a padded segment cannot be shorter than its logical segment")


@dataclass(frozen=True)
class PackedSeqParamsMetadata:
    """Value snapshot of the PackedSeqParams fields consumed by GDN and FA."""

    qkv_format: str
    cu_seqlens_q: tuple[int, ...] | None
    cu_seqlens_kv: tuple[int, ...] | None
    cu_seqlens_q_padded: tuple[int, ...] | None
    cu_seqlens_kv_padded: tuple[int, ...] | None
    max_seqlen_q: int | None
    max_seqlen_kv: int | None


def packed_seq_params_metadata(packed_seq_params):
    def offsets(field):
        value = getattr(packed_seq_params, field, None)
        if value is None:
            return None
        return tuple(int(item) for item in value.reshape(-1).tolist())

    def scalar(field):
        value = getattr(packed_seq_params, field, None)
        return None if value is None else int(value)

    return PackedSeqParamsMetadata(
        qkv_format=packed_seq_params.qkv_format,
        cu_seqlens_q=offsets("cu_seqlens_q"),
        cu_seqlens_kv=offsets("cu_seqlens_kv"),
        cu_seqlens_q_padded=offsets("cu_seqlens_q_padded"),
        cu_seqlens_kv_padded=offsets("cu_seqlens_kv_padded"),
        max_seqlen_q=scalar("max_seqlen_q"),
        max_seqlen_kv=scalar("max_seqlen_kv"),
    )


def packed_boundary_metadata(input_ids, packed_seq_params):
    """Capture logical boundaries separately from PackedSeqParams' padded offsets."""
    actual = tuple(int(value) for value in input_ids.offsets().tolist())
    padded_tensor = packed_seq_params.cu_seqlens_q_padded
    if padded_tensor is None:
        padded_tensor = packed_seq_params.cu_seqlens_q
    padded = tuple(int(value) for value in padded_tensor.tolist())
    metadata = PackedBoundaryMetadata(
        actual_cu_seqlens=actual,
        padded_cu_seqlens=padded,
        segment_ids=tuple(range(len(actual) - 1)),
    )
    metadata.validate()
    return metadata


def _call_supports(function, argument):
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot inspect Stage 1 primitive {function!r}") from exc
    parameter = parameters.get(argument)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _supported_keyword_arguments(function, arguments):
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot inspect Stage 1 primitive {function!r}") from exc
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(arguments)
    return {
        name: value
        for name, value in arguments.items()
        if name in parameters and parameters[name].kind is not inspect.Parameter.POSITIONAL_ONLY
    }


def stage1_stateful_causal_conv(
    x,
    weight,
    bias=None,
    *,
    residual=None,
    activation=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_list=None,
    **_unused,
):
    """FLA-shaped adapter over the Stage 1 MindSpeed-Ops CausalConv primitive."""
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError(f"expected x=[B,S,D], weight=[D,W], got {x.shape}, {weight.shape}")
    # MindSpeed GDN stores [D,W]; the Stage 1 primitive consumes [W,D].
    primitive_weight = weight.transpose(0, 1).contiguous()
    optional_arguments = {
        "bias": bias,
        "residual": residual,
        "initial_state": initial_state,
        "activation": activation,
        "output_final_state": output_final_state,
    }
    if cu_seqlens is None:
        return STAGE1_CAUSAL_CONV1D(
            x,
            primitive_weight,
            **_supported_keyword_arguments(STAGE1_CAUSAL_CONV1D, optional_arguments),
        )
    if _call_supports(STAGE1_CAUSAL_CONV1D, "cu_seqlens"):
        optional_arguments["cu_seqlens"] = cu_seqlens
        return STAGE1_CAUSAL_CONV1D(
            x,
            primitive_weight,
            **_supported_keyword_arguments(STAGE1_CAUSAL_CONV1D, optional_arguments),
        )

    if x.shape[0] != 1:
        raise ValueError("segmented packed CausalConv fallback requires flattened batch size 1")
    offsets = cu_seqlens_list
    if offsets is None:
        offsets = [int(value) for value in cu_seqlens.tolist()]
    outputs = []
    final_states = []
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        segment_state = None if initial_state is None else initial_state[index : index + 1]
        segment_residual = None if residual is None else residual[:, start:end]
        segment_arguments = {
            "bias": bias,
            "residual": segment_residual,
            "initial_state": segment_state,
            "activation": activation,
            "output_final_state": output_final_state,
        }
        output, final_state = STAGE1_CAUSAL_CONV1D(
            x[:, start:end],
            primitive_weight,
            **_supported_keyword_arguments(STAGE1_CAUSAL_CONV1D, segment_arguments),
        )
        outputs.append(output)
        if output_final_state:
            final_states.append(final_state)
    final_state = torch.cat(final_states, dim=0) if output_final_state else None
    return torch.cat(outputs, dim=1), final_state


def stage1_stateful_gated_delta_rule(
    query,
    key,
    value,
    *,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_list=None,
    chunk_size=64,
    **_unused,
):
    """MindSpeed GDN-shaped adapter over the Stage 1 stateful GDR primitive."""
    stage1_arguments = {
        "q": query,
        "k": key,
        "v": value,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "chunk_size": chunk_size,
        "head_first": False,
    }
    if scale is not None:
        stage1_arguments["scale"] = scale
    if use_qk_l2norm_in_kernel:
        stage1_arguments["use_qk_l2norm_in_kernel"] = True
    if cu_seqlens is None:
        return STAGE1_CHUNK_GATED_DELTA_RULE(**stage1_arguments)
    if _call_supports(STAGE1_CHUNK_GATED_DELTA_RULE, "cu_seqlens"):
        stage1_arguments["cu_seqlens"] = cu_seqlens
        return STAGE1_CHUNK_GATED_DELTA_RULE(**stage1_arguments)

    if query.shape[0] != 1:
        raise ValueError("segmented packed GDR fallback requires flattened batch size 1")
    offsets = cu_seqlens_list
    if offsets is None:
        offsets = [int(value) for value in cu_seqlens.tolist()]
    outputs = []
    final_states = []
    for index, (start, end) in enumerate(zip(offsets, offsets[1:])):
        segment_state = None if initial_state is None else initial_state[index : index + 1]
        segment_arguments = {
            "q": query[:, start:end],
            "k": key[:, start:end],
            "v": value[:, start:end],
            "g": g[:, start:end],
            "beta": beta[:, start:end],
            "initial_state": segment_state,
            "output_final_state": output_final_state,
            "chunk_size": chunk_size,
            "head_first": False,
        }
        if scale is not None:
            segment_arguments["scale"] = scale
        if use_qk_l2norm_in_kernel:
            segment_arguments["use_qk_l2norm_in_kernel"] = True
        output, final_state = STAGE1_CHUNK_GATED_DELTA_RULE(**segment_arguments)
        outputs.append(output)
        if output_final_state:
            final_states.append(final_state)
    final_state = torch.cat(final_states, dim=0) if output_final_state else None
    return torch.cat(outputs, dim=1), final_state


@contextmanager
def bind_stage1_gdn_primitives(gdn_module, *models):
    """Temporarily route MindSpeed 0.16 GDN through the verified Stage 1 primitives."""
    gdn_layers = []
    for model in models:
        if isinstance(model, gdn_module.GatedDeltaNet):
            gdn_layers.append(model)
        else:
            gdn_layers.extend(
                module for module in model.modules() if isinstance(module, gdn_module.GatedDeltaNet)
            )
    if not gdn_layers:
        raise AssertionError("no MindSpeed GatedDeltaNet layers found for Stage 1 binding")

    original_causal_conv = gdn_module.causal_conv1d
    original_recurrent_backends = [layer.gated_delta_rule for layer in gdn_layers]
    gdn_module.causal_conv1d = stage1_stateful_causal_conv
    for layer in gdn_layers:
        layer.gated_delta_rule = stage1_stateful_gated_delta_rule
    try:
        yield {
            "layers": tuple(gdn_layers),
            "causal_conv": f"{stage1_stateful_causal_conv.__module__}.{stage1_stateful_causal_conv.__name__}",
            "gated_delta_rule": (
                f"{stage1_stateful_gated_delta_rule.__module__}."
                f"{stage1_stateful_gated_delta_rule.__name__}"
            ),
        }
    finally:
        gdn_module.causal_conv1d = original_causal_conv
        for layer, backend in zip(gdn_layers, original_recurrent_backends):
            layer.gated_delta_rule = backend


def qwen35_config(*, cp_size: int):
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=24,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=3584,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        attention_output_gate=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        qk_layernorm=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        rotary_percent=0.25,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        use_cpu_initialization=False,
        params_dtype=DTYPE,
        pipeline_dtype=DTYPE,
        autocast_dtype=DTYPE,
        bf16=True,
        deterministic_mode=False,
        # core_v0.16.1's experimental hybrid block spec only accepts the TE
        # provider. MindSpeed repatches that provider for NPU execution.
        transformer_impl="transformer_engine",
        apply_rope_fusion=False,
        bias_dropout_fusion=False,
    )


def make_qwen35_model(runtime, *, cp_size: int, tpr: bool = False, num_layers: int = 24):
    """Build the 24-layer Qwen stack (default) or its 3-GDN/1-FA gate."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_block_with_experimental_attention_variant_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel

    config = qwen35_config(cp_size=cp_size)
    if num_layers not in (4, 24):
        raise ValueError("baseline supports the 4-layer gate or full 24-layer Qwen pattern")
    config.num_layers = num_layers
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    if tpr:
        if cp_size not in (1, 2):
            raise NotImplementedError("Hybrid TPR baseline requires CP=1/2")
        from verl.models.mcore.tpr.module_spec import replace_self_attention_with_tpr

        spec = replace_self_attention_with_tpr(spec)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=max(SEQUENCE_LENGTH, 128),
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=0.25,
        rotary_base=10_000_000.0,
        pg_collection=process_groups(runtime, cp_size=cp_size),
    ).to(device=runtime.device, dtype=DTYPE)
    if hasattr(model.rotary_pos_emb, "inv_freq"):
        model.rotary_pos_emb.inv_freq = model.rotary_pos_emb.inv_freq.to(runtime.device)
    for module in model.modules():
        if hasattr(module, "tp_group") and module.tp_group is None:
            module.tp_group = runtime.tp_group
    model.train()
    return model


def assert_hybrid_architecture(model):
    import mindspeed.core.ssm.gated_delta_net as mindspeed_gdn

    layers = tuple(model.decoder.layers)
    gdn_layers = tuple(
        layer for layer in layers if isinstance(layer.self_attention, mindspeed_gdn.GatedDeltaNet)
    )
    fa_layers = tuple(layer for layer in layers if layer not in gdn_layers)
    assert len(layers) == 24
    assert len(gdn_layers) == 18
    assert len(fa_layers) == 6
    assert tuple(layer.layer_number for layer in fa_layers) == (4, 8, 12, 16, 20, 24)
    return gdn_layers, fa_layers


def full_tokens(device, length=SEQUENCE_LENGTH):
    tokens = (torch.arange(length, device=device, dtype=torch.long) * 17 + 23) % VOCAB_SIZE
    positions = torch.arange(length, device=device, dtype=torch.long)
    labels = torch.roll(tokens, shifts=-1)
    valid = torch.ones(length, device=device, dtype=torch.bool)
    valid[-1] = False
    return tokens.unsqueeze(0), positions.unsqueeze(0), labels, valid


def zigzag_indices(length: int, *, cp_rank: int, cp_size: int, device):
    assert length % (2 * cp_size) == 0
    chunk = length // (2 * cp_size)
    first = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    mirror = 2 * cp_size - cp_rank - 1
    second = torch.arange(mirror * chunk, (mirror + 1) * chunk, device=device)
    return torch.cat((first, second)).long()


def selected_output_and_loss(logits, labels, valid):
    """Return a compact output probe and globally sum-normalized CE loss."""
    if logits.dim() != 3:
        raise AssertionError(f"expected 3-D logits, got {tuple(logits.shape)}")
    # MCore post-process output is [S, B, V].
    if logits.shape[1] == 1:
        flat = logits[:, 0]
    elif logits.shape[0] == 1:
        flat = logits[0]
    else:
        raise AssertionError(f"cannot identify singleton batch dimension: {tuple(logits.shape)}")
    probe_ids = torch.tensor([0, 1, 17, 1024, VOCAB_SIZE - 1], device=flat.device)
    probe = flat.index_select(-1, probe_ids)
    losses = F.cross_entropy(flat.float(), labels, reduction="none")
    loss = losses[valid].sum() / float(SEQUENCE_LENGTH - 1)
    return probe, loss


def allreduce_parameter_gradients(model, group):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise AssertionError(f"missing parameter gradient: {name}")
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)


def clone_parameter_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise AssertionError(f"missing parameter gradient: {name}")
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().clone()
    return gradients


@dataclass(frozen=True)
class GradientPairMetrics:
    reference_norm: float
    actual_norm: float
    norm_ratio: float
    absolute_l2: float
    relative_l2: float
    cosine: float


@dataclass(frozen=True)
class GradientMapDiagnostics:
    aggregate: GradientPairMetrics
    worst_name: str
    worst: GradientPairMetrics


def _gradient_pair_metrics(*, reference_l2, actual_l2, difference_l2, dot):
    reference_norm = reference_l2**0.5
    actual_norm = actual_l2**0.5
    absolute_l2 = difference_l2**0.5
    if reference_norm == 0.0:
        norm_ratio = 1.0 if actual_norm == 0.0 else float("inf")
        relative_l2 = 0.0 if absolute_l2 == 0.0 else float("inf")
    else:
        norm_ratio = actual_norm / reference_norm
        relative_l2 = absolute_l2 / reference_norm
    if reference_norm == 0.0 and actual_norm == 0.0:
        cosine = 1.0
    elif reference_norm == 0.0 or actual_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (reference_norm * actual_norm)
    return GradientPairMetrics(
        reference_norm=reference_norm,
        actual_norm=actual_norm,
        norm_ratio=norm_ratio,
        absolute_l2=absolute_l2,
        relative_l2=relative_l2,
        cosine=cosine,
    )


def gradient_map_diagnostics(reference, actual):
    """Summarize a gradient-map comparison without applying pass/fail thresholds."""
    if reference.keys() != actual.keys():
        raise AssertionError(
            f"gradient parameter sets differ: missing={reference.keys() - actual.keys()}, "
            f"unexpected={actual.keys() - reference.keys()}"
        )
    if not reference:
        raise AssertionError("cannot compare empty gradient maps")

    reference_l2 = actual_l2 = difference_l2 = dot = 0.0
    worst_name = None
    worst_metrics = None
    for name in reference:
        left_f = reference[name].detach().float()
        right_f = actual[name].detach().float()
        if left_f.shape != right_f.shape:
            raise AssertionError(
                f"gradient shape differs for {name}: {left_f.shape} versus {right_f.shape}"
            )
        local_reference_l2 = torch.sum(left_f.square()).item()
        local_actual_l2 = torch.sum(right_f.square()).item()
        local_difference_l2 = torch.sum((right_f - left_f).square()).item()
        local_dot = torch.sum(left_f * right_f).item()
        local_metrics = _gradient_pair_metrics(
            reference_l2=local_reference_l2,
            actual_l2=local_actual_l2,
            difference_l2=local_difference_l2,
            dot=local_dot,
        )
        if worst_metrics is None or local_metrics.relative_l2 > worst_metrics.relative_l2:
            worst_name = name
            worst_metrics = local_metrics
        reference_l2 += local_reference_l2
        actual_l2 += local_actual_l2
        difference_l2 += local_difference_l2
        dot += local_dot

    return GradientMapDiagnostics(
        aggregate=_gradient_pair_metrics(
            reference_l2=reference_l2,
            actual_l2=actual_l2,
            difference_l2=difference_l2,
            dot=dot,
        ),
        worst_name=worst_name,
        worst=worst_metrics,
    )


def assert_gradient_maps_close(reference, actual, *, rtol=8e-2, cosine_min=0.995):
    if reference.keys() != actual.keys():
        raise AssertionError(
            f"gradient parameter sets differ: missing={reference.keys() - actual.keys()}, "
            f"unexpected={actual.keys() - reference.keys()}"
        )
    diff2 = ref2 = act2 = dot = 0.0
    worst = (0.0, "")
    for name in reference:
        left = reference[name]
        right = actual[name]
        left_f, right_f = left.float(), right.float()
        local_diff2 = torch.sum((right_f - left_f).square()).item()
        local_ref2 = torch.sum(left_f.square()).item()
        diff2 += local_diff2
        ref2 += local_ref2
        act2 += torch.sum(right_f.square()).item()
        dot += torch.sum(left_f * right_f).item()
        relative = (local_diff2 / max(local_ref2, 1e-24)) ** 0.5
        # Segment probes use integer keys; parameter maps use strings. Rank
        # only by error, never compare heterogeneous names when errors tie.
        worst = max(worst, (relative, name), key=lambda item: item[0])
    relative_l2 = (diff2 / max(ref2, 1e-24)) ** 0.5
    cosine = dot / max((ref2 * act2) ** 0.5, 1e-24)
    if relative_l2 > rtol or cosine < cosine_min:
        raise AssertionError(
            f"gradient mismatch: relative_l2={relative_l2:.6e}, cosine={cosine:.9f}, "
            f"worst={worst[1]}:{worst[0]:.6e}"
        )
    return relative_l2, cosine, worst


def assert_gradients_close(reference, actual, *, rtol=8e-2, cosine_min=0.995):
    reference_gradients = {
        name: parameter.grad for name, parameter in reference.named_parameters()
    }
    actual_gradients = {
        name: parameter.grad for name, parameter in actual.named_parameters()
    }
    missing = [
        name
        for name, gradient in (*reference_gradients.items(), *actual_gradients.items())
        if gradient is None
    ]
    if missing:
        raise AssertionError(f"missing compared parameter gradients: {missing}")
    return assert_gradient_maps_close(
        reference_gradients,
        actual_gradients,
        rtol=rtol,
        cosine_min=cosine_min,
    )


def assert_tensor_close_by_norm(
    reference,
    actual,
    *,
    rtol=8e-2,
    cosine_min=0.995,
    max_abs=None,
    label="tensor",
):
    reference_flat = reference.detach().float().reshape(-1)
    actual_flat = actual.detach().float().reshape(-1)
    if reference_flat.shape != actual_flat.shape:
        raise AssertionError(
            f"{label} shapes differ: {reference.shape} versus {actual.shape}"
        )
    reference_norm = torch.linalg.vector_norm(reference_flat)
    actual_norm = torch.linalg.vector_norm(actual_flat)
    difference = actual_flat - reference_flat
    difference_norm = torch.linalg.vector_norm(difference)
    relative_l2 = (difference_norm / reference_norm.clamp_min(1e-24)).item()
    cosine = (
        torch.dot(reference_flat, actual_flat)
        / (reference_norm * actual_norm).clamp_min(1e-24)
    ).item()
    max_abs_diff = difference.abs().max().item() if difference.numel() else 0.0
    if relative_l2 > rtol or cosine < cosine_min or (
        max_abs is not None and max_abs_diff > max_abs
    ):
        raise AssertionError(
            f"{label} mismatch: relative_l2={relative_l2:.6e} (limit {rtol:.6e}), "
            f"cosine={cosine:.9f} (minimum {cosine_min:.9f}), "
            f"max_abs={max_abs_diff:.6e}"
            + ("" if max_abs is None else f" (limit {max_abs:.6e})")
        )
    return relative_l2, cosine, max_abs_diff


def assert_tensor_gradient_close(reference, actual, *, rtol=8e-2, cosine_min=0.995):
    relative_l2, cosine, _ = assert_tensor_close_by_norm(
        reference,
        actual,
        rtol=rtol,
        cosine_min=cosine_min,
        label="input-gradient",
    )
    return relative_l2, cosine
