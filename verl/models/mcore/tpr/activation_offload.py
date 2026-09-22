"""CP1 and Ring CP2/CP4 Full-Attention adapter for MindSpeed swap-attention.

One Visit/Pop is a complete forward/backward microbatch. Native SwapPrefetch
owns selection, host allocations, streams, transfer and release. Hooks are
scoped to that microbatch so graph-free Push never enters native swap queues.
"""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import sys
from types import ModuleType, SimpleNamespace

import torch

from .context import get_tpr_attention_context


def _args():
    try:
        from megatron.training import get_args
        return get_args()
    except (ImportError, AssertionError):
        # VERL's megatron backend initializes MindSpeed via repatch, not the
        # Megatron training launcher. Do not import/repatch MindSpeed when off.
        module = sys.modules.get("mindspeed.args_utils")
        return None if module is None else module.get_full_args()


def _native_prefetch():
    """Import the installed native implementation with Core-only Megatron.

    Older MindSpeed prefetch.py imports training.get_args at module scope even
    though its only training dependency is that accessor. Supply it only during
    this import; never leave a fake training package visible to other callers.
    """
    name = "mindspeed.core.memory.swap_attention.prefetch"
    try:
        return import_module(name)
    except ModuleNotFoundError as error:
        if error.name != "megatron.training" or "megatron.training" in sys.modules:
            raise
    from mindspeed.args_utils import get_full_args

    training = ModuleType("megatron.training")
    training.get_args = get_full_args
    sys.modules["megatron.training"] = training
    try:
        return import_module(name)
    finally:
        del sys.modules["megatron.training"]


def swap_enabled(model):
    config = getattr(model, "config", None)
    explicit = getattr(config, "swap_attention", None)
    return bool(getattr(_args(), "swap_attention", False) if explicit is None else explicit)


def validate_activation_offload(model, cp_size, cp_backend=None):
    if not swap_enabled(model):
        return
    config = getattr(model, "config", None)
    args = _args()
    if args is None:
        raise RuntimeError("MindSpeed swap-attention requires initialized Megatron/MindSpeed args")
    if any(getattr(module, "tpr_state_kind", None) == "gdn" for module in model.modules()):
        raise NotImplementedError(
            "TPR swap-attention supports Full Attention only; "
            "GDN/Hybrid support is reserved for a later phase"
        )
    backend_name = cp_backend if isinstance(cp_backend, str) else getattr(cp_backend, "backend_name", None)
    if cp_size not in (1, 2, 4) or (cp_size == 1 and cp_backend is not None) or (
        cp_size > 1 and backend_name != "ring"
    ):
        raise NotImplementedError("TPR swap-attention supports CP=1 or Ring CP=2/4 only")
    if cp_backend is not None and not isinstance(cp_backend, str):
        if getattr(cp_backend, "parallel_size", None) != cp_size:
            raise ValueError("swap-attention backend parallel_size does not match actual CP size")
    if getattr(config, "context_parallel_size", 1) != cp_size:
        raise ValueError("swap-attention model context_parallel_size does not match actual CP size")
    for source in (config, args):
        if getattr(source, "context_parallel_size", cp_size) != cp_size:
            raise ValueError("swap-attention configured context_parallel_size does not match actual CP size")
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size"):
            if getattr(source, name, 1) != 1:
                raise NotImplementedError(f"TPR swap-attention requires {name}=1")
        for name in ("recompute_granularity", "recompute_method", "recompute_num_layers",
                     "virtual_pipeline_model_parallel_size", "cpu_offloading",
                     "fine_grained_activation_offloading", "adaptive_memory_optimization",
                     "adaptive_recompute_device_swap", "lora_target_modules"):
            if getattr(source, name, None):
                raise NotImplementedError(f"TPR swap-attention cannot be combined with {name}")
        if getattr(source, "cuda_graph_impl", "none") not in (None, "none"):
            raise NotImplementedError("TPR swap-attention does not support graph capture")


def _storage_key(tensor):
    return (tensor.device, tensor.untyped_storage().data_ptr())


def _iter_extra_external_tensors(context):
    """Reserved extension seam for future non-FA Prefix state exports.

    Phase A is intentionally Full-Attention only, so no additional state is
    exported here. A later Hybrid phase can extend this seam without changing
    the native swap lifecycle or KV protection policy.
    """
    return ()


def _external_storages():
    """Exported FA KV roots and reusable Prefix KV must stay resident."""
    context = get_tpr_attention_context()
    if context is None:
        return set()
    tensors = []
    for mapping in (context.past_key_values, context.new_key_values):
        tensors.extend(t for pair in mapping.values() for t in pair)
    tensors.extend(_iter_extra_external_tensors(context))
    return {_storage_key(t) for t in tensors}


def _protect_exports(native, protected):
    """Native resize_(0) is unsafe for state exported outside module outputs.

    Keep these native handles resident and remove them from the pending queue
    before native sync_d2h frees storage. All aliases of a protected storage
    are excluded. Already submitted copies finish on the native stream.
    """
    remaining = []
    for item in native.swap_tensors:
        if _storage_key(item.tensor) in protected:
            item.tpr_resident = True
        else:
            remaining.append(item)
    if len(remaining) == len(native.swap_tensors):
        return
    native.prefetch_stream.synchronize()
    native.swap_tensors = remaining
    native.data_ptr = {item.storage_data_ptr: i for i, item in enumerate(remaining)}
    for i, item in enumerate(remaining):
        item.first_tensor = i == 0
        item.last_tensor = False


@contextmanager
def mindspeed_swap_attention(model, *, cp_size=1, cp_backend=None, persistent_kv_storages=None):
    """Temporarily install the native PP1 swap schedule on a constructed model.

    VERL bypasses megatron.training.setup_model_and_optimizer, where upstream
    normally installs these hooks. This adapter deliberately keeps native
    swap_modules and the native tensor-size/view/leaf filters.
    """
    if not swap_enabled(model):
        yield None
        return
    validate_activation_offload(model, cp_size, cp_backend)
    prefetch = _native_prefetch()
    SwapPrefetch, get_layer_id = prefetch.SwapPrefetch, prefetch.get_layer_id

    if getattr(SwapPrefetch, "swap_prefetch", None) is not None:
        raise RuntimeError("Native global swap-attention is already installed; do not install it twice for TPR")
    modules = tuple(model.modules())
    if any(hasattr(m, "no_checkpoint_adaptive_recompute_forward") for m in modules):
        raise RuntimeError("Model already has MindSpeed adaptive/swap wrappers")
    config, args = getattr(model, "config", None), _args()
    selected = getattr(config, "swap_modules", getattr(args, "swap_modules", "input_norm,self_attention,post_attention_norm"))
    selected = set(selected.split(",") if isinstance(selected, str) else selected)
    layers = [(name, module) for name, module in model.named_modules()
              if name.rsplit(".", 1)[-1].isdigit() and hasattr(module, "self_attention")]
    if not layers:
        raise RuntimeError("swap-attention found no numbered Transformer layers")
    layer_ids = [str(get_layer_id(name)) for name, _ in layers]
    if len(set(layer_ids)) != len(layer_ids):
        raise RuntimeError("swap-attention requires unique native layer IDs")
    targets = [(f"{name}.{child_name}", child) for name, layer in layers
               for child_name, child in layer.named_children() if child_name in selected]
    if not targets:
        raise RuntimeError(f"swap_modules={sorted(selected)} matched no Transformer submodules")
    protected = {_storage_key(p) for p in model.parameters()}
    # Only bridge the argument accessor when the training launcher is absent.
    # Native pack_hook imports get_args directly; it cannot see get_full_args.
    native_args = SimpleNamespace(pipeline_model_parallel_size=1, eval_interval=0, curr_iteration=0)
    vars(native_args).update(vars(args))
    original_get_args = prefetch.get_args
    prefetch.get_args = lambda: native_args
    try:
        native = SwapPrefetch([[layer_ids], 1, 0, len(layers)])
    except Exception:
        prefetch.get_args = original_get_args
        raise
    original_unpack = native.unpack_hook
    handles, forwards = [], []

    def unpack(item):
        if isinstance(item, torch.Tensor):
            return original_unpack(item)
        if getattr(item, "tpr_resident", False):
            return item.tensor
        # Pop's direct dKV roots can bypass the layer backward hook.
        # Use native same-layer reload as a correctness fallback, no new policy.
        # Native duplicate handles are labelled h2d without recording their
        # own event: the last alias owns the actual transfer. Reload the layer
        # even for such handles, then wait on the native stream, not that event.
        native.h2d(item.layer_name)
        if item.stat != "h2d":
            raise RuntimeError(f"swap-attention tensor was not restored: {item.stat}")
        torch.npu.current_stream().wait_stream(native.prefetch_stream)
        return original_unpack(item)

    native.unpack_hook = unpack

    def after_layer(name):
        def hook(module, inputs, output):
            protected.update(_external_storages())
            if persistent_kv_storages is not None:
                protected.update(persistent_kv_storages())
            _protect_exports(native, protected)
            native.sync_d2h(name)
        return hook

    def before_layer_backward(name):
        def hook(module, *unused):
            native.h2d(name)
        return hook

    try:
        for name, module in targets:
            original = module.forward
            forwards.append((module, original))
            module.forward = native.hook_swap_manager_forward(original, name)
        for name, layer in layers:
            handles.append(layer.register_forward_hook(after_layer(name)))
            # Same hook kind as upstream, avoiding full-backward-hook view changes.
            handles.append(layer.register_backward_hook(before_layer_backward(name)))
        yield native
    finally:
        prefetch.get_args = original_get_args
        for handle in handles:
            handle.remove()
        for module, original in forwards:
            module.forward = original
        # Drain native transfers before dropping host/source buffer references,
        # including tensors whose branches were not traversed by autograd.
        native.prefetch_stream.synchronize()
        for name in ("swap_tensors", "prefetch_list", "prefetch_data_ptr_list", "slice_tensor_storage_ptr_list"):
            getattr(native, name).clear()
        native.data_ptr.clear()
        native.slice_tensor_storage_ptr.clear()
        native.unpack_hook = original_unpack


def with_activation_offload(method):
    """Scope native hooks to a full differentiable Visit or recomputing Pop."""
    @wraps(method)
    def run(executor, *args, **kwargs):
        executor._ensure_healthy()
        try:
            def persistent_kv_storages():
                # Read the live stack after Pop removed/released its entry.
                # Keep only storage identities, never extra owning references.
                # Anchors are not enumerated: native leaf/view filters apply;
                # any alias of persistent KV inherits its storage protection.
                return {
                    _storage_key(tensor)
                    for segment_id in executor.kv_stack.segment_ids
                    for pair in executor.kv_stack.get(segment_id).kv.key_values.values()
                    for tensor in pair
                }

            with mindspeed_swap_attention(
                executor.model, cp_size=executor.cp_size,
                cp_backend=executor.cp_backend, persistent_kv_storages=persistent_kv_storages,
            ):
                return method(executor, *args, **kwargs)
        except Exception:
            executor._failed = True
            raise
    return run
