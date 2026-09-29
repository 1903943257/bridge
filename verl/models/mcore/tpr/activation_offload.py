"""Full-Attention adapter for MindSpeed swap-attention.

One Visit/Pop is a complete forward/backward microbatch. Native SwapPrefetch
owns selection, host allocations, streams, transfer and release. Hooks are
scoped to that microbatch so graph-free Push never enters native swap queues.

Phase C additionally retains one native payload per Prefix Push. Its module
wrappers are installed only while the Push graph is captured; the native
queues and pinned host tensors survive until the matching Pop backward.
"""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import os
import sys
from types import ModuleType, SimpleNamespace

import torch

from .context import get_tpr_attention_context


_active_swap_installation = None


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


def _external_storages(*, include_new_key_values=True):
    """Return external FA storages that native swap must not release.

    Reusable/past KV is always external state. Current/new KV is external only
    when the caller still needs it as a direct autograd root (Pop). VisitLeaf
    consumes current KV entirely inside its forward/backward graph, so native
    swap may offload/reload it like any other transient activation.
    """
    context = get_tpr_attention_context()
    if context is None:
        return set()
    tensors = [
        tensor
        for pair in context.past_key_values.values()
        for tensor in pair
    ]
    if include_new_key_values:
        tensors.extend(
            tensor
            for pair in context.new_key_values.values()
            for tensor in pair
        )
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
def _claim_swap_installation(owner):
    """Serialize module/global hook installation, not suspended payloads."""
    global _active_swap_installation
    if _active_swap_installation is not None:
        raise RuntimeError("Another TPR native swap-attention capture is already installed")
    _active_swap_installation = owner
    try:
        yield
    finally:
        if _active_swap_installation is owner:
            _active_swap_installation = None


def _clear_native_payload(native, original_unpack):
    """Release native queue and pinned-buffer ownership after backward/abort."""
    native.prefetch_stream.synchronize()
    for name in (
        "swap_tensors",
        "prefetch_list",
        "prefetch_data_ptr_list",
        "slice_tensor_storage_ptr_list",
    ):
        getattr(native, name).clear()
    native.data_ptr.clear()
    native.slice_tensor_storage_ptr.clear()
    native.unpack_hook = original_unpack
    if hasattr(native, "_tpr_original_unpack"):
        del native._tpr_original_unpack


@contextmanager
def _installed_mindspeed_swap_attention(
    model,
    *,
    persistent_kv_storages=None,
    protect_new_key_values=True,
    defer_payload_cleanup=False,
    capture_outer_saved_tensors=False,
    capture_decoder_saved_tensors=False,
    compact_saved_views=False,
    unpack_allowed=None,
):
    """Install native wrappers/hooks; validation and serialization are external.

    VERL bypasses megatron.training.setup_model_and_optimizer, where upstream
    normally installs these hooks. This adapter deliberately keeps native
    swap_modules and the native tensor-size/view/leaf filters.
    """
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
    original_pack = native.pack_hook
    original_no_swap_tensor = native.no_swap_tensor
    original_unpack = native.unpack_hook
    native._tpr_original_unpack = original_unpack
    handles, forwards = [], []
    force_swap_storages = set()

    def no_swap_tensor(tensor):
        if _storage_key(tensor) in force_swap_storages:
            return False
        return original_no_swap_tensor(tensor)

    if compact_saved_views:
        native.no_swap_tensor = no_swap_tensor

    def pack(tensor):
        if not compact_saved_views:
            return original_pack(tensor)
        try:
            # Native SwapPrefetch intentionally skips slice/view tensors because
            # resize_(0) on shared backing storage would invalidate sibling
            # aliases. Retained Prefix graphs keep those skipped views alive
            # across all sibling Visits. Give saved-tensor offload an independent
            # compact copy instead, so native can release only that copy while
            # preserving ordinary SwapPrefetch scheduling and restore semantics.
            storage_size = tensor.storage().size()
            if (
                tensor.grad_fn is not None
                and storage_size
                and storage_size != tensor.numel()
                and tensor.numel() * tensor.element_size() * 2 >= 1024 * 1024
            ):
                tensor = tensor.clone(memory_format=torch.contiguous_format)
                storage_key = _storage_key(tensor)
                force_swap_storages.add(storage_key)
                try:
                    return original_pack(tensor)
                finally:
                    force_swap_storages.discard(storage_key)
        except (AttributeError, RuntimeError, TypeError):
            # Keep native filtering authoritative for unusual tensor wrappers.
            pass
        return original_pack(tensor)

    if compact_saved_views:
        native.pack_hook = pack

    def unpack(item):
        if unpack_allowed is not None and not unpack_allowed():
            raise RuntimeError("Retained swap tensor unpack requires resume_for_backward()")
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
        # Diagnostic only: CP4 padded Ring occasionally shows sparse backward
        # mismatches after restore. Force host-side completion to distinguish a
        # cross-stream/HCCL visibility issue from ordinary BF16 Ring noise.
        # Default behavior stays identical to native-style async prefetch.
        if os.getenv("TPR_OFFLOAD_B_FORCE_RESTORE_SYNC", "0") == "1":
            native.prefetch_stream.synchronize()
        else:
            torch.npu.current_stream().wait_stream(native.prefetch_stream)
        return original_unpack(item)

    native.unpack_hook = unpack

    def sync_d2h(name):
        protected.update(
            _external_storages(include_new_key_values=protect_new_key_values)
        )
        if persistent_kv_storages is not None:
            protected.update(persistent_kv_storages())
        _protect_exports(native, protected)
        native.sync_d2h(name)

    decoder_capture_enabled = False
    outer_capture_enabled = False
    last_layer_name = layers[-1][0]
    outer_layer_id = max(int(layer_id) for layer_id in layer_ids) + 1
    outer_layer_name = f"tpr.outer.{outer_layer_id}"

    def before_layer(name):
        def hook(module, inputs):
            nonlocal decoder_capture_enabled
            decoder_capture_enabled = True
            native.layer_name = name
        return hook

    def after_layer(name):
        def hook(module, inputs, output):
            nonlocal decoder_capture_enabled, outer_capture_enabled
            sync_d2h(name)
            decoder_capture_enabled = False
            if capture_outer_saved_tensors and name == last_layer_name:
                # Continue the same native queue through final norm/LM-head/CE,
                # but use a synthetic layer id after the decoder.
                outer_capture_enabled = True
        return hook

    def before_layer_backward(name):
        def hook(module, *unused):
            native.h2d(name)
        return hook

    def outer_pack(tensor):
        if capture_decoder_saved_tensors and decoder_capture_enabled:
            # Child swap_modules install nested saved-tensor hooks, so this
            # catches only layer-level work outside those children (norms,
            # residual paths, etc.) without double-packing child activations.
            return native.pack_hook(tensor)
        if not outer_capture_enabled:
            return tensor
        # Native SwapPrefetch groups work by monotonically increasing layer id.
        # Treat final-norm/LM-head/CE saved tensors as one synthetic layer after
        # the decoder instead of reusing the last Transformer layer id, which
        # would otherwise start a second native microbatch queue.
        native.layer_name = outer_layer_name
        return native.pack_hook(tensor)

    yielded = False
    try:
        for name, module in targets:
            # Preserve the exact attribute state, not just the currently bound
            # callable. Most nn.Module implementations inherit forward from the
            # class, so assigning the saved bound method on exit would leave a
            # new instance-level "forward" attribute behind.
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")
            original = module.forward
            forwards.append((module, had_instance_forward, instance_forward))
            module.forward = native.hook_swap_manager_forward(original, name)
        for name, layer in layers:
            if capture_decoder_saved_tensors:
                handles.append(layer.register_forward_pre_hook(before_layer(name)))
            handles.append(layer.register_forward_hook(after_layer(name)))
            # Same hook kind as upstream, avoiding full-backward-hook view changes.
            handles.append(layer.register_backward_hook(before_layer_backward(name)))
        if capture_outer_saved_tensors:
            with torch.autograd.graph.saved_tensors_hooks(outer_pack, native.unpack_hook):
                yielded = True
                yield native
            # CE/loss nodes may save tensors after the final Transformer-layer
            # forward hook. Flush them as a synthetic post-decoder layer so the
            # retained session stays in the same native microbatch queue.
            if native.swap_tensors:
                sync_d2h(outer_layer_name)
        else:
            yielded = True
            yield native
    finally:
        prefetch.get_args = original_get_args
        for handle in handles:
            handle.remove()
        for module, had_instance_forward, instance_forward in forwards:
            if had_instance_forward:
                module.forward = instance_forward
            else:
                module.__dict__.pop("forward", None)
        native.pack_hook = original_pack
        native.no_swap_tensor = original_no_swap_tensor
        # A retained Prefix session suspends here: module/global hooks are gone,
        # but saved autograd handles still own the native queues and pinned CPU
        # payload. Ordinary Visit/Pop scopes release that ownership immediately.
        if defer_payload_cleanup and yielded:
            native.prefetch_stream.synchronize()
        else:
            _clear_native_payload(native, original_unpack)


@contextmanager
def mindspeed_swap_attention(
    model,
    *,
    cp_size=1,
    cp_backend=None,
    persistent_kv_storages=None,
    protect_new_key_values=True,
    compact_saved_views=False,
):
    """Temporarily install native swap for one complete forward/backward scope."""
    if not swap_enabled(model):
        yield None
        return
    validate_activation_offload(model, cp_size, cp_backend)
    owner = object()
    with _claim_swap_installation(owner):
        with _installed_mindspeed_swap_attention(
            model,
            persistent_kv_storages=persistent_kv_storages,
            protect_new_key_values=protect_new_key_values,
            compact_saved_views=compact_saved_views,
        ) as native:
            yield native


class RetainedSwapSession:
    """One Push-owned native payload retained until its matching Pop.

    ``capture`` is the only interval that mutates model forwards or registers
    module hooks. Leaving it suspends the session while keeping native queues
    and pinned buffers alive. ``resume_for_backward`` is an explicit lifecycle
    check; actual restores stay native/lazy through the unpack callbacks saved
    in the original autograd graph.
    """

    def __init__(
        self,
        model,
        *,
        cp_size=1,
        cp_backend=None,
        persistent_kv_storages=None,
        protect_new_key_values=True,
    ):
        if not swap_enabled(model):
            raise RuntimeError("Retained Prefix graph offload requires native swap-attention")
        validate_activation_offload(model, cp_size, cp_backend)
        self.model = model
        self.cp_size = cp_size
        self.cp_backend = cp_backend
        self.persistent_kv_storages = persistent_kv_storages
        self.protect_new_key_values = protect_new_key_values
        self.native = None
        self._original_unpack = None
        self._state = "new"

    @property
    def state(self):
        return self._state

    @property
    def closed(self):
        return self._state == "closed"

    @contextmanager
    def capture(self):
        """Capture one Push forward and suspend its native payload on exit."""
        if self._state != "new":
            raise RuntimeError(f"Retained swap session cannot capture from state {self._state!r}")
        owner = object()
        self._state = "capturing"
        try:
            with _claim_swap_installation(owner):
                with _installed_mindspeed_swap_attention(
                    self.model,
                    persistent_kv_storages=self.persistent_kv_storages,
                    protect_new_key_values=self.protect_new_key_values,
                    defer_payload_cleanup=True,
                    capture_outer_saved_tensors=True,
                    capture_decoder_saved_tensors=True,
                    compact_saved_views=True,
                    unpack_allowed=lambda: self._state == "backward",
                ) as native:
                    self.native = native
                    self._original_unpack = native._tpr_original_unpack
                    yield native
            self._state = "suspended"
        except BaseException:
            # The installed scope has already removed model wrappers/hooks here.
            if self.native is not None:
                self._state = "suspended"
                self.close()
            else:
                self._state = "closed"
            raise

    def finish_forward(self):
        """Wait for capture-side D2H; idempotent after ``capture`` exits."""
        if self._state not in ("capturing", "suspended"):
            raise RuntimeError(f"Cannot finish retained forward from state {self._state!r}")
        if self.native is not None:
            self.native.prefetch_stream.synchronize()

    def resume_for_backward(self):
        """Mark the matching Pop backward; native unpack restores lazily."""
        if self._state != "suspended":
            raise RuntimeError(f"Cannot resume retained backward from state {self._state!r}")
        self._state = "backward"
        return self.native

    def close(self):
        """Release native queues/pinned buffers after backward or capture abort."""
        if self._state == "closed":
            return
        if self._state == "capturing":
            raise RuntimeError("Cannot close a retained swap session while capture hooks are installed")
        if self.native is not None:
            _clear_native_payload(self.native, self._original_unpack)
        self._state = "closed"

    abort = close


def create_retained_swap_session(
    model,
    *,
    cp_size=1,
    cp_backend=None,
    persistent_kv_storages=None,
    protect_new_key_values=True,
):
    """Create an unstarted Push-to-Pop native SwapPrefetch session."""
    return RetainedSwapSession(
        model,
        cp_size=cp_size,
        cp_backend=cp_backend,
        persistent_kv_storages=persistent_kv_storages,
        protect_new_key_values=protect_new_key_values,
    )


def with_activation_offload(method):
    """Scope native hooks to a full differentiable Visit or recomputing Pop."""
    @wraps(method)
    def run(executor, *args, **kwargs):
        executor._ensure_healthy()
        try:
            # Phase C Pop consumes the Push-owned retained session. Installing a
            # second short-lived manager would neither capture a forward nor own
            # the saved graph handles, and can corrupt the retained queue.
            if (
                method.__name__ == "pop"
                and getattr(executor, "prefix_backward_policy", "recompute") == "offload"
            ):
                return method(executor, *args, **kwargs)

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

            # VisitLeaf never exports its current/new KV beyond this
            # forward/backward graph, so native swap may release/reload it.
            # Pop still needs recomputed new KV as explicit dKV roots after
            # forward, therefore those storages must remain device-resident.
            protect_new_key_values = method.__name__ == "pop"
            compact_saved_views = (
                method.__name__ == "pop"
                and getattr(executor, "prefix_backward_policy", "recompute") == "recompute"
                and os.getenv("TPR_RECOMPUTE_AGGRESSIVE_SAVED_VIEWS", "0") == "1"
            )
            with mindspeed_swap_attention(
                executor.model,
                cp_size=executor.cp_size,
                cp_backend=executor.cp_backend,
                persistent_kv_storages=persistent_kv_storages,
                protect_new_key_values=protect_new_key_values,
                compact_saved_views=compact_saved_views,
            ):
                return method(executor, *args, **kwargs)
        except Exception:
            mark_failed = getattr(executor, "_mark_failed", None)
            if callable(mark_failed):
                mark_failed()
            else:
                executor._failed = True
            raise
    return run
