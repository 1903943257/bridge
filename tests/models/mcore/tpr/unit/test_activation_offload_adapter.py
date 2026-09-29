"""Dependency-free lifecycle tests. Tensor/stream stubs do not prove NPU correctness."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[5] / "verl/models/mcore/tpr/activation_offload.py"


class Tensor:
    device = "npu:0"

    def __init__(self, pointer, *, numel=1, storage_size=None, grad_fn=None):
        self.pointer = pointer
        self._numel = numel
        self._storage_size = numel if storage_size is None else storage_size
        self.grad_fn = grad_fn

    def untyped_storage(self):
        return NS(data_ptr=lambda: self.pointer)

    def storage(self):
        return NS(size=lambda: self._storage_size)

    def numel(self):
        return self._numel

    def element_size(self):
        return 2

    def clone(self, memory_format=None):
        return Tensor(
            self.pointer + 100000,
            numel=self._numel,
            storage_size=self._numel,
            grad_fn=None,
        )


class Module:
    def __init__(self):
        self.forward = Mock()
        self.handles = []

    def register_forward_pre_hook(self, hook):
        self.forward_pre_hook = hook
        handle = NS(remove=Mock())
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, hook):
        self.forward_hook = hook
        handle = NS(remove=Mock())
        self.handles.append(handle)
        return handle

    def register_backward_hook(self, hook):
        self.backward_hook = hook
        handle = NS(remove=Mock())
        self.handles.append(handle)
        return handle


class Context:
    def __init__(self, on_enter=None):
        self.on_enter = on_enter

    def __enter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class AdapterTest(unittest.TestCase):
    def setUp(self):
        context = ModuleType("_offload_test.context")
        context.get_tpr_attention_context = lambda: None
        torch = ModuleType("torch")
        torch.Tensor = Tensor
        torch.contiguous_format = object()
        self.stream = NS(wait_stream=Mock())
        torch.npu = NS(current_stream=lambda: self.stream)
        self.saved_hooks = Mock(side_effect=lambda pack, unpack: Context())
        torch.autograd = NS(graph=NS(saved_tensors_hooks=self.saved_hooks))
        self.args = NS(swap_attention=False, pipeline_model_parallel_size=1)
        training = ModuleType("megatron.training")
        training.get_args = lambda: self.args
        native_module = ModuleType("mindspeed.core.memory.swap_attention.prefetch")
        native_module.get_args = training.get_args
        native_package = ModuleType("mindspeed.core.memory.swap_attention")
        native_package.prefetch = native_module
        self.native = NS(
            swap_tensors=[], prefetch_list=[], prefetch_data_ptr_list=[],
            slice_tensor_storage_ptr_list=[], data_ptr={}, slice_tensor_storage_ptr={},
            unpack_hook=Mock(side_effect=lambda item: item if isinstance(item, Tensor) else item.tensor),
            h2d=Mock(), sync_d2h=Mock(), prefetch_stream=NS(synchronize=Mock()),
            pack_hook=Mock(side_effect=lambda tensor: tensor), layer_name="",
            no_swap_tensor=Mock(
                side_effect=lambda tensor: tensor.grad_fn is None
                or tensor.storage().size() != tensor.numel()
            ),
            hook_swap_manager_forward=Mock(side_effect=lambda f, name: Mock(wraps=f)),
        )
        self.native_factory = Mock(return_value=self.native)
        native_module.SwapPrefetch = self.native_factory
        native_module.SwapPrefetch.swap_prefetch = None
        native_module.get_layer_id = lambda name: "0"
        self.imports = patch.dict(sys.modules, {
            "torch": torch, "_offload_test.context": context,
            "megatron.training": training,
            "mindspeed.core.memory.swap_attention.prefetch": native_module,
            "mindspeed.core.memory.swap_attention": native_package,
        })
        self.imports.start()
        self.addCleanup(self.imports.stop)
        spec = importlib.util.spec_from_file_location("_offload_test.activation_offload", SOURCE)
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.layer, self.attention = Module(), Module()
        self.layer.self_attention = self.attention
        self.layer.named_children = lambda: [("self_attention", self.attention)]
        self.model = NS(
            config=NS(swap_attention=True),
            modules=lambda: [self.layer, self.attention],
            named_modules=lambda: [("decoder.layers.0", self.layer)],
            parameters=lambda: [],
        )

    def test_off_does_not_import_native_or_install_hooks(self):
        self.model.config.swap_attention = False
        with self.adapter.mindspeed_swap_attention(self.model, cp_size=4) as native:
            self.assertIsNone(native)
        self.assertFalse(self.layer.handles)

    def test_global_flag_and_explicit_override(self):
        self.model.config = NS()
        self.args.swap_attention = True
        self.assertTrue(self.adapter.swap_enabled(self.model))
        self.model.config.swap_attention = False
        self.assertFalse(self.adapter.swap_enabled(self.model))

    def test_hybrid_gdn_rejected(self):
        self.attention.tpr_state_kind = "gdn"
        with self.assertRaisesRegex(NotImplementedError, "Full Attention only"):
            self.adapter.validate_activation_offload(self.model, 1)
        del self.attention.tpr_state_kind

    def test_cp_and_checkpoint_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "CP=1"):
            self.adapter.validate_activation_offload(self.model, 2)
        for name in ("recompute_granularity", "cpu_offloading", "fine_grained_activation_offloading"):
            setattr(self.model.config, name, True)
            with self.assertRaises(NotImplementedError):
                self.adapter.validate_activation_offload(self.model, 1)
            delattr(self.model.config, name)

    def test_ring_cp_whitelist_and_topology(self):
        for size in (2, 4):
            self.model.config.context_parallel_size = size
            self.args.context_parallel_size = size
            backend = NS(backend_name="ring", parallel_size=size)
            self.adapter.validate_activation_offload(self.model, size, backend)
            with self.adapter.mindspeed_swap_attention(self.model, cp_size=size, cp_backend=backend):
                pass
            for name in (None, "allgather", "ulysses", "hybrid"):
                with self.assertRaises(NotImplementedError):
                    self.adapter.validate_activation_offload(self.model, size, name)
            with self.assertRaises(ValueError):
                self.adapter.validate_activation_offload(self.model, size, NS(backend_name="ring", parallel_size=8))
            self.model.config.context_parallel_size = 1
            with self.assertRaises(ValueError):
                self.adapter.validate_activation_offload(self.model, size, backend)
        for size in (0, 3, 8):
            with self.assertRaises(NotImplementedError):
                self.adapter.validate_activation_offload(self.model, size, "ring")

    def test_persistent_cp_storage_is_protected_without_protecting_other_anchors(self):
        persistent = NS(tensor=Tensor(3), storage_data_ptr=3)
        independent_anchor = NS(tensor=Tensor(4), storage_data_ptr=4)
        self.native.swap_tensors = [persistent, independent_anchor]
        with self.adapter.mindspeed_swap_attention(
            self.model, persistent_kv_storages=lambda: {("npu:0", 3)}
        ):
            self.layer.forward_hook(self.layer, (), None)
            self.assertTrue(persistent.tpr_resident)
            self.assertFalse(getattr(independent_anchor, "tpr_resident", False))
            self.assertEqual(self.native.swap_tensors, [independent_anchor])
            self.native.sync_d2h.assert_called_once()

    def test_decorator_uses_actual_backend_and_live_stack_after_pop(self):
        self.model.config.context_parallel_size = 2
        self.args.context_parallel_size = 2
        stack = NS(segment_ids=(0,), get=Mock(side_effect=AssertionError("popped KV must not be read")))
        executor = NS(model=self.model, cp_size=2, cp_backend=NS(backend_name="ring", parallel_size=2),
                      kv_stack=stack, _ensure_healthy=Mock(), _failed=False)
        candidate = NS(tensor=Tensor(3), storage_data_ptr=3)

        @self.adapter.with_activation_offload
        def pop(executor):
            executor.kv_stack.segment_ids = ()
            self.native.swap_tensors = [candidate]
            self.layer.forward_hook(self.layer, (), None)
            self.assertEqual(self.native.swap_tensors, [candidate])
            return "done"

        self.assertEqual(pop(executor), "done")
        stack.get.assert_not_called()
        self.assertFalse(executor._failed)

    def test_launcher_cp_mismatch_rejected(self):
        self.model.config.context_parallel_size = 2
        self.args.context_parallel_size = 4
        with self.assertRaisesRegex(ValueError, "configured context_parallel_size"):
            self.adapter.validate_activation_offload(self.model, 2, "ring")

    def test_class_defined_forward_is_restored_without_instance_shadow(self):
        class ClassForwardModule(Module):
            def __init__(self):
                self.handles = []

            def forward(self, *args, **kwargs):
                return args, kwargs

        attention = ClassForwardModule()
        layer = Module()
        layer.self_attention = attention
        layer.named_children = lambda: [("self_attention", attention)]
        model = NS(
            config=NS(swap_attention=True),
            modules=lambda: [layer, attention],
            named_modules=lambda: [("decoder.layers.0", layer)],
            parameters=lambda: [],
        )
        self.assertNotIn("forward", attention.__dict__)
        with self.adapter.mindspeed_swap_attention(model):
            self.assertIn("forward", attention.__dict__)
        self.assertNotIn("forward", attention.__dict__)

    def test_hooks_restore_after_success_and_error(self):
        original = self.attention.forward
        for fail in (False, True):
            try:
                with self.adapter.mindspeed_swap_attention(self.model) as native:
                    self.assertIs(native, self.native)
                    self.assertIsNot(self.attention.forward, original)
                    if fail:
                        raise ValueError("forward failed")
            except ValueError:
                pass
            self.assertIs(self.attention.forward, original)
            for handle in self.layer.handles:
                handle.remove.assert_called_once()
            self.layer.handles.clear()
        self.assertEqual(self.native.prefetch_stream.synchronize.call_count, 2)

    def test_retained_session_suspends_payload_until_matching_backward(self):
        original = self.attention.forward
        session = self.adapter.create_retained_swap_session(self.model)
        payload = object()
        with session.capture() as native:
            self.assertIs(native, self.native)
            self.assertEqual(session.state, "capturing")
            self.assertIsNot(self.attention.forward, original)
            self.native.prefetch_list.append([payload])

        self.assertEqual(session.state, "suspended")
        self.assertIs(self.attention.forward, original)
        self.assertEqual(self.native.prefetch_list, [[payload]])
        self.saved_hooks.assert_called_once()
        self.assertTrue(callable(self.saved_hooks.call_args.args[0]))
        self.assertIs(self.saved_hooks.call_args.args[1], self.native.unpack_hook)
        retained_unpack = self.saved_hooks.call_args.args[1]
        with self.assertRaisesRegex(RuntimeError, "resume_for_backward"):
            retained_unpack(Tensor(9))
        for handle in self.layer.handles:
            handle.remove.assert_called_once()

        session.finish_forward()  # Explicit use is allowed and idempotent.
        self.assertIs(session.resume_for_backward(), self.native)
        resident = Tensor(9)
        self.assertIs(retained_unpack(resident), resident)
        self.assertEqual(session.state, "backward")
        session.close()
        session.close()
        self.assertTrue(session.closed)
        self.assertEqual(self.native.prefetch_list, [])

    def test_retained_capture_flushes_decoder_and_loss_scope(self):
        candidate = NS(tensor=Tensor(8), storage_data_ptr=8)
        original_pack = self.native.pack_hook
        session = self.adapter.create_retained_swap_session(self.model)
        with session.capture():
            outer_pack = self.saved_hooks.call_args.args[0]
            before_decoder = Tensor(6)
            self.assertIs(outer_pack(before_decoder), before_decoder)
            original_pack.assert_not_called()

            self.layer.forward_pre_hook(self.layer, ())
            decoder_tensor = Tensor(7)
            self.assertIs(outer_pack(decoder_tensor), decoder_tensor)
            original_pack.assert_called_once_with(decoder_tensor)

            # Retained compacting preserves native's 1 MiB minimum-swap threshold.
            view = Tensor(
                10,
                numel=1024 * 1024,
                storage_size=2 * 1024 * 1024,
                grad_fn=object(),
            )
            forced_checks = []
            original_side_effect = original_pack.side_effect

            def observe_forced_compact(tensor):
                if tensor is not view and tensor.storage().size() == tensor.numel():
                    forced_checks.append(self.native.no_swap_tensor(tensor))
                return original_side_effect(tensor)

            original_pack.side_effect = observe_forced_compact
            packed_view = self.native.pack_hook(view)
            compact = original_pack.call_args.args[0]
            self.assertIsNot(compact, view)
            self.assertEqual(compact.storage().size(), compact.numel())
            self.assertIsNone(compact.grad_fn)
            self.assertEqual(forced_checks[-1], False)
            self.assertIs(packed_view, compact)
            original_pack.side_effect = original_side_effect

            self.layer.forward_hook(self.layer, (), None)
            loss_tensor = Tensor(9)
            self.assertIs(outer_pack(loss_tensor), loss_tensor)
            self.assertEqual(original_pack.call_args.args[0], loss_tensor)
            self.assertEqual(self.native.layer_name, "tpr.outer.1")
            # Represents a tensor saved by CE after the final layer hook.
            self.native.swap_tensors = [candidate]
        self.assertEqual(
            [call.args[0] for call in self.native.sync_d2h.call_args_list],
            ["decoder.layers.0", "tpr.outer.1"],
        )
        self.assertIs(self.native.pack_hook, original_pack)
        self.assertEqual(self.native.swap_tensors, [candidate])
        session.abort()

    def test_retained_capture_is_exclusive_but_suspended_payload_is_not_installed(self):
        first = self.adapter.create_retained_swap_session(self.model)
        second = self.adapter.create_retained_swap_session(self.model)
        with first.capture():
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                with second.capture():
                    pass
        self.assertEqual(first.state, "suspended")
        self.assertTrue(second.closed)
        first.close()

    def test_retained_capture_error_aborts_payload_and_restores_model(self):
        original = self.attention.forward
        session = self.adapter.create_retained_swap_session(self.model)
        with self.assertRaisesRegex(ValueError, "failed Push"):
            with session.capture():
                self.native.prefetch_list.append([object()])
                raise ValueError("failed Push")
        self.assertTrue(session.closed)
        self.assertEqual(self.native.prefetch_list, [])
        self.assertIs(self.attention.forward, original)

    def test_retained_session_requires_enabled_swap_and_valid_order(self):
        self.model.config.swap_attention = False
        with self.assertRaisesRegex(RuntimeError, "requires native swap-attention"):
            self.adapter.create_retained_swap_session(self.model)
        self.model.config.swap_attention = True
        session = self.adapter.create_retained_swap_session(self.model)
        with self.assertRaisesRegex(RuntimeError, "Cannot resume"):
            session.resume_for_backward()
        with session.capture():
            pass
        with self.assertRaisesRegex(RuntimeError, "cannot capture"):
            with session.capture():
                pass
        session.close()

    def test_recompute_pop_can_enable_aggressive_saved_view_offload(self):
        executor = NS(
            prefix_backward_policy="recompute",
            model=self.model,
            cp_size=1,
            cp_backend=None,
            kv_stack=NS(segment_ids=()),
            _ensure_healthy=Mock(),
            _mark_failed=Mock(),
        )
        calls = []

        def manager(*args, **kwargs):
            calls.append(kwargs)
            return Context()

        @self.adapter.with_activation_offload
        def pop(executor):
            return "recomputed"

        with patch.object(self.adapter, "mindspeed_swap_attention", side_effect=manager):
            with patch.dict(self.adapter.os.environ, {"TPR_RECOMPUTE_AGGRESSIVE_SAVED_VIEWS": "1"}):
                self.assertEqual(pop(executor), "recomputed")
            self.assertTrue(calls[-1]["compact_saved_views"])
            self.assertTrue(calls[-1]["capture_decoder_saved_tensors"])
            self.assertTrue(calls[-1]["capture_outer_saved_tensors"])

            calls.clear()
            with patch.dict(self.adapter.os.environ, {}, clear=False):
                self.adapter.os.environ.pop("TPR_RECOMPUTE_AGGRESSIVE_SAVED_VIEWS", None)
                self.assertEqual(pop(executor), "recomputed")
            self.assertFalse(calls[-1]["compact_saved_views"])
            self.assertFalse(calls[-1]["capture_decoder_saved_tensors"])
            self.assertFalse(calls[-1]["capture_outer_saved_tensors"])

    def test_visit_does_not_enable_recompute_aggressive_saved_views(self):
        executor = NS(
            prefix_backward_policy="recompute",
            model=self.model,
            cp_size=1,
            cp_backend=None,
            kv_stack=NS(segment_ids=()),
            _ensure_healthy=Mock(),
            _mark_failed=Mock(),
        )
        calls = []

        def manager(*args, **kwargs):
            calls.append(kwargs)
            return Context()

        @self.adapter.with_activation_offload
        def visit_leaf(executor):
            return "visited"

        with patch.object(self.adapter, "mindspeed_swap_attention", side_effect=manager):
            with patch.dict(self.adapter.os.environ, {"TPR_RECOMPUTE_AGGRESSIVE_SAVED_VIEWS": "1"}):
                self.assertEqual(visit_leaf(executor), "visited")
        self.assertFalse(calls[-1]["compact_saved_views"])
        self.assertFalse(calls[-1]["capture_decoder_saved_tensors"])
        self.assertFalse(calls[-1]["capture_outer_saved_tensors"])

    def test_offload_policy_pop_uses_retained_session_not_short_manager(self):
        executor = NS(
            prefix_backward_policy="offload",
            _ensure_healthy=Mock(),
            _mark_failed=Mock(),
        )

        @self.adapter.with_activation_offload
        def pop(executor):
            return "retained"

        self.assertEqual(pop(executor), "retained")
        self.native_factory.assert_not_called()
        executor._mark_failed.assert_not_called()

    def test_external_storage_extension_seam_is_fa_only_by_default(self):
        context = NS(
            past_key_values={1: (Tensor(1), Tensor(2))},
            new_key_values={1: (Tensor(3), Tensor(4))},
        )
        with patch.object(self.adapter, "get_tpr_attention_context", return_value=context):
            self.assertEqual(
                self.adapter._external_storages(),
                {("npu:0", 1), ("npu:0", 2), ("npu:0", 3), ("npu:0", 4)},
            )
            self.assertEqual(
                self.adapter._external_storages(include_new_key_values=False),
                {("npu:0", 1), ("npu:0", 2)},
            )
            with patch.object(
                self.adapter, "_iter_extra_external_tensors", return_value=(Tensor(5),)
            ):
                self.assertIn(
                    ("npu:0", 5),
                    self.adapter._external_storages(include_new_key_values=False),
                )

    def test_visit_can_swap_new_kv_but_pop_keeps_direct_roots_resident(self):
        context = NS(
            past_key_values={},
            new_key_values={1: (Tensor(3), Tensor(4))},
        )
        items = [
            NS(tensor=Tensor(3), storage_data_ptr=3, first_tensor=False, last_tensor=False),
            NS(tensor=Tensor(4), storage_data_ptr=4, first_tensor=False, last_tensor=False),
        ]
        with patch.object(self.adapter, "get_tpr_attention_context", return_value=context):
            self.native.swap_tensors = list(items)
            with self.adapter.mindspeed_swap_attention(
                self.model, protect_new_key_values=False
            ):
                self.layer.forward_hook(self.layer, (), None)
                self.assertEqual(self.native.swap_tensors, items)
                self.assertFalse(any(getattr(item, "tpr_resident", False) for item in items))

            items = [
                NS(tensor=Tensor(3), storage_data_ptr=3, first_tensor=False, last_tensor=False),
                NS(tensor=Tensor(4), storage_data_ptr=4, first_tensor=False, last_tensor=False),
            ]
            self.native.swap_tensors = list(items)
            with self.adapter.mindspeed_swap_attention(
                self.model, protect_new_key_values=True
            ):
                self.layer.forward_hook(self.layer, (), None)
                self.assertEqual(self.native.swap_tensors, [])
                self.assertTrue(all(item.tpr_resident for item in items))

    def test_exported_storage_and_aliases_stay_resident(self):
        def item(ptr):
            return NS(tensor=Tensor(ptr), storage_data_ptr=ptr, first_tensor=False, last_tensor=False)
        first, alias, ordinary = item(3), item(3), item(4)
        self.native.swap_tensors = [first, alias, ordinary]
        self.adapter._protect_exports(self.native, {("npu:0", 3)})
        self.assertTrue(first.tpr_resident and alias.tpr_resident)
        self.assertEqual(self.native.swap_tensors, [ordinary])
        self.assertEqual(self.native.data_ptr, {4: 0})
        self.assertTrue(ordinary.first_tensor)
        self.native.prefetch_stream.synchronize.assert_called_once()

    def test_pending_outer_d2h_flushes_before_recompute_unpack(self):
        tensor = Tensor(42)
        item = NS(
            tensor=tensor,
            storage_data_ptr=42,
            stat="d2h",
            layer_name="tpr.outer.1",
            first_tensor=False,
            last_tensor=False,
        )
        self.native.swap_tensors = [item]
        order = []

        def finish_d2h(name):
            order.append(("d2h", name))
            item.stat = "host"

        def launch_h2d(name):
            order.append(("h2d", name))
            item.stat = "h2d"

        self.native.sync_d2h.side_effect = finish_d2h
        self.native.h2d.side_effect = launch_h2d

        with self.adapter.mindspeed_swap_attention(
            self.model,
            capture_outer_saved_tensors=True,
        ):
            self.assertIs(self.native.unpack_hook(item), tensor)

        self.assertEqual(
            order[:2],
            [("d2h", item.layer_name), ("h2d", item.layer_name)],
        )

    def test_direct_kv_root_reloads_with_native_h2d(self):
        tensor = Tensor(42)
        item = NS(tensor=tensor, stat="host", layer_name="decoder.layers.0.self_attention", h2d_event=object())
        self.native.h2d.side_effect = lambda name: setattr(item, "stat", "h2d")
        with self.adapter.mindspeed_swap_attention(self.model):
            self.assertIs(self.native.unpack_hook(item), tensor)
        self.native.h2d.assert_called_once_with(item.layer_name)
        self.stream.wait_stream.assert_called_once_with(self.native.prefetch_stream)

    def test_force_restore_sync_waits_for_prefetch_completion(self):
        tensor = Tensor(42)
        item = NS(tensor=tensor, stat="host", layer_name="decoder.layers.0.self_attention")
        self.native.h2d.side_effect = lambda name: setattr(item, "stat", "h2d")
        with patch.object(self.adapter.os, "getenv", return_value="1"):
            with self.adapter.mindspeed_swap_attention(self.model):
                self.assertIs(self.native.unpack_hook(item), tensor)
        self.native.prefetch_stream.synchronize.assert_called()
        self.stream.wait_stream.assert_not_called()

    def test_exported_handle_does_not_reload(self):
        tensor = Tensor(42)
        with self.adapter.mindspeed_swap_attention(self.model):
            self.assertIs(self.native.unpack_hook(NS(tensor=tensor, tpr_resident=True)), tensor)
        self.native.h2d.assert_not_called()

    def test_duplicate_handle_without_own_event_reloads_owner(self):
        item = NS(tensor=Tensor(42), stat="h2d", layer_name="decoder.layers.0.self_attention")
        with self.adapter.mindspeed_swap_attention(self.model):
            self.assertIs(self.native.unpack_hook(item), item.tensor)
        self.native.h2d.assert_called_once_with(item.layer_name)
        self.stream.wait_stream.assert_called_once_with(self.native.prefetch_stream)

    def test_mindspeed_args_without_training_launcher(self):
        training = sys.modules["megatron.training"]
        training.get_args = Mock(side_effect=AssertionError("args is not initialized."))
        mind_args = ModuleType("mindspeed.args_utils")
        mind_args.get_full_args = lambda: self.args
        self.args.swap_attention = True
        self.model.config = NS()
        with patch.dict(sys.modules, {"mindspeed.args_utils": mind_args}):
            self.assertTrue(self.adapter.swap_enabled(self.model))
            with self.adapter.mindspeed_swap_attention(self.model):
                pass

    def test_core_only_imports_real_native_prefetch_without_leaking_training(self):
        native_dir = SOURCE.parents[5] / "MindSpeed/mindspeed/core/memory/swap_attention"
        if not (native_dir / "prefetch.py").is_file():
            self.skipTest("requires the adjacent MindSpeed checkout for the native import regression")
        megatron = ModuleType("megatron")
        megatron.__path__ = []  # Core-only installation: no training package.
        mind_args = ModuleType("mindspeed.args_utils")
        mind_args.get_full_args = lambda: self.args
        torch_npu = ModuleType("torch_npu")
        torch_npu.npu = NS(Stream=lambda **kwargs: self.native.prefetch_stream)
        torch = sys.modules["torch"]
        torch.npu.current_device = lambda: 0
        package = sys.modules["mindspeed.core.memory.swap_attention"]
        package.__path__ = [str(native_dir)]
        with patch.dict(sys.modules, {"megatron": megatron, "mindspeed.args_utils": mind_args,
                                      "torch_npu": torch_npu}):
            del sys.modules["megatron.training"]
            del sys.modules["mindspeed.core.memory.swap_attention.prefetch"]
            prefetch = self.adapter._native_prefetch()
            self.assertEqual(Path(prefetch.__file__).resolve(), (native_dir / "prefetch.py").resolve())
            self.assertIs(prefetch.get_args(), self.args)
            self.assertNotIn("megatron.training", sys.modules)
            self.assertFalse(hasattr(megatron, "training"))
            original_get_args = prefetch.get_args
            with self.adapter.mindspeed_swap_attention(self.model) as native:
                self.assertIsInstance(native, prefetch.SwapPrefetch)
            retained = self.adapter.create_retained_swap_session(self.model)
            with retained.capture() as native:
                self.assertIsInstance(native, prefetch.SwapPrefetch)
            retained.resume_for_backward()
            retained.close()
            self.assertIs(prefetch.get_args, original_get_args)
            self.assertNotIn("megatron.training", sys.modules)

    def test_native_import_does_not_hide_other_missing_dependencies(self):
        error = ModuleNotFoundError("missing torch_npu", name="torch_npu")
        with patch.object(self.adapter, "import_module", side_effect=error):
            with self.assertRaises(ModuleNotFoundError) as caught:
                self.adapter._native_prefetch()
        self.assertIs(caught.exception, error)

    def test_temporary_training_removed_when_native_retry_fails(self):
        mind_args = ModuleType("mindspeed.args_utils")
        mind_args.get_full_args = lambda: self.args
        missing_training = ModuleNotFoundError("missing training", name="megatron.training")
        missing_other = ModuleNotFoundError("missing torch_npu", name="torch_npu")
        with patch.dict(sys.modules, {"mindspeed.args_utils": mind_args}):
            del sys.modules["megatron.training"]
            with patch.object(self.adapter, "import_module", side_effect=[missing_training, missing_other]):
                with self.assertRaises(ModuleNotFoundError) as caught:
                    self.adapter._native_prefetch()
            self.assertIs(caught.exception, missing_other)
            self.assertNotIn("megatron.training", sys.modules)

    def test_missing_targets_and_double_install_rejected(self):
        self.model.config.swap_modules = "missing"
        with self.assertRaisesRegex(RuntimeError, "matched no"):
            with self.adapter.mindspeed_swap_attention(self.model):
                pass
        self.attention.no_checkpoint_adaptive_recompute_forward = object()
        with self.assertRaisesRegex(RuntimeError, "already has"):
            with self.adapter.mindspeed_swap_attention(self.model):
                pass


if __name__ == "__main__":
    unittest.main()
