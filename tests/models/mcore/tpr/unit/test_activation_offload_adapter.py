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

    def __init__(self, pointer):
        self.pointer = pointer

    def untyped_storage(self):
        return NS(data_ptr=lambda: self.pointer)


class Module:
    def __init__(self):
        self.forward = Mock()
        self.handles = []

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


class AdapterTest(unittest.TestCase):
    def setUp(self):
        context = ModuleType("_offload_test.context")
        context.get_tpr_attention_context = lambda: None
        torch = ModuleType("torch")
        torch.Tensor = Tensor
        self.stream = NS(wait_stream=Mock())
        torch.npu = NS(current_stream=lambda: self.stream)
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
            hook_swap_manager_forward=Mock(side_effect=lambda f, name: Mock(wraps=f)),
        )
        native_module.SwapPrefetch = Mock(return_value=self.native)
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
