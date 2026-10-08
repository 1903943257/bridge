"""No Megatron installation needed: regression tests for Phase-4 Engine patching."""

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
SCRIPT = ROOT / "patches" / "apply_tpr_phase4.py"
_spec = importlib.util.spec_from_file_location("apply_tpr_phase4", SCRIPT)
assert _spec is not None and _spec.loader is not None
_patcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_patcher)


def _engine(*, with_routed: bool = True, with_bshd: bool = True,
            with_ce_request: bool = True) -> str:
    request = (
        "        if False:\n"
        "            return run_tpr_forward_backward(self, request, forward_only=False)\n"
    ) if with_ce_request else ""
    routed = (
        "        if self.tf_config.calculate_per_token_loss:\n"
        "            routed_num_tokens = self._routed_num_tokens(data)\n"
        "            tu.assign_non_tensor(data, routed_num_tokens=routed_num_tokens.item())\n"
    ) if with_routed else ""
    prelude = (
        "        # BSHD path only: optional prepadding\n"
        "        pad_bshd_to_minibatch_max = False\n"
    ) if with_bshd else (
        "        vpp_size = 1\n"
    )
    return (
        "class MegatronEngine:\n"
        "    def forward_backward_batch(self, data, loss_function, forward_only=False):\n"
        f"{request}"
        "        loss_mask = data['loss_mask']\n"
        "        tu.assign_non_tensor(data, batch_num_tokens=loss_mask.sum().item())\n"
        "        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())\n"
        f"{routed}"
        f"{prelude}"
        "        micro_batches, indices = prepare_micro_batches(data=data)\n"
        "        return micro_batches\n"
    )


class Phase4PatcherTests(unittest.TestCase):
    def test_native_with_routed_metadata(self):
        original = _engine()
        result, status = _patcher.transform(original)
        self.assertEqual(status, "patched")
        self.assertEqual(result.count("run_tpr_forward_backward_batch"), 2)
        self.assertLess(
            result.index("routed_num_tokens=routed_num_tokens.item()"),
            result.index("from verl.models.mcore.tpr.megatron_adapter import run_tpr_forward_backward_batch"),
        )
        self.assertLess(result.index("return run_tpr_forward_backward_batch("),
                        result.index("prepare_micro_batches(data=data)"))
        self.assertIn("run_tpr_forward_backward(self, request", result)
        ast.parse(result)

    def test_fallback_without_bshd_or_routed_branch(self):
        updated, status = _patcher.transform(_engine(with_routed=False, with_bshd=False))
        self.assertEqual(status, "patched")
        self.assertLess(updated.index("return run_tpr_forward_backward_batch("),
                        updated.index("vpp_size = 1"))
        ast.parse(updated)

    def test_repeat_is_idempotent(self):
        once, _ = _patcher.transform(_engine())
        twice, status = _patcher.transform(once)
        self.assertEqual(status, "already-patched")
        self.assertEqual(once, twice)

    def test_does_not_guess_missing_normalization(self):
        bad = _engine().replace(
            "tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())",
            "pass",
        )
        with self.assertRaisesRegex(_patcher.PatchError, "batch_num_tokens"):
            _patcher.transform(bad)

    def test_does_not_insert_after_missing_split(self):
        bad = _engine().replace(
            "micro_batches, indices = prepare_micro_batches(data=data)",
            "micro_batches = []",
        )
        with self.assertRaisesRegex(_patcher.PatchError, "prepare_micro_batches"):
            _patcher.transform(bad)


if __name__ == "__main__":
    unittest.main()
