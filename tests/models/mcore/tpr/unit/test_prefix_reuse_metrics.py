"""Reporting tests runnable with unittest alone (no PyTorch/NPU imports)."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parents[1] / "parallel" / "_prefix_reuse_metrics.py"
spec = importlib.util.spec_from_file_location("prefix_reuse_metrics", path)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class PrefixReuseMetricsTest(unittest.TestCase):
    def test_default_matrix(self):
        cases = metrics.benchmark_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({n for n, _, _ in cases}, {2, 4, 8, 16})
        self.assertEqual({(p, s) for _, p, s in cases}, {(16384, 1024), (16384, 2048), (8192, 8192)})
        self.assertEqual({p / s for _, p, s in cases}, {16, 8, 1})
        self.assertEqual(max(p + s for _, p, s in cases), 18432)

    def test_reject_invalid_matrix(self):
        for n, lengths in (("1", "64:64"), ("2,2", "64:64"),
                           ("2", "65:64"), ("2", "64:0"),
                           ("2", "64:64:64"), ("2", "64:64,64:64")):
            with self.subTest(n=n, lengths=lengths), self.assertRaises(ValueError):
                metrics.benchmark_cases(n, lengths)

    def test_ratio_of_medians_and_saving_fraction(self):
        ref = [dict(total_ms=t, loss=10) for t in (90, 100, 800)]
        tpr = [dict(total_ms=t, loss=10.01) for t in (40, 50, 600)]
        row = metrics.summarize_case(4, 128, 128, ref, tpr)
        self.assertEqual(row["measured_speedup"], 2)
        self.assertEqual(row["ideal_token_speedup"], 1.6)
        self.assertAlmostEqual(row["saving_realization_fraction"], 0.5 / 0.375)
        self.assertAlmostEqual(row["loss_relative_diff"], 0.001)
        self.assertLess(row["recompute_adjusted_token_speedup_F1_B2"], row["ideal_token_speedup"])

    def test_slowdown_is_not_clipped(self):
        ref = [dict(total_ms=10, loss=1)] * 3
        tpr = [dict(total_ms=20, loss=1)] * 3
        row = metrics.summarize_case(2, 64, 64, ref, tpr)
        self.assertEqual(row["measured_speedup"], 0.5)
        self.assertLess(row["saving_realization_fraction"], 0)

    def test_requires_three_matching_samples(self):
        good = [dict(total_ms=10, loss=1)] * 3
        for bad in (good[:2], [*good[:2], dict(total_ms=10)]):
            with self.assertRaises(ValueError):
                metrics.summarize_case(2, 64, 64, bad, good)


if __name__ == "__main__":
    unittest.main()
