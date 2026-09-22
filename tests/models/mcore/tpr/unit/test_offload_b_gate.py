"""Dependency-free regression tests for Phase B's numerical noise budget."""

import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "profiling/_offload_b_gate.py"
spec = importlib.util.spec_from_file_location("offload_b_gate", PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def metrics(l2=0.006, absolute=0.001, fraction=0.02, finite=True):
    return dict(relative_l2=l2, max_abs=absolute, mismatch_fraction=fraction, finite=finite)


class GateTest(unittest.TestCase):
    def test_supplied_repeat_noise_is_accepted(self):
        for baseline, on, repeated in ((0.0060, 0.0065, 0.0059),
                                      (0.0123, 0.0117, 0.0119),
                                      (0.0108, 0.0116, 0.0105)):
            for value in (on, repeated):
                self.assertTrue(gate.parameter_gate(metrics(value), metrics(baseline))[0])

    def test_each_metric_can_fail_independently(self):
        baseline = metrics()
        for key in gate.NOISE_FLOORS:
            actual = baseline.copy()
            actual[key] *= 3
            self.assertFalse(gate.parameter_gate(actual, baseline)[0], key)

    def test_zero_noise_does_not_produce_zero_budget(self):
        baseline = metrics(0, 0, 0)
        passed, limits, _ = gate.parameter_gate(dict(finite=True, **gate.NOISE_FLOORS), baseline)
        self.assertTrue(passed)
        self.assertEqual(limits, gate.NOISE_FLOORS)
        self.assertFalse(gate.parameter_gate(metrics(0.003, 0, 0), baseline)[0])

    def test_nonfinite_reference_baseline_or_actual_always_fails(self):
        for invalid in (metrics(finite=False), metrics(float("nan")), metrics(float("inf"))):
            self.assertFalse(gate.parameter_gate(invalid, None)[0])
            self.assertFalse(gate.parameter_gate(invalid, metrics())[0])
            self.assertFalse(gate.parameter_gate(metrics(), invalid)[0])

    def test_calibration_does_not_apply_pointwise_gate(self):
        self.assertTrue(gate.parameter_gate(metrics(0.0123), None)[0])


if __name__ == "__main__":
    unittest.main()
