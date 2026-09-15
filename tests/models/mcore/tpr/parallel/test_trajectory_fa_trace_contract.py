"""Trace classification tests runnable without Torch or NPU."""

import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "trajectory_trace_contract", Path(__file__).with_name("_trajectory_fa_trace.py"))
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TraceContract(unittest.TestCase):
    def native(self, rank):
        remote = ((32, 1, 2048), (64, 1, 512)) if rank == 0 else (
            (64, 1, 2048), (32, 1, 512))
        forward = [((64, 1, 2048), (64, 1, 512), "SBH", 3), (*remote, "SBH", 0)]
        return dict(native_entry=1, fwd=forward, bwd=list(reversed(forward)))

    def test_both_native_ranks(self):
        for rank in (0, 1):
            trace.check_whole_trace(self.native(rank), "native_ring", rank)

    def test_old_schedule_cannot_be_labeled_native(self):
        old = dict(native_entry=0,
                   fwd=[((32, 8, 256), (32, 2, 256), "TND", 0)] * 5,
                   bwd=[((32, 8, 256), (32, 2, 256), "TND", 0)] * 5)
        trace.check_whole_trace(old, "ring", 0)
        with self.assertRaises(AssertionError):
            trace.check_whole_trace(old, "native_ring", 0)

    def test_forward_order_backward_is_rejected(self):
        records = self.native(0)
        records["bwd"] = records["fwd"]
        with self.assertRaises(AssertionError):
            trace.check_whole_trace(records, "native_ring", 0)


if __name__ == "__main__":
    unittest.main()
