"""CPU-only checks: preserve first exception and never sync on unwind."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

source = Path(__file__).parents[1] / "parallel" / "_stage45_sync_trace.py"
spec = importlib.util.spec_from_file_location("stage45_trace", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TraceTest(unittest.TestCase):
    def make_trace(self, fail_at=None):
        logs, syncs = [], []
        error = RuntimeError("device failure")

        def synchronize():
            syncs.append(1)
            if len(syncs) == fail_at:
                raise error

        return module.SyncTrace(synchronize, logs.append, "r=1 rep=2"), logs, syncs, error

    def test_success(self):
        trace, logs, syncs, _ = self.make_trace()
        with trace.span("pop"):
            with trace.span("backward"):
                pass
        self.assertEqual(len(syncs), 4)
        self.assertIn("path=pop/backward END", logs[-3])
        self.assertFalse(trace.failed)
        self.assertEqual(trace.stack, [])

    def test_body_failure_no_exit_sync(self):
        trace, logs, syncs, error = self.make_trace()
        with self.assertRaises(RuntimeError) as raised:
            with trace.span("pop"):
                with trace.span("backward"):
                    raise error
        self.assertIs(raised.exception, error)
        self.assertEqual(len(syncs), 2)
        self.assertEqual(sum("FIRST_ERROR" in line for line in logs), 1)
        self.assertIn("phase=body", logs[-1])
        self.assertEqual(trace.stack, [])

    def test_post_sync_failure_preserves_origin(self):
        trace, logs, syncs, error = self.make_trace(fail_at=3)
        with self.assertRaises(RuntimeError) as raised:
            with trace.span("visit"):
                with trace.span("backward"):
                    pass
        self.assertIs(raised.exception, error)
        self.assertEqual(len(syncs), 3)
        self.assertIn("path=visit/backward FIRST_ERROR phase=post_sync", logs[-1])
        self.assertEqual(trace.stack, [])

    def test_pre_sync_does_not_run_body(self):
        trace, logs, syncs, error = self.make_trace(fail_at=1)
        with self.assertRaises(RuntimeError) as raised:
            with trace.span("forward"):
                self.fail("body should not run")
        self.assertIs(raised.exception, error)
        self.assertIn("phase=pre_sync", logs[-1])
        self.assertEqual(len(syncs), 1)

    def test_installed_segment_and_backward_nesting(self):
        trace, logs, _, _ = self.make_trace()
        autograd = SimpleNamespace(backward=lambda: 7)

        class Executor:
            plan = SimpleNamespace(get=lambda sid: SimpleNamespace(segment_id=sid, length=1024, prefix_length=8192))

            def push(self, sid):
                return autograd.backward()

            visit_leaf = pop = _forward = _compute_loss = push

        patches = []

        def setattr_(owner, attr, value):
            p = patch.object(owner, attr, value)
            p.start()
            patches.append(p)

        try:
            trace.install(SimpleNamespace(setattr=setattr_), Executor, autograd)
            self.assertEqual(Executor().visit_leaf(2), 7)
            self.assertTrue(any("visit_leaf(sid=2,T=1024,P=8192,no_grad=NA)/backward BEGIN" in line for line in logs))
        finally:
            for p in reversed(patches):
                p.stop()


if __name__ == "__main__":
    unittest.main()
