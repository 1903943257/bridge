"""Dependency-free lifecycle tests for retained Prefix backward graphs."""

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).resolve().parents[5] / "verl/models/mcore/tpr/prefix_graph.py"
SPEC = importlib.util.spec_from_file_location("tpr_prefix_graph", PATH)
prefix_graph = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefix_graph
SPEC.loader.exec_module(prefix_graph)

PrefixGraphStore = prefix_graph.PrefixGraphStore


class _Session:
    def __init__(self, *, fail=False, close_order=None, name=None):
        self.close_calls = 0
        self.fail = fail
        self.close_order = close_order
        self.name = name

    def close(self):
        self.close_calls += 1
        if self.close_order is not None:
            self.close_order.append(self.name)
        if self.fail:
            raise RuntimeError(f"close failed: {self.name}")


def _push(store, segment_id, parent_signature, session=None):
    session = _Session() if session is None else session
    objects = {
        "loss": object(),
        "kv": object(),
        "anchors": object(),
        "detached": object(),
    }
    record = store.push(
        segment_id=segment_id,
        parent_stack_signature=parent_signature,
        backward_loss_root=objects["loss"],
        graph_key_values={1: objects["kv"]},
        parent_anchors=objects["anchors"],
        detached_loss=objects["detached"],
        native_session=session,
    )
    return record, session, objects


class PrefixGraphStoreTest(unittest.TestCase):
    def test_push_assigns_generation_and_tracks_parent_signature(self):
        store = PrefixGraphStore()
        root, _, _ = _push(store, 4, ())
        child, _, _ = _push(store, 7, (4,))

        self.assertEqual(root.generation, 0)
        self.assertEqual(child.generation, 1)
        self.assertEqual(child.parent_stack_signature, (4,))
        self.assertEqual(store.segment_ids, (4, 7))
        self.assertIs(store.top(), child)

    def test_consume_is_lifo_and_validates_post_kv_pop_signature(self):
        store = PrefixGraphStore()
        root, _, _ = _push(store, 4, ())
        child, _, _ = _push(store, 7, (4,))

        with self.assertRaisesRegex(RuntimeError, "store top"):
            store.consume(segment_id=4, parent_stack_signature=())
        with self.assertRaisesRegex(RuntimeError, "stale"):
            store.consume(segment_id=7, parent_stack_signature=())
        self.assertEqual(store.segment_ids, (4, 7))

        consumed = store.consume(segment_id=7, parent_stack_signature=(4,))
        self.assertIs(consumed, child)
        self.assertTrue(consumed.consumed)
        self.assertEqual(store.segment_ids, (4,))
        consumed.close()
        store.abort_top(segment_id=4, parent_stack_signature=())
        store.assert_empty()
        self.assertTrue(root.closed)

    def test_consumed_record_context_closes_once_and_drops_graph_references(self):
        store = PrefixGraphStore()
        record, session, objects = _push(store, 1, ())

        with store.consume(segment_id=1, parent_stack_signature=()) as consumed:
            self.assertIs(consumed.backward_loss_root, objects["loss"])
            self.assertIs(consumed.graph_key_values[1], objects["kv"])
            self.assertIs(consumed.parent_anchors, objects["anchors"])
            self.assertIs(consumed.detached_loss, objects["detached"])
            self.assertIs(consumed.native_session, session)

        self.assertTrue(record.closed)
        self.assertEqual(session.close_calls, 1)
        record.close()
        self.assertEqual(session.close_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = record.backward_loss_root

    def test_backward_scope_exception_still_closes_session(self):
        store = PrefixGraphStore()
        record, session, _ = _push(store, 1, ())

        with self.assertRaisesRegex(ValueError, "backward failed"):
            with store.consume(segment_id=1, parent_stack_signature=()):
                raise ValueError("backward failed")

        self.assertTrue(record.closed)
        self.assertEqual(session.close_calls, 1)
        store.assert_empty()

    def test_abort_closes_active_record_without_marking_it_consumed(self):
        store = PrefixGraphStore()
        record, session, _ = _push(store, 1, ())

        store.abort_top(segment_id=1, parent_stack_signature=())

        self.assertTrue(record.closed)
        self.assertFalse(record.consumed)
        self.assertEqual(session.close_calls, 1)
        store.assert_empty()

    def test_close_all_is_reverse_order_and_continues_after_close_failure(self):
        order = []
        store = PrefixGraphStore()
        root_session = _Session(close_order=order, name="root")
        child_session = _Session(fail=True, close_order=order, name="child")
        _push(store, 1, (), root_session)
        _push(store, 2, (1,), child_session)

        with self.assertRaisesRegex(RuntimeError, "1 Prefix graph session"):
            store.close_all()

        self.assertEqual(order, ["child", "root"])
        self.assertEqual(child_session.close_calls, 1)
        self.assertEqual(root_session.close_calls, 1)
        store.assert_empty()

    def test_rejected_push_transfers_and_cleans_session_ownership(self):
        store = PrefixGraphStore()
        _push(store, 1, ())
        rejected = _Session()

        with self.assertRaisesRegex(RuntimeError, "parent stack mismatch"):
            _push(store, 2, (), rejected)

        self.assertEqual(rejected.close_calls, 1)
        self.assertEqual(store.segment_ids, (1,))
        store.close_all()

    def test_assert_empty_reports_leaked_segments(self):
        store = PrefixGraphStore()
        _push(store, 3, ())
        with self.assertRaisesRegex(RuntimeError, r"active segments \(3,\)"):
            store.assert_empty()
        store.close_all()


if __name__ == "__main__":
    unittest.main()
