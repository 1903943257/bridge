"""Run real transport loops against threaded scalar RingP2P, without torch/NPU.

Checks direction, step/owner routing and fixed allocation counts. This does
not emulate asynchronous device streams or validate fused-attention numerics.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace as NS
import unittest


class Cell:
    def __init__(self, storage, index):
        self.storage, self.index = storage, index

    @property
    def value(self):
        return self.storage[self.index]

    def add_(self, value):
        self.storage[self.index] += value


class Buffer:
    def __init__(self, values):
        self.values = list(values)

    def contiguous(self):
        return self

    def __getitem__(self, index):
        return Cell(self.values, index)


class TransportTest(unittest.TestCase):
    def run_ring(self, cp, backward):
        queues = [Queue() for _ in range(cp)]
        path = Path(__file__).resolve().parents[5] / "verl/models/mcore/tpr/parallel/ring_attention.py"
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                 and n.name in ("_circulate_kv", "_reduce_ring_gradients_to_owner")]

        def worker(rank):
            allocations, visited, sends = [], [], []

            def allocate(values):
                result = Buffer(values)
                allocations.append(result)
                return result

            class Ring:
                def __init__(self, ranks, group, is_backward=False):
                    self.next = (rank - 1 if is_backward else rank + 1) % cp
                    self.pending = None

                def async_send_recv(self, send, recv):
                    assert self.pending is None
                    queues[self.next].put(list(send.values))
                    sends.append((id(send), id(recv)))
                    self.pending = recv

                def wait(self):
                    if self.pending is not None:
                        self.pending.values[:] = queues[rank].get(timeout=5)
                        self.pending = None

            torch = NS(stack=lambda pair, dim: allocate([x.value for x in pair]),
                       empty_like=lambda t: allocate([0, 0]), zeros_like=lambda t: allocate([0, 0]))
            ns = dict(torch=torch, _load_mindspeed_ring_primitives=lambda: (Ring, None),
                      _observe_ring_storage=lambda *args: None)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
            key, value = Cell([rank + 1], 0), Cell([10 * (rank + 1)], 0)
            config = NS(cp_rank=rank, cp_size=cp, global_ranks=tuple(range(cp)), cp_group=None)

            def consume(source, k, v, dk=None, dv=None):
                self.assertEqual(k.value, source + 1)
                self.assertEqual(v.value, 10 * (source + 1))
                visited.append(source)
                if dk is not None:
                    dk.add_(100 * (rank + 1) + source)
                    dv.add_(1000 * (rank + 1) + source)

            if backward:
                dk, dv = ns["_reduce_ring_gradients_to_owner"](key, value, config, consume)
                self.assertEqual(dk.value, 100 * cp * (cp + 1) // 2 + cp * rank)
                self.assertEqual(dv.value, 1000 * cp * (cp + 1) // 2 + cp * rank)
                self.assertEqual(visited, [(rank + step + 1) % cp for step in range(cp)])
                self.assertEqual(len(sends), 2 * (cp - 1))
                self.assertEqual(len(allocations), 4)
            else:
                ns["_circulate_kv"](key, value, config, consume)
                self.assertEqual(visited, [(rank - step) % cp for step in range(cp)])
                self.assertEqual(len(sends), cp - 1)
                self.assertEqual(len(allocations), 2)
            # Neither ping-pong path overwrites the original local inputs.
            self.assertEqual((key.value, value.value), (rank + 1, 10 * (rank + 1)))

        with ThreadPoolExecutor(max_workers=cp) as pool:
            list(pool.map(worker, range(cp)))
        self.assertTrue(all(q.empty() for q in queues))

    def test_forward_all_sources_constant_buffers(self):
        for cp in (2, 4, 8):
            self.run_ring(cp, False)

    def test_backward_reverse_replay_and_owner_reduction(self):
        for cp in (2, 4, 8):
            self.run_ring(cp, True)


if __name__ == "__main__":
    unittest.main()
