"""Opt-in no-per-event-synchronize TPR Push/Pop/Visit NPU event + allocator profiler.

NPU timing is measured on the current stream (not a sum of all Ring/HCCL
device streams), so use the max-rank end-to-end synchronized elapsed time as
the critical path. Memory snapshots use allocator per-event reset; aggregate
maxima must be reconstructed across ALL events. No persistent tensor refs.
"""
from __future__ import annotations

import time

import torch


class SegmentEventProfiler:
    def __init__(self):
        self._pending = []
        self.max_allocated_bytes = 0
        self.max_reserved_bytes = 0

    def begin(self, segment, *, kind: str, tree_index: int, depth: int):
        self.max_allocated_bytes = max(
            self.max_allocated_bytes, int(torch.npu.max_memory_allocated())
        )
        self.max_reserved_bytes = max(
            self.max_reserved_bytes, int(torch.npu.max_memory_reserved())
        )
        # Reset the peak to attribute this event, without synchronizing the
        # device. Caller must reconstruct the overall max after execution.
        torch.npu.reset_peak_memory_stats()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        return {
            "segment_id": int(segment.segment_id),
            "parent_id": segment.parent_id,
            "tree_index": int(tree_index),
            "depth": int(depth),
            "kind": kind,
            "length": int(segment.length),
            "prefix_length": int(segment.prefix_length),
            "start_event": start,
            "end_event": end,
            "cpu_start": time.perf_counter(),
            "alloc_before_bytes": int(torch.npu.memory_allocated()),
            "reserved_before_bytes": int(torch.npu.memory_reserved()),
        }

    def end(self, pending):
        pending["end_event"].record()
        pending["cpu_enqueue_ms"] = (
            time.perf_counter() - pending.pop("cpu_start")
        ) * 1000.0
        pending["alloc_after_bytes"] = int(torch.npu.memory_allocated())
        pending["reserved_after_bytes"] = int(torch.npu.memory_reserved())
        pending["peak_allocated_bytes"] = int(torch.npu.max_memory_allocated())
        pending["peak_reserved_bytes"] = int(torch.npu.max_memory_reserved())
        self.max_allocated_bytes = max(
            self.max_allocated_bytes, pending["peak_allocated_bytes"]
        )
        self.max_reserved_bytes = max(
            self.max_reserved_bytes, pending["peak_reserved_bytes"]
        )
        self._pending.append(pending)

    def materialize(self):
        """Call after torch.npu.synchronize() ONCE per mini-batch."""
        result = []
        for pending in self._pending:
            record = {k: v for k, v in pending.items()
                      if k not in ("start_event", "end_event")}
            record["npu_event_ms"] = float(
                pending["start_event"].elapsed_time(pending["end_event"])
            )
            result.append(record)
        return result
