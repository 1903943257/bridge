"""Pure CPU arithmetic regression for the observed unguarded dh0 store."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from parallel._stateful_ops_probe import conv_dh0_allocation_audit


class BoundsTest(unittest.TestCase):
    def test_qwen_cp2_t1024(self):
        result = conv_dh0_allocation_audit(1024, 8, 1, 3072, 4, 3072 * 4)
        self.assertEqual(result["time_tiles"], 128)
        self.assertEqual(result["unguarded_store_extent"], 128 * 3072 * 4)
        self.assertTrue(result["unguarded_store_out_of_bounds"])

    def test_short_sequence_can_also_have_invalid_store(self):
        result = conv_dh0_allocation_audit(64, 2, 1, 3072, 4, 2 * 3072 * 4)
        self.assertTrue(result["unguarded_store_out_of_bounds"])

    def test_one_tile_fits(self):
        result = conv_dh0_allocation_audit(4, 8, 1, 3072, 4, 3072 * 4)
        self.assertFalse(result["unguarded_store_out_of_bounds"])

    def test_full_sized_buffer_fits(self):
        result = conv_dh0_allocation_audit(1024, 8, 1, 3072, 4, 128 * 3072 * 4)
        self.assertFalse(result["unguarded_store_out_of_bounds"])

    def test_head_tile_guard_is_in_bounds_and_keeps_every_contribution(self):
        for t in (4, 64, 128, 256, 512, 1024):
            for bt in (1, 2, 4, 8, 16, 32):
                for w in (2, 3, 4):
                    nt = (t + bt - 1) // bt
                    allocated_tiles = min(nt, (w + bt - 1) // bt)
                    writers = [tile for tile in range(nt) if tile * bt < w - 1]
                    self.assertTrue(all(tile < allocated_tiles for tile in writers))
                    for slot in range(1, w):
                        covered = [time for tile in writers for time in range(min(slot, t))
                                   if tile * bt <= time < (tile + 1) * bt]
                        self.assertEqual(covered, list(range(min(slot, t))))


if __name__ == "__main__":
    unittest.main()
