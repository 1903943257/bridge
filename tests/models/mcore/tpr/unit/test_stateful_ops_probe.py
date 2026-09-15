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


if __name__ == "__main__":
    unittest.main()
