import unittest

from router.rl.gpu_layout import GPULayout, parse_gpu_ids
from router.rl.models import ValidationError


class GPULayoutTests(unittest.TestCase):
    def test_four_gpu_two_plus_two_non_contiguous(self):
        layout = GPULayout.from_strings("1,3", "5,7")
        self.assertEqual(layout.trainer_gpu_ids, (1, 3))
        self.assertEqual(layout.rollout_gpu_ids, (5, 7))
        self.assertEqual(layout.all_gpu_ids, (1, 3, 5, 7))
        self.assertEqual(layout.trainer_cuda_visible_devices, "1,3")

    def test_four_gpu_three_plus_one_is_valid(self):
        layout = GPULayout.from_strings("1,3,5", "7")
        self.assertEqual(len(layout.trainer_gpu_ids), 3)
        self.assertEqual(len(layout.rollout_gpu_ids), 1)

    def test_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "overlap"):
            GPULayout.from_strings("1,3", "3,5")

    def test_wrong_total_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "expected 4"):
            GPULayout.from_strings("1", "5,7")

    def test_duplicate_id_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            parse_gpu_ids("1,1")


if __name__ == "__main__":
    unittest.main()
