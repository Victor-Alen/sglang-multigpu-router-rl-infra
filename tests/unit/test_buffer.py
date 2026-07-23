import unittest

from router.rl.buffer import BoundedRolloutBuffer, VersionedBatch
from router.rl.models import ValidationError


class BufferTests(unittest.TestCase):
    def test_one_step_overlap(self):
        buffer = BoundedRolloutBuffer(max_batches=1, max_policy_lag=1)
        batch = VersionedBatch("b1", 4, {"samples": 8}, 1000)
        buffer.put(batch, trainer_policy_version=5)
        self.assertEqual(buffer.get(5), batch)

    def test_stale_batch_rejected(self):
        buffer = BoundedRolloutBuffer(max_policy_lag=1)
        with self.assertRaisesRegex(ValidationError, "max_policy_lag"):
            buffer.put(VersionedBatch("b1", 3, None, 100), trainer_policy_version=5)
        self.assertEqual(buffer.stale_tokens, 100)

    def test_capacity_is_hard_bound(self):
        buffer = BoundedRolloutBuffer(max_batches=1)
        buffer.put(VersionedBatch("b1", 1, None, 1), 1)
        with self.assertRaises(BufferError):
            buffer.put(VersionedBatch("b2", 1, None, 1), 1)


if __name__ == "__main__":
    unittest.main()
