import unittest

from router.policy import CacheTracker


class ServingPolicyTests(unittest.TestCase):
    def test_repeated_fingerprint_respects_capacity(self):
        tracker = CacheTracker(max_entries=3)
        for _ in range(100):
            tracker.record("same")
        self.assertEqual(len(tracker._queue), 3)
        self.assertEqual(tracker.score("same"), 3)


if __name__ == "__main__":
    unittest.main()
