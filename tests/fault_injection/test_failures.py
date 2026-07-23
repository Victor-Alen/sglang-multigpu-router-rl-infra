import time
import unittest

from router.rl.models import ValidationError, WorkerLifecycle
from router.rl.routing import FixedEvenSplitPolicy
from tests.helpers import requests, workers


class FailureInjectionTests(unittest.TestCase):
    def test_failed_worker_removed_from_candidates(self):
        candidates = workers()
        candidates[0].lifecycle = WorkerLifecycle.FAILED
        decision = FixedEvenSplitPolicy().place_group(requests(), candidates)
        self.assertEqual(set(decision.assignments), {"w1"})

    def test_all_workers_failed_fails_closed(self):
        candidates = workers()
        for worker in candidates:
            worker.lifecycle = WorkerLifecycle.FAILED
        with self.assertRaisesRegex(ValidationError, "no READY worker"):
            FixedEvenSplitPolicy().place_group(requests(), candidates)

    def test_wrong_version_healthy_worker_fails_closed(self):
        candidates = workers(version=4)
        with self.assertRaisesRegex(ValidationError, "policy version 3"):
            FixedEvenSplitPolicy().place_group(requests(version=3), candidates)


if __name__ == "__main__":
    unittest.main()
