import tempfile
import unittest
from pathlib import Path

from router.rl.models import ValidationError, WorkerLifecycle
from router.rl.versioning import PolicyVersionCoordinator
from tests.helpers import workers


class VersioningTests(unittest.TestCase):
    def test_stop_the_world_update(self):
        candidates = workers(version=3)
        coordinator = PolicyVersionCoordinator(candidates)
        coordinator.begin_update(3, 4, expected_checksum="abc")
        for worker in candidates:
            self.assertEqual(worker.lifecycle, WorkerLifecycle.DRAINING)
            coordinator.mark_drained(worker.worker_id)
            coordinator.verify_worker(worker.worker_id, 4, "abc")
        update = coordinator.commit()
        self.assertEqual(update.status, "COMMITTED")
        self.assertTrue(all(worker.loaded_policy_version == 4 for worker in candidates))

    def test_checksum_failure_isolated(self):
        candidates = workers(version=3)
        coordinator = PolicyVersionCoordinator(candidates)
        coordinator.begin_update(3, 4, expected_checksum="abc")
        coordinator.mark_drained("w0")
        with self.assertRaisesRegex(ValidationError, "checksum"):
            coordinator.verify_worker("w0", 4, "bad")
        self.assertEqual(candidates[0].lifecycle, WorkerLifecycle.FAILED)

    def test_inflight_worker_cannot_update(self):
        candidates = workers(version=3)
        coordinator = PolicyVersionCoordinator(candidates)
        coordinator.begin_update(3, 4)
        candidates[0].running_requests = 1
        with self.assertRaisesRegex(ValidationError, "in-flight"):
            coordinator.mark_drained("w0")

    def test_rolling_only_drains_one_worker(self):
        candidates = workers(version=3)
        coordinator = PolicyVersionCoordinator(candidates)
        coordinator.begin_update(3, 4, mode="rolling")
        self.assertEqual(candidates[0].lifecycle, WorkerLifecycle.DRAINING)
        self.assertEqual(candidates[1].lifecycle, WorkerLifecycle.READY)
        coordinator.mark_drained("w0")
        coordinator.verify_worker("w0", 4, None)
        self.assertEqual(candidates[1].lifecycle, WorkerLifecycle.DRAINING)

    def test_audit_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            candidates = workers(version=3)
            PolicyVersionCoordinator(candidates, path).begin_update(3, 4)
            self.assertIn("begin_weight_update", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
