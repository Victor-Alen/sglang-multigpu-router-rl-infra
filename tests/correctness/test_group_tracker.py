import tempfile
import unittest
from pathlib import Path

from router.rl.models import ResponseMetadata, ValidationError
from router.rl.tracker import GroupTracker
from tests.helpers import requests


def response(request, served_version=None):
    return ResponseMetadata(
        request_id=request.request_id,
        group_id=request.group_id,
        sample_index=request.sample_index,
        worker_id="w0",
        requested_policy_version=request.policy_version,
        served_policy_version=(
            request.policy_version if served_version is None else served_version
        ),
        prompt_tokens=request.prompt_tokens,
        generated_tokens=50,
        generation_seed=request.generation_seed,
        tokenizer_revision=request.tokenizer_revision,
        chat_template_hash=request.chat_template_hash,
    )


class GroupTrackerTests(unittest.TestCase):
    def test_barrier_opens_only_after_all_samples(self):
        tracker = GroupTracker()
        items = requests()
        tracker.register(items)
        for item in items[:-1]:
            self.assertEqual(tracker.record_response(response(item)).status.value, "OPEN")
        self.assertEqual(tracker.record_response(response(items[-1])).status.value, "READY")

    def test_mixed_version_response_rejected(self):
        tracker = GroupTracker()
        items = requests()
        tracker.register(items)
        with self.assertRaisesRegex(ValidationError, "mixed-version"):
            tracker.record_response(response(items[0], served_version=4))

    def test_duplicate_training_sample_rejected(self):
        tracker = GroupTracker()
        items = requests()
        tracker.register(items)
        tracker.record_response(response(items[0]))
        duplicate = response(items[0])
        duplicate.request_id = "different-request"
        with self.assertRaisesRegex(ValidationError, "duplicate training sample"):
            tracker.record_response(duplicate)

    def test_router_restart_recovers_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.json"
            items = requests()
            tracker = GroupTracker(path)
            tracker.register(items)
            tracker.record_response(response(items[0]))
            recovered = GroupTracker(path)
            group = recovered.require(items[0].group_id)
            self.assertEqual(group.completed_samples, {0})
            self.assertEqual(group.policy_version, 3)

    def test_retry_seed_modes(self):
        tracker = GroupTracker()
        item = requests()[0]
        self.assertEqual(tracker.retry_seed(item, 1, "same_seed"), item.generation_seed)
        self.assertNotEqual(tracker.retry_seed(item, 1, "new_seed"), item.generation_seed)


if __name__ == "__main__":
    unittest.main()
