import unittest

from router.rl.models import RolloutRequest, ValidationError, ensure_homogeneous_group
from tests.helpers import requests


class ModelTests(unittest.TestCase):
    def test_request_round_trip(self):
        request = requests()[0]
        self.assertEqual(RolloutRequest.from_dict(request.to_dict()), request)

    def test_invalid_index_rejected(self):
        value = requests()[0].to_dict()
        value["sample_index"] = value["group_size"]
        with self.assertRaises(ValidationError):
            RolloutRequest.from_dict(value)

    def test_mixed_policy_version_rejected(self):
        items = requests()
        value = items[-1].to_dict()
        value["policy_version"] += 1
        items[-1] = RolloutRequest.from_dict(value)
        with self.assertRaisesRegex(ValidationError, "mixed policy_version"):
            ensure_homogeneous_group(items)

    def test_duplicate_sample_index_rejected(self):
        items = requests()
        value = items[-1].to_dict()
        value["sample_index"] = 0
        items[-1] = RolloutRequest.from_dict(value)
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            ensure_homogeneous_group(items)


if __name__ == "__main__":
    unittest.main()
