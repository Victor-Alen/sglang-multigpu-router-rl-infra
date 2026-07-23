import time
import unittest

from router.rl.models import ValidationError, WorkerLifecycle
from router.rl.routing import (
    AdaptiveGroupPolicy,
    FixedEvenSplitPolicy,
    FixedPackPolicy,
    build_group_policy,
)
from tests.helpers import requests, workers


class RoutingTests(unittest.TestCase):
    def test_pack_assigns_entire_group(self):
        decision = FixedPackPolicy().place_group(requests(), workers())
        self.assertEqual(sum(map(len, decision.assignments.values())), 4)
        self.assertEqual(len([v for v in decision.assignments.values() if v]), 1)

    def test_even_split_balances_group(self):
        decision = FixedEvenSplitPolicy().place_group(requests(), workers())
        self.assertEqual(sorted(map(len, decision.assignments.values())), [2, 2])

    def test_adaptive_packs_long_prompt_short_output(self):
        decision = AdaptiveGroupPolicy().place_group(
            requests(prompt_tokens=4096, predicted=24, p90=32), workers()
        )
        self.assertTrue(decision.reason["selected_candidate"].startswith("pack_"))

    def test_adaptive_splits_long_outputs(self):
        decision = AdaptiveGroupPolicy().place_group(
            requests(prompt_tokens=128, predicted=800, p90=1000), workers()
        )
        self.assertTrue(decision.reason["selected_candidate"].startswith("split_"))

    def test_version_filter_is_hard_constraint(self):
        candidates = workers()
        candidates[0].loaded_policy_version = 4
        decision = FixedEvenSplitPolicy().place_group(requests(version=3), candidates)
        self.assertEqual(list(decision.assignments), ["w1"])

    def test_draining_worker_is_not_eligible(self):
        candidates = workers()
        candidates[0].lifecycle = WorkerLifecycle.DRAINING
        decision = FixedPackPolicy().place_group(requests(), candidates)
        self.assertEqual(list(decision.assignments), ["w1"])

    def test_stale_heartbeat_rejected(self):
        candidates = workers()
        for worker in candidates:
            worker.heartbeat_timestamp_ns = time.time_ns() - 60_000_000_000
        with self.assertRaisesRegex(ValidationError, "no READY worker"):
            FixedPackPolicy(heartbeat_timeout_seconds=1).place_group(requests(), candidates)

    def test_all_declared_policies_build(self):
        names = [
            "round-robin",
            "least-requests",
            "least-queued-tokens",
            "cache-aware-group",
            "power-of-two",
            "fixed-pack",
            "fixed-even-split",
            "load-proportional-split",
            "adaptive-group",
            "offline-oracle",
        ]
        for name in names:
            self.assertIsNotNone(build_group_policy(name))


if __name__ == "__main__":
    unittest.main()
