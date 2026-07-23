import tempfile
import unittest
import json
from pathlib import Path

from router.rl.models import ResponseMetadata
from router.rl.runtime import RLRouterRuntime
from router.rl.trace import TraceReplayer
from tests.helpers import requests, workers


class RuntimeReplayTests(unittest.TestCase):
    def test_online_and_offline_decisions_match(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = RLRouterRuntime(workers(), state_dir=directory)
            items = requests(prompt_tokens=128, predicted=800, p90=1000)
            decision = runtime.route(items)
            summary = TraceReplayer("adaptive-group").replay(Path(directory) / "trace.jsonl")
            self.assertEqual(summary.groups, 1)
            self.assertEqual(summary.assignment_matches, 1)
            self.assertEqual(sum(map(len, decision.assignments.values())), 4)

    def test_fresh_token_throughput_records_complete_group(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = RLRouterRuntime(workers(), state_dir=directory)
            items = requests()
            runtime.route(items)
            for item in items:
                runtime.complete(
                    ResponseMetadata(
                        request_id=item.request_id,
                        group_id=item.group_id,
                        sample_index=item.sample_index,
                        worker_id="w0",
                        requested_policy_version=3,
                        served_policy_version=3,
                        prompt_tokens=item.prompt_tokens,
                        generated_tokens=10,
                        tokenizer_revision=item.tokenizer_revision,
                        chat_template_hash=item.chat_template_hash,
                    ),
                    consumed_at_policy_version=4,
                )
            snapshot = runtime.metrics.snapshot()
            self.assertEqual(snapshot["groups_completed"], 1)
            self.assertEqual(snapshot["fresh_tokens"], 40)

    def test_replay_rebases_historical_heartbeat_age(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = RLRouterRuntime(workers(), state_dir=directory)
            runtime.route(requests())
            path = Path(directory) / "trace.jsonl"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["timestamp_ns"] = 1_000_000_000
            for worker in record["payload"]["workers"]:
                worker["heartbeat_timestamp_ns"] = 1_000_000_000
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary = TraceReplayer("adaptive-group").replay(path)
            self.assertEqual(summary.assignment_matches, 1)


if __name__ == "__main__":
    unittest.main()
