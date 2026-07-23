from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .models import RolloutRequest, RolloutWorkerState
from .routing import PlacementDecision, build_group_policy


class JsonlTraceWriter:
    def __init__(self, path: Path, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict) -> None:
        record = {
            "schema_version": 1,
            "timestamp_ns": time.time_ns(),
            "event": event,
            "payload": payload,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


@dataclass
class ReplaySummary:
    groups: int = 0
    exact_matches: int = 0
    assignment_matches: int = 0
    decision_latency_ms: float = 0.0
    predicted_duplicated_prefill_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "groups": self.groups,
            "exact_matches": self.exact_matches,
            "assignment_matches": self.assignment_matches,
            "exact_match_ratio": self.exact_matches / max(1, self.groups),
            "assignment_match_ratio": self.assignment_matches / max(1, self.groups),
            "decision_latency_ms": self.decision_latency_ms,
            "predicted_duplicated_prefill_tokens": self.predicted_duplicated_prefill_tokens,
        }


class TraceReplayer:
    def __init__(self, policy_name: str) -> None:
        self.policy_name = policy_name

    def replay(self, path: Path) -> ReplaySummary:
        summary = ReplaySummary()
        for record in read_jsonl(path):
            if record.get("event") != "routing_decision":
                continue
            payload = record["payload"]
            requests = [RolloutRequest.from_dict(item) for item in payload["requests"]]
            replay_now_ns = time.time_ns()
            recorded_now_ns = int(record.get("timestamp_ns", replay_now_ns))
            workers = []
            for item in payload["workers"]:
                worker_data = dict(item)
                recorded_heartbeat_ns = int(
                    worker_data.get("heartbeat_timestamp_ns", recorded_now_ns)
                )
                recorded_age_ns = max(0, recorded_now_ns - recorded_heartbeat_ns)
                worker_data["heartbeat_timestamp_ns"] = replay_now_ns - recorded_age_ns
                workers.append(RolloutWorkerState.from_dict(worker_data))
            expected = payload["decision"]
            decision = build_group_policy(self.policy_name).place_group(requests, workers)
            summary.groups += 1
            summary.decision_latency_ms += decision.decision_latency_ms
            if decision.strategy == expected["strategy"]:
                summary.exact_matches += 1
            if decision.assignments == expected["assignments"]:
                summary.assignment_matches += 1
            selected = decision.reason.get("selected_candidate")
            if selected and selected in decision.candidate_scores:
                summary.predicted_duplicated_prefill_tokens += decision.candidate_scores[
                    selected
                ].duplicated_prefill_tokens
        return summary
