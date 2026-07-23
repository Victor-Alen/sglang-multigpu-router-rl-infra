from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from .models import ResponseMetadata, RolloutRequest, RolloutWorkerState, WorkerLifecycle
from .routing import ObjectiveWeights, PlacementDecision, build_group_policy
from .telemetry import RLRouterMetrics
from .trace import JsonlTraceWriter
from .tracker import GroupTracker
from .versioning import PolicyVersionCoordinator


class RLRouterRuntime:
    def __init__(
        self,
        workers: Iterable[RolloutWorkerState],
        policy_name: str = "adaptive-group",
        state_dir: str | Path = "results/rl_state",
        weights: Optional[ObjectiveWeights] = None,
        heartbeat_timeout_seconds: float = 15.0,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        worker_list = list(workers)
        kwargs = {"heartbeat_timeout_seconds": heartbeat_timeout_seconds}
        if policy_name in {"adaptive", "adaptive-group", "offline-oracle"}:
            kwargs["weights"] = weights or ObjectiveWeights()
        self.policy = build_group_policy(policy_name, **kwargs)
        self.workers: Dict[str, RolloutWorkerState] = {
            worker.worker_id: worker for worker in worker_list
        }
        self.tracker = GroupTracker(self.state_dir / "groups.json")
        self.trace = JsonlTraceWriter(self.state_dir / "trace.jsonl", fsync=True)
        self.metrics = RLRouterMetrics()
        self.versioning = PolicyVersionCoordinator(
            self.workers.values(),
            self.state_dir / "version_audit.jsonl",
            self.state_dir / "version_state.json",
        )
        self._lock = threading.RLock()

    def register_worker(self, worker: RolloutWorkerState) -> None:
        with self._lock:
            self.workers[worker.worker_id] = worker
            self.versioning.workers[worker.worker_id] = worker
            self.trace.write("worker_registered", worker.to_dict())

    def heartbeat(self, worker_id: str, fields: dict) -> RolloutWorkerState:
        with self._lock:
            worker = self.workers[worker_id]
            allowed = {
                "lifecycle",
                "loaded_policy_version",
                "running_requests",
                "queued_requests",
                "queued_prefill_tokens",
                "active_decode_tokens",
                "available_kv_tokens",
                "kv_cache_utilization",
                "prefill_tokens_per_second_ema",
                "decode_tokens_per_second_ema",
                "prefix_matched_tokens",
                "parameter_checksum",
            }
            for key, value in fields.items():
                if key not in allowed:
                    continue
                if key == "lifecycle":
                    value = WorkerLifecycle(value)
                setattr(worker, key, value)
            worker.heartbeat_timestamp_ns = time.time_ns()
            worker.__post_init__()
            return worker

    def route(self, requests: Iterable[RolloutRequest], deadline_seconds: float = 300.0) -> PlacementDecision:
        items = list(requests)
        group = self.tracker.register(items, deadline_seconds)
        workers_snapshot = [RolloutWorkerState.from_dict(worker.to_dict()) for worker in self.workers.values()]
        decision = self.policy.place_group(items, workers_snapshot)
        self.tracker.assign(group.group_id, decision.assignments)
        duplicated = 0
        selected = decision.reason.get("selected_candidate")
        if selected and selected in decision.candidate_scores:
            duplicated = decision.candidate_scores[selected].duplicated_prefill_tokens
        self.metrics.record_decision(
            decision.strategy, decision.assignments, decision.decision_latency_ms, duplicated
        )
        self.trace.write(
            "routing_decision",
            {
                "requests": [request.to_dict() for request in items],
                "workers": [worker.to_dict() for worker in workers_snapshot],
                "decision": decision.to_dict(),
            },
        )
        return decision

    def complete(self, response: ResponseMetadata, consumed_at_policy_version: Optional[int] = None) -> dict:
        before = self.tracker.require(response.group_id).status
        group = self.tracker.record_response(response)
        self.trace.write("response_completed", response.to_dict())
        if group.status.value == "READY" and before != group.status:
            completion = (time.time_ns() - group.creation_timestamp_ns) / 1e9
            tokens = sum(item.generated_tokens for item in group.responses.values())
            consumed_version = consumed_at_policy_version or group.policy_version
            policy_lag = consumed_version - group.policy_version
            self.metrics.record_group("READY", completion, tokens, policy_lag, 1)
        return group.to_dict()

    def snapshot(self) -> dict:
        return {
            "workers": {key: value.to_dict() for key, value in self.workers.items()},
            "groups": {key: value.to_dict() for key, value in self.tracker.groups.items()},
            "versions": self.versioning.snapshot(),
            "metrics": self.metrics.snapshot(),
        }
