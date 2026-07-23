from __future__ import annotations

import itertools
import hashlib
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Mapping, Protocol, Sequence

from .models import (
    RolloutRequest,
    RolloutWorkerState,
    ValidationError,
    ensure_homogeneous_group,
)


@dataclass(frozen=True)
class ObjectiveWeights:
    makespan: float = 1.0
    duplicated_prefill: float = 0.0005
    worker_skew: float = 0.15
    queue_wait: float = 0.2
    risk: float = 0.35


@dataclass
class CandidateScore:
    name: str
    score: float
    makespan_seconds: float
    duplicated_prefill_tokens: int
    worker_skew: float
    queue_wait_seconds: float
    assignment_counts: Dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlacementDecision:
    group_id: str
    policy_version: int
    strategy: str
    assignments: Dict[str, list[str]]
    candidate_scores: Dict[str, CandidateScore] = field(default_factory=dict)
    reason: Dict[str, object] = field(default_factory=dict)
    decision_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "policy_version": self.policy_version,
            "strategy": self.strategy,
            "assignments": self.assignments,
            "candidate_scores": {k: v.to_dict() for k, v in self.candidate_scores.items()},
            "reason": self.reason,
            "decision_latency_ms": self.decision_latency_ms,
        }


class GroupRoutingPolicy(Protocol):
    name: str

    def place_group(
        self,
        requests: Sequence[RolloutRequest],
        workers: Sequence[RolloutWorkerState],
    ) -> PlacementDecision:
        ...


def eligible_workers(
    requests: Sequence[RolloutRequest],
    workers: Sequence[RolloutWorkerState],
    heartbeat_timeout_seconds: float,
) -> tuple[list[RolloutRequest], list[RolloutWorkerState]]:
    items = ensure_homogeneous_group(requests)
    now_ns = time.time_ns()
    timeout_ns = int(heartbeat_timeout_seconds * 1e9)
    selected = [
        worker
        for worker in workers
        if worker.is_eligible(items[0].policy_version, now_ns, timeout_ns)
    ]
    if not selected:
        states = {
            worker.worker_id: {
                "lifecycle": worker.lifecycle.value,
                "loaded_policy_version": worker.loaded_policy_version,
                "heartbeat_age_seconds": (now_ns - worker.heartbeat_timestamp_ns) / 1e9,
            }
            for worker in workers
        }
        raise ValidationError(
            f"no READY worker for policy version {items[0].policy_version}: {states}"
        )
    return items, sorted(selected, key=lambda worker: worker.worker_id)


def _decision(
    name: str,
    requests: Sequence[RolloutRequest],
    assignment: Mapping[str, Sequence[RolloutRequest]],
    start: float,
    reason: dict | None = None,
) -> PlacementDecision:
    return PlacementDecision(
        group_id=requests[0].group_id,
        policy_version=requests[0].policy_version,
        strategy=name,
        assignments={key: [request.request_id for request in value] for key, value in assignment.items()},
        reason=reason or {},
        decision_latency_ms=(time.perf_counter() - start) * 1000,
    )


class RoundRobinGroupPolicy:
    name = "round-robin"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self._counter = itertools.count()

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        assignment = {worker.worker_id: [] for worker in ready}
        offset = next(self._counter)
        for index, request in enumerate(items):
            assignment[ready[(offset + index) % len(ready)].worker_id].append(request)
        return _decision(self.name, items, assignment, start)


class FixedPackPolicy:
    name = "fixed-pack"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        request = items[0]
        chosen = min(
            ready,
            key=lambda worker: (
                worker.queue_time_seconds()
                + max(0, request.prompt_tokens - worker.matched_prefix_tokens(
                    request.prefix_fingerprint, request.prompt_tokens
                )) / worker.prefill_tokens_per_second_ema,
                worker.running_requests,
                worker.worker_id,
            ),
        )
        return _decision(
            self.name,
            items,
            {chosen.worker_id: items},
            start,
            {"selected_worker": chosen.worker_id},
        )


class FixedEvenSplitPolicy:
    name = "fixed-even-split"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        assignment = {worker.worker_id: [] for worker in ready}
        for index, request in enumerate(items):
            assignment[ready[index % len(ready)].worker_id].append(request)
        return _decision(self.name, items, assignment, start)


class LeastQueuedTokensPolicy:
    name = "least-queued-tokens"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        virtual = {
            worker.worker_id: worker.queued_prefill_tokens + worker.active_decode_tokens
            for worker in ready
        }
        assignment = {worker.worker_id: [] for worker in ready}
        for request in sorted(items, key=lambda item: item.predicted_output_p90, reverse=True):
            worker_id = min(virtual, key=lambda key: (virtual[key], key))
            assignment[worker_id].append(request)
            virtual[worker_id] += request.prompt_tokens + int(request.predicted_output_p90)
        return _decision(self.name, items, assignment, start)


class LeastRequestsGroupPolicy:
    name = "least-requests"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        virtual = {worker.worker_id: worker.running_requests + worker.queued_requests for worker in ready}
        assignment = {worker.worker_id: [] for worker in ready}
        for request in items:
            worker_id = min(virtual, key=lambda key: (virtual[key], key))
            assignment[worker_id].append(request)
            virtual[worker_id] += 1
        return _decision(self.name, items, assignment, start)


class CacheAwareGroupPolicy:
    name = "cache-aware-group"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        assignment = {worker.worker_id: [] for worker in ready}
        virtual = {worker.worker_id: worker.queue_time_seconds() for worker in ready}
        for request in items:
            chosen = min(
                ready,
                key=lambda worker: (
                    -worker.matched_prefix_tokens(request.prefix_fingerprint, request.prompt_tokens),
                    virtual[worker.worker_id],
                    worker.worker_id,
                ),
            )
            assignment[chosen.worker_id].append(request)
            virtual[chosen.worker_id] += request.predicted_output_p90 / chosen.decode_tokens_per_second_ema
        return _decision(self.name, items, assignment, start)


class PowerOfTwoGroupPolicy:
    name = "power-of-two"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        assignment = {worker.worker_id: [] for worker in ready}
        virtual = {
            worker.worker_id: worker.queued_prefill_tokens + worker.active_decode_tokens
            for worker in ready
        }
        for request in items:
            digest = hashlib.sha256(request.request_id.encode("utf-8")).digest()
            first = int.from_bytes(digest[:4], "big") % len(ready)
            second = int.from_bytes(digest[4:8], "big") % len(ready)
            if len(ready) > 1 and second == first:
                second = (second + 1) % len(ready)
            candidates = [ready[first], ready[second]]
            chosen = min(candidates, key=lambda worker: (virtual[worker.worker_id], worker.worker_id))
            assignment[chosen.worker_id].append(request)
            virtual[chosen.worker_id] += request.prompt_tokens + int(request.predicted_output_p90)
        return _decision(self.name, items, assignment, start)


class LoadProportionalSplitPolicy:
    name = "load-proportional-split"

    def __init__(self, heartbeat_timeout_seconds: float = 15.0) -> None:
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        assignment = {worker.worker_id: [] for worker in ready}
        finish = {worker.worker_id: worker.queue_time_seconds() for worker in ready}
        lookup = {worker.worker_id: worker for worker in ready}
        for request in sorted(items, key=lambda item: item.predicted_output_p90, reverse=True):
            worker_id = min(finish, key=lambda key: (finish[key], key))
            worker = lookup[worker_id]
            finish[worker_id] += request.predicted_output_p90 / worker.decode_tokens_per_second_ema
            assignment[worker_id].append(request)
        return _decision(self.name, items, assignment, start)


class AdaptiveGroupPolicy:
    name = "adaptive-group"

    def __init__(
        self,
        weights: ObjectiveWeights | None = None,
        heartbeat_timeout_seconds: float = 15.0,
    ) -> None:
        self.weights = weights or ObjectiveWeights()
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds

    @staticmethod
    def _candidate_assignments(requests, workers):
        candidates: Dict[str, Dict[str, list[RolloutRequest]]] = {}
        for worker in workers:
            candidates[f"pack_{worker.worker_id}"] = {worker.worker_id: list(requests)}

        even = {worker.worker_id: [] for worker in workers}
        for index, request in enumerate(requests):
            even[workers[index % len(workers)].worker_id].append(request)
        candidates["split_even"] = even

        finish = {worker.worker_id: worker.queue_time_seconds() for worker in workers}
        proportional = {worker.worker_id: [] for worker in workers}
        lookup = {worker.worker_id: worker for worker in workers}
        for request in sorted(requests, key=lambda item: item.predicted_output_p90, reverse=True):
            worker_id = min(finish, key=lambda key: (finish[key], key))
            proportional[worker_id].append(request)
            finish[worker_id] += request.predicted_output_p90 / lookup[worker_id].decode_tokens_per_second_ema
        candidates["split_load_proportional"] = proportional
        return candidates

    def _score(self, name, assignment, request, workers) -> CandidateScore:
        lookup = {worker.worker_id: worker for worker in workers}
        finish_times: list[float] = []
        queue_waits: list[float] = []
        duplicated_prefill = 0
        assigned_loads = []
        used_workers = 0

        for worker_id, assigned in assignment.items():
            if not assigned:
                continue
            used_workers += 1
            worker = lookup[worker_id]
            queue = worker.queue_time_seconds()
            queue_waits.append(queue)
            matched = worker.matched_prefix_tokens(
                request.prefix_fingerprint, request.prompt_tokens
            )
            prefill = max(0, request.prompt_tokens - matched)
            decode_mean = sum(item.predicted_output_tokens for item in assigned)
            decode_p90 = sum(item.predicted_output_p90 for item in assigned)
            expected_decode = (
                (1 - self.weights.risk) * decode_mean + self.weights.risk * decode_p90
            )
            finish_times.append(
                queue
                + prefill / worker.prefill_tokens_per_second_ema
                + expected_decode / worker.decode_tokens_per_second_ema
            )
            assigned_loads.append(decode_p90 + prefill)
            duplicated_prefill += prefill

        if used_workers:
            duplicated_prefill -= min(
                max(
                    0,
                    request.prompt_tokens
                    - lookup[worker_id].matched_prefix_tokens(
                        request.prefix_fingerprint, request.prompt_tokens
                    ),
                )
                for worker_id, assigned in assignment.items()
                if assigned
            )
        makespan = max(finish_times, default=math.inf)
        mean_load = sum(assigned_loads) / max(1, len(assigned_loads))
        skew = max(assigned_loads, default=0.0) / max(1.0, mean_load)
        queue_wait = max(queue_waits, default=0.0)
        score = (
            self.weights.makespan * makespan
            + self.weights.duplicated_prefill * duplicated_prefill
            + self.weights.worker_skew * skew
            + self.weights.queue_wait * queue_wait
        )
        return CandidateScore(
            name=name,
            score=score,
            makespan_seconds=makespan,
            duplicated_prefill_tokens=duplicated_prefill,
            worker_skew=skew,
            queue_wait_seconds=queue_wait,
            assignment_counts={key: len(value) for key, value in assignment.items()},
        )

    def place_group(self, requests, workers) -> PlacementDecision:
        start = time.perf_counter()
        items, ready = eligible_workers(requests, workers, self.heartbeat_timeout_seconds)
        candidates = self._candidate_assignments(items, ready)
        scores = {
            name: self._score(name, assignment, items[0], ready)
            for name, assignment in candidates.items()
        }
        selected_name = min(scores, key=lambda name: (scores[name].score, name))
        assignment = candidates[selected_name]
        decision = _decision(
            self.name,
            items,
            assignment,
            start,
            {
                "selected_candidate": selected_name,
                "prompt_tokens": items[0].prompt_tokens,
                "predicted_output_p90": max(item.predicted_output_p90 for item in items),
                "eligible_workers": [worker.worker_id for worker in ready],
                "weights": asdict(self.weights),
            },
        )
        decision.candidate_scores = scores
        return decision


class OfflineOraclePolicy(AdaptiveGroupPolicy):
    """Adaptive scorer fed with actual output lengths by the replay harness."""

    name = "offline-oracle"

    def place_group(self, requests, workers) -> PlacementDecision:
        exact = [
            RolloutRequest(
                **{
                    **request.to_dict(),
                    "predicted_output_tokens": request.predicted_output_p90,
                }
            )
            for request in requests
        ]
        decision = super().place_group(exact, workers)
        decision.strategy = self.name
        return decision


def build_group_policy(name: str, **kwargs) -> GroupRoutingPolicy:
    aliases = {
        "rr": RoundRobinGroupPolicy,
        "round-robin": RoundRobinGroupPolicy,
        "least-load": LeastQueuedTokensPolicy,
        "least-queued-tokens": LeastQueuedTokensPolicy,
        "least-requests": LeastRequestsGroupPolicy,
        "cache-aware": CacheAwareGroupPolicy,
        "cache-aware-group": CacheAwareGroupPolicy,
        "power-of-two": PowerOfTwoGroupPolicy,
        "fixed-pack": FixedPackPolicy,
        "pack": FixedPackPolicy,
        "fixed-even-split": FixedEvenSplitPolicy,
        "split": FixedEvenSplitPolicy,
        "load-proportional-split": LoadProportionalSplitPolicy,
        "adaptive": AdaptiveGroupPolicy,
        "adaptive-group": AdaptiveGroupPolicy,
        "offline-oracle": OfflineOraclePolicy,
    }
    try:
        return aliases[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown group routing policy: {name}") from exc
