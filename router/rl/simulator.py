from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import asdict, dataclass

from .models import RolloutRequest, RolloutWorkerState, WorkerLifecycle
from .routing import build_group_policy


@dataclass(frozen=True)
class SimulationScenario:
    policy: str
    group_size: int
    prompt_bucket: tuple[int, int]
    output_bucket: tuple[int, int]
    seed: int
    groups: int = 64
    arrival_interval_seconds: float = 0.05
    prefix_pool_size: int = 8
    prefill_tokens_per_second: float = 2500.0
    decode_tokens_per_second: float = 100.0


@dataclass
class SimulatedGroup:
    group_id: str
    completion_seconds: float
    generated_tokens: int
    duplicated_prefill_tokens: int
    total_prefill_tokens: int
    decision_latency_ms: float
    selected_candidate: str | None
    worker_counts: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResult:
    scenario: SimulationScenario
    groups: list[SimulatedGroup]
    wall_seconds: float

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    def summary(self) -> dict:
        latencies = [group.completion_seconds for group in self.groups]
        generated = sum(group.generated_tokens for group in self.groups)
        duplicated = sum(group.duplicated_prefill_tokens for group in self.groups)
        prefill = sum(group.total_prefill_tokens for group in self.groups)
        worker_counts: dict[str, int] = {}
        for group in self.groups:
            for worker_id, count in group.worker_counts.items():
                worker_counts[worker_id] = worker_counts.get(worker_id, 0) + count
        loads = list(worker_counts.values())
        skew = max(loads, default=0) / max(1.0, statistics.fmean(loads) if loads else 1.0)
        duplicate_ratio = duplicated / max(1, prefill)
        return {
            "scenario": asdict(self.scenario),
            "group_completion_seconds_mean": statistics.fmean(latencies),
            "group_completion_seconds_p50": self._quantile(latencies, 0.50),
            "group_completion_seconds_p95": self._quantile(latencies, 0.95),
            "group_completion_seconds_p99": self._quantile(latencies, 0.99),
            "group_goodput": len(self.groups) / max(self.wall_seconds, 1e-9),
            "fresh_token_throughput": generated / max(self.wall_seconds, 1e-9),
            "cache_efficient_group_goodput": (
                len(self.groups) / max(self.wall_seconds * (1 + duplicate_ratio), 1e-9)
            ),
            "duplicated_prefill_tokens": duplicated,
            "duplicated_prefill_ratio": duplicate_ratio,
            "worker_skew": skew,
            "router_decision_ms_mean": statistics.fmean(
                group.decision_latency_ms for group in self.groups
            ),
            "generated_tokens": generated,
            "wall_seconds": self.wall_seconds,
        }


class OfflineGroupSimulator:
    """Small deterministic discrete-event simulator for policy screening."""

    def run(self, scenario: SimulationScenario) -> SimulationResult:
        if scenario.group_size <= 0 or scenario.groups <= 0:
            raise ValueError("group_size and groups must be positive")
        rng = random.Random(scenario.seed)
        policy = build_group_policy(scenario.policy)
        now_ns = time.time_ns()
        workers = [
            RolloutWorkerState(
                worker_id=f"w{index}",
                endpoint=f"http://worker-{index}",
                lifecycle=WorkerLifecycle.READY,
                loaded_policy_version=0,
                prefill_tokens_per_second_ema=scenario.prefill_tokens_per_second,
                decode_tokens_per_second_ema=scenario.decode_tokens_per_second,
                heartbeat_timestamp_ns=now_ns,
            )
            for index in range(2)
        ]
        available_at = {worker.worker_id: 0.0 for worker in workers}
        prefix_cache = {worker.worker_id: set() for worker in workers}
        groups: list[SimulatedGroup] = []

        for group_index in range(scenario.groups):
            arrival = group_index * scenario.arrival_interval_seconds
            prompt_tokens = rng.randint(*scenario.prompt_bucket)
            output_low, output_high = scenario.output_bucket
            output_mid = math.sqrt(output_low * output_high)
            actual_outputs = [
                max(
                    output_low,
                    min(output_high, int(rng.lognormvariate(math.log(output_mid), 0.65))),
                )
                for _ in range(scenario.group_size)
            ]
            predicted_mean = float(output_mid)
            predicted_p90 = min(float(output_high), predicted_mean * 1.5)
            fingerprint = f"prefix-{group_index % max(1, scenario.prefix_pool_size)}"

            for worker in workers:
                remaining = max(0.0, available_at[worker.worker_id] - arrival)
                worker.active_decode_tokens = int(
                    remaining * worker.decode_tokens_per_second_ema
                )
                worker.queued_requests = int(remaining > 0)
                worker.running_requests = int(remaining > 0)
                worker.heartbeat_timestamp_ns = time.time_ns()
                worker.prefix_matched_tokens = {
                    item: prompt_tokens for item in prefix_cache[worker.worker_id]
                }

            group_id = f"sim-{scenario.seed}-{group_index}"
            requests = [
                RolloutRequest(
                    request_id=f"{group_id}:{sample_index}",
                    prompt_id=f"prompt-{group_index}",
                    group_id=group_id,
                    sample_index=sample_index,
                    group_size=scenario.group_size,
                    policy_version=0,
                    rollout_step=0,
                    generation_seed=scenario.seed * 1_000_000 + group_index * 100 + sample_index,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=output_high,
                    predicted_output_tokens=(
                        float(actual_outputs[sample_index])
                        if scenario.policy == "offline-oracle"
                        else predicted_mean
                    ),
                    predicted_output_p90=(
                        float(actual_outputs[sample_index])
                        if scenario.policy == "offline-oracle"
                        else predicted_p90
                    ),
                    prefix_fingerprint=fingerprint,
                    tokenizer_revision="sim-tokenizer",
                    chat_template_hash="sim-template",
                )
                for sample_index in range(scenario.group_size)
            ]
            decision = policy.place_group(requests, workers)
            request_index = {request.request_id: request.sample_index for request in requests}
            finishes = []
            total_prefill = 0
            worker_prefills = []
            worker_counts = {}
            for worker in workers:
                assigned = decision.assignments.get(worker.worker_id, [])
                worker_counts[worker.worker_id] = len(assigned)
                if not assigned:
                    continue
                start = max(arrival, available_at[worker.worker_id])
                prefill_tokens = 0 if fingerprint in prefix_cache[worker.worker_id] else prompt_tokens
                decode_tokens = sum(actual_outputs[request_index[request_id]] for request_id in assigned)
                finish = (
                    start
                    + prefill_tokens / worker.prefill_tokens_per_second_ema
                    + decode_tokens / worker.decode_tokens_per_second_ema
                )
                available_at[worker.worker_id] = finish
                prefix_cache[worker.worker_id].add(fingerprint)
                finishes.append(finish)
                total_prefill += prefill_tokens
                worker_prefills.append(prefill_tokens)
            duplicated = total_prefill - min(worker_prefills, default=0)
            completion = max(finishes) - arrival
            groups.append(
                SimulatedGroup(
                    group_id=group_id,
                    completion_seconds=completion,
                    generated_tokens=sum(actual_outputs),
                    duplicated_prefill_tokens=duplicated,
                    total_prefill_tokens=total_prefill,
                    decision_latency_ms=decision.decision_latency_ms,
                    selected_candidate=decision.reason.get("selected_candidate"),
                    worker_counts=worker_counts,
                )
            )

        wall = max(available_at.values(), default=0.0)
        return SimulationResult(scenario, groups, wall)


def bootstrap_mean_ci(
    values: list[float], confidence: float = 0.95, resamples: int = 500, seed: int = 0
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    low = means[max(0, int(tail * resamples))]
    high = means[min(resamples - 1, int((1 - tail) * resamples))]
    return low, high
