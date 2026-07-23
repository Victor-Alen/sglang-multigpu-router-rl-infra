from __future__ import annotations

import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RLRouterMetrics:
    started_at: float = field(default_factory=time.time)
    groups_routed: int = 0
    groups_completed: int = 0
    groups_failed: int = 0
    groups_dropped: int = 0
    mixed_version_rejections: int = 0
    generated_tokens: int = 0
    fresh_tokens: int = 0
    duplicated_prefill_tokens: int = 0
    weight_updates_completed: int = 0
    weight_updates_failed: int = 0
    weight_update_seconds_sum: float = 0.0
    weight_update_drain_seconds_sum: float = 0.0
    routing_decision_seconds_sum: float = 0.0
    group_completion_seconds: list[float] = field(default_factory=list)
    strategy_counts: Counter = field(default_factory=Counter)
    worker_request_counts: Counter = field(default_factory=Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_decision(self, strategy: str, assignments: Dict[str, list[str]], latency_ms: float, duplicated: int) -> None:
        with self.lock:
            self.groups_routed += 1
            self.strategy_counts[strategy] += 1
            self.routing_decision_seconds_sum += latency_ms / 1000
            self.duplicated_prefill_tokens += duplicated
            for worker_id, requests in assignments.items():
                self.worker_request_counts[worker_id] += len(requests)

    def record_group(self, status: str, completion_seconds: float, tokens: int, policy_lag: int, max_lag: int) -> None:
        with self.lock:
            if status == "READY":
                self.groups_completed += 1
                self.group_completion_seconds.append(completion_seconds)
                self.generated_tokens += tokens
                if policy_lag <= max_lag:
                    self.fresh_tokens += tokens
            elif status == "DROPPED":
                self.groups_dropped += 1
            else:
                self.groups_failed += 1

    def record_weight_update(self, success: bool, total_seconds: float, drain_seconds: float) -> None:
        with self.lock:
            if success:
                self.weight_updates_completed += 1
            else:
                self.weight_updates_failed += 1
            self.weight_update_seconds_sum += total_seconds
            self.weight_update_drain_seconds_sum += drain_seconds

    @staticmethod
    def _quantile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]

    def snapshot(self) -> dict:
        with self.lock:
            wall = max(1e-9, time.time() - self.started_at)
            loads = list(self.worker_request_counts.values())
            skew = max(loads, default=0) / max(1.0, sum(loads) / max(1, len(loads)))
            return {
                "uptime_seconds": wall,
                "groups_routed": self.groups_routed,
                "groups_completed": self.groups_completed,
                "groups_failed": self.groups_failed,
                "groups_dropped": self.groups_dropped,
                "mixed_version_rejections": self.mixed_version_rejections,
                "group_goodput": self.groups_completed / wall,
                "fresh_token_throughput": self.fresh_tokens / wall,
                "generated_tokens": self.generated_tokens,
                "fresh_tokens": self.fresh_tokens,
                "duplicated_prefill_tokens": self.duplicated_prefill_tokens,
                "weight_updates_completed": self.weight_updates_completed,
                "weight_updates_failed": self.weight_updates_failed,
                "weight_update_seconds_sum": self.weight_update_seconds_sum,
                "weight_update_drain_seconds_sum": self.weight_update_drain_seconds_sum,
                "group_completion_seconds_p50": self._quantile(self.group_completion_seconds, 0.5),
                "group_completion_seconds_p95": self._quantile(self.group_completion_seconds, 0.95),
                "group_completion_seconds_p99": self._quantile(self.group_completion_seconds, 0.99),
                "worker_skew": skew,
                "strategy_counts": dict(self.strategy_counts),
                "worker_request_counts": dict(self.worker_request_counts),
            }

    def prometheus(self) -> str:
        values = self.snapshot()
        scalar_names = {
            "groups_routed": "rl_router_groups_routed_total",
            "groups_completed": "rl_router_groups_completed_total",
            "groups_failed": "rl_router_groups_failed_total",
            "groups_dropped": "rl_router_groups_dropped_total",
            "mixed_version_rejections": "rl_router_mixed_version_rejections_total",
            "group_goodput": "rl_router_group_goodput",
            "fresh_token_throughput": "rl_router_fresh_token_throughput",
            "duplicated_prefill_tokens": "rl_router_duplicated_prefill_tokens_total",
            "weight_updates_completed": "rl_router_weight_updates_completed_total",
            "weight_updates_failed": "rl_router_weight_updates_failed_total",
            "weight_update_seconds_sum": "rl_router_weight_update_seconds_sum",
            "weight_update_drain_seconds_sum": "rl_router_weight_update_drain_seconds_sum",
            "worker_skew": "rl_router_worker_skew",
            "group_completion_seconds_p50": "rl_router_group_completion_seconds_p50",
            "group_completion_seconds_p95": "rl_router_group_completion_seconds_p95",
            "group_completion_seconds_p99": "rl_router_group_completion_seconds_p99",
        }
        lines = [f"{metric} {values[key]}" for key, metric in scalar_names.items()]
        lines.extend(
            f'rl_router_strategy_groups_total{{strategy="{key}"}} {value}'
            for key, value in values["strategy_counts"].items()
        )
        lines.extend(
            f'rl_router_worker_requests_total{{worker="{key}"}} {value}'
            for key, value in values["worker_request_counts"].items()
        )
        return "\n".join(lines) + "\n"
