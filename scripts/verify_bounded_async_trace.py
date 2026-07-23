#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def verify_trace(
    rows: list[dict[str, Any]],
    *,
    expected_batches: int | None = None,
    max_policy_lag: int = 1,
    max_buffered_batches: int = 1,
    require_overlap: bool = False,
    expected_prefetch_failures: int | None = None,
    expected_strict_fallbacks: int | None = None,
    require_fault_injection: bool = False,
    require_actor_death_recovery: bool = False,
    require_network_partition_recovery: bool = False,
) -> dict[str, Any]:
    failures = [row for row in rows if row.get("event") == "pipeline_failed"]
    if failures:
        raise ValueError(f"pipeline reported failure: {failures[-1].get('error', 'unknown')}")
    completions = [row for row in rows if row.get("event") == "pipeline_complete"]
    if len(completions) != 1:
        raise ValueError(f"expected exactly one pipeline_complete event, found {len(completions)}")

    consumed = [row for row in rows if row.get("event") == "batch_consumed"]
    if expected_batches is not None and len(consumed) != expected_batches:
        raise ValueError(f"expected {expected_batches} consumed batches, found {len(consumed)}")
    rollout_ids = [int(row["rollout_id"]) for row in consumed]
    duplicates = [key for key, count in Counter(rollout_ids).items() if count != 1]
    if duplicates:
        raise ValueError(f"rollout batches were consumed more than once: {duplicates}")
    if rollout_ids != sorted(rollout_ids):
        raise ValueError("rollout batches were not consumed in FIFO order")

    lags = [int(row["policy_lag"]) for row in consumed]
    if any(lag < 0 or lag > max_policy_lag for lag in lags):
        raise ValueError(f"policy lag bound violated: {lags}")
    for row in consumed:
        generated = int(row["generated_policy_version"])
        consumed_version = int(row["consumed_policy_version"])
        if consumed_version - generated != int(row["policy_lag"]):
            raise ValueError(f"inconsistent version accounting for rollout {row['rollout_id']}")

    completion = completions[0]
    observed_buffer = int(completion.get("max_buffered_batches", 0))
    if observed_buffer > max_buffered_batches:
        raise ValueError(
            f"buffer capacity violated: observed={observed_buffer}, allowed={max_buffered_batches}"
        )
    if int(completion.get("max_observed_lag", 0)) > max_policy_lag:
        raise ValueError("summary max_observed_lag exceeds the acceptance bound")
    if float(completion.get("fresh_token_throughput", 0.0)) <= 0:
        raise ValueError("fresh token throughput was not positive")

    prefetches = [row for row in rows if row.get("event") == "prefetch_started"]
    positive_lag = sum(lag > 0 for lag in lags)
    if require_overlap and (not prefetches or positive_lag == 0):
        raise ValueError("trace does not prove one-step rollout/training overlap")

    prefetch_failures = int(completion.get("prefetch_failures", 0))
    strict_fallbacks = int(completion.get("strict_fallbacks", 0))
    if expected_prefetch_failures is not None and prefetch_failures != expected_prefetch_failures:
        raise ValueError(
            f"expected {expected_prefetch_failures} prefetch failures, found {prefetch_failures}"
        )
    if expected_strict_fallbacks is not None and strict_fallbacks != expected_strict_fallbacks:
        raise ValueError(
            f"expected {expected_strict_fallbacks} strict fallbacks, found {strict_fallbacks}"
        )
    injected = [row for row in rows if row.get("event") == "fault_injected"]
    if require_fault_injection and len(injected) != 1:
        raise ValueError(f"expected exactly one fault_injected event, found {len(injected)}")

    def require_chaos_sequence(events: tuple[str, ...]) -> tuple[float, float]:
        matches = [[row for row in rows if row.get("event") == event] for event in events]
        counts = [len(found) for found in matches]
        if counts != [1] * len(events):
            raise ValueError(
                f"expected exactly one {'/'.join(events)} event, found {counts}"
            )
        indexes = [rows.index(found[0]) for found in matches]
        if indexes != sorted(indexes):
            raise ValueError(f"chaos events are out of order: {'/'.join(events)}")
        def timestamp_seconds(row: dict[str, Any]) -> float:
            if "timestamp_s" in row:
                return float(row["timestamp_s"])
            return float(row.get("timestamp_ns", 0)) / 1_000_000_000

        injected_s = timestamp_seconds(matches[0][0])
        recovered_s = timestamp_seconds(matches[-1][0])
        if injected_s and recovered_s and recovered_s <= injected_s:
            raise ValueError(f"non-positive recovery interval for {events[0]}")
        return injected_s, recovered_s

    actor_death_mttr = None
    if require_actor_death_recovery:
        injected_s, recovered_s = require_chaos_sequence(
            (
                "actor_death_injected",
                "actor_death_confirmed",
                "actor_replacement_ready",
                "actor_death_recovered",
            )
        )
        if injected_s and recovered_s:
            actor_death_mttr = recovered_s - injected_s

    network_partition_mttr = None
    if require_network_partition_recovery:
        injected_s, recovered_s = require_chaos_sequence(
            (
                "network_partition_started",
                "network_partition_ended",
                "network_partition_recovered",
            )
        )
        if injected_s and recovered_s:
            network_partition_mttr = recovered_s - injected_s

    return {
        "status": "accepted",
        "consumed_batches": len(consumed),
        "accepted_tokens": int(completion.get("accepted_tokens", 0)),
        "fresh_token_throughput": float(completion["fresh_token_throughput"]),
        "max_observed_lag": max(lags, default=0),
        "max_buffered_batches": observed_buffer,
        "prefetches_started": len(prefetches),
        "strict_fallbacks": strict_fallbacks,
        "prefetch_failures": prefetch_failures,
        "fault_injections": len(injected),
        "positive_lag_batches": positive_lag,
        "final_policy_version": int(completion["final_policy_version"]),
        "actor_death_recovered": require_actor_death_recovery,
        "actor_death_mttr_seconds": actor_death_mttr,
        "network_partition_recovered": require_network_partition_recovery,
        "network_partition_mttr_seconds": network_partition_mttr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a completed bounded-async slime trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int)
    parser.add_argument("--max-policy-lag", type=int, default=1)
    parser.add_argument("--max-buffered-batches", type=int, default=1)
    parser.add_argument("--require-overlap", action="store_true")
    parser.add_argument("--expected-prefetch-failures", type=int)
    parser.add_argument("--expected-strict-fallbacks", type=int)
    parser.add_argument("--require-fault-injection", action="store_true")
    parser.add_argument("--require-actor-death-recovery", action="store_true")
    parser.add_argument("--require-network-partition-recovery", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = verify_trace(
        rows,
        expected_batches=args.expected_batches,
        max_policy_lag=args.max_policy_lag,
        max_buffered_batches=args.max_buffered_batches,
        require_overlap=args.require_overlap,
        expected_prefetch_failures=args.expected_prefetch_failures,
        expected_strict_fallbacks=args.expected_strict_fallbacks,
        require_fault_injection=args.require_fault_injection,
        require_actor_death_recovery=args.require_actor_death_recovery,
        require_network_partition_recovery=args.require_network_partition_recovery,
    )
    rendered = json.dumps(result, sort_keys=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
