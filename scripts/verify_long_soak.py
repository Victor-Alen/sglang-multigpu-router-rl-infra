#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _window_rate(rows: list[dict[str, Any]]) -> tuple[float, float]:
    consumed = [row for row in rows if row.get("event") == "batch_consumed"]
    if len(consumed) < 20:
        raise ValueError("at least 20 consumed batches are required for drift analysis")
    window = max(10, len(consumed) // 10)

    def timestamp_seconds(row: dict[str, Any]) -> float:
        if "timestamp_s" in row:
            return float(row["timestamp_s"])
        return int(row["timestamp_ns"]) / 1_000_000_000

    def rate(part: list[dict[str, Any]]) -> float:
        elapsed = timestamp_seconds(part[-1]) - timestamp_seconds(part[0])
        if elapsed <= 0:
            raise ValueError("non-positive soak trace interval")
        return sum(int(row["generated_tokens"]) for row in part[1:]) / elapsed

    return rate(consumed[:window]), rate(consumed[-window:])


def _trace_end_ns(rows: list[dict[str, Any]]) -> int:
    consumed = [row for row in rows if row.get("event") == "batch_consumed"]
    if not consumed:
        raise ValueError("soak trace contains no consumed batches")
    row = consumed[-1]
    if "timestamp_ns" in row:
        return int(row["timestamp_ns"])
    return int(float(row["timestamp_s"]) * 1_000_000_000)


def verify_long_soak(
    acceptance: dict[str, Any],
    router: dict[str, Any],
    timing: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, str]],
    *,
    expected_batches: int,
    expected_gpus: set[int],
    min_wall_seconds: float,
    min_throughput_ratio: float,
    max_temperature_c: float,
    max_steady_memory_growth_mib: float,
    min_samples_per_gpu: int,
) -> dict[str, Any]:
    wall_seconds = float(timing["wall_seconds"])
    if wall_seconds < min_wall_seconds:
        raise ValueError(f"soak was too short: {wall_seconds:.3f}s < {min_wall_seconds:.3f}s")
    if acceptance.get("status") != "accepted":
        raise ValueError("bounded-async acceptance was not accepted")
    if int(acceptance.get("consumed_batches", -1)) != expected_batches:
        raise ValueError("soak did not consume the expected number of batches")
    if int(acceptance.get("max_observed_lag", 999)) > 1:
        raise ValueError("policy lag exceeded one")
    if int(acceptance.get("max_buffered_batches", 999)) > 1:
        raise ValueError("prefetch buffer exceeded one")
    if int(acceptance.get("prefetch_failures", 999)) != 0:
        raise ValueError("natural prefetch failure occurred during soak")
    if int(acceptance.get("strict_fallbacks", 999)) != 0:
        raise ValueError("strict fallback occurred during soak")
    if int(router.get("mixed_version_responses", 999)) != 0:
        raise ValueError("mixed policy-version response occurred during soak")
    if int(router.get("complete_groups", -1)) != int(router.get("groups", -2)):
        raise ValueError("router trace contains incomplete groups")

    first_rate, last_rate = _window_rate(trace_rows)
    throughput_ratio = last_rate / first_rate
    if throughput_ratio < min_throughput_ratio:
        raise ValueError(
            f"terminal throughput regressed: ratio={throughput_ratio:.3f} < {min_throughput_ratio:.3f}"
        )

    by_gpu: dict[int, list[dict[str, str]]] = defaultdict(list)
    final_batch_ns = _trace_end_ns(trace_rows)
    for row in telemetry_rows:
        # The final checkpoint is written after the last batch is consumed.
        # Exclude those high-water samples: they are checkpoint I/O state, not
        # the steady training workload and have no post-clear counterpart.
        if int(row["timestamp_ns"]) <= final_batch_ns:
            by_gpu[int(row["gpu"])].append(row)
    if set(by_gpu) != expected_gpus:
        raise ValueError(f"telemetry GPU set mismatch: found={sorted(by_gpu)}")

    max_temperature = 0.0
    steady_growth: dict[int, float] = {}
    steady_baselines: dict[int, float] = {}
    steady_terminals: dict[int, float] = {}
    samples: dict[int, int] = {}
    for gpu, rows in by_gpu.items():
        rows.sort(key=lambda row: int(row["timestamp_ns"]))
        samples[gpu] = len(rows)
        if len(rows) < min_samples_per_gpu:
            raise ValueError(f"insufficient telemetry for GPU {gpu}: {len(rows)}")
        max_temperature = max(max_temperature, max(float(row["temperature_c"]) for row in rows))
        window = max(5, len(rows) // 10)
        baseline_start = len(rows) // 4
        baseline = rows[baseline_start : baseline_start + window]
        terminal = rows[-window:]
        baseline_median = median(float(row["memory_used_mib"]) for row in baseline)
        terminal_median = median(float(row["memory_used_mib"]) for row in terminal)
        growth = max(0.0, terminal_median - baseline_median)
        steady_baselines[gpu] = baseline_median
        steady_terminals[gpu] = terminal_median
        steady_growth[gpu] = growth
        if growth > max_steady_memory_growth_mib:
            raise ValueError(
                f"GPU {gpu} steady memory growth is too high: {growth:.1f} MiB"
            )
    if max_temperature > max_temperature_c:
        raise ValueError(f"GPU temperature exceeded limit: {max_temperature:.1f} C")

    return {
        "status": "accepted",
        "wall_seconds": wall_seconds,
        "wall_hours": wall_seconds / 3600,
        "consumed_batches": expected_batches,
        "first_window_tokens_per_second": first_rate,
        "last_window_tokens_per_second": last_rate,
        "terminal_throughput_ratio": throughput_ratio,
        "max_temperature_c": max_temperature,
        "steady_memory_baseline_median_mib": steady_baselines,
        "steady_memory_terminal_median_mib": steady_terminals,
        "steady_memory_growth_mib": steady_growth,
        "telemetry_samples_per_gpu": samples,
        "telemetry_cutoff_timestamp_ns": final_batch_ns,
        "policy_lag_bound": 1,
        "buffer_bound": 1,
        "prefetch_failures": 0,
        "strict_fallbacks": 0,
        "mixed_version_responses": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a multi-hour 4-GPU stability run")
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--router-acceptance", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--bounded-trace", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int, required=True)
    parser.add_argument("--expected-gpus", required=True)
    parser.add_argument("--min-wall-seconds", type=float, default=7200)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.5)
    parser.add_argument("--max-temperature-c", type=float, default=90)
    parser.add_argument("--max-steady-memory-growth-mib", type=float, default=2048)
    parser.add_argument("--min-samples-per-gpu", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    traces = [
        json.loads(line)
        for line in args.bounded_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with args.telemetry.open(encoding="utf-8", newline="") as handle:
        telemetry = list(csv.DictReader(handle))
    result = verify_long_soak(
        _load_json(args.acceptance),
        _load_json(args.router_acceptance),
        _load_json(args.timing),
        traces,
        telemetry,
        expected_batches=args.expected_batches,
        expected_gpus={int(value) for value in args.expected_gpus.split(",")},
        min_wall_seconds=args.min_wall_seconds,
        min_throughput_ratio=args.min_throughput_ratio,
        max_temperature_c=args.max_temperature_c,
        max_steady_memory_growth_mib=args.max_steady_memory_growth_mib,
        min_samples_per_gpu=args.min_samples_per_gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
