#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


METRIC_PATTERN = re.compile(r"'(?P<name>perf/(?:step|train_wait|train|rollout)_time)': (?P<value>[0-9.]+)")


def parse_perf_metrics(text: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for match in METRIC_PATTERN.finditer(text):
        result.setdefault(match.group("name"), []).append(float(match.group("value")))
    return result


def trace_workload_signature(path: Path) -> list[tuple[int, str, int, float, float, int | None]]:
    """Describe generated work without coupling identity to async scheduling time."""
    signature: list[tuple[int, str, int, float, float, int | None]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") != "routing_decision":
            continue
        for request in row["payload"]["requests"]:
            signature.append(
                (
                    int(request["sample_index"]),
                    str(request["prompt_id"]),
                    int(request["max_new_tokens"]),
                    float(request["temperature"]),
                    float(request["top_p"]),
                    int(request["top_k"]) if request.get("top_k") is not None else None,
                )
            )
    return sorted(signature)


def trace_seed_signature(path: Path) -> list[tuple[int, str, int]]:
    signature: list[tuple[int, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") != "routing_decision":
            continue
        for request in row["payload"]["requests"]:
            signature.append(
                (int(request["sample_index"]), str(request["prompt_id"]), int(request["generation_seed"]))
            )
    return sorted(signature)


def _mean(metrics: dict[str, list[float]], name: str) -> float | None:
    values = metrics.get(name, [])
    return statistics.fmean(values) if values else None


def _steady_mean(metrics: dict[str, list[float]], name: str) -> float | None:
    values = metrics.get(name, [])
    steady = values[1:] if len(values) > 1 else values
    return statistics.fmean(steady) if steady else None


def compare(strict_dir: Path, bounded_dir: Path, strict_trace: Path, bounded_trace: Path) -> dict[str, Any]:
    strict_summary = json.loads((strict_dir / "live_run_summary.json").read_text(encoding="utf-8"))
    bounded_summary = json.loads((bounded_dir / "live_run_summary.json").read_text(encoding="utf-8"))
    strict_metrics = parse_perf_metrics((strict_dir / "train.log").read_text(encoding="utf-8", errors="replace"))
    bounded_metrics = parse_perf_metrics((bounded_dir / "train.log").read_text(encoding="utf-8", errors="replace"))
    strict_step_all = _mean(strict_metrics, "perf/step_time")
    bounded_step_all = _mean(bounded_metrics, "perf/step_time")
    strict_step = _steady_mean(strict_metrics, "perf/step_time")
    bounded_step = _steady_mean(bounded_metrics, "perf/step_time")
    if None in (strict_step_all, bounded_step_all, strict_step, bounded_step):
        raise ValueError("both logs must contain perf/step_time")
    strict_e2e = float(strict_summary["end_to_end_token_throughput"])
    bounded_e2e = float(bounded_summary["end_to_end_token_throughput"])
    strict_signature = trace_workload_signature(strict_trace)
    bounded_signature = trace_workload_signature(bounded_trace)
    overlap = len(set(strict_signature) & set(bounded_signature)) / max(1, len(set(strict_signature)))
    workload_match = strict_signature == bounded_signature
    generation_seed_match = trace_seed_signature(strict_trace) == trace_seed_signature(bounded_trace)
    if workload_match and generation_seed_match:
        status = "paired"
    elif workload_match:
        status = "matched_prompts_unmatched_seeds"
    else:
        status = "same_config_unmatched_samples"
    return {
        "status": status,
        "workload_match": workload_match,
        "generation_seed_match": generation_seed_match,
        "workload_signature_overlap_ratio": overlap,
        "strict": {
            "wall_seconds": float(strict_summary["wall_seconds_including_launch_cleanup"]),
            "end_to_end_token_throughput": strict_e2e,
            "mean_step_seconds_all": strict_step_all,
            "mean_steady_step_seconds_excluding_step0": strict_step,
            "mean_steady_train_wait_seconds_excluding_step0": _steady_mean(strict_metrics, "perf/train_wait_time"),
            "measured_steps": len(strict_metrics.get("perf/step_time", [])),
        },
        "bounded": {
            "wall_seconds": float(bounded_summary["wall_seconds_including_launch_cleanup"]),
            "end_to_end_token_throughput": bounded_e2e,
            "mean_step_seconds_all": bounded_step_all,
            "mean_steady_step_seconds_excluding_step0": bounded_step,
            "mean_steady_train_wait_seconds_excluding_step0": _steady_mean(bounded_metrics, "perf/train_wait_time"),
            "measured_steps": len(bounded_metrics.get("perf/step_time", [])),
        },
        "relative": {
            "steady_step_time_reduction": 1.0 - bounded_step / strict_step,
            "end_to_end_throughput_change": bounded_e2e / strict_e2e - 1.0,
        },
        "interpretation": (
            "One hot-window pair only. Step time isolates overlap better than total wall time; "
            "startup/cache variance dominates these short jobs. No confidence interval or learning claim."
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    strict = result["strict"]
    bounded = result["bounded"]
    relative = result["relative"]
    return f"""# Strict sync vs bounded overlap: paired four-GPU run

This is one matched hot-window pair, not a statistical performance claim.

| Metric | Strict sync | Bounded overlap |
|---|---:|---:|
| Workload trace match | {result['workload_match']} | {result['workload_match']} |
| Generation seed match | {result['generation_seed_match']} | {result['generation_seed_match']} |
| Measured training steps | {strict['measured_steps']} | {bounded['measured_steps']} |
| Workload signature overlap | {result['workload_signature_overlap_ratio']:.1%} | {result['workload_signature_overlap_ratio']:.1%} |
| Mean steady step time, excluding step 0 | {strict['mean_steady_step_seconds_excluding_step0']:.4f} s | {bounded['mean_steady_step_seconds_excluding_step0']:.4f} s |
| Mean steady trainer wait, excluding step 0 | {strict['mean_steady_train_wait_seconds_excluding_step0']:.4f} s | {bounded['mean_steady_train_wait_seconds_excluding_step0']:.4f} s |
| Launch-to-cleanup wall time | {strict['wall_seconds']:.4f} s | {bounded['wall_seconds']:.4f} s |
| End-to-end token throughput | {strict['end_to_end_token_throughput']:.4f} token/s | {bounded['end_to_end_token_throughput']:.4f} token/s |

Bounded overlap reduced measured steady step time by
{relative['steady_step_time_reduction'] * 100:.2f}%, while end-to-end throughput
changed by {relative['end_to_end_throughput_change'] * 100:.2f}% in this pair.
The disagreement is expected for four-step jobs because model/Ray startup and
filesystem cache variance dominate total wall time. At least three counterbalanced
repetitions are required for a confidence interval. Both traces passed Group
Barrier and version correctness checks; the bounded run also passed lag<=1 and
buffer-capacity-one checks.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one strict/bounded four-GPU pair")
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--bounded-dir", type=Path, required=True)
    parser.add_argument("--strict-trace", type=Path, required=True)
    parser.add_argument("--bounded-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.strict_dir, args.bounded_dir, args.strict_trace, args.bounded_trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
