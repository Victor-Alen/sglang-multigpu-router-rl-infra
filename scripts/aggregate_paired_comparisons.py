#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


T_975 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.3646242516,
    8: 2.3060041352,
    9: 2.2621571629,
    10: 2.2281388520,
}


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        raise ValueError("at least one value is required")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "sample_stddev": None, "ci95_low": None, "ci95_high": None}
    stddev = statistics.stdev(values)
    df = len(values) - 1
    critical = T_975.get(df, 1.96 if df >= 30 else 2.0)
    half_width = critical * stddev / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_stddev": stddev,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("at least two paired comparisons are required")
    pairs = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "paired":
            raise ValueError(f"unpaired comparison: {path}")
        pairs.append(
            {
                "path": str(path),
                "steady_step_time_reduction": float(result["relative"]["steady_step_time_reduction"]),
                "end_to_end_throughput_change": float(result["relative"]["end_to_end_throughput_change"]),
                "strict_steady_step_seconds": float(result["strict"]["mean_steady_step_seconds_excluding_step0"]),
                "bounded_steady_step_seconds": float(result["bounded"]["mean_steady_step_seconds_excluding_step0"]),
                "strict_end_to_end_token_throughput": float(result["strict"]["end_to_end_token_throughput"]),
                "bounded_end_to_end_token_throughput": float(result["bounded"]["end_to_end_token_throughput"]),
            }
        )
    return {
        "status": "descriptive_paired_summary",
        "pair_count": len(pairs),
        "all_workloads_and_seeds_matched": True,
        "pairs": pairs,
        "steady_step_time_reduction": summarize([p["steady_step_time_reduction"] for p in pairs]),
        "end_to_end_throughput_change": summarize([p["end_to_end_throughput_change"] for p in pairs]),
        "interpretation": (
            f"Student-t 95% intervals over {len(pairs)} adjacent paired windows. "
            "This is still a small runtime sample; the interval describes repeatability "
            "and is not a learning-quality claim."
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    rows = []
    for index, pair in enumerate(result["pairs"], start=1):
        rows.append(
            f"| {index} | {pair['strict_steady_step_seconds']:.4f} | "
            f"{pair['bounded_steady_step_seconds']:.4f} | "
            f"{pair['steady_step_time_reduction'] * 100:.2f}% | "
            f"{pair['end_to_end_throughput_change'] * 100:.2f}% |"
        )
    step = result["steady_step_time_reduction"]
    e2e = result["end_to_end_throughput_change"]
    return f"""# Counterbalanced strict vs bounded four-GPU validation

All {result['pair_count']} adjacent pairs matched prompts, decode parameters and generation seeds.

| Pair | Strict steady step (s) | Bounded steady step (s) | Step reduction | E2E throughput change |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

- Mean steady-step reduction: {step['mean'] * 100:.2f}%
- 95% paired interval: [{step['ci95_low'] * 100:.2f}%, {step['ci95_high'] * 100:.2f}%]
- Mean end-to-end throughput change: {e2e['mean'] * 100:.2f}%
- 95% paired interval: [{e2e['ci95_low'] * 100:.2f}%, {e2e['ci95_high'] * 100:.2f}%]

These are Student-t intervals over a small runtime sample, not learning-quality evidence.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired strict/bounded comparisons")
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
