#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.rl.simulator import OfflineGroupSimulator, SimulationScenario, bootstrap_mean_ci


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/rl_fixed_weight_matrix.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rl_offline_matrix"))
    parser.add_argument("--groups-per-run", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_runs.jsonl"
    simulator = OfflineGroupSimulator()
    rows = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for policy in config["policies"]:
            for group_size in config["group_sizes"]:
                for prompt_bucket in config["prompt_token_buckets"]:
                    for output_bucket in config["output_token_buckets"]:
                        for repetition in range(config.get("repetitions", 1)):
                            for seed in config["seeds"]:
                                effective_seed = int(seed) + repetition * 100_003
                                scenario = SimulationScenario(
                                    policy=policy,
                                    group_size=int(group_size),
                                    prompt_bucket=tuple(prompt_bucket),
                                    output_bucket=tuple(output_bucket),
                                    seed=effective_seed,
                                    groups=args.groups_per_run,
                                )
                                result = simulator.run(scenario)
                                row = result.summary()
                                row["repetition"] = repetition
                                rows.append(row)
                                raw.write(json.dumps(row, sort_keys=True) + "\n")

    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        by_policy.setdefault(row["scenario"]["policy"], []).append(row)
    summary = {"schema_version": 1, "runs": len(rows), "policies": {}}
    for policy, policy_rows in sorted(by_policy.items()):
        p95_values = [row["group_completion_seconds_p95"] for row in policy_rows]
        ci_low, ci_high = bootstrap_mean_ci(p95_values, seed=17)
        summary["policies"][policy] = {
            "runs": len(policy_rows),
            "group_p95_mean": statistics.fmean(p95_values),
            "group_p95_bootstrap_ci95": [ci_low, ci_high],
            "group_goodput_mean": statistics.fmean(row["group_goodput"] for row in policy_rows),
            "fresh_token_throughput_mean": statistics.fmean(
                row["fresh_token_throughput"] for row in policy_rows
            ),
            "duplicated_prefill_ratio_mean": statistics.fmean(
                row["duplicated_prefill_ratio"] for row in policy_rows
            ),
            "worker_skew_mean": statistics.fmean(row["worker_skew"] for row in policy_rows),
            "router_decision_ms_mean": statistics.fmean(
                row["router_decision_ms_mean"] for row in policy_rows
            ),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ranked = sorted(summary["policies"].items(), key=lambda item: item[1]["group_p95_mean"])
    lines = [
        "# Offline RL rollout routing matrix",
        "",
        f"Runs: {summary['runs']}. These are deterministic discrete-event screening results, not GPU measurements.",
        "",
        "| Policy | Mean group p95 (s) | Mean goodput | Dup prefill ratio | Worker skew | Decision ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, metrics in ranked:
        lines.append(
            f"| {policy} | {metrics['group_p95_mean']:.4f} | "
            f"{metrics['group_goodput_mean']:.4f} | {metrics['duplicated_prefill_ratio_mean']:.4f} | "
            f"{metrics['worker_skew_mean']:.4f} | {metrics['router_decision_ms_mean']:.4f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(rows), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
