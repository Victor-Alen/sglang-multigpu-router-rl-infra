#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required production evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required production trace is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _events(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("event") == name]


def verify_production_gate(
    config: dict[str, Any],
    *,
    automated_tests: int,
    actor_death: dict[str, Any],
    actor_trace: list[dict[str, Any]],
    network_partition: dict[str, Any],
    network_trace: list[dict[str, Any]],
    host_loss: dict[str, Any],
    long_soak: dict[str, Any],
) -> dict[str, Any]:
    slo = config["production"]
    actor_mttr = float(actor_death.get("actor_death_mttr_seconds") or float("inf"))
    network_mttr = float(
        network_partition.get("network_partition_mttr_seconds") or float("inf")
    )
    host_recovery = float(host_loss.get("recovery_seconds") or float("inf"))
    memory_growth = {
        int(gpu): float(value)
        for gpu, value in long_soak.get("steady_memory_growth_mib", {}).items()
    }
    actor_injected = _events(actor_trace, "actor_death_injected")
    actor_confirmed = _events(actor_trace, "actor_death_confirmed")
    actor_recovered = _events(actor_trace, "actor_death_recovered")
    actor_target_consistent = (
        len(actor_injected) == len(actor_confirmed) == len(actor_recovered) == 1
        and actor_injected[0].get("engine_index") == actor_confirmed[0].get("engine_index")
        == actor_recovered[0].get("engine_index")
    )
    network_started = _events(network_trace, "network_partition_started")
    network_ended = _events(network_trace, "network_partition_ended")
    network_recovered = _events(network_trace, "network_partition_recovered")
    network_target_consistent = (
        len(network_started) == len(network_ended) == len(network_recovered) == 1
        and all(
            network_started[0].get(key) == network_ended[0].get(key)
            for key in ("host", "port", "label")
        )
    )
    checkpoint_iteration = int(host_loss.get("checkpoint_iteration", -1))
    final_iteration = int(host_loss.get("final_iteration", -1))
    checks = {
        "automated_tests": automated_tests >= int(slo["min_automated_tests"]),
        "actor_death_status": actor_death.get("status") == "accepted",
        "actor_death_recovered": actor_death.get("actor_death_recovered") is True,
        "actor_death_mttr": actor_mttr <= float(slo["max_actor_death_mttr_seconds"]),
        "actor_trace_sequence": actor_target_consistent,
        "network_partition_status": network_partition.get("status") == "accepted",
        "network_partition_recovered": network_partition.get("network_partition_recovered")
        is True,
        "network_partition_mttr": network_mttr
        <= float(slo["max_network_partition_mttr_seconds"]),
        "network_trace_sequence": network_target_consistent,
        "network_partition_duration": network_target_consistent
        and float(network_started[0].get("duration_seconds", 0))
        >= float(slo["min_network_partition_seconds"]),
        "host_loss_status": host_loss.get("status") == "accepted",
        "host_failure_scope": host_loss.get("failure_scope")
        == slo["required_host_failure_scope"],
        "host_phase1_interrupted": int(host_loss.get("phase1_exit_code", 0)) != 0,
        "host_iteration_advanced": final_iteration - checkpoint_iteration
        >= int(slo["min_host_recovery_iteration_advance"]),
        "host_state_restored": bool(host_loss.get("state_restored"))
        if slo.get("require_host_state_restore", True)
        else True,
        "host_dataset_cursor_restored": bool(host_loss.get("dataset_cursor_restored"))
        if slo.get("require_dataset_cursor_restore", True)
        else True,
        "host_loss_recovery": host_recovery <= float(slo["max_host_loss_recovery_seconds"]),
        "long_soak_status": long_soak.get("status") == "accepted",
        "long_soak_duration": float(long_soak.get("wall_seconds", 0))
        >= float(slo["min_soak_seconds"]),
        "long_soak_batches": int(long_soak.get("consumed_batches", 0))
        >= int(slo["min_soak_batches"]),
        "terminal_throughput": float(long_soak.get("terminal_throughput_ratio", 0))
        >= float(slo["min_terminal_throughput_ratio"]),
        "gpu_temperature": float(long_soak.get("max_temperature_c", float("inf")))
        <= float(slo["max_gpu_temperature_c"]),
        "gpu_memory_growth": bool(memory_growth)
        and max(memory_growth.values()) <= float(slo["max_steady_memory_growth_mib"]),
        "policy_lag": int(long_soak.get("policy_lag_bound", 999))
        <= int(slo["max_policy_lag"]),
        "prefetch_buffer": int(long_soak.get("buffer_bound", 999))
        <= int(slo["max_buffered_batches"]),
        "no_soak_prefetch_failure": int(long_soak.get("prefetch_failures", 999))
        <= int(slo["max_soak_prefetch_failures"]),
        "no_soak_strict_fallback": int(long_soak.get("strict_fallbacks", 999))
        <= int(slo["max_soak_strict_fallbacks"]),
        "no_mixed_policy_version": int(long_soak.get("mixed_version_responses", 999))
        <= int(slo["max_version_mismatches"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "status": "accepted" if not failed else "rejected",
        "checks": checks,
        "failed_checks": failed,
        "automated_tests": automated_tests,
        "actor_death_mttr_seconds": actor_mttr,
        "network_partition_mttr_seconds": network_mttr,
        "host_loss_recovery_seconds": host_recovery,
        "soak_wall_seconds": float(long_soak.get("wall_seconds", 0)),
        "soak_batches": int(long_soak.get("consumed_batches", 0)),
        "terminal_throughput_ratio": float(long_soak.get("terminal_throughput_ratio", 0)),
        "max_temperature_c": float(long_soak.get("max_temperature_c", 0)),
        "steady_memory_growth_mib": memory_growth,
    }
    if failed:
        raise ValueError(f"production gate rejected: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the production evidence gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--automated-tests", type=int, required=True)
    parser.add_argument("--actor-death", type=Path, required=True)
    parser.add_argument("--actor-trace", type=Path, required=True)
    parser.add_argument("--network-partition", type=Path, required=True)
    parser.add_argument("--network-trace", type=Path, required=True)
    parser.add_argument("--host-loss", type=Path, required=True)
    parser.add_argument("--long-soak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_production_gate(
        yaml.safe_load(args.config.read_text(encoding="utf-8")),
        automated_tests=args.automated_tests,
        actor_death=_load(args.actor_death),
        actor_trace=_load_jsonl(args.actor_trace),
        network_partition=_load(args.network_partition),
        network_trace=_load_jsonl(args.network_trace),
        host_loss=_load(args.host_loss),
        long_soak=_load(args.long_soak),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
