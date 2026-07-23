#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_gate(
    config: dict[str, Any],
    *,
    automated_tests: int,
    checkpoint: dict[str, Any],
    fault: dict[str, Any],
    soak: dict[str, Any],
    soak_router: dict[str, Any],
) -> dict[str, Any]:
    slo = config["release_candidate"]
    checks = {
        "automated_tests": automated_tests >= int(slo["min_automated_tests"]),
        "checkpoint_status": checkpoint.get("status") == "accepted",
        "checkpoint_state_restored": bool(checkpoint.get("model_optimizer_scheduler_rng_restored"))
        if slo.get("require_checkpoint_model_optimizer_scheduler_rng", True)
        else True,
        "dataset_cursor_restored": bool(checkpoint.get("dataset_cursor_restored"))
        if slo.get("require_dataset_cursor_restore", True)
        else True,
        "fault_status": fault.get("status") == "accepted",
        "fault_was_injected": int(fault.get("fault_injections", 0)) == 1,
        "fault_prefetch_failure": int(fault.get("prefetch_failures", -1))
        == int(slo["required_fault_prefetch_failures"]),
        "fault_strict_fallback": int(fault.get("strict_fallbacks", -1))
        == int(slo["required_fault_strict_fallbacks"]),
        "soak_status": soak.get("status") == "accepted",
        "soak_batches": int(soak.get("consumed_batches", 0)) >= int(slo["min_soak_batches"]),
        "soak_policy_lag": int(soak.get("max_observed_lag", 999)) <= int(slo["max_policy_lag"]),
        "soak_buffer_bound": int(soak.get("max_buffered_batches", 999))
        <= int(slo["max_buffered_batches"]),
        "soak_no_prefetch_failure": int(soak.get("prefetch_failures", 999))
        <= int(slo["max_soak_prefetch_failures"]),
        "soak_no_fallback": int(soak.get("strict_fallbacks", 999))
        <= int(slo["max_soak_strict_fallbacks"]),
        "router_all_groups_complete": int(soak_router.get("complete_groups", -1))
        == int(soak_router.get("groups", -2)),
        "router_no_mixed_version": int(soak_router.get("mixed_version_responses", 999))
        <= int(slo["max_version_mismatches"]),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "status": "accepted" if not failed else "rejected",
        "checks": checks,
        "failed_checks": failed,
        "automated_tests": automated_tests,
        "soak_batches": int(soak.get("consumed_batches", 0)),
        "max_policy_lag": int(soak.get("max_observed_lag", 0)),
        "max_buffered_batches": int(soak.get("max_buffered_batches", 0)),
    }
    if failed:
        raise ValueError(f"maturity gate rejected: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the production-maturity evidence gate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--automated-tests", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--soak-router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = verify_gate(
        config,
        automated_tests=args.automated_tests,
        checkpoint=_load(args.checkpoint),
        fault=_load(args.fault),
        soak=_load(args.soak),
        soak_router=_load(args.soak_router),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
