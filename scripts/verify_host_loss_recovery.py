#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify_host_loss(
    checkpoint: dict[str, Any],
    *,
    phase1_exit_code: int,
    interrupted_ns: int,
    recovered_ns: int,
    max_recovery_seconds: float,
) -> dict[str, Any]:
    if phase1_exit_code == 0:
        raise ValueError("interrupted Ray job unexpectedly exited successfully")
    if checkpoint.get("status") != "accepted":
        raise ValueError("checkpoint cold-resume evidence was not accepted")
    if recovered_ns <= interrupted_ns:
        raise ValueError("recovery completion did not follow interruption")
    recovery_seconds = (recovered_ns - interrupted_ns) / 1_000_000_000
    if recovery_seconds > max_recovery_seconds:
        raise ValueError(
            f"host-loss recovery exceeded SLO: {recovery_seconds:.3f}s > {max_recovery_seconds:.3f}s"
        )
    return {
        "status": "accepted",
        "failure_scope": "private_single_node_ray_control_plane_and_job",
        "phase1_exit_code": phase1_exit_code,
        "recovery_seconds": recovery_seconds,
        "max_recovery_seconds": max_recovery_seconds,
        "checkpoint_iteration": checkpoint["first"]["iteration"],
        "final_iteration": checkpoint["final"]["iteration"],
        "state_restored": bool(checkpoint["model_optimizer_scheduler_rng_restored"]),
        "dataset_cursor_restored": bool(checkpoint["dataset_cursor_restored"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cold recovery after private Ray host loss")
    parser.add_argument("--checkpoint-acceptance", type=Path, required=True)
    parser.add_argument("--phase1-exit-code", type=int, required=True)
    parser.add_argument("--interrupted-ns", type=int, required=True)
    parser.add_argument("--recovered-ns", type=int, required=True)
    parser.add_argument("--max-recovery-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint_acceptance.read_text(encoding="utf-8"))
    result = verify_host_loss(
        checkpoint,
        phase1_exit_code=args.phase1_exit_code,
        interrupted_ns=args.interrupted_ns,
        recovered_ns=args.recovered_ns,
        max_recovery_seconds=args.max_recovery_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
