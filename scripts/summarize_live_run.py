#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]], *, mode: str, wall_seconds: float) -> dict[str, Any]:
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    responses = [row["payload"] for row in rows if row.get("event") == "response_completed"]
    decisions = [row["payload"] for row in rows if row.get("event") == "routing_decision"]
    generated_tokens = sum(int(row.get("generated_tokens", 0)) for row in responses)
    served_versions = Counter(str(row["served_policy_version"]) for row in responses)
    requested_mismatches = sum(
        row.get("requested_policy_version") != row.get("served_policy_version") for row in responses
    )
    return {
        "mode": mode,
        "wall_seconds_including_launch_cleanup": wall_seconds,
        "groups": len(decisions),
        "responses": len(responses),
        "generated_tokens": generated_tokens,
        "end_to_end_token_throughput": generated_tokens / wall_seconds,
        "served_version_response_counts": dict(sorted(served_versions.items())),
        "requested_served_version_mismatches": requested_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one measured live router run")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = summarize(rows, mode=args.mode, wall_seconds=args.wall_seconds)
    rendered = json.dumps(result, sort_keys=True, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
