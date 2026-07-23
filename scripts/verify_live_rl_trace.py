#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def verify_trace(
    path: Path,
    expected_groups: int | None = None,
    min_policy_versions: int = 1,
    *,
    allow_failed_groups: bool = False,
    expected_complete_groups: int | None = None,
    min_complete_groups: int | None = None,
) -> dict:
    decisions = {}
    expected_sizes = {}
    responses = defaultdict(dict)
    strategy_counts = Counter()
    candidate_counts = Counter()
    failed_groups = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        event = record.get("event")
        payload = record.get("payload", {})
        if event == "routing_decision":
            decision = payload["decision"]
            group_id = decision["group_id"]
            if group_id in decisions:
                raise ValueError(f"duplicate routing decision for {group_id}")
            decisions[group_id] = decision
            expected_sizes[group_id] = len(payload["requests"])
            strategy_counts[decision["strategy"]] += 1
            selected = decision.get("reason", {}).get("selected_candidate")
            if selected:
                candidate_counts[selected] += 1
        elif event == "response_completed":
            group_id = payload["group_id"]
            index = int(payload["sample_index"])
            if index in responses[group_id]:
                raise ValueError(f"duplicate completed sample {group_id}:{index}")
            responses[group_id][index] = payload
        elif event == "slime_sample_failed":
            failed_groups.add(payload["group_id"])

    if expected_groups is not None and len(decisions) != expected_groups:
        raise ValueError(f"expected {expected_groups} groups, found {len(decisions)}")
    incomplete = {
        group_id: {"expected": expected_sizes[group_id], "completed": len(responses[group_id])}
        for group_id in decisions
        if len(responses[group_id]) != expected_sizes[group_id]
    }
    unaccounted_incomplete = {
        group_id: counts for group_id, counts in incomplete.items() if group_id not in failed_groups
    }
    if unaccounted_incomplete or (incomplete and not allow_failed_groups):
        raise ValueError(f"incomplete groups: {incomplete}")
    completed_groups = set(decisions) - set(incomplete)
    if expected_complete_groups is not None and len(completed_groups) != expected_complete_groups:
        raise ValueError(
            f"expected {expected_complete_groups} complete groups, found {len(completed_groups)}"
        )
    if min_complete_groups is not None and len(completed_groups) < min_complete_groups:
        raise ValueError(
            f"expected at least {min_complete_groups} complete groups, found {len(completed_groups)}"
        )
    failed_without_decision = sorted(failed_groups - set(decisions))
    if failed_without_decision:
        raise ValueError(f"failed samples without routing decisions: {failed_without_decision}")
    unexpected = sorted(set(responses) - set(decisions))
    if unexpected:
        raise ValueError(f"responses without routing decisions: {unexpected}")

    versions = Counter()
    mixed = []
    for group_id in completed_groups:
        samples = responses[group_id]
        group_versions = set()
        for response in samples.values():
            requested = int(response["requested_policy_version"])
            served = int(response["served_policy_version"])
            versions[served] += 1
            group_versions.add(served)
            if requested != served:
                mixed.append(
                    {
                        "group_id": group_id,
                        "sample_index": response["sample_index"],
                        "requested": requested,
                        "served": served,
                    }
                )
        if len(group_versions) != 1:
            mixed.append({"group_id": group_id, "served_versions": sorted(group_versions)})
    if mixed:
        raise ValueError(f"mixed policy versions: {mixed}")
    if len(versions) < min_policy_versions:
        raise ValueError(
            f"expected at least {min_policy_versions} served policy versions, found {sorted(versions)}"
        )
    return {
        "groups": len(decisions),
        "responses": sum(len(responses[group_id]) for group_id in completed_groups),
        "complete_groups": len(completed_groups),
        "discarded_groups": len(incomplete),
        "failed_groups": len(failed_groups),
        "mixed_version_responses": 0,
        "served_version_response_counts": dict(sorted(versions.items())),
        "strategy_counts": dict(strategy_counts),
        "selected_candidate_counts": dict(candidate_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int)
    parser.add_argument("--min-policy-versions", type=int, default=1)
    parser.add_argument("--allow-failed-groups", action="store_true")
    parser.add_argument("--expected-complete-groups", type=int)
    parser.add_argument("--min-complete-groups", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify_trace(
        args.trace,
        args.expected_groups,
        args.min_policy_versions,
        allow_failed_groups=args.allow_failed_groups,
        expected_complete_groups=args.expected_complete_groups,
        min_complete_groups=args.min_complete_groups,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
