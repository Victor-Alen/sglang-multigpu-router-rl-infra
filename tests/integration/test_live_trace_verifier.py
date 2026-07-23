import json

import pytest

from scripts.verify_live_rl_trace import verify_trace


def write_trace(path, served_versions=(1, 1)):
    records = [
        {
            "event": "routing_decision",
            "payload": {
                "requests": [{"request_id": "g:0"}, {"request_id": "g:1"}],
                "decision": {
                    "group_id": "g",
                    "strategy": "adaptive-group",
                    "reason": {"selected_candidate": "split_even"},
                },
            },
        }
    ]
    for index, version in enumerate(served_versions):
        records.append(
            {
                "event": "response_completed",
                "payload": {
                    "group_id": "g",
                    "sample_index": index,
                    "requested_policy_version": 1,
                    "served_policy_version": version,
                },
            }
        )
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_live_trace_verifier_accepts_complete_homogeneous_group(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path)
    summary = verify_trace(path, expected_groups=1)
    assert summary["mixed_version_responses"] == 0
    assert summary["served_version_response_counts"] == {1: 2}


def test_live_trace_verifier_rejects_mixed_group(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path, served_versions=(1, 2))
    with pytest.raises(ValueError, match="mixed policy"):
        verify_trace(path)


def test_live_trace_verifier_allows_failed_discarded_group(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("event") != "response_completed"]
    rows.append(
        {
            "event": "slime_sample_failed",
            "payload": {"group_id": "g", "request_id": "g:0", "error": "connection reset"},
        }
    )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary = verify_trace(
        path,
        expected_groups=1,
        min_policy_versions=0,
        allow_failed_groups=True,
        expected_complete_groups=0,
    )
    assert summary["complete_groups"] == 0
    assert summary["discarded_groups"] == 1


def test_live_trace_verifier_accepts_minimum_complete_groups(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path)
    summary = verify_trace(path, min_complete_groups=1)
    assert summary["complete_groups"] == 1


def test_live_trace_verifier_rejects_unaccounted_incomplete_group(tmp_path):
    path = tmp_path / "trace.jsonl"
    write_trace(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows.pop()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete groups"):
        verify_trace(path, allow_failed_groups=True)
