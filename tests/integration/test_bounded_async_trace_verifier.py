from __future__ import annotations

import pytest

from scripts.verify_bounded_async_trace import verify_trace


def _valid_rows() -> list[dict]:
    return [
        {"event": "pipeline_started"},
        {"event": "batch_consumed", "rollout_id": 0, "generated_policy_version": 1, "consumed_policy_version": 1, "policy_lag": 0},
        {"event": "prefetch_started", "rollout_id": 1, "generated_policy_version": 1},
        {"event": "batch_consumed", "rollout_id": 1, "generated_policy_version": 1, "consumed_policy_version": 2, "policy_lag": 1},
        {
            "event": "pipeline_complete",
            "accepted_tokens": 100,
            "fresh_token_throughput": 5.0,
            "max_observed_lag": 1,
            "max_buffered_batches": 1,
            "strict_fallbacks": 0,
            "prefetch_failures": 0,
            "final_policy_version": 3,
        },
    ]


def test_verifier_accepts_bounded_overlap_trace() -> None:
    result = verify_trace(_valid_rows(), expected_batches=2, require_overlap=True)
    assert result["status"] == "accepted"
    assert result["positive_lag_batches"] == 1


def test_verifier_rejects_lag_violation() -> None:
    rows = _valid_rows()
    rows[3]["consumed_policy_version"] = 3
    rows[3]["policy_lag"] = 2
    with pytest.raises(ValueError, match="lag bound"):
        verify_trace(rows, expected_batches=2)


def test_verifier_rejects_smoke_trace_without_overlap() -> None:
    rows = _valid_rows()
    rows[3]["generated_policy_version"] = 2
    rows[3]["policy_lag"] = 0
    rows[:] = [row for row in rows if row.get("event") != "prefetch_started"]
    with pytest.raises(ValueError, match="does not prove"):
        verify_trace(rows, expected_batches=2, require_overlap=True)


def test_verifier_accepts_one_injected_prefetch_failure_and_fallback() -> None:
    rows = _valid_rows()
    rows.insert(2, {"event": "fault_injected", "fault": "prefetch_result_failure", "rollout_id": 1})
    rows[-1]["prefetch_failures"] = 1
    rows[-1]["strict_fallbacks"] = 1
    result = verify_trace(
        rows,
        expected_batches=2,
        expected_prefetch_failures=1,
        expected_strict_fallbacks=1,
        require_fault_injection=True,
    )
    assert result["fault_injections"] == 1


def test_verifier_rejects_missing_required_fault() -> None:
    with pytest.raises(ValueError, match="fault_injected"):
        verify_trace(_valid_rows(), require_fault_injection=True)


def test_verifier_accepts_actor_death_recovery_sequence() -> None:
    rows = _valid_rows()
    rows[1:1] = [
        {"event": "actor_death_injected", "timestamp_ns": 10},
        {"event": "actor_death_confirmed", "timestamp_ns": 11},
        {"event": "actor_replacement_ready", "timestamp_ns": 15},
        {"event": "actor_death_recovered", "timestamp_ns": 20},
    ]
    result = verify_trace(rows, require_actor_death_recovery=True)
    assert result["actor_death_recovered"] is True
    assert result["actor_death_mttr_seconds"] == 10 / 1_000_000_000


def test_verifier_rejects_out_of_order_network_recovery() -> None:
    rows = _valid_rows()
    rows[1:1] = [
        {"event": "network_partition_ended", "timestamp_ns": 20},
        {"event": "network_partition_started", "timestamp_ns": 10},
        {"event": "network_partition_recovered", "timestamp_ns": 30},
    ]
    with pytest.raises(ValueError, match="out of order"):
        verify_trace(rows, require_network_partition_recovery=True)
