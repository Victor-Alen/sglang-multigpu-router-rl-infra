from __future__ import annotations

import json

import pytest

from integrations.bounded_pipeline import (
    BoundedPipelineStats,
    PipelineTraceWriter,
    count_generated_tokens,
    normalize_weight_versions,
)


def test_normalize_weight_versions_requires_homogeneous_integer_versions() -> None:
    assert normalize_weight_versions(["v7", "7", 7]) == 7
    with pytest.raises(ValueError, match="mixed"):
        normalize_weight_versions([7, 8])
    with pytest.raises(ValueError, match="did not report"):
        normalize_weight_versions([7, None])


def test_count_generated_tokens_sums_disjoint_dp_partitions() -> None:
    partitions = [
        {"response_lengths": [10, 4]},
        {"response_lengths": [3, 8]},
    ]
    assert count_generated_tokens(partitions) == 25


def test_pipeline_stats_reports_fresh_token_throughput_and_bounds() -> None:
    stats = BoundedPipelineStats(started_at_s=10.0)
    stats.observe_batch(tokens=120, lag=0, buffered_batches=0)
    stats.observe_batch(tokens=80, lag=1, buffered_batches=1)
    summary = stats.summary(now_s=20.0)
    assert summary["completed_batches"] == 2
    assert summary["accepted_tokens"] == 200
    assert summary["max_observed_lag"] == 1
    assert summary["max_buffered_batches"] == 1
    assert summary["fresh_token_throughput"] == 20.0


def test_pipeline_trace_is_append_only_jsonl(tmp_path) -> None:
    path = tmp_path / "run" / "bounded_async_trace.jsonl"
    writer = PipelineTraceWriter(path)
    writer.write("prefetch_started", rollout_id=1, generated_policy_version=3)
    writer.write("batch_consumed", rollout_id=1, policy_lag=1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["prefetch_started", "batch_consumed"]
    assert rows[0]["generated_policy_version"] == 3
