import json

import pytest

from scripts.aggregate_paired_comparisons import aggregate, summarize


def test_summarize_reports_student_t_interval() -> None:
    result = summarize([0.1, 0.2, 0.3])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(0.2)
    assert result["ci95_low"] < result["mean"] < result["ci95_high"]


def test_aggregate_requires_multiple_pairs(tmp_path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"status": "matched_prompts_unmatched_seeds"}), encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        aggregate([path])


def test_aggregate_rejects_unpaired_input(tmp_path) -> None:
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps({"status": "matched_prompts_unmatched_seeds"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unpaired"):
        aggregate([path, path])
