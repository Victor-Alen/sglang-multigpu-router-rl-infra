from scripts.summarize_live_run import summarize


def test_summarize_live_run_counts_tokens_versions_and_wall_time() -> None:
    rows = [
        {"event": "routing_decision", "payload": {"decision": {}}},
        {
            "event": "response_completed",
            "payload": {
                "generated_tokens": 100,
                "requested_policy_version": 2,
                "served_policy_version": 2,
            },
        },
        {
            "event": "response_completed",
            "payload": {
                "generated_tokens": 50,
                "requested_policy_version": 2,
                "served_policy_version": 2,
            },
        },
    ]
    result = summarize(rows, mode="strict", wall_seconds=10.0)
    assert result["groups"] == 1
    assert result["responses"] == 2
    assert result["generated_tokens"] == 150
    assert result["end_to_end_token_throughput"] == 15.0
    assert result["served_version_response_counts"] == {"2": 2}
    assert result["requested_served_version_mismatches"] == 0
