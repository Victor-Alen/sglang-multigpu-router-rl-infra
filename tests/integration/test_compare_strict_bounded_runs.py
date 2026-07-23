import json

from scripts.compare_strict_bounded_runs import (
    parse_perf_metrics,
    trace_seed_signature,
    trace_workload_signature,
)


def test_parse_perf_metrics_extracts_repeated_step_values() -> None:
    text = """
    perf 0: {'perf/train_wait_time': 4.8, 'perf/train_time': 2.9, 'perf/step_time': 7.7}
    perf 1: {'perf/train_wait_time': 3.0, 'perf/train_time': 2.9, 'perf/step_time': 5.9}
    """
    metrics = parse_perf_metrics(text)
    assert metrics["perf/step_time"] == [7.7, 5.9]
    assert metrics["perf/train_wait_time"] == [4.8, 3.0]


def test_workload_signature_ignores_async_rollout_timing_but_seed_check_does_not(tmp_path) -> None:
    def row(step: int, seed: int) -> str:
        return json.dumps(
            {
                "event": "routing_decision",
                "payload": {
                    "requests": [
                        {
                            "rollout_step": step,
                            "sample_index": 0,
                            "prompt_id": "prompt-sha",
                            "max_new_tokens": 128,
                            "temperature": 0.8,
                            "top_p": 1.0,
                            "top_k": None,
                            "generation_seed": seed,
                        }
                    ]
                },
            }
        )

    strict = tmp_path / "strict.jsonl"
    bounded = tmp_path / "bounded.jsonl"
    strict.write_text(row(3, 10) + "\n", encoding="utf-8")
    bounded.write_text(row(0, 11) + "\n", encoding="utf-8")
    assert trace_workload_signature(strict) == trace_workload_signature(bounded)
    assert trace_seed_signature(strict) != trace_seed_signature(bounded)
