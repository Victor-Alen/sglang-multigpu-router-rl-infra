from __future__ import annotations

import pytest

from scripts.verify_long_soak import verify_long_soak


def _verify(telemetry: list[dict[str, str]]) -> dict[str, object]:
    trace = [
        {
            "event": "batch_consumed",
            "timestamp_ns": index * 1_000_000_000,
            "generated_tokens": 100,
        }
        for index in range(20)
    ]
    return verify_long_soak(
        {
            "status": "accepted",
            "consumed_batches": 20,
            "max_observed_lag": 1,
            "max_buffered_batches": 1,
            "prefetch_failures": 0,
            "strict_fallbacks": 0,
        },
        {"groups": 40, "complete_groups": 40, "mixed_version_responses": 0},
        {"wall_seconds": 7200},
        trace,
        telemetry,
        expected_batches=20,
        expected_gpus={1, 2, 3, 4},
        min_wall_seconds=7200,
        min_throughput_ratio=0.5,
        max_temperature_c=90,
        max_steady_memory_growth_mib=2048,
        min_samples_per_gpu=20,
    )


def test_long_soak_verifier_accepts_stable_run() -> None:
    telemetry = [
        {
            "timestamp_ns": str(sample * 1_000_000_000),
            "gpu": str(gpu),
            "memory_used_mib": str(10000 + (sample % 2) * 12000),
            "temperature_c": "70",
            "power_w": "200",
        }
        for gpu in (1, 2, 3, 4)
        for sample in range(20)
    ]
    result = _verify(telemetry)
    assert result["status"] == "accepted"
    assert result["terminal_throughput_ratio"] == 1
    assert result["steady_memory_growth_mib"] == {1: 0, 2: 0, 3: 0, 4: 0}


def test_long_soak_verifier_rejects_progressive_memory_growth() -> None:
    telemetry = [
        {
            "timestamp_ns": str(sample * 1_000_000_000),
            "gpu": str(gpu),
            "memory_used_mib": str(10000 + sample * 300),
            "temperature_c": "70",
            "power_w": "200",
        }
        for gpu in (1, 2, 3, 4)
        for sample in range(20)
    ]
    with pytest.raises(ValueError, match="steady memory growth"):
        _verify(telemetry)
