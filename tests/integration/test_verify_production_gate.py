from __future__ import annotations

import pytest

from scripts.verify_production_gate import verify_production_gate


CONFIG = {
    "production": {
        "min_automated_tests": 93,
        "min_soak_seconds": 7200,
        "min_soak_batches": 1200,
        "min_terminal_throughput_ratio": 0.5,
        "max_gpu_temperature_c": 90,
        "max_steady_memory_growth_mib": 2048,
        "max_policy_lag": 1,
        "max_buffered_batches": 1,
        "max_version_mismatches": 0,
        "max_soak_prefetch_failures": 0,
        "max_soak_strict_fallbacks": 0,
        "max_actor_death_mttr_seconds": 120,
        "min_network_partition_seconds": 10,
        "max_network_partition_mttr_seconds": 60,
        "max_host_loss_recovery_seconds": 1200,
        "required_host_failure_scope": "private_single_node_ray_control_plane_and_job",
        "min_host_recovery_iteration_advance": 2,
        "require_host_state_restore": True,
        "require_dataset_cursor_restore": True,
    }
}


def _evidence() -> dict[str, dict[str, object]]:
    return {
        "actor_death": {
            "status": "accepted",
            "actor_death_recovered": True,
            "actor_death_mttr_seconds": 56.7,
        },
        "actor_trace": [
            {"event": "actor_death_injected", "engine_index": 0},
            {"event": "actor_death_confirmed", "engine_index": 0},
            {"event": "actor_death_recovered", "engine_index": 0},
        ],
        "network_partition": {
            "status": "accepted",
            "network_partition_recovered": True,
            "network_partition_mttr_seconds": 16.0,
        },
        "network_trace": [
            {
                "event": "network_partition_started",
                "host": "127.0.1.1",
                "port": 15000,
                "label": "chaos-test",
                "duration_seconds": 10,
            },
            {
                "event": "network_partition_ended",
                "host": "127.0.1.1",
                "port": 15000,
                "label": "chaos-test",
            },
            {"event": "network_partition_recovered"},
        ],
        "host_loss": {
            "status": "accepted",
            "failure_scope": "private_single_node_ray_control_plane_and_job",
            "phase1_exit_code": 1,
            "checkpoint_iteration": 2,
            "final_iteration": 4,
            "state_restored": True,
            "dataset_cursor_restored": True,
            "recovery_seconds": 769,
        },
        "long_soak": {
            "status": "accepted",
            "wall_seconds": 8000,
            "consumed_batches": 1200,
            "terminal_throughput_ratio": 0.9,
            "max_temperature_c": 80,
            "steady_memory_growth_mib": {"1": 0, "2": 100, "3": 0, "4": 0},
            "policy_lag_bound": 1,
            "buffer_bound": 1,
            "prefetch_failures": 0,
            "strict_fallbacks": 0,
            "mixed_version_responses": 0,
        },
    }


def test_production_gate_accepts_complete_evidence() -> None:
    evidence = _evidence()
    result = verify_production_gate(CONFIG, automated_tests=93, **evidence)
    assert result["status"] == "accepted"
    assert all(result["checks"].values())


def test_production_gate_rejects_slow_actor_recovery() -> None:
    evidence = _evidence()
    evidence["actor_death"]["actor_death_mttr_seconds"] = 121
    with pytest.raises(ValueError, match="actor_death_mttr"):
        verify_production_gate(CONFIG, automated_tests=93, **evidence)


def test_production_gate_rejects_memory_growth() -> None:
    evidence = _evidence()
    evidence["long_soak"]["steady_memory_growth_mib"] = {"1": 4096}
    with pytest.raises(ValueError, match="gpu_memory_growth"):
        verify_production_gate(CONFIG, automated_tests=93, **evidence)


def test_production_gate_rejects_unproven_network_partition() -> None:
    evidence = _evidence()
    evidence["network_trace"][0]["duration_seconds"] = 9
    with pytest.raises(ValueError, match="network_partition_duration"):
        verify_production_gate(CONFIG, automated_tests=93, **evidence)
