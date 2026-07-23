import pytest

from scripts.verify_maturity_gate import verify_gate


def _config() -> dict:
    return {
        "release_candidate": {
            "min_automated_tests": 10,
            "min_soak_batches": 8,
            "max_policy_lag": 1,
            "max_buffered_batches": 1,
            "max_version_mismatches": 0,
            "max_soak_prefetch_failures": 0,
            "max_soak_strict_fallbacks": 0,
            "required_fault_prefetch_failures": 1,
            "required_fault_strict_fallbacks": 1,
            "require_checkpoint_model_optimizer_scheduler_rng": True,
            "require_dataset_cursor_restore": True,
        }
    }


def _evidence() -> tuple[dict, dict, dict, dict]:
    checkpoint = {
        "status": "accepted",
        "model_optimizer_scheduler_rng_restored": True,
        "dataset_cursor_restored": True,
    }
    fault = {"status": "accepted", "fault_injections": 1, "prefetch_failures": 1, "strict_fallbacks": 1}
    soak = {
        "status": "accepted",
        "consumed_batches": 8,
        "max_observed_lag": 1,
        "max_buffered_batches": 1,
        "prefetch_failures": 0,
        "strict_fallbacks": 0,
    }
    router = {"groups": 16, "complete_groups": 16, "mixed_version_responses": 0}
    return checkpoint, fault, soak, router


def test_maturity_gate_accepts_complete_evidence() -> None:
    checkpoint, fault, soak, router = _evidence()
    result = verify_gate(
        _config(), automated_tests=10, checkpoint=checkpoint, fault=fault, soak=soak, soak_router=router
    )
    assert result["status"] == "accepted"


def test_maturity_gate_rejects_version_mismatch() -> None:
    checkpoint, fault, soak, router = _evidence()
    router["mixed_version_responses"] = 1
    with pytest.raises(ValueError, match="router_no_mixed_version"):
        verify_gate(
            _config(), automated_tests=10, checkpoint=checkpoint, fault=fault, soak=soak, soak_router=router
        )
