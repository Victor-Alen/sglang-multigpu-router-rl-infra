from __future__ import annotations

import pytest

from scripts.verify_host_loss_recovery import verify_host_loss


def _checkpoint() -> dict:
    return {
        "status": "accepted",
        "first": {"iteration": 2},
        "final": {"iteration": 4},
        "model_optimizer_scheduler_rng_restored": True,
        "dataset_cursor_restored": True,
    }


def test_host_loss_verifier_accepts_nonzero_failure_and_resume() -> None:
    result = verify_host_loss(
        _checkpoint(),
        phase1_exit_code=1,
        interrupted_ns=1_000_000_000,
        recovered_ns=3_000_000_000,
        max_recovery_seconds=3,
    )
    assert result["status"] == "accepted"
    assert result["recovery_seconds"] == 2


def test_host_loss_verifier_rejects_clean_phase1_exit() -> None:
    with pytest.raises(ValueError, match="unexpectedly exited successfully"):
        verify_host_loss(
            _checkpoint(),
            phase1_exit_code=0,
            interrupted_ns=1,
            recovered_ns=2,
            max_recovery_seconds=3,
        )
