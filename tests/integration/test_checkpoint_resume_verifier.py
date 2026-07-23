import json

import pytest

from scripts.verify_checkpoint_resume import verify_resume


def _checkpoint(root, iteration: int, global_step: int) -> None:
    checkpoint = root / f"iter_{iteration:07d}"
    for component in ("model", "optimizer", "lr_scheduler"):
        directory = checkpoint / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "state.distcp").write_bytes(b"state")
    (checkpoint / "rng.pt").write_bytes(b"rng")
    (checkpoint / "meta.json").write_text(
        json.dumps(
            {
                "iteration": iteration,
                "next_rollout_id": iteration,
                "global_step": global_step,
                "micro_step": global_step,
                "world_size": 2,
            }
        ),
        encoding="utf-8",
    )
    rollout = root / "rollout"
    rollout.mkdir(exist_ok=True)
    (rollout / f"global_dataset_state_dict_{iteration - 1}.pt").write_bytes(b"dataset")


def test_verifier_accepts_complete_resume(tmp_path) -> None:
    _checkpoint(tmp_path, 2, 2)
    _checkpoint(tmp_path, 4, 4)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4", encoding="utf-8")
    log = tmp_path / "resume.log"
    log.write_text(
        "[FSDP] Loaded model from x\n[FSDP] Loaded optimizer from x\n"
        "[FSDP] Loaded LR scheduler from x\n[time] actor.py - rollout 2: {}\n",
        encoding="utf-8",
    )
    result = verify_resume(tmp_path, 2, 4, log)
    assert result["status"] == "accepted"
    assert result["completed_after_resume"] == 2


def test_verifier_rejects_restart_from_zero(tmp_path) -> None:
    _checkpoint(tmp_path, 2, 2)
    _checkpoint(tmp_path, 4, 4)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4", encoding="utf-8")
    log = tmp_path / "resume.log"
    log.write_text(
        "[FSDP] Loaded model from x\n[FSDP] Loaded optimizer from x\n"
        "[FSDP] Loaded LR scheduler from x\n[x] actor.py - rollout 2: {}\n"
        "[x] actor.py - rollout 0: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="restarted"):
        verify_resume(tmp_path, 2, 4, log)
