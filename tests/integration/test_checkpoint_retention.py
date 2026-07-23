from scripts.prune_checkpoints import apply_plan, retention_plan


def test_retention_keeps_latest_and_tracker_target(tmp_path) -> None:
    for iteration in (2, 4, 6):
        (tmp_path / f"iter_{iteration:07d}").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("4", encoding="utf-8")
    plan = retention_plan(tmp_path, keep_last=1)
    assert {path.split("_")[-1] for path in plan["keep"]} == {"0000004", "0000006"}
    assert [path.split("_")[-1] for path in plan["delete"]] == ["0000002"]
    apply_plan(plan)
    assert not (tmp_path / "iter_0000002").exists()
    assert (tmp_path / "iter_0000004").is_dir()


def test_dry_run_does_not_delete(tmp_path) -> None:
    for iteration in (1, 2):
        (tmp_path / f"iter_{iteration:07d}").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2", encoding="utf-8")
    retention_plan(tmp_path, keep_last=1)
    assert (tmp_path / "iter_0000001").is_dir()
