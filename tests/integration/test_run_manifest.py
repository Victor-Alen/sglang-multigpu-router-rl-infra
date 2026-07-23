import shutil

import pytest

from scripts.write_run_manifest import build_manifest


def test_manifest_hashes_artifacts(tmp_path) -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("GPU diagnostics are unavailable on this runner")
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}\n", encoding="utf-8")
    result = build_manifest(
        tmp_path,
        run_id="run-1",
        status="accepted",
        trainer_gpus="1,2",
        rollout_gpus="3,4",
        automated_tests=12,
        artifacts=[artifact],
    )
    assert result["artifacts"][0]["bytes"] == 3
    assert len(result["artifacts"][0]["sha256"]) == 64
