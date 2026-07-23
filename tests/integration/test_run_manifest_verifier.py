import hashlib

import pytest

from scripts.verify_run_manifest import verify_manifest


def test_manifest_verifier_accepts_matching_artifact(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_bytes(b"{}\n")
    manifest = {
        "artifacts": [
            {
                "path": str(artifact),
                "bytes": 3,
                "sha256": hashlib.sha256(b"{}\n").hexdigest(),
            }
        ]
    }
    assert verify_manifest(manifest)["verified_artifacts"] == 1


def test_manifest_verifier_rejects_tampering(tmp_path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_bytes(b"tampered")
    manifest = {"artifacts": [{"path": str(artifact), "bytes": 8, "sha256": "0" * 64}]}
    with pytest.raises(ValueError, match="sha256_mismatch"):
        verify_manifest(manifest)
