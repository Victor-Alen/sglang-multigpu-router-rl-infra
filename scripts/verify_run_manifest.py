#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for artifact in manifest.get("artifacts", []):
        path = Path(artifact["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        if path.stat().st_size != int(artifact["bytes"]):
            failures.append({"path": str(path), "reason": "size_mismatch"})
            continue
        if _sha256(path) != artifact["sha256"]:
            failures.append({"path": str(path), "reason": "sha256_mismatch"})
    if failures:
        raise ValueError(f"manifest verification failed: {failures}")
    return {"status": "accepted", "verified_artifacts": len(manifest.get("artifacts", []))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify evidence files against a run manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(verify_manifest(manifest), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
