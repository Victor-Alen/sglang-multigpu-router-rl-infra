#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _command(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    project_root: Path,
    *,
    run_id: str,
    status: str,
    trainer_gpus: str,
    rollout_gpus: str,
    automated_tests: int,
    artifacts: list[Path],
) -> dict[str, Any]:
    artifact_records = []
    for path in artifacts:
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {path}")
        artifact_records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    status_text = _command("git", "status", "--porcelain", cwd=project_root)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": status,
        "project_commit": _command("git", "rev-parse", "HEAD", cwd=project_root),
        "working_tree_dirty": bool(status_text),
        "working_tree_status_sha256": hashlib.sha256(status_text.encode()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "trainer_gpu_ids": trainer_gpus,
        "rollout_gpu_ids": rollout_gpus,
        "automated_tests": automated_tests,
        "gpu_snapshot": _command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ).splitlines(),
        "artifacts": artifact_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an immutable evidence manifest")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--status", choices=("accepted", "rejected"), required=True)
    parser.add_argument("--trainer-gpus", required=True)
    parser.add_argument("--rollout-gpus", required=True)
    parser.add_argument("--automated-tests", type=int, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(
        args.project_root,
        run_id=args.run_id,
        status=args.status,
        trainer_gpus=args.trainer_gpus,
        rollout_gpus=args.rollout_gpus,
        automated_tests=args.automated_tests,
        artifacts=args.artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
