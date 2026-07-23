#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


ITERATION = re.compile(r"iter_(\d{7})$")


def retention_plan(root: Path, keep_last: int) -> dict[str, Any]:
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    resolved = root.resolve()
    tracker = resolved / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(f"checkpoint tracker is missing: {tracker}")
    tracked_iteration = int(tracker.read_text(encoding="utf-8").strip())
    checkpoints: list[tuple[int, Path]] = []
    for child in resolved.iterdir():
        match = ITERATION.fullmatch(child.name)
        if match and child.is_dir() and not child.is_symlink():
            checkpoints.append((int(match.group(1)), child))
    checkpoints.sort()
    keep_iterations = {iteration for iteration, _ in checkpoints[-keep_last:]}
    keep_iterations.add(tracked_iteration)
    delete = [path for iteration, path in checkpoints if iteration not in keep_iterations]
    keep = [path for iteration, path in checkpoints if iteration in keep_iterations]
    return {
        "root": str(resolved),
        "tracked_iteration": tracked_iteration,
        "keep": [str(path) for path in keep],
        "delete": [str(path) for path in delete],
    }


def apply_plan(plan: dict[str, Any]) -> None:
    root = Path(plan["root"]).resolve()
    tracked = root / f"iter_{int(plan['tracked_iteration']):07d}"
    for value in plan["delete"]:
        target = Path(value).resolve()
        if target.parent != root or ITERATION.fullmatch(target.name) is None:
            raise ValueError(f"refusing to delete path outside checkpoint root: {target}")
        if target == tracked:
            raise ValueError("refusing to delete the tracker-selected checkpoint")
        shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely apply FSDP checkpoint retention")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--apply", action="store_true", help="Delete candidates; default is dry-run")
    args = parser.parse_args()
    plan = retention_plan(args.checkpoint_root, args.keep_last)
    if args.apply:
        apply_plan(plan)
        plan["applied"] = True
    else:
        plan["applied"] = False
    print(json.dumps(plan, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
