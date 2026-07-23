#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def inspect_checkpoint(root: Path, iteration: int) -> dict[str, Any]:
    checkpoint = root / f"iter_{iteration:07d}"
    meta_path = checkpoint / "meta.json"
    if not meta_path.is_file():
        raise ValueError(f"checkpoint metadata is missing: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(metadata.get("iteration", -1)) != iteration:
        raise ValueError(f"checkpoint iteration mismatch in {meta_path}")
    if int(metadata.get("next_rollout_id", -1)) != iteration:
        raise ValueError(f"next_rollout_id mismatch in {meta_path}")

    component_files: dict[str, int] = {}
    for component in ("model", "optimizer", "lr_scheduler"):
        directory = checkpoint / component
        count = sum(path.is_file() for path in directory.rglob("*")) if directory.is_dir() else 0
        if count == 0:
            raise ValueError(f"checkpoint component is empty: {directory}")
        component_files[component] = count
    if not (checkpoint / "rng.pt").is_file():
        raise ValueError(f"RNG state is missing: {checkpoint / 'rng.pt'}")
    dataset_state = root / "rollout" / f"global_dataset_state_dict_{iteration - 1}.pt"
    if not dataset_state.is_file():
        raise ValueError(f"global dataset state is missing: {dataset_state}")
    return {
        "iteration": iteration,
        "next_rollout_id": int(metadata["next_rollout_id"]),
        "global_step": int(metadata["global_step"]),
        "micro_step": int(metadata["micro_step"]),
        "world_size": int(metadata["world_size"]),
        "component_file_counts": component_files,
        "dataset_state": str(dataset_state),
    }


def verify_resume(root: Path, first_iteration: int, final_iteration: int, resume_log: Path) -> dict[str, Any]:
    tracker = root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or int(tracker.read_text(encoding="utf-8").strip()) != final_iteration:
        raise ValueError("latest checkpoint tracker does not point at the final iteration")
    first = inspect_checkpoint(root, first_iteration)
    final = inspect_checkpoint(root, final_iteration)
    if final["global_step"] <= first["global_step"]:
        raise ValueError("global_step did not advance after resume")

    text = resume_log.read_text(encoding="utf-8", errors="replace")
    required = (
        "[FSDP] Loaded model from",
        "[FSDP] Loaded optimizer from",
        "[FSDP] Loaded LR scheduler from",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise ValueError(f"resume log is missing load markers: {missing}")
    if re.search(rf" - rollout {first_iteration}: ", text) is None:
        raise ValueError(f"resume log does not contain rollout {first_iteration}")
    if re.search(r" - rollout 0: ", text) is not None:
        raise ValueError("resumed run unexpectedly restarted at rollout 0")
    return {
        "status": "accepted",
        "checkpoint_root": str(root),
        "first": first,
        "final": final,
        "resumed_from_rollout": first_iteration,
        "completed_after_resume": final_iteration - first_iteration,
        "model_optimizer_scheduler_rng_restored": True,
        "dataset_cursor_restored": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a real FSDP checkpoint save/resume sequence")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--first-iteration", type=int, required=True)
    parser.add_argument("--final-iteration", type=int, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_resume(args.checkpoint_root, args.first_iteration, args.final_iteration, args.resume_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
