#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
from pathlib import Path

import psutil


def _csv(command: list[str]) -> list[list[str]]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(completed.stdout))
        if row
    ]


def gpu_processes(gpu_ids: set[int]) -> set[int]:
    uuid_to_index = {
        uuid: int(index)
        for index, uuid in _csv(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
        )
    }
    result = set()
    for uuid, pid in _csv(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    ):
        if uuid_to_index.get(uuid) in gpu_ids:
            result.add(int(pid))
    return result


def _has_run_id(process: psutil.Process, run_id: str) -> bool:
    try:
        return process.uids().effective == os.getuid() and process.environ().get(
            "RL_ROUTER_RUN_ID"
        ) == run_id
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def matching_root(pid: int, run_id: str) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return None
    root = None
    while process is not None:
        if _has_run_id(process, run_id):
            root = process
        try:
            process = process.parent()
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            break
    return root


def cleanup(gpu_ids: set[int], run_id: str, *, dry_run: bool = False) -> dict:
    roots = {}
    for pid in gpu_processes(gpu_ids):
        root = matching_root(pid, run_id)
        if root is not None:
            roots[root.pid] = root

    killed = []
    for root in roots.values():
        try:
            processes = root.children(recursive=True) + [root]
        except psutil.NoSuchProcess:
            continue
        killed.extend(process.pid for process in processes)
        if dry_run:
            continue
        for process in reversed(processes):
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    return {
        "run_id": run_id,
        "gpu_ids": sorted(gpu_ids),
        "matched_roots": sorted(roots),
        "cleaned_pids": sorted(set(killed)),
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean only GPU processes carrying an exact run ID")
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_id or len(args.run_id) > 200:
        raise ValueError("invalid run ID")
    result = cleanup(
        {int(value) for value in args.gpu_ids.split(",")},
        args.run_id,
        dry_run=args.dry_run,
    )
    rendered = json.dumps(result, sort_keys=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
