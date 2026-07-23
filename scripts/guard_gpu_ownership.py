#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time

import psutil


def _query_rows(fields: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{fields.split(':', 1)[0]}={fields.split(':', 1)[1]}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [column.strip() for column in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip() and "No running processes" not in line
    ]


def foreign_gpu_processes(gpu_ids: set[int], allowed_uid: int) -> list[dict]:
    uuid_to_index = {
        uuid: int(index)
        for index, uuid in _query_rows("gpu:index,uuid")
        if int(index) in gpu_ids
    }
    offenders = []
    for row in _query_rows("compute-apps:gpu_uuid,pid,process_name,used_gpu_memory"):
        if len(row) < 4 or row[0] not in uuid_to_index:
            continue
        try:
            pid = int(row[1])
            uid = psutil.Process(pid).uids().effective
        except (ValueError, psutil.Error):
            continue
        if uid != allowed_uid:
            offenders.append(
                {
                    "gpu": uuid_to_index[row[0]],
                    "pid": pid,
                    "uid": uid,
                    "process": row[2],
                    "used_mib": row[3],
                }
            )
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--ray-bin", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    gpu_ids = {int(item) for item in args.gpu_ids.split(",") if item.strip()}
    while True:
        offenders = foreign_gpu_processes(gpu_ids, os.geteuid())
        if offenders:
            print(f"Foreign process claimed selected GPU(s): {offenders}", flush=True)
            if not args.once:
                subprocess.run([args.ray_bin, "stop", "--force"], check=False)
            return 3
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
