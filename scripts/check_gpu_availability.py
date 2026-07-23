#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail closed unless requested GPUs are idle")
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5")
    parser.add_argument("--max-used-mib", type=int, default=512)
    args = parser.parse_args()
    requested = {int(value) for value in args.gpu_ids.split(",")}
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {}
    for row in csv.reader(io.StringIO(output)):
        index = int(row[0].strip())
        rows[index] = {
            "name": row[1].strip(),
            "total_mib": int(row[2].strip()),
            "used_mib": int(row[3].strip()),
        }
    missing = requested - rows.keys()
    busy = {index: rows[index] for index in requested & rows.keys() if rows[index]["used_mib"] > args.max_used_mib}
    for index in sorted(requested & rows.keys()):
        state = "BUSY" if index in busy else "IDLE"
        print(f"GPU {index}: {state}, used={rows[index]['used_mib']} MiB, {rows[index]['name']}")
    if missing:
        print(f"Missing GPUs: {sorted(missing)}", file=sys.stderr)
    if busy:
        print("Refusing to start: requested GPUs are occupied.", file=sys.stderr)
    return 1 if missing or busy else 0


if __name__ == "__main__":
    raise SystemExit(main())
