#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.rl.gpu_layout import GPULayout


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate disjoint Trainer/Rollout GPU roles")
    parser.add_argument("--trainer-gpu-ids", required=True)
    parser.add_argument("--rollout-gpu-ids", required=True)
    parser.add_argument("--expected-total", type=int, default=4)
    args = parser.parse_args()
    try:
        layout = GPULayout.from_strings(
            args.trainer_gpu_ids,
            args.rollout_gpu_ids,
            args.expected_total,
        )
    except ValueError as exc:
        print(f"Invalid GPU layout: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(layout.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
