#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.rl.models import RolloutRequest, RolloutWorkerState, WorkerLifecycle
from router.rl.runtime import RLRouterRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/golden_trace.jsonl"))
    parser.add_argument("--groups", type=int, default=32)
    args = parser.parse_args()
    now = time.time_ns()
    workers = [
        RolloutWorkerState(
            worker_id=f"rollout-{index}",
            endpoint=f"http://127.0.0.1:{30004 + index}",
            lifecycle=WorkerLifecycle.READY,
            loaded_policy_version=0,
            prefill_tokens_per_second_ema=2500,
            decode_tokens_per_second_ema=100,
            heartbeat_timestamp_ns=now,
        )
        for index in range(2)
    ]
    with tempfile.TemporaryDirectory() as directory:
        runtime = RLRouterRuntime(workers, state_dir=directory)
        for group_index in range(args.groups):
            group_size = 4
            prompt_tokens = [128, 512, 2048, 4096][group_index % 4]
            predicted = [64, 256, 800, 32][group_index % 4]
            p90 = min(2048, int(predicted * 1.3))
            group_id = f"golden-{group_index:03d}"
            requests = [
                RolloutRequest(
                    request_id=f"{group_id}:{sample}",
                    prompt_id=f"prompt-{group_index:03d}",
                    group_id=group_id,
                    sample_index=sample,
                    group_size=group_size,
                    policy_version=0,
                    rollout_step=0,
                    generation_seed=group_index * 100 + sample,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=2048,
                    predicted_output_tokens=predicted,
                    predicted_output_p90=p90,
                    prefix_fingerprint=f"prefix-{group_index:03d}",
                    tokenizer_revision="golden-tokenizer",
                    chat_template_hash="golden-template",
                )
                for sample in range(group_size)
            ]
            runtime.route(requests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((Path(directory) / "trace.jsonl").read_bytes())
    print(f"wrote {args.groups} groups to {args.output}")


if __name__ == "__main__":
    main()
