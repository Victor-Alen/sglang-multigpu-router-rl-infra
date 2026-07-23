#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from router.app import create_app, parse_args


async def run() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        sys.argv = [
            "router.app",
            "--config",
            "configs/rl_router_6xa6000.yaml",
            "--policy",
            "rr",
            "--rl-policy",
            "adaptive-group",
            "--rl-state-dir",
            directory,
        ]
        app = create_app(parse_args())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/health")).status_code == 200
            workers = (await client.get("/rl/workers")).json()
            assert set(workers) == {"rollout-0", "rollout-1"}
            requests = [
                {
                    "request_id": f"smoke:{index}",
                    "prompt_id": "smoke-prompt",
                    "group_id": "smoke-group",
                    "sample_index": index,
                    "group_size": 4,
                    "policy_version": 0,
                    "rollout_step": 0,
                    "generation_seed": index,
                    "prompt_tokens": 128,
                    "max_new_tokens": 256,
                    "predicted_output_tokens": 200,
                    "predicted_output_p90": 240,
                    "prefix_fingerprint": "smoke-prefix",
                    "tokenizer_revision": "smoke-tokenizer",
                    "chat_template_hash": "smoke-template",
                }
                for index in range(4)
            ]
            route = await client.post("/rl/route", json={"requests": requests})
            assert route.status_code == 200, route.text
            for item in requests:
                response = {
                    "request_id": item["request_id"],
                    "group_id": item["group_id"],
                    "sample_index": item["sample_index"],
                    "worker_id": "rollout-0",
                    "requested_policy_version": 0,
                    "served_policy_version": 0,
                    "prompt_tokens": 128,
                    "generated_tokens": 20,
                    "tokenizer_revision": "smoke-tokenizer",
                    "chat_template_hash": "smoke-template",
                }
                completed = await client.post("/rl/responses", json=response)
                assert completed.status_code == 200, completed.text
            begin = await client.post(
                "/rl/begin-weight-update",
                json={"from_version": 0, "to_version": 1, "mode": "drain"},
            )
            assert begin.status_code == 200, begin.text
            for worker_id in workers:
                drained = await client.post(f"/rl/weight-update/{worker_id}/drained")
                assert drained.status_code == 200, drained.text
                verified = await client.post(
                    f"/rl/weight-update/{worker_id}/verified",
                    json={"loaded_policy_version": 1},
                )
                assert verified.status_code == 200, verified.text
            committed = await client.post("/rl/commit-weight-update")
            assert committed.status_code == 200, committed.text
            metrics = (await client.get("/rl/metrics")).json()
            prometheus = (await client.get("/rl/metrics/prometheus")).text
            assert metrics["groups_completed"] == 1
            assert metrics["fresh_tokens"] == 80
            assert "rl_router_groups_completed_total 1" in prometheus
            result = {
                "route": route.json(),
                "weight_update": committed.json(),
                "metrics": metrics,
            }
        await app.state.client.aclose()
        return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2, sort_keys=True))
