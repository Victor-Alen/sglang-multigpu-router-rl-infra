import argparse
import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Deque, Dict, List, Optional

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .metrics import RouterMetrics
    from .policy import CacheTracker, build_policy, fingerprint_prompt
    from .rl_api import install_rl_routes
except ImportError:  # Supports `python router/app.py`.
    from metrics import RouterMetrics
    from policy import CacheTracker, build_policy, fingerprint_prompt
    from rl_api import install_rl_routes


@dataclass
class WorkerState:
    name: str
    base_url: str
    in_flight: int
    healthy: bool
    cache_tracker: CacheTracker


def extract_prompt(payload: Dict[str, Any]) -> str:
    if "messages" in payload:
        parts = []
        for msg in payload.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}:{content}")
        return "\n".join(parts)
    return payload.get("prompt", "") or ""


def load_workers(config_path: str, cache_entries: int) -> List[WorkerState]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    workers = []
    for item in raw.get("workers", []):
        name = item.get("name") or item["base_url"]
        workers.append(
            WorkerState(
                name=name,
                base_url=item["base_url"].rstrip("/"),
                in_flight=0,
                healthy=True,
                cache_tracker=CacheTracker(cache_entries),
            )
        )
    if not workers:
        raise ValueError("No workers found in config")
    return workers


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI()
    workers = load_workers(args.config, args.cache_entries)
    policy = build_policy(
        args.policy,
        cache_load_tolerance=args.cache_load_tolerance,
        cache_hit_weight=args.cache_hit_weight,
        cache_load_weight=args.cache_load_weight,
    )
    fallback_policy = build_policy(args.fallback_policy)
    metrics = RouterMetrics()
    semaphore = asyncio.Semaphore(args.max_in_flight)
    recent_cache_hits: Deque[int] = deque(maxlen=args.fallback_window)
    fallback_active = False

    app.state.workers = workers
    app.state.policy = policy
    app.state.fallback_policy = fallback_policy
    app.state.metrics = metrics
    app.state.semaphore = semaphore
    app.state.args = args
    app.state.client = httpx.AsyncClient(timeout=args.timeout)
    install_rl_routes(app, args, workers)

    async def healthcheck_loop() -> None:
        while True:
            await asyncio.sleep(args.health_interval)
            for worker in workers:
                try:
                    resp = await app.state.client.get(f"{worker.base_url}/health")
                    worker.healthy = resp.status_code == 200
                except httpx.HTTPError:
                    worker.healthy = False

    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(healthcheck_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.client.aclose()

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def router_metrics() -> Dict[str, object]:
        return metrics.snapshot()

    def choose_worker(fingerprint: Optional[str]) -> tuple[WorkerState, str, int, bool]:
        nonlocal fallback_active
        healthy = [w for w in workers if w.healthy]
        if not healthy:
            healthy = workers

        fallback_used = False
        if (
            args.policy == "cache-aware"
            and args.fallback_hit_rate_threshold >= 0
            and len(recent_cache_hits) >= args.fallback_min_samples
        ):
            hit_rate = sum(recent_cache_hits) / max(1, len(recent_cache_hits))

            if fallback_active:
                if hit_rate >= args.fallback_recover_hit_rate_threshold:
                    fallback_active = False
            else:
                if hit_rate < args.fallback_hit_rate_threshold:
                    fallback_active = True

            if fallback_active:
                selected = fallback_policy.select(healthy, fingerprint)
                predicted_hit = 0
                if fingerprint:
                    predicted_hit = 1 if any(w.cache_tracker.score(fingerprint) > 0 for w in healthy) else 0
                reason = f"cache-aware:fallback-{args.fallback_policy}"
                return selected, reason, predicted_hit, True

        selected = policy.select(healthy, fingerprint)
        if args.policy != "cache-aware":
            return selected, args.policy, 0, fallback_used
        if not fingerprint:
            return selected, "cache-aware:missing-prefix", 0, fallback_used

        selected_hit = 1 if selected.cache_tracker.score(fingerprint) > 0 else 0
        any_hit = any(w.cache_tracker.score(fingerprint) > 0 for w in healthy)
        if selected_hit:
            return selected, "cache-aware:hit", 1, fallback_used
        if any_hit:
            return selected, "cache-aware:spillover", 0, fallback_used
        return selected, "cache-aware:miss", 0, fallback_used

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        payload = await request.json()
        prompt = extract_prompt(payload)
        fingerprint = fingerprint_prompt(prompt, args.prefix_chars) if prompt else None

        async with semaphore:
            select_start = time.perf_counter()
            worker, reason, cache_hit_pred, fallback_used = choose_worker(fingerprint)
            select_latency_ms = (time.perf_counter() - select_start) * 1000
            metrics.record_routing_decision(
                worker=worker.name,
                reason=reason,
                decision_latency_ms=select_latency_ms,
                fallback_used=fallback_used,
            )
            worker.in_flight += 1
            metrics.set_in_flight(worker.name, worker.in_flight)

            if args.policy == "cache-aware" and fingerprint:
                recent_cache_hits.append(cache_hit_pred)

            async def finalize(latency_ms: float, ok: bool) -> None:
                worker.in_flight -= 1
                metrics.set_in_flight(worker.name, worker.in_flight)
                metrics.record_request(worker.name, latency_ms, ok)
                if ok and fingerprint:
                    worker.cache_tracker.record(fingerprint)

            for attempt in range(args.max_retries + 1):
                start = time.time()
                try:
                    url = f"{worker.base_url}/v1/chat/completions"
                    if payload.get("stream"):
                        req = app.state.client.build_request("POST", url, json=payload)
                        stream = await app.state.client.send(req, stream=True)
                        ok = stream.status_code < 400

                        async def streamer() -> AsyncIterator[bytes]:
                            try:
                                async for chunk in stream.aiter_bytes():
                                    yield chunk
                            finally:
                                latency_ms = (time.time() - start) * 1000
                                await finalize(latency_ms, ok)
                                await stream.aclose()

                        headers = {
                            "x-router-worker": worker.name,
                            "x-router-reason": reason,
                            "x-router-cache-hit-pred": str(cache_hit_pred),
                        }
                        return StreamingResponse(streamer(), headers=headers, status_code=stream.status_code)

                    resp = await app.state.client.post(url, json=payload)
                    ok = resp.status_code < 400
                    latency_ms = (time.time() - start) * 1000
                    await finalize(latency_ms, ok)

                    headers = {
                        "x-router-worker": worker.name,
                        "x-router-reason": reason,
                        "x-router-cache-hit-pred": str(cache_hit_pred),
                    }
                    return JSONResponse(resp.json(), status_code=resp.status_code, headers=headers)
                except httpx.HTTPError as exc:
                    if attempt >= args.max_retries:
                        latency_ms = (time.time() - start) * 1000
                        await finalize(latency_ms, False)
                        raise HTTPException(status_code=502, detail=str(exc))
                    await asyncio.sleep(args.retry_backoff)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        payload = await request.json()
        prompt = payload.get("prompt") if isinstance(payload.get("prompt"), str) else ""
        fingerprint = fingerprint_prompt(prompt, args.prefix_chars) if prompt else None

        async with semaphore:
            select_start = time.perf_counter()
            worker, reason, cache_hit_pred, fallback_used = choose_worker(fingerprint)
            select_latency_ms = (time.perf_counter() - select_start) * 1000
            metrics.record_routing_decision(
                worker=worker.name,
                reason=reason,
                decision_latency_ms=select_latency_ms,
                fallback_used=fallback_used,
            )
            worker.in_flight += 1
            metrics.set_in_flight(worker.name, worker.in_flight)

            if args.policy == "cache-aware" and fingerprint:
                recent_cache_hits.append(cache_hit_pred)

            async def finalize(latency_ms: float, ok: bool) -> None:
                worker.in_flight -= 1
                metrics.set_in_flight(worker.name, worker.in_flight)
                metrics.record_request(worker.name, latency_ms, ok)
                if ok and fingerprint:
                    worker.cache_tracker.record(fingerprint)

            for attempt in range(args.max_retries + 1):
                start = time.time()
                try:
                    url = f"{worker.base_url}/v1/completions"
                    if payload.get("stream"):
                        req = app.state.client.build_request("POST", url, json=payload)
                        stream = await app.state.client.send(req, stream=True)
                        ok = stream.status_code < 400

                        async def streamer() -> AsyncIterator[bytes]:
                            try:
                                async for chunk in stream.aiter_bytes():
                                    yield chunk
                            finally:
                                latency_ms = (time.time() - start) * 1000
                                await finalize(latency_ms, ok)
                                await stream.aclose()

                        headers = {
                            "x-router-worker": worker.name,
                            "x-router-reason": reason,
                            "x-router-cache-hit-pred": str(cache_hit_pred),
                        }
                        return StreamingResponse(streamer(), headers=headers, status_code=stream.status_code)

                    resp = await app.state.client.post(url, json=payload)
                    ok = resp.status_code < 400
                    latency_ms = (time.time() - start) * 1000
                    await finalize(latency_ms, ok)

                    headers = {
                        "x-router-worker": worker.name,
                        "x-router-reason": reason,
                        "x-router-cache-hit-pred": str(cache_hit_pred),
                    }
                    return JSONResponse(resp.json(), status_code=resp.status_code, headers=headers)
                except httpx.HTTPError as exc:
                    if attempt >= args.max_retries:
                        latency_ms = (time.time() - start) * 1000
                        await finalize(latency_ms, False)
                        raise HTTPException(status_code=502, detail=str(exc))
                    await asyncio.sleep(args.retry_backoff)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/cluster.yaml")
    parser.add_argument("--policy", default="rr", choices=["rr", "least-load", "cache-aware"])
    parser.add_argument("--max-in-flight", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-backoff", type=float, default=0.1)
    parser.add_argument("--prefix-chars", type=int, default=512)
    parser.add_argument("--cache-entries", type=int, default=2048)
    parser.add_argument("--cache-load-tolerance", type=int, default=1)
    parser.add_argument("--cache-hit-weight", type=float, default=1.0)
    parser.add_argument("--cache-load-weight", type=float, default=1.0)
    parser.add_argument("--fallback-policy", choices=["rr", "least-load"], default="least-load")
    parser.add_argument("--fallback-hit-rate-threshold", type=float, default=-1.0)
    parser.add_argument("--fallback-recover-hit-rate-threshold", type=float, default=0.15)
    parser.add_argument("--fallback-window", type=int, default=200)
    parser.add_argument("--fallback-min-samples", type=int, default=50)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--rl-policy",
        choices=[
            "round-robin",
            "least-requests",
            "least-queued-tokens",
            "cache-aware-group",
            "power-of-two",
            "fixed-pack",
            "fixed-even-split",
            "load-proportional-split",
            "adaptive-group",
            "offline-oracle",
        ],
        default="adaptive-group",
    )
    parser.add_argument("--rl-initial-policy-version", type=int, default=0)
    parser.add_argument("--rl-state-dir", default="results/rl_state")
    parser.add_argument("--rl-heartbeat-timeout", type=float, default=15.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
