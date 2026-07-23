from __future__ import annotations

import time
from typing import Any, Dict, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from integrations.sglang_adapter import SGLangWorkerAdapter
from integrations.weight_update_executor import WeightUpdateExecutor

try:
    from .rl.models import ResponseMetadata, RolloutRequest, RolloutWorkerState, ValidationError, WorkerLifecycle
    from .rl.runtime import RLRouterRuntime
except ImportError:  # Supports `python router/app.py`.
    from rl.models import ResponseMetadata, RolloutRequest, RolloutWorkerState, ValidationError, WorkerLifecycle
    from rl.runtime import RLRouterRuntime


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (ValidationError, ValueError)):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def install_rl_routes(app: FastAPI, args: Any, serving_workers: Iterable[Any]) -> RLRouterRuntime:
    workers = [
        RolloutWorkerState(
            worker_id=worker.name,
            endpoint=worker.base_url,
            lifecycle=WorkerLifecycle.READY,
            loaded_policy_version=args.rl_initial_policy_version,
        )
        for worker in serving_workers
    ]
    runtime = RLRouterRuntime(
        workers=workers,
        policy_name=args.rl_policy,
        state_dir=args.rl_state_dir,
        heartbeat_timeout_seconds=args.rl_heartbeat_timeout,
    )
    app.state.rl_runtime = runtime

    @app.post("/rl/workers/register")
    async def register_worker(request: Request) -> Dict[str, object]:
        try:
            worker = RolloutWorkerState.from_dict(await request.json())
            runtime.register_worker(worker)
            return worker.to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/workers/{worker_id}/heartbeat")
    async def worker_heartbeat(worker_id: str, request: Request) -> Dict[str, object]:
        try:
            worker = runtime.heartbeat(worker_id, await request.json())
            return worker.to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/rl/workers")
    async def rl_workers() -> Dict[str, object]:
        return {key: value.to_dict() for key, value in runtime.workers.items()}

    @app.post("/rl/groups")
    async def create_group(request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            requests = [RolloutRequest.from_dict(item) for item in body["requests"]]
            group = runtime.tracker.register(requests, body.get("deadline_seconds", 300.0))
            runtime.trace.write("group_registered", group.to_dict())
            return group.to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/route")
    async def route_group(request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            requests = [RolloutRequest.from_dict(item) for item in body["requests"]]
            return runtime.route(requests, body.get("deadline_seconds", 300.0)).to_dict()
        except Exception as exc:
            if "mixed" in str(exc).lower() or "version" in str(exc).lower():
                runtime.metrics.mixed_version_rejections += 1
            raise _error(exc) from exc

    @app.post("/rl/responses")
    async def complete_response(request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            consumed = body.pop("consumed_at_policy_version", None)
            return runtime.complete(ResponseMetadata.from_dict(body), consumed)
        except Exception as exc:
            if "mixed-version" in str(exc).lower() or "version" in str(exc).lower():
                runtime.metrics.mixed_version_rejections += 1
            raise _error(exc) from exc

    @app.get("/rl/groups/{group_id}")
    async def get_group(group_id: str) -> Dict[str, object]:
        try:
            return runtime.tracker.require(group_id).to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/begin-weight-update")
    async def begin_weight_update(request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            update = runtime.versioning.begin_update(
                from_version=body["from_version"],
                to_version=body["to_version"],
                mode=body.get("mode", "stop-the-world").replace("drain", "stop-the-world"),
                timeout_seconds=body.get("timeout_seconds", 120.0),
                expected_checksum=body.get("expected_checksum"),
                worker_ids=body.get("workers"),
            )
            return {
                "update_id": update.update_id,
                "accepted": True,
                "workers": list(update.workers),
                "status": update.status,
            }
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/execute-weight-update")
    async def execute_weight_update(request: Request) -> Dict[str, object]:
        """Drain, update, probe, verify and commit real SGLang workers."""
        started = time.perf_counter()
        try:
            body = await request.json()
            worker_ids = body.get("workers") or list(runtime.workers)
            adapters = {
                worker_id: SGLangWorkerAdapter(
                    runtime.workers[worker_id].endpoint,
                    timeout_seconds=float(body.get("timeout_seconds", 120.0)),
                )
                for worker_id in worker_ids
            }
            executor = WeightUpdateExecutor(runtime.versioning, adapters)
            result = await executor.execute_disk_update(
                model_path=body["model_path"],
                from_version=int(body["from_version"]),
                to_version=int(body["to_version"]),
                mode=body.get("mode", "stop-the-world").replace("drain", "stop-the-world"),
                timeout_seconds=float(body.get("timeout_seconds", 120.0)),
                expected_checksum=body.get("expected_checksum"),
                worker_ids=worker_ids,
                load_format=body.get("load_format"),
                probe_prompt=body.get("probe_prompt"),
            )
            runtime.metrics.record_weight_update(True, result.total_seconds, result.drain_seconds)
            runtime.trace.write("weight_update_completed", result.to_dict())
            return result.to_dict()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            runtime.metrics.record_weight_update(False, elapsed, 0.0)
            runtime.trace.write(
                "weight_update_failed",
                {"elapsed_seconds": elapsed, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise _error(exc) from exc

    @app.post("/rl/weight-update/{worker_id}/drained")
    async def worker_drained(worker_id: str) -> Dict[str, object]:
        try:
            runtime.versioning.mark_drained(worker_id)
            return runtime.versioning.snapshot()
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/weight-update/{worker_id}/verified")
    async def worker_verified(worker_id: str, request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            runtime.versioning.verify_worker(
                worker_id, body["loaded_policy_version"], body.get("checksum")
            )
            return runtime.versioning.snapshot()
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/commit-weight-update")
    async def commit_weight_update() -> Dict[str, object]:
        try:
            update = runtime.versioning.commit()
            return {"update_id": update.update_id, "status": update.status}
        except Exception as exc:
            raise _error(exc) from exc

    @app.post("/rl/abort-weight-update")
    async def abort_weight_update(request: Request) -> Dict[str, object]:
        try:
            body = await request.json()
            update = runtime.versioning.abort(body.get("reason", "operator requested"))
            return {"update_id": update.update_id, "status": update.status}
        except Exception as exc:
            raise _error(exc) from exc

    @app.get("/rl/policy-versions")
    async def policy_versions() -> Dict[str, object]:
        return runtime.versioning.snapshot()

    @app.get("/rl/metrics")
    async def rl_metrics() -> Dict[str, object]:
        return runtime.metrics.snapshot()

    @app.get("/rl/metrics/prometheus", response_class=PlainTextResponse)
    async def rl_metrics_prometheus() -> str:
        return runtime.metrics.prometheus()

    return runtime
