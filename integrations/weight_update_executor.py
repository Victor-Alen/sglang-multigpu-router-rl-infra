from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Mapping, Protocol

from router.rl.models import ValidationError, WorkerLifecycle
from router.rl.versioning import PolicyVersionCoordinator


class WeightUpdateAdapter(Protocol):
    def update_weights_from_disk(
        self,
        model_path: str,
        policy_version: int,
        load_format: str | None = None,
        abort_all_requests: bool = False,
        keep_pause: bool = False,
        flush_cache: bool = True,
    ) -> dict: ...

    def fixed_input_probe(self, prompt: str, max_new_tokens: int = 1) -> dict: ...


@dataclass
class WorkerUpdateResult:
    worker_id: str
    policy_version: int
    update_seconds: float
    probe_digest: str | None = None
    model_info: dict = field(default_factory=dict)


@dataclass
class WeightUpdateResult:
    update_id: str
    status: str
    mode: str
    from_version: int
    to_version: int
    total_seconds: float
    drain_seconds: float
    workers: dict[str, WorkerUpdateResult]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "workers": {key: asdict(value) for key, value in self.workers.items()},
        }


def fixed_probe_digest(response: dict) -> str:
    """Hash stable token/log-prob content, excluding request IDs and timestamps."""
    choices = response.get("choices", [])
    stable = []
    for choice in choices:
        stable.append(
            {
                "text": choice.get("text"),
                "finish_reason": choice.get("finish_reason"),
                "logprobs": choice.get("logprobs"),
            }
        )
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WeightUpdateExecutor:
    """Execute the coordinator state machine against real SGLang workers."""

    def __init__(
        self,
        coordinator: PolicyVersionCoordinator,
        adapters: Mapping[str, WeightUpdateAdapter],
        poll_interval_seconds: float = 0.1,
    ) -> None:
        self.coordinator = coordinator
        self.adapters = dict(adapters)
        self.poll_interval_seconds = poll_interval_seconds

    async def _wait_drained(self, worker_id: str, deadline: float) -> float:
        started = time.perf_counter()
        worker = self.coordinator.workers[worker_id]
        while worker.running_requests or worker.queued_requests:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"worker {worker_id} did not drain before deadline")
            await asyncio.sleep(self.poll_interval_seconds)
        return time.perf_counter() - started

    async def execute_disk_update(
        self,
        *,
        model_path: str,
        from_version: int,
        to_version: int,
        mode: str = "stop-the-world",
        timeout_seconds: float = 120.0,
        expected_checksum: str | None = None,
        worker_ids: list[str] | None = None,
        load_format: str | None = None,
        probe_prompt: str | None = None,
    ) -> WeightUpdateResult:
        started = time.perf_counter()
        update = self.coordinator.begin_update(
            from_version=from_version,
            to_version=to_version,
            mode=mode,
            timeout_seconds=timeout_seconds,
            expected_checksum=expected_checksum,
            worker_ids=worker_ids,
        )
        deadline = time.monotonic() + timeout_seconds
        results: dict[str, WorkerUpdateResult] = {}
        drain_seconds = 0.0
        current_worker: str | None = None
        try:
            for worker_id in update.workers:
                current_worker = worker_id
                adapter = self.adapters.get(worker_id)
                if adapter is None:
                    raise ValidationError(f"no SGLang adapter configured for {worker_id}")
                if self.coordinator.workers[worker_id].lifecycle != WorkerLifecycle.DRAINING:
                    raise ValidationError(f"worker {worker_id} was not selected for draining")
                drain_seconds += await self._wait_drained(worker_id, deadline)
                self.coordinator.mark_drained(worker_id)

                update_started = time.perf_counter()
                response = await asyncio.to_thread(
                    adapter.update_weights_from_disk,
                    model_path,
                    to_version,
                    load_format,
                    False,
                    False,
                    True,
                )
                probe_digest = None
                if probe_prompt is not None:
                    probe = await asyncio.to_thread(adapter.fixed_input_probe, probe_prompt, 1)
                    probe_digest = fixed_probe_digest(probe)
                self.coordinator.verify_worker(worker_id, to_version, expected_checksum)
                results[worker_id] = WorkerUpdateResult(
                    worker_id=worker_id,
                    policy_version=to_version,
                    update_seconds=time.perf_counter() - update_started,
                    probe_digest=probe_digest,
                    model_info=dict(response.get("model_info", {})),
                )

            probe_digests = {
                result.probe_digest for result in results.values() if result.probe_digest is not None
            }
            if len(probe_digests) > 1:
                raise ValidationError(f"fixed-input probe mismatch across workers: {probe_digests}")
            committed = self.coordinator.commit()
            return WeightUpdateResult(
                update_id=committed.update_id,
                status=committed.status,
                mode=mode,
                from_version=from_version,
                to_version=to_version,
                total_seconds=time.perf_counter() - started,
                drain_seconds=drain_seconds,
                workers=results,
            )
        except Exception as exc:
            if current_worker is not None:
                worker_update = update.workers[current_worker]
                if worker_update.state not in {"FAILED", "VERIFIED"}:
                    self.coordinator.fail_worker(current_worker, f"{type(exc).__name__}: {exc}")
            if update.status != "ABORTED":
                self.coordinator.abort(f"{type(exc).__name__}: {exc}")
            raise
