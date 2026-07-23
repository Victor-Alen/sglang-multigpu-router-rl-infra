from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

from .models import RolloutWorkerState, ValidationError, WorkerLifecycle


@dataclass
class WorkerUpdate:
    worker_id: str
    previous_version: int
    target_version: int
    state: str = "PENDING"
    checksum: Optional[str] = None
    error: Optional[str] = None


@dataclass
class WeightUpdate:
    update_id: str
    from_version: int
    to_version: int
    mode: str
    created_at_ns: int
    deadline_ns: int
    status: str = "DRAINING"
    expected_checksum: Optional[str] = None
    workers: Dict[str, WorkerUpdate] = field(default_factory=dict)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


class PolicyVersionCoordinator:
    """Auditable READY/DRAINING/UPDATING coordinator.

    It deliberately does not perform framework RPCs. Adapters call these state
    transitions only after the corresponding SGLang operation succeeds.
    """

    def __init__(
        self,
        workers: Iterable[RolloutWorkerState],
        audit_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ):
        worker_list = list(workers)
        self.workers = {worker.worker_id: worker for worker in worker_list}
        if len(self.workers) != len(worker_list):
            raise ValidationError("worker ids must be unique")
        self.audit_path = Path(audit_path) if audit_path else None
        self.state_path = Path(state_path) if state_path else None
        self.active: Optional[WeightUpdate] = None
        self._lock = threading.RLock()
        self._restore_after_restart()

    @staticmethod
    def _weight_update_from_dict(value: dict) -> WeightUpdate:
        data = dict(value)
        data["workers"] = {
            worker_id: WorkerUpdate(**worker)
            for worker_id, worker in data.get("workers", {}).items()
        }
        return WeightUpdate(**data)

    def _persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "saved_at_ns": time.time_ns(),
            "active_update": asdict(self.active) if self.active else None,
            # This snapshot is for audit only. Worker versions are always
            # re-probed after restart and are never restored from this file.
            "workers": {key: value.to_dict() for key, value in self.workers.items()},
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _restore_after_restart(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValidationError("unsupported policy-version state schema")
        active = value.get("active_update")
        if active is None:
            return
        self.active = self._weight_update_from_dict(active)
        if self.active.status not in {"COMMITTED", "ABORTED"}:
            previous_status = self.active.status
            self.active.status = "ABORTED"
            for worker_id in self.active.workers:
                if worker_id in self.workers:
                    self.workers[worker_id].lifecycle = WorkerLifecycle.RECOVERING
            self._audit(
                "restart_aborted_incomplete_update",
                update_id=self.active.update_id,
                previous_status=previous_status,
            )
            self._persist()

    def _audit(self, event: str, **fields) -> None:
        if not self.audit_path:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_ns": time.time_ns(), "event": event, **fields}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()

    def begin_update(
        self,
        from_version: int,
        to_version: int,
        mode: str = "stop-the-world",
        timeout_seconds: float = 120.0,
        expected_checksum: Optional[str] = None,
        worker_ids: Optional[Iterable[str]] = None,
    ) -> WeightUpdate:
        if mode not in {"stop-the-world", "rolling"}:
            raise ValidationError("mode must be stop-the-world or rolling")
        if to_version <= from_version:
            raise ValidationError("to_version must be greater than from_version")
        with self._lock:
            if self.active and self.active.status not in {"COMMITTED", "ABORTED"}:
                raise ValidationError("another weight update is active")
            ids = list(worker_ids or self.workers)
            if not ids:
                raise ValidationError("at least one worker is required")
            for worker_id in ids:
                worker = self.workers[worker_id]
                if worker.lifecycle != WorkerLifecycle.READY:
                    raise ValidationError(f"worker {worker_id} is not READY")
                if worker.loaded_policy_version != from_version:
                    raise ValidationError(f"worker {worker_id} is not at version {from_version}")
            now = time.time_ns()
            self.active = WeightUpdate(
                update_id=f"update-{to_version}-{uuid.uuid4().hex[:8]}",
                from_version=from_version,
                to_version=to_version,
                mode=mode,
                created_at_ns=now,
                deadline_ns=now + int(timeout_seconds * 1e9),
                expected_checksum=expected_checksum,
                workers={
                    worker_id: WorkerUpdate(worker_id, from_version, to_version)
                    for worker_id in ids
                },
            )
            if mode == "stop-the-world":
                for worker_id in ids:
                    self.workers[worker_id].lifecycle = WorkerLifecycle.DRAINING
            else:
                self.workers[ids[0]].lifecycle = WorkerLifecycle.DRAINING
            self._audit("begin_weight_update", update=asdict(self.active))
            self._persist()
            return self.active

    def mark_drained(self, worker_id: str) -> None:
        with self._lock:
            update = self._require_active()
            worker = self.workers[worker_id]
            if worker.lifecycle != WorkerLifecycle.DRAINING:
                raise ValidationError(f"worker {worker_id} is not DRAINING")
            if worker.running_requests or worker.queued_requests:
                raise ValidationError(f"worker {worker_id} still has in-flight work")
            worker.lifecycle = WorkerLifecycle.UPDATING
            update.workers[worker_id].state = "UPDATING"
            update.status = "UPDATING"
            self._audit("worker_drained", update_id=update.update_id, worker_id=worker_id)
            self._persist()

    def verify_worker(self, worker_id: str, loaded_version: int, checksum: Optional[str]) -> None:
        with self._lock:
            update = self._require_active()
            worker_update = update.workers[worker_id]
            worker = self.workers[worker_id]
            if worker.lifecycle != WorkerLifecycle.UPDATING:
                raise ValidationError(f"worker {worker_id} is not UPDATING")
            if loaded_version != update.to_version:
                self.fail_worker(worker_id, f"loaded version {loaded_version}, expected {update.to_version}")
                raise ValidationError("worker loaded the wrong policy version")
            if update.expected_checksum and checksum != update.expected_checksum:
                self.fail_worker(worker_id, "parameter checksum mismatch")
                raise ValidationError("parameter checksum mismatch")
            worker.loaded_policy_version = loaded_version
            worker.parameter_checksum = checksum
            worker.lifecycle = WorkerLifecycle.READY
            worker_update.checksum = checksum
            worker_update.state = "VERIFIED"
            self._audit(
                "worker_verified",
                update_id=update.update_id,
                worker_id=worker_id,
                version=loaded_version,
                checksum=checksum,
            )
            if update.mode == "rolling":
                pending = [
                    item.worker_id for item in update.workers.values() if item.state == "PENDING"
                ]
                if pending:
                    self.workers[pending[0]].lifecycle = WorkerLifecycle.DRAINING
            self._persist()

    def commit(self) -> WeightUpdate:
        with self._lock:
            update = self._require_active()
            incomplete = [
                worker.worker_id for worker in update.workers.values() if worker.state != "VERIFIED"
            ]
            if incomplete:
                raise ValidationError(f"cannot commit; unverified workers: {incomplete}")
            update.status = "COMMITTED"
            self._audit("commit_weight_update", update=asdict(update))
            self._persist()
            return update

    def fail_worker(self, worker_id: str, error: str) -> None:
        update = self._require_active()
        self.workers[worker_id].lifecycle = WorkerLifecycle.FAILED
        update.workers[worker_id].state = "FAILED"
        update.workers[worker_id].error = error
        update.status = "FAILED"
        self._audit("worker_update_failed", update_id=update.update_id, worker_id=worker_id, error=error)
        self._persist()

    def abort(self, reason: str) -> WeightUpdate:
        with self._lock:
            update = self._require_active()
            update.status = "ABORTED"
            for worker_id, worker_update in update.workers.items():
                worker = self.workers[worker_id]
                if worker_update.state == "VERIFIED":
                    worker.lifecycle = WorkerLifecycle.FAILED
                elif worker.lifecycle in {WorkerLifecycle.DRAINING, WorkerLifecycle.UPDATING}:
                    worker.lifecycle = WorkerLifecycle.RECOVERING
            self._audit("abort_weight_update", update_id=update.update_id, reason=reason)
            self._persist()
            return update

    def check_timeout(self, now_ns: Optional[int] = None) -> bool:
        with self._lock:
            update = self._require_active()
            if (now_ns or time.time_ns()) <= update.deadline_ns:
                return False
            self.abort("deadline exceeded")
            return True

    def _require_active(self) -> WeightUpdate:
        if not self.active or self.active.status in {"COMMITTED", "ABORTED"}:
            raise ValidationError("no active weight update")
        return self.active

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "workers": {key: value.to_dict() for key, value in self.workers.items()},
                "active_update": asdict(self.active) if self.active else None,
            }
