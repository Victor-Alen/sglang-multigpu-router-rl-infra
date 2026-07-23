from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

from .models import (
    GroupStatus,
    ResponseMetadata,
    RolloutGroupState,
    RolloutRequest,
    ValidationError,
    ensure_homogeneous_group,
)


def sampling_signature(request: RolloutRequest) -> str:
    payload = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "max_new_tokens": request.max_new_tokens,
        "tokenizer_revision": request.tokenizer_revision,
        "chat_template_hash": request.chat_template_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class GroupTracker:
    """Thread-safe group barrier with durable, atomic snapshots."""

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path else None
        self.groups: Dict[str, RolloutGroupState] = {}
        self._lock = threading.RLock()
        if self.state_path and self.state_path.exists():
            self.restore()

    def register(self, requests: Iterable[RolloutRequest], deadline_seconds: float = 300.0) -> RolloutGroupState:
        items = ensure_homogeneous_group(requests)
        first = items[0]
        if len(items) != first.group_size:
            raise ValidationError(
                f"group {first.group_id} has {len(items)} requests, expected {first.group_size}"
            )
        with self._lock:
            existing = self.groups.get(first.group_id)
            if existing:
                if (
                    existing.policy_version != first.policy_version
                    or existing.expected_size != first.group_size
                ):
                    raise ValidationError(f"conflicting re-registration of {first.group_id}")
                return existing
            now = time.time_ns()
            group = RolloutGroupState(
                group_id=first.group_id,
                prompt_id=first.prompt_id,
                expected_size=first.group_size,
                policy_version=first.policy_version,
                rollout_step=first.rollout_step,
                creation_timestamp_ns=now,
                deadline_timestamp_ns=now + int(deadline_seconds * 1e9),
                sampling_signature=sampling_signature(first),
            )
            self.groups[group.group_id] = group
            self.persist()
            return group

    def assign(self, group_id: str, assignments: Dict[str, list[str]]) -> None:
        with self._lock:
            group = self.require(group_id)
            group.assigned_workers = sorted(worker for worker, ids in assignments.items() if ids)
            self.persist()

    def record_response(self, response: ResponseMetadata) -> RolloutGroupState:
        with self._lock:
            group = self.require(response.group_id)
            group.ensure_indices([response.sample_index])
            if response.requested_policy_version != group.policy_version:
                raise ValidationError("response requested version does not match group")
            if response.served_policy_version != group.policy_version:
                raise ValidationError(
                    f"mixed-version response for {group.group_id}: requested "
                    f"{group.policy_version}, served {response.served_policy_version}"
                )
            existing = group.responses.get(response.sample_index)
            if existing:
                if existing.request_id != response.request_id:
                    raise ValidationError(
                        f"duplicate training sample index {response.sample_index} in {group.group_id}"
                    )
                return group
            revisions = (
                response.tokenizer_revision,
                response.chat_template_hash,
            )
            completed_revisions = {
                (item.tokenizer_revision, item.chat_template_hash)
                for item in group.responses.values()
            }
            if completed_revisions and revisions not in completed_revisions:
                raise ValidationError(f"tokenizer/chat template mismatch in {group.group_id}")
            group.responses[response.sample_index] = response
            group.completed_samples.add(response.sample_index)
            group.failed_samples.discard(response.sample_index)
            group.failure_reasons.pop(response.sample_index, None)
            if group.is_complete:
                group.status = GroupStatus.READY
            self.persist()
            return group

    def record_failure(self, group_id: str, sample_index: int, reason: str, terminal: bool = False) -> None:
        with self._lock:
            group = self.require(group_id)
            group.ensure_indices([sample_index])
            group.failed_samples.add(sample_index)
            group.failure_reasons[sample_index] = reason
            if terminal:
                group.status = GroupStatus.FAILED
            self.persist()

    def expire(self, now_ns: Optional[int] = None, policy: str = "drop_entire_group") -> list[str]:
        now_ns = now_ns or time.time_ns()
        expired = []
        with self._lock:
            for group in self.groups.values():
                if group.status == GroupStatus.OPEN and now_ns >= group.deadline_timestamp_ns:
                    expired.append(group.group_id)
                    group.status = (
                        GroupStatus.DROPPED if policy == "drop_entire_group" else GroupStatus.FAILED
                    )
            if expired:
                self.persist()
        return expired

    def retry_seed(self, request: RolloutRequest, retry_count: int, mode: str) -> int:
        if mode == "same_seed":
            return request.generation_seed
        if mode == "new_seed":
            digest = hashlib.sha256(
                f"{request.request_id}:{request.generation_seed}:{retry_count}".encode("utf-8")
            ).digest()
            return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
        raise ValueError("retry mode must be same_seed or new_seed")

    def require(self, group_id: str) -> RolloutGroupState:
        try:
            return self.groups[group_id]
        except KeyError as exc:
            raise KeyError(f"unknown group: {group_id}") from exc

    def persist(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "saved_at_ns": time.time_ns(),
            "groups": {key: value.to_dict() for key, value in self.groups.items()},
        }
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temp.replace(self.state_path)

    def restore(self) -> None:
        if not self.state_path:
            raise ValueError("state_path is not configured")
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValidationError("unsupported group state schema")
        with self._lock:
            self.groups = {
                key: RolloutGroupState.from_dict(item)
                for key, item in value.get("groups", {}).items()
            }
