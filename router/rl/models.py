from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional


class ValidationError(ValueError):
    """Raised when rollout metadata violates a correctness invariant."""


class WorkerLifecycle(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    DRAINING = "DRAINING"
    UPDATING = "UPDATING"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"


class GroupStatus(str, Enum):
    OPEN = "OPEN"
    READY = "READY"
    FAILED = "FAILED"
    DROPPED = "DROPPED"


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_new_tokens: int = 256
    tokenizer_revision: str = "unknown"
    chat_template_hash: str = "unknown"

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValidationError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValidationError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValidationError("top_k must be positive when set")
        if self.max_new_tokens <= 0:
            raise ValidationError("max_new_tokens must be positive")


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    prompt_id: str
    group_id: str
    sample_index: int
    group_size: int
    policy_version: int
    rollout_step: int
    generation_seed: int
    prompt_tokens: int
    max_new_tokens: int
    predicted_output_tokens: float
    predicted_output_p90: float
    prefix_fingerprint: str
    enqueue_timestamp_ns: int = field(default_factory=time.time_ns)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    tokenizer_revision: str = "unknown"
    chat_template_hash: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("request_id", "prompt_id", "group_id"):
            if not getattr(self, name):
                raise ValidationError(f"{name} must not be empty")
        if self.group_size <= 0:
            raise ValidationError("group_size must be positive")
        if not 0 <= self.sample_index < self.group_size:
            raise ValidationError("sample_index must be within group_size")
        if self.policy_version < 0 or self.rollout_step < 0:
            raise ValidationError("policy_version and rollout_step must be non-negative")
        if self.prompt_tokens < 0 or self.max_new_tokens <= 0:
            raise ValidationError("token counts are invalid")
        if self.predicted_output_tokens < 0 or self.predicted_output_p90 < 0:
            raise ValidationError("predicted output lengths must be non-negative")
        if self.predicted_output_p90 < self.predicted_output_tokens:
            raise ValidationError("predicted_output_p90 must be >= predicted_output_tokens")
        SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
            tokenizer_revision=self.tokenizer_revision,
            chat_template_hash=self.chat_template_hash,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutRequest":
        return cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutWorkerState:
    worker_id: str
    endpoint: str
    lifecycle: WorkerLifecycle = WorkerLifecycle.STARTING
    loaded_policy_version: int = 0
    running_requests: int = 0
    queued_requests: int = 0
    queued_prefill_tokens: int = 0
    active_decode_tokens: int = 0
    available_kv_tokens: int = 0
    kv_cache_utilization: float = 0.0
    prefill_tokens_per_second_ema: float = 1.0
    decode_tokens_per_second_ema: float = 1.0
    heartbeat_timestamp_ns: int = field(default_factory=time.time_ns)
    prefix_matched_tokens: Dict[str, int] = field(default_factory=dict)
    parameter_checksum: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.worker_id or not self.endpoint:
            raise ValidationError("worker_id and endpoint must not be empty")
        if isinstance(self.lifecycle, str):
            self.lifecycle = WorkerLifecycle(self.lifecycle)
        if self.loaded_policy_version < 0:
            raise ValidationError("loaded_policy_version must be non-negative")
        for name in (
            "running_requests",
            "queued_requests",
            "queued_prefill_tokens",
            "active_decode_tokens",
            "available_kv_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValidationError(f"{name} must be non-negative")
        if not 0 <= self.kv_cache_utilization <= 1:
            raise ValidationError("kv_cache_utilization must be in [0, 1]")
        if self.prefill_tokens_per_second_ema <= 0 or self.decode_tokens_per_second_ema <= 0:
            raise ValidationError("worker throughput estimates must be positive")

    def matched_prefix_tokens(self, fingerprint: str, prompt_tokens: int) -> int:
        return min(prompt_tokens, max(0, self.prefix_matched_tokens.get(fingerprint, 0)))

    def queue_time_seconds(self) -> float:
        return (
            self.queued_prefill_tokens / self.prefill_tokens_per_second_ema
            + self.active_decode_tokens / self.decode_tokens_per_second_ema
        )

    def is_eligible(self, policy_version: int, now_ns: int, heartbeat_timeout_ns: int) -> bool:
        return (
            self.lifecycle == WorkerLifecycle.READY
            and self.loaded_policy_version == policy_version
            and now_ns - self.heartbeat_timestamp_ns <= heartbeat_timeout_ns
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutWorkerState":
        data = dict(value)
        data["lifecycle"] = WorkerLifecycle(data.get("lifecycle", WorkerLifecycle.STARTING))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["lifecycle"] = self.lifecycle.value
        return result


@dataclass
class ResponseMetadata:
    request_id: str
    group_id: str
    sample_index: int
    worker_id: str
    requested_policy_version: int
    served_policy_version: int
    prompt_tokens: int
    generated_tokens: int
    prefix_matched_tokens: int = 0
    queue_wait_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    finish_reason: str = "stop"
    retry_count: int = 0
    generation_seed: Optional[int] = None
    tokenizer_revision: str = "unknown"
    chat_template_hash: str = "unknown"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResponseMetadata":
        return cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutGroupState:
    group_id: str
    prompt_id: str
    expected_size: int
    policy_version: int
    rollout_step: int
    assigned_workers: list[str] = field(default_factory=list)
    completed_samples: set[int] = field(default_factory=set)
    failed_samples: set[int] = field(default_factory=set)
    creation_timestamp_ns: int = field(default_factory=time.time_ns)
    deadline_timestamp_ns: int = 0
    status: GroupStatus = GroupStatus.OPEN
    sampling_signature: Optional[str] = None
    responses: Dict[int, ResponseMetadata] = field(default_factory=dict)
    failure_reasons: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expected_size <= 0:
            raise ValidationError("expected_size must be positive")
        if self.deadline_timestamp_ns == 0:
            self.deadline_timestamp_ns = self.creation_timestamp_ns + 300_000_000_000
        if isinstance(self.status, str):
            self.status = GroupStatus(self.status)

    @property
    def is_complete(self) -> bool:
        return len(self.completed_samples) == self.expected_size and not self.failed_samples

    def ensure_indices(self, indices: Iterable[int]) -> None:
        invalid = [i for i in indices if not 0 <= i < self.expected_size]
        if invalid:
            raise ValidationError(f"sample indices outside group: {invalid}")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        result["completed_samples"] = sorted(self.completed_samples)
        result["failed_samples"] = sorted(self.failed_samples)
        result["responses"] = {str(k): v.to_dict() for k, v in self.responses.items()}
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutGroupState":
        data = dict(value)
        data["completed_samples"] = set(data.get("completed_samples", []))
        data["failed_samples"] = set(data.get("failed_samples", []))
        data["responses"] = {
            int(k): ResponseMetadata.from_dict(v) for k, v in data.get("responses", {}).items()
        }
        data["status"] = GroupStatus(data.get("status", GroupStatus.OPEN))
        return cls(**data)


def ensure_homogeneous_group(requests: Iterable[RolloutRequest]) -> list[RolloutRequest]:
    items = list(requests)
    if not items:
        raise ValidationError("a group must contain at least one request")
    first = items[0]
    expected = {
        "group_id": first.group_id,
        "prompt_id": first.prompt_id,
        "group_size": first.group_size,
        "policy_version": first.policy_version,
        "rollout_step": first.rollout_step,
        "tokenizer_revision": first.tokenizer_revision,
        "chat_template_hash": first.chat_template_hash,
        "temperature": first.temperature,
        "top_p": first.top_p,
        "top_k": first.top_k,
    }
    for request in items[1:]:
        for name, value in expected.items():
            other = getattr(request, name)
            equal = math.isclose(other, value) if isinstance(value, float) else other == value
            if not equal:
                raise ValidationError(f"mixed {name} in group {first.group_id}")
    indices = [request.sample_index for request in items]
    if len(indices) != len(set(indices)):
        raise ValidationError(f"duplicate sample_index in group {first.group_id}")
    return sorted(items, key=lambda item: item.sample_index)
