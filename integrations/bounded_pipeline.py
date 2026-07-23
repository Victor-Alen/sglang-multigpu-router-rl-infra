from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def normalize_weight_versions(values: Sequence[Any]) -> int:
    """Return the common integer policy version reported by rollout engines."""
    if not values:
        raise ValueError("no rollout weight versions were reported")
    parsed: list[int] = []
    for value in values:
        if value is None:
            raise ValueError("a rollout engine did not report a weight version")
        text = str(value).strip()
        if text.startswith("v"):
            text = text[1:]
        try:
            parsed.append(int(text))
        except ValueError as exc:
            raise ValueError(f"invalid rollout weight version: {value!r}") from exc
    if len(set(parsed)) != 1:
        raise ValueError(f"rollout engines have mixed weight versions: {parsed}")
    return parsed[0]


def count_generated_tokens(partitions: Iterable[Mapping[str, Any]]) -> int:
    """Count response tokens once across disjoint data-parallel partitions."""
    total = 0
    for partition in partitions:
        lengths = partition.get("response_lengths", ())
        total += sum(int(length) for length in lengths)
    return total


@dataclass
class BoundedPipelineStats:
    started_at_s: float
    completed_batches: int = 0
    accepted_tokens: int = 0
    stale_batches: int = 0
    stale_tokens: int = 0
    prefetch_failures: int = 0
    strict_fallbacks: int = 0
    max_observed_lag: int = 0
    max_buffered_batches: int = 0

    @classmethod
    def start(cls) -> "BoundedPipelineStats":
        return cls(started_at_s=time.time())

    def observe_batch(self, *, tokens: int, lag: int, buffered_batches: int) -> None:
        if lag < 0:
            raise ValueError("policy lag cannot be negative")
        self.completed_batches += 1
        self.accepted_tokens += tokens
        self.max_observed_lag = max(self.max_observed_lag, lag)
        self.max_buffered_batches = max(self.max_buffered_batches, buffered_batches)

    def summary(self, *, now_s: float | None = None) -> dict[str, Any]:
        elapsed = max(1e-9, (time.time() if now_s is None else now_s) - self.started_at_s)
        result = asdict(self)
        result.update(
            {
                "elapsed_s": elapsed,
                "fresh_token_throughput": self.accepted_tokens / elapsed,
            }
        )
        return result


class PipelineTraceWriter:
    """Process-local, fsync-backed JSONL event writer for async acceptance."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        record = {"event": event, "timestamp_s": time.time(), **fields}
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
