import hashlib
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Protocol, Sequence, TypeVar


@dataclass
class CacheTracker:
    max_entries: int
    _queue: Deque[str]
    _counts: Dict[str, int]

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self._queue = deque()
        self._counts = {}

    def record(self, fingerprint: str) -> None:
        if fingerprint in self._counts:
            self._counts[fingerprint] += 1
            self._queue.append(fingerprint)
            self._evict_if_needed()
            return
        self._counts[fingerprint] = 1
        self._queue.append(fingerprint)
        self._evict_if_needed()

    def score(self, fingerprint: str) -> int:
        return self._counts.get(fingerprint, 0)

    def _evict_if_needed(self) -> None:
        while len(self._queue) > self.max_entries:
            old = self._queue.popleft()
            count = self._counts.get(old, 0)
            if count <= 1:
                self._counts.pop(old, None)
            else:
                self._counts[old] = count - 1


def fingerprint_prompt(text: str, prefix_chars: int) -> str:
    prefix = text[:prefix_chars] if text else ""
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


class WorkerLike(Protocol):
    in_flight: int
    cache_tracker: CacheTracker


TWorker = TypeVar("TWorker", bound=WorkerLike)


class BasePolicy:
    name = "base"

    def select(self, workers: Sequence[TWorker], fingerprint: Optional[str]) -> TWorker:
        raise NotImplementedError


class RoundRobinPolicy(BasePolicy):
    name = "rr"

    def __init__(self) -> None:
        self._counter = itertools.count()

    def select(self, workers: Sequence[TWorker], fingerprint: Optional[str]) -> TWorker:
        idx = next(self._counter) % len(workers)
        return workers[idx]


class LeastLoadPolicy(BasePolicy):
    name = "least-load"

    def __init__(self) -> None:
        self._counter = itertools.count()

    def _pick_min_load_rr(self, workers: Sequence[TWorker]) -> TWorker:
        min_load = min(w.in_flight for w in workers)
        candidates = [w for w in workers if w.in_flight == min_load]
        idx = next(self._counter) % len(candidates)
        return candidates[idx]

    def select(self, workers: Sequence[TWorker], fingerprint: Optional[str]) -> TWorker:
        return self._pick_min_load_rr(workers)


class CacheAwarePolicy(BasePolicy):
    name = "cache-aware"

    def __init__(
        self,
        load_tolerance: int = 1,
        hit_weight: float = 1.0,
        load_weight: float = 1.0,
    ) -> None:
        self.load_tolerance = load_tolerance
        self.hit_weight = hit_weight
        self.load_weight = load_weight
        self._counter = itertools.count()

    def _pick_min_load_rr(self, workers: Sequence[TWorker]) -> TWorker:
        min_load = min(w.in_flight for w in workers)
        candidates = [w for w in workers if w.in_flight == min_load]
        idx = next(self._counter) % len(candidates)
        return candidates[idx]

    def select(self, workers: Sequence[TWorker], fingerprint: Optional[str]) -> TWorker:
        if not fingerprint:
            return self._pick_min_load_rr(workers)

        def score(worker: TWorker) -> float:
            hit = 1.0 if worker.cache_tracker.score(fingerprint) > 0 else 0.0
            return (self.hit_weight * hit) - (self.load_weight * float(worker.in_flight))

        scored = [(worker, score(worker)) for worker in workers]
        best_score = max(item[1] for item in scored)
        candidates = [worker for worker, value in scored if value == best_score]
        idx = next(self._counter) % len(candidates)
        return candidates[idx]


def build_policy(
    name: str,
    cache_load_tolerance: int = 1,
    cache_hit_weight: float = 1.0,
    cache_load_weight: float = 1.0,
) -> BasePolicy:
    if name == RoundRobinPolicy.name:
        return RoundRobinPolicy()
    if name == LeastLoadPolicy.name:
        return LeastLoadPolicy()
    if name == CacheAwarePolicy.name:
        return CacheAwarePolicy(
            load_tolerance=cache_load_tolerance,
            hit_weight=cache_hit_weight,
            load_weight=cache_load_weight,
        )
    raise ValueError(f"Unknown policy: {name}")
