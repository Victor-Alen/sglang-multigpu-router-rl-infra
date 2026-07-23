from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple


@dataclass(frozen=True)
class LengthPrediction:
    mean: float
    p90: float
    samples: int


class BucketedOutputLengthPredictor:
    """Online prompt/dataset bucket predictor with bounded history."""

    def __init__(self, alpha: float = 0.2, max_history: int = 256, default_ratio: float = 0.6):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_history = max_history
        self.default_ratio = default_ratio
        self._means: Dict[Tuple[str, int, int], float] = {}
        self._history: Dict[Tuple[str, int, int], Deque[int]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    @staticmethod
    def key(dataset: str, prompt_tokens: int, max_new_tokens: int) -> Tuple[str, int, int]:
        prompt_bucket = max(0, int(math.log2(max(1, prompt_tokens))))
        output_bucket = max(0, int(math.log2(max(1, max_new_tokens))))
        return dataset or "unknown", prompt_bucket, output_bucket

    def observe(self, dataset: str, prompt_tokens: int, max_new_tokens: int, output_tokens: int) -> None:
        if output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        key = self.key(dataset, prompt_tokens, max_new_tokens)
        previous = self._means.get(key, float(output_tokens))
        self._means[key] = self.alpha * output_tokens + (1 - self.alpha) * previous
        self._history[key].append(output_tokens)

    def predict(self, dataset: str, prompt_tokens: int, max_new_tokens: int) -> LengthPrediction:
        key = self.key(dataset, prompt_tokens, max_new_tokens)
        history = self._history.get(key)
        if not history:
            estimate = min(max_new_tokens, max(1.0, max_new_tokens * self.default_ratio))
            return LengthPrediction(estimate, min(max_new_tokens, estimate * 1.35), 0)
        ordered = sorted(history)
        rank = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
        mean = min(float(max_new_tokens), self._means[key])
        p90 = min(float(max_new_tokens), max(mean, float(ordered[rank])))
        return LengthPrediction(mean, p90, len(history))

    def export(self) -> dict:
        return {
            "alpha": self.alpha,
            "max_history": self.max_history,
            "default_ratio": self.default_ratio,
            "buckets": [
                {
                    "key": list(key),
                    "mean": self._means[key],
                    "history": list(history),
                }
                for key, history in self._history.items()
            ],
        }

    @classmethod
    def restore(cls, value: dict) -> "BucketedOutputLengthPredictor":
        predictor = cls(value["alpha"], value["max_history"], value["default_ratio"])
        for bucket in value.get("buckets", []):
            key = tuple(bucket["key"])
            predictor._means[key] = float(bucket["mean"])
            predictor._history[key].extend(int(v) for v in bucket["history"])
        return predictor
