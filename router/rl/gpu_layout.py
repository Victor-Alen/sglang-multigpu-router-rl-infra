from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from .models import ValidationError


def parse_gpu_ids(value: str | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        try:
            result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as exc:
            raise ValidationError(f"invalid GPU id list: {value!r}") from exc
    else:
        result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValidationError("GPU ids must be non-negative")
    if len(result) != len(set(result)):
        raise ValidationError(f"duplicate GPU ids: {result}")
    return result


@dataclass(frozen=True)
class GPULayout:
    trainer_gpu_ids: Tuple[int, ...]
    rollout_gpu_ids: Tuple[int, ...]
    expected_total: int = 4

    def __post_init__(self) -> None:
        trainer = parse_gpu_ids(self.trainer_gpu_ids)
        rollout = parse_gpu_ids(self.rollout_gpu_ids)
        object.__setattr__(self, "trainer_gpu_ids", trainer)
        object.__setattr__(self, "rollout_gpu_ids", rollout)
        if not trainer:
            raise ValidationError("at least one Trainer GPU is required")
        if not rollout:
            raise ValidationError("at least one Rollout GPU is required")
        overlap = set(trainer) & set(rollout)
        if overlap:
            raise ValidationError(f"Trainer and Rollout GPU ids overlap: {sorted(overlap)}")
        if len(trainer) + len(rollout) != self.expected_total:
            raise ValidationError(
                f"layout uses {len(trainer) + len(rollout)} GPUs, expected {self.expected_total}"
            )

    @classmethod
    def from_strings(
        cls,
        trainer_gpu_ids: str,
        rollout_gpu_ids: str,
        expected_total: int = 4,
    ) -> "GPULayout":
        return cls(
            parse_gpu_ids(trainer_gpu_ids),
            parse_gpu_ids(rollout_gpu_ids),
            expected_total,
        )

    @property
    def all_gpu_ids(self) -> Tuple[int, ...]:
        return self.trainer_gpu_ids + self.rollout_gpu_ids

    @property
    def trainer_cuda_visible_devices(self) -> str:
        return ",".join(map(str, self.trainer_gpu_ids))

    @property
    def rollout_cuda_visible_devices(self) -> str:
        return ",".join(map(str, self.rollout_gpu_ids))

    def to_dict(self) -> dict:
        return {
            "trainer_gpu_ids": list(self.trainer_gpu_ids),
            "rollout_gpu_ids": list(self.rollout_gpu_ids),
            "all_gpu_ids": list(self.all_gpu_ids),
            "trainer_world_size": len(self.trainer_gpu_ids),
            "rollout_worker_count": len(self.rollout_gpu_ids),
            "expected_total": self.expected_total,
            "trainer_cuda_visible_devices": self.trainer_cuda_visible_devices,
            "rollout_cuda_visible_devices": self.rollout_cuda_visible_devices,
        }
