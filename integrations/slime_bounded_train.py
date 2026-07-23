"""slime v0.2.1 training entrypoint with one-batch bounded rollout overlap.

The next rollout is generated while the current batch trains.  Weight updates
wait for both operations, so no request can straddle an update.  A failed or
invalid prefetch is discarded and regenerated synchronously after the update.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from integrations.bounded_pipeline import (
    BoundedPipelineStats,
    PipelineTraceWriter,
    count_generated_tokens,
    normalize_weight_versions,
)
from integrations.chaos import OutputPortReject, schedule_partition_removal
from router.rl.buffer import BoundedRolloutBuffer, VersionedBatch
from router.rl.models import ValidationError
from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking


logger = logging.getLogger(__name__)


def _resolve_partitions(rollout_data_ref: list[Any]) -> list[dict[str, Any]]:
    partitions: list[dict[str, Any]] = []
    for boxed in rollout_data_ref:
        value = getattr(boxed, "inner", boxed)
        if isinstance(value, ray.ObjectRef):
            value = ray.get(value)
        if not isinstance(value, dict):
            raise TypeError(f"unexpected rollout partition type: {type(value)!r}")
        partitions.append(value)
    return partitions


def _get_policy_version(rollout_manager: Any) -> int:
    versions = ray.get(rollout_manager.get_weight_versions.remote())
    return normalize_weight_versions(versions)


def _trace_path() -> Path:
    state_dir = Path(os.environ.get("RL_ROUTER_STATE_DIR", "results/rl_state"))
    run_id = os.environ.get("RL_ROUTER_RUN_ID", "bounded-async")
    return state_dir / run_id / "bounded_async_trace.jsonl"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    target = path.with_name("bounded_async_summary.json")
    temporary = target.with_suffix(".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def train(args: Any) -> None:
    configure_logger()
    max_batches = int(os.environ.get("RL_MAX_PREFETCHED_BATCHES", "1"))
    max_lag = int(os.environ.get("RL_MAX_POLICY_LAG", "1"))
    strict_fallback = os.environ.get("RL_ASYNC_STRICT_FALLBACK", "1") != "0"
    fault_prefetch_rollout_id = int(os.environ.get("RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID", "-1"))
    fault_injected = False
    chaos_engine_rollout_id = int(os.environ.get("RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID", "-1"))
    chaos_engine_index = int(os.environ.get("RL_CHAOS_ENGINE_INDEX", "0"))
    chaos_network_rollout_id = int(os.environ.get("RL_CHAOS_NETWORK_ROLLOUT_ID", "-1"))
    destructive_chaos = os.environ.get("RL_ENABLE_DESTRUCTIVE_CHAOS", "0") == "1"
    actor_death_injected = False
    actor_death_recovered = False
    network_partition: OutputPortReject | None = None
    network_timer = None
    network_recovered = False
    if max_batches != 1:
        raise ValueError("bounded async requires RL_MAX_PREFETCHED_BATCHES=1")
    if max_lag not in (0, 1):
        raise ValueError("RL_MAX_POLICY_LAG must be 0 or 1")
    if args.offload_rollout:
        raise ValueError("bounded async requires dedicated, non-offloaded rollout GPUs")

    trace_path = _trace_path()
    trace = PipelineTraceWriter(trace_path)
    stats = BoundedPipelineStats.start()
    buffer: BoundedRolloutBuffer[list[Any]] = BoundedRolloutBuffer(
        max_batches=max_batches,
        max_policy_lag=max_lag,
    )

    pgs = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    def train_batch(rollout_id: int, rollout_data_ref: list[Any]) -> None:
        if args.use_critic:
            critic_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

    def save_batch(rollout_id: int) -> None:
        if not should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch):
            return
        if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
            actor_model.save_model(rollout_id)
        if args.use_critic:
            critic_model.save_model(rollout_id)
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    def clear_train_memory(rollout_id: int) -> None:
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def generate_sync(rollout_id: int, expected_version: int, reason: str) -> VersionedBatch[list[Any]]:
        observed_before = _get_policy_version(rollout_manager)
        if observed_before != expected_version:
            raise RuntimeError(
                f"rollout version changed before generation: expected={expected_version}, observed={observed_before}"
            )
        payload = ray.get(rollout_manager.generate.remote(rollout_id))
        observed_after = _get_policy_version(rollout_manager)
        if observed_after != observed_before:
            raise RuntimeError(
                f"rollout version changed during generation: before={observed_before}, after={observed_after}"
            )
        tokens = count_generated_tokens(_resolve_partitions(payload))
        trace.write(
            "rollout_generated",
            rollout_id=rollout_id,
            generated_policy_version=observed_before,
            generated_tokens=tokens,
            mode=reason,
        )
        return VersionedBatch(str(rollout_id), observed_before, payload, tokens)

    trace.write(
        "pipeline_started",
        max_prefetched_batches=max_batches,
        max_policy_lag=max_lag,
        strict_fallback=strict_fallback,
    )

    try:
        actor_model.update_weights()
        current_version = _get_policy_version(rollout_manager)
        trace.write("weight_update_complete", policy_version=current_version, phase="initial")

        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))
            return
        if args.start_rollout_id >= args.num_rollout:
            return

        if args.eval_interval is not None and args.start_rollout_id == 0:
            ray.get(rollout_manager.eval.remote(0))

        first = generate_sync(args.start_rollout_id, current_version, "initial")
        buffer.put(first, trainer_policy_version=current_version)

        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            batch = buffer.get(trainer_policy_version=current_version)
            if batch is None:
                raise RuntimeError(f"bounded rollout buffer is empty at rollout {rollout_id}")
            lag = current_version - batch.generated_policy_version
            trace.write(
                "batch_consumed",
                rollout_id=rollout_id,
                generated_policy_version=batch.generated_policy_version,
                consumed_policy_version=current_version,
                policy_lag=lag,
                generated_tokens=batch.generated_tokens,
                buffered_batches=len(buffer),
            )

            next_rollout_id = rollout_id + 1
            prefetch_handle = None
            prefetch_version = current_version
            if next_rollout_id < args.num_rollout and max_lag == 1:
                observed = _get_policy_version(rollout_manager)
                if observed == current_version:
                    prefetch_handle = rollout_manager.generate.remote(next_rollout_id)
                    trace.write(
                        "prefetch_started",
                        rollout_id=next_rollout_id,
                        generated_policy_version=prefetch_version,
                    )

                    if next_rollout_id == chaos_engine_rollout_id and not actor_death_injected:
                        if not destructive_chaos:
                            raise RuntimeError("actor death injection requires RL_ENABLE_DESTRUCTIVE_CHAOS=1")
                        engines, _, _ = ray.get(rollout_manager.get_rollout_engines_and_lock.remote())
                        if not 0 <= chaos_engine_index < len(engines):
                            raise ValueError(f"invalid chaos engine index: {chaos_engine_index}")
                        actor_death_injected = True
                        trace.write(
                            "actor_death_injected",
                            rollout_id=next_rollout_id,
                            engine_index=chaos_engine_index,
                        )
                        try:
                            ray.get(engines[chaos_engine_index].fault_inject_actor_death.remote())
                            raise RuntimeError("actor-death injection unexpectedly returned")
                        except ray.exceptions.RayError:
                            trace.write(
                                "actor_death_confirmed",
                                rollout_id=next_rollout_id,
                                engine_index=chaos_engine_index,
                            )
                        replacement = ray.get(
                            rollout_manager.fault_recover_rollout_engine.remote(chaos_engine_index)
                        )
                        trace.write(
                            "actor_replacement_ready",
                            rollout_id=next_rollout_id,
                            engine_index=chaos_engine_index,
                            new_engines=int(replacement["new_engines"]),
                        )

                    if next_rollout_id == chaos_network_rollout_id and network_partition is None:
                        if not destructive_chaos:
                            raise RuntimeError("network partition requires RL_ENABLE_DESTRUCTIVE_CHAOS=1")
                        network_host = os.environ["RL_CHAOS_NETWORK_HOST"]
                        network_port = int(os.environ.get("RL_CHAOS_NETWORK_PORT", "15000"))
                        network_seconds = float(os.environ.get("RL_CHAOS_NETWORK_SECONDS", "10"))
                        run_id = os.environ.get("RL_ROUTER_RUN_ID", "bounded")
                        label = "sglang-chaos-" + hashlib.sha256(run_id.encode()).hexdigest()[:16]
                        network_partition = OutputPortReject(network_host, network_port, label)
                        network_partition.install()
                        trace.write(
                            "network_partition_started",
                            rollout_id=next_rollout_id,
                            host=network_host,
                            port=network_port,
                            duration_seconds=network_seconds,
                            label=label,
                        )
                        ended_rollout_id = next_rollout_id
                        ended_host = network_host
                        ended_port = network_port
                        ended_label = label
                        network_timer = schedule_partition_removal(
                            network_partition,
                            network_seconds,
                            lambda: trace.write(
                                "network_partition_ended",
                                rollout_id=ended_rollout_id,
                                host=ended_host,
                                port=ended_port,
                                label=ended_label,
                            ),
                        )

            train_batch(rollout_id, batch.payload)
            stats.observe_batch(tokens=batch.generated_tokens, lag=lag, buffered_batches=len(buffer))

            prefetched: VersionedBatch[list[Any]] | None = None
            if prefetch_handle is not None:
                try:
                    payload = ray.get(prefetch_handle)
                    if next_rollout_id == fault_prefetch_rollout_id and not fault_injected:
                        fault_injected = True
                        trace.write(
                            "fault_injected",
                            fault="prefetch_result_failure",
                            rollout_id=next_rollout_id,
                        )
                        raise RuntimeError(
                            f"injected prefetch result failure at rollout {next_rollout_id}"
                        )
                    tokens = count_generated_tokens(_resolve_partitions(payload))
                    observed = _get_policy_version(rollout_manager)
                    if observed != prefetch_version:
                        raise RuntimeError(
                            f"prefetch version changed: expected={prefetch_version}, observed={observed}"
                        )
                    prefetched = VersionedBatch(str(next_rollout_id), prefetch_version, payload, tokens)
                    trace.write(
                        "rollout_generated",
                        rollout_id=next_rollout_id,
                        generated_policy_version=prefetch_version,
                        generated_tokens=tokens,
                        mode="prefetch",
                    )
                except Exception as exc:
                    stats.prefetch_failures += 1
                    trace.write(
                        "prefetch_failed",
                        rollout_id=next_rollout_id,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    if not strict_fallback:
                        raise

            save_batch(rollout_id)
            clear_train_memory(rollout_id)
            actor_model.update_weights()
            updated_version = _get_policy_version(rollout_manager)
            if updated_version != current_version + 1:
                raise RuntimeError(
                    f"non-monotonic policy update: previous={current_version}, updated={updated_version}"
                )
            current_version = updated_version
            trace.write("weight_update_complete", policy_version=current_version, rollout_id=rollout_id)
            if actor_death_injected and not actor_death_recovered:
                actor_death_recovered = True
                trace.write(
                    "actor_death_recovered",
                    rollout_id=rollout_id,
                    policy_version=current_version,
                    engine_index=chaos_engine_index,
                )
            if (
                network_partition is not None
                and network_timer is not None
                and not network_recovered
                and not network_timer.is_alive()
            ):
                network_recovered = True
                trace.write(
                    "network_partition_recovered",
                    rollout_id=rollout_id,
                    policy_version=current_version,
                )

            if next_rollout_id < args.num_rollout:
                if prefetched is not None:
                    try:
                        buffer.put(prefetched, trainer_policy_version=current_version)
                    except ValidationError as exc:
                        stats.stale_batches += 1
                        stats.stale_tokens += prefetched.generated_tokens
                        trace.write(
                            "prefetch_rejected",
                            rollout_id=next_rollout_id,
                            generated_policy_version=prefetched.generated_policy_version,
                            trainer_policy_version=current_version,
                            error=str(exc),
                        )
                        prefetched = None
                        if not strict_fallback:
                            raise
                if prefetched is None:
                    stats.strict_fallbacks += 1
                    fallback = generate_sync(next_rollout_id, current_version, "strict_fallback")
                    buffer.put(fallback, trainer_policy_version=current_version)
                    trace.write("strict_fallback_complete", rollout_id=next_rollout_id)
                stats.max_buffered_batches = max(stats.max_buffered_batches, len(buffer))

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))

        summary = stats.summary()
        summary.update(
            {
                "final_policy_version": current_version,
                "buffer_stale_batches": buffer.stale_batches,
                "buffer_stale_tokens": buffer.stale_tokens,
                "status": "complete",
            }
        )
        _write_summary(trace_path, summary)
        trace.write("pipeline_complete", **summary)
        logger.info("Bounded async summary: %s", json.dumps(summary, sort_keys=True))
    except Exception as exc:
        summary = stats.summary()
        summary.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        _write_summary(trace_path, summary)
        trace.write("pipeline_failed", **summary)
        raise
    finally:
        if network_timer is not None and network_timer.is_alive():
            network_timer.cancel()
        if network_partition is not None:
            network_partition.remove()
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    train(parse_args())
