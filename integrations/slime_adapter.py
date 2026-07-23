from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from router.rl.models import (
    GroupStatus,
    ResponseMetadata,
    RolloutRequest,
    RolloutWorkerState,
    WorkerLifecycle,
)
from router.rl.predictor import BucketedOutputLengthPredictor
from router.rl.runtime import RLRouterRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlimeMetadataAdapter:
    """Small compatibility boundary for external slime rollout engines."""

    tokenizer_revision: str
    chat_template_hash: str

    def to_rollout_requests(
        self,
        group_id: str,
        prompt_id: str,
        prompt_tokens: int,
        group_size: int,
        policy_version: int,
        rollout_step: int,
        prefix_fingerprint: str,
        sampling: Mapping[str, Any],
        seeds: Iterable[int],
    ) -> list[RolloutRequest]:
        seed_list = list(seeds)
        if len(seed_list) != group_size:
            raise ValueError("one generation seed is required per group sample")
        max_new_tokens = int(sampling.get("max_new_tokens", 256))
        predicted = float(sampling.get("predicted_output_tokens", max_new_tokens * 0.6))
        p90 = float(sampling.get("predicted_output_p90", min(max_new_tokens, predicted * 1.35)))
        return [
            RolloutRequest(
                request_id=f"{group_id}:{index}",
                prompt_id=prompt_id,
                group_id=group_id,
                sample_index=index,
                group_size=group_size,
                policy_version=policy_version,
                rollout_step=rollout_step,
                generation_seed=seed,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                predicted_output_tokens=predicted,
                predicted_output_p90=p90,
                prefix_fingerprint=prefix_fingerprint,
                temperature=float(sampling.get("temperature", 1.0)),
                top_p=float(sampling.get("top_p", 1.0)),
                top_k=(
                    int(sampling["top_k"])
                    if sampling.get("top_k") is not None and int(sampling["top_k"]) > 0
                    else None
                ),
                tokenizer_revision=self.tokenizer_revision,
                chat_template_hash=self.chat_template_hash,
            )
            for index, seed in enumerate(seed_list)
        ]


def parse_policy_version(value: Any, default: int = 0) -> int:
    """Normalize SGLang's string weight version into the router integer domain."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def endpoint_to_worker_id(endpoint: str) -> str:
    digest = hashlib.sha256(endpoint.rstrip("/").encode("utf-8")).hexdigest()[:10]
    return f"sglang-{digest}"


def _stable_seed(workload_id: str, sample_index: int) -> int:
    """Derive a replayable seed from workload identity, never from a run id."""
    digest = hashlib.sha256(f"{workload_id}:{sample_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


@dataclass
class _LiveGroupDecision:
    requests: list[RolloutRequest]
    worker_by_request: dict[str, str]


class SlimeRLRouteCoordinator:
    """Bridge slime's per-sample hook to a group-level routing decision.

    slime invokes the custom generation function concurrently for every sample
    in a group. The first caller constructs the complete virtual group and
    persists one decision; all other callers reuse it. Requests are then sent
    directly to the selected SGLang engine, so this is the real rollout data
    path rather than an offline replay.
    """

    def __init__(self, args: Any) -> None:
        run_id = os.environ.get("RL_ROUTER_RUN_ID", f"pid-{os.getpid()}")
        configured_state = os.environ.get("RL_ROUTER_STATE_DIR")
        state_dir = Path(configured_state or Path(args.save) / "rl_router_state") / run_id
        self.run_id = run_id
        self.policy_name = os.environ.get("RL_ROUTER_POLICY", "adaptive-group")
        self.dataset = os.environ.get("RL_ROUTER_DATASET", "dapo-math-17k")
        self.tokenizer_revision = os.environ.get("RL_TOKENIZER_REVISION", "unknown")
        self.chat_template_hash = os.environ.get("RL_CHAT_TEMPLATE_HASH", "unknown")
        self.heartbeat_timeout = float(os.environ.get("RL_ROUTER_HEARTBEAT_TIMEOUT", "30"))
        self.prefill_tps = float(os.environ.get("RL_ROUTER_PREFILL_TPS", "2000"))
        self.decode_tps = float(os.environ.get("RL_ROUTER_DECODE_TPS", "50"))
        self.state_dir = state_dir
        self.runtime: RLRouterRuntime | None = None
        self.predictor = BucketedOutputLengthPredictor()
        self._groups: dict[str, _LiveGroupDecision] = {}
        self._lock = asyncio.Lock()

    async def _discover_workers(self, args: Any) -> list[RolloutWorkerState]:
        from slime.utils.http_utils import get

        retries = int(os.environ.get("RL_ROUTER_DISCOVERY_RETRIES", "15"))
        retry_delay = float(os.environ.get("RL_ROUTER_DISCOVERY_RETRY_SECONDS", "1"))
        if retries < 1 or retry_delay < 0:
            raise ValueError("invalid worker discovery retry policy")

        async def probe(url: str) -> RolloutWorkerState:
            model_info, loads = await asyncio.gather(
                get(f"{url}/model_info"),
                get(f"{url}/get_load"),
            )
            if isinstance(loads, dict):
                loads = [loads]
            num_reqs = sum(int(item.get("num_reqs", 0)) for item in loads)
            num_waiting = sum(int(item.get("num_waiting_reqs", 0)) for item in loads)
            num_tokens = sum(int(item.get("num_tokens", 0)) for item in loads)
            worker_id = endpoint_to_worker_id(url)
            existing = self.runtime.workers.get(worker_id) if self.runtime else None
            return RolloutWorkerState(
                worker_id=worker_id,
                endpoint=url,
                lifecycle=WorkerLifecycle.READY,
                loaded_policy_version=parse_policy_version(model_info.get("weight_version")),
                running_requests=max(0, num_reqs - num_waiting),
                queued_requests=num_waiting,
                active_decode_tokens=num_tokens,
                prefill_tokens_per_second_ema=(
                    existing.prefill_tokens_per_second_ema if existing else self.prefill_tps
                ),
                decode_tokens_per_second_ema=(
                    existing.decode_tokens_per_second_ema if existing else self.decode_tps
                ),
                prefix_matched_tokens=(dict(existing.prefix_matched_tokens) if existing else {}),
            )

        router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        for attempt in range(1, retries + 1):
            try:
                response = await get(f"{router}/list_workers")
                urls = sorted(url.rstrip("/") for url in response.get("urls", []))
                if not urls:
                    raise RuntimeError("slime router has no registered rollout workers")
                return list(await asyncio.gather(*(probe(url) for url in urls)))
            except Exception as exc:
                if attempt == retries:
                    raise
                logger.warning(
                    "Worker discovery attempt %s/%s failed; retrying in %.2fs: %s",
                    attempt,
                    retries,
                    retry_delay,
                    exc,
                )
                if self.runtime is not None:
                    self.runtime.trace.write(
                        "worker_discovery_retry",
                        {
                            "attempt": attempt,
                            "max_attempts": retries,
                            "retry_seconds": retry_delay,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                await asyncio.sleep(retry_delay)
        raise AssertionError("unreachable")

    def _refresh_runtime(self, workers: list[RolloutWorkerState]) -> None:
        if self.runtime is None:
            self.runtime = RLRouterRuntime(
                workers,
                policy_name=self.policy_name,
                state_dir=self.state_dir,
                heartbeat_timeout_seconds=self.heartbeat_timeout,
            )
            return

        discovered_ids = {worker.worker_id for worker in workers}
        for worker in workers:
            if worker.worker_id not in self.runtime.workers:
                self.runtime.register_worker(worker)
                continue
            previous = self.runtime.workers[worker.worker_id].loaded_policy_version
            self.runtime.heartbeat(worker.worker_id, worker.to_dict())
            if previous != worker.loaded_policy_version:
                self.runtime.trace.write(
                    "framework_weight_version_observed",
                    {
                        "worker_id": worker.worker_id,
                        "from_version": previous,
                        "to_version": worker.loaded_policy_version,
                    },
                )
        for worker_id, worker in self.runtime.workers.items():
            if worker_id not in discovered_ids:
                worker.lifecycle = WorkerLifecycle.FAILED

    def _build_group_requests(
        self,
        args: Any,
        sample: Any,
        sampling_params: Mapping[str, Any],
        prompt_ids: list[int],
        policy_version: int,
    ) -> list[RolloutRequest]:
        group_size = int(args.n_samples_per_prompt)
        sample_index = int(sample.index) % group_size
        group_id = f"{self.run_id}:group-{sample.group_index}"
        prompt_digest = hashlib.sha256(
            ",".join(str(token) for token in prompt_ids).encode("utf-8")
        ).hexdigest()
        current_seed = sampling_params.get("sampling_seed")
        if current_seed is None:
            workload_id = f"group-{sample.group_index}:{prompt_digest}"
            seeds = [_stable_seed(workload_id, index) for index in range(group_size)]
        else:
            seed_base = int(current_seed) - sample_index
            seeds = [seed_base + index for index in range(group_size)]
        max_new_tokens = int(sampling_params["max_new_tokens"])
        prediction = self.predictor.predict(self.dataset, len(prompt_ids), max_new_tokens)
        adapter = SlimeMetadataAdapter(self.tokenizer_revision, self.chat_template_hash)
        requests = adapter.to_rollout_requests(
            group_id=group_id,
            prompt_id=prompt_digest,
            prompt_tokens=len(prompt_ids),
            group_size=group_size,
            policy_version=policy_version,
            rollout_step=max(0, policy_version - 1),
            prefix_fingerprint=prompt_digest,
            sampling={
                **sampling_params,
                "predicted_output_tokens": prediction.mean,
                "predicted_output_p90": prediction.p90,
            },
            seeds=seeds,
        )
        return requests

    async def assignment_for(
        self,
        args: Any,
        sample: Any,
        sampling_params: Mapping[str, Any],
        prompt_ids: list[int],
    ) -> tuple[RolloutRequest, RolloutWorkerState]:
        group_size = int(args.n_samples_per_prompt)
        sample_index = int(sample.index) % group_size
        group_key = f"{self.run_id}:group-{sample.group_index}"
        async with self._lock:
            context = self._groups.get(group_key)
            if context is None:
                workers = await self._discover_workers(args)
                self._refresh_runtime(workers)
                target_version = max(worker.loaded_policy_version for worker in workers)
                requests = self._build_group_requests(
                    args, sample, sampling_params, prompt_ids, target_version
                )
                assert self.runtime is not None
                decision = self.runtime.route(requests)
                worker_by_request = {
                    request_id: worker_id
                    for worker_id, request_ids in decision.assignments.items()
                    for request_id in request_ids
                }
                context = _LiveGroupDecision(requests, worker_by_request)
                self._groups[group_key] = context
            request = context.requests[sample_index]
            worker_id = context.worker_by_request[request.request_id]
            worker = self.runtime.workers[worker_id]  # type: ignore[union-attr]
            worker.running_requests += 1
            return request, worker

    async def finish(
        self,
        request: RolloutRequest,
        worker: RolloutWorkerState,
        sample: Any,
        generated_tokens: int,
    ) -> None:
        assert self.runtime is not None
        served_version = parse_policy_version(
            sample.weight_versions[-1] if sample.weight_versions else request.policy_version,
            request.policy_version,
        )
        response = ResponseMetadata(
            request_id=request.request_id,
            group_id=request.group_id,
            sample_index=request.sample_index,
            worker_id=worker.worker_id,
            requested_policy_version=request.policy_version,
            served_policy_version=served_version,
            prompt_tokens=request.prompt_tokens,
            generated_tokens=generated_tokens,
            prefix_matched_tokens=worker.matched_prefix_tokens(
                request.prefix_fingerprint, request.prompt_tokens
            ),
            finish_reason=sample.status.value,
            generation_seed=request.generation_seed,
            tokenizer_revision=request.tokenizer_revision,
            chat_template_hash=request.chat_template_hash,
        )
        group = self.runtime.complete(response, consumed_at_policy_version=request.policy_version)
        worker.prefix_matched_tokens[request.prefix_fingerprint] = request.prompt_tokens
        self.predictor.observe(
            self.dataset, request.prompt_tokens, request.max_new_tokens, generated_tokens
        )
        sample.metadata.update(
            {
                "rl_group_id": request.group_id,
                "rl_sample_index": request.sample_index,
                "rl_worker_id": worker.worker_id,
                "rl_worker_endpoint": worker.endpoint,
                "requested_policy_version": request.policy_version,
                "served_policy_version": served_version,
                "generation_seed": request.generation_seed,
            }
        )
        if group["status"] == GroupStatus.READY.value:
            self._groups.pop(request.group_id, None)

    async def fail(self, request: RolloutRequest, worker: RolloutWorkerState, exc: Exception) -> None:
        assert self.runtime is not None
        self.runtime.tracker.record_failure(
            request.group_id, request.sample_index, f"{type(exc).__name__}: {exc}"
        )
        self.runtime.trace.write(
            "slime_sample_failed",
            {
                "request_id": request.request_id,
                "group_id": request.group_id,
                "worker_id": worker.worker_id,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    async def release(self, worker: RolloutWorkerState) -> None:
        async with self._lock:
            worker.running_requests = max(0, worker.running_requests - 1)
            worker.heartbeat_timestamp_ns = time.time_ns()


_COORDINATORS: dict[int, SlimeRLRouteCoordinator] = {}


async def generate_with_rl_router(args: Any, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """slime ``--custom-generate-function-path`` entrypoint."""
    from slime.rollout.sglang_rollout import GenerateState, generate
    from slime.utils.processing_utils import prepare_model_inputs

    coordinator = _COORDINATORS.get(id(args))
    if coordinator is None:
        coordinator = SlimeRLRouteCoordinator(args)
        _COORDINATORS[id(args)] = coordinator
    state = GenerateState(args)
    prompt_ids, _ = prepare_model_inputs(
        sample.prompt,
        state.tokenizer,
        state.processor,
        sample.metadata,
        args.apply_chat_template_kwargs,
    )
    request, worker = await coordinator.assignment_for(args, sample, sampling_params, prompt_ids)
    sampling_params["sampling_seed"] = request.generation_seed
    endpoint = urlsplit(worker.endpoint)
    if not endpoint.hostname or not endpoint.port:
        raise RuntimeError(f"invalid SGLang worker endpoint: {worker.endpoint}")
    direct_args = copy.copy(args)
    direct_args.sglang_router_ip = endpoint.hostname
    direct_args.sglang_router_port = endpoint.port
    direct_args.use_slime_router = False
    before_tokens = sample.response_length
    try:
        result = await generate(direct_args, sample, sampling_params)
        await coordinator.finish(request, worker, result, result.response_length - before_tokens)
        return result
    except Exception as exc:
        await coordinator.fail(request, worker, exc)
        raise
    finally:
        await coordinator.release(worker)
