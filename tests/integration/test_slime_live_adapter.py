import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from integrations.slime_adapter import (
    SlimeRLRouteCoordinator,
    endpoint_to_worker_id,
    parse_policy_version,
)
from router.rl.models import RolloutWorkerState, WorkerLifecycle


def test_policy_version_and_worker_identity_are_stable():
    endpoint = "http://127.0.0.1:15000"
    assert parse_policy_version("7") == 7
    assert parse_policy_version("default", default=3) == 3
    assert endpoint_to_worker_id(endpoint) == endpoint_to_worker_id(endpoint + "/")


def test_slime_group_metadata_is_constructed_once_for_all_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_ROUTER_RUN_ID", "test-run")
    monkeypatch.setenv("RL_ROUTER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("RL_TOKENIZER_REVISION", "tokenizer-rev")
    monkeypatch.setenv("RL_CHAT_TEMPLATE_HASH", "template-sha")
    args = Namespace(save=str(tmp_path), n_samples_per_prompt=4)
    sample = SimpleNamespace(index=9, group_index=2)
    coordinator = SlimeRLRouteCoordinator(args)

    requests = coordinator._build_group_requests(
        args,
        sample,
        {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": -1,
            "max_new_tokens": 256,
            "sampling_seed": 1001,
        },
        [10, 20, 30],
        policy_version=3,
    )

    assert [request.sample_index for request in requests] == [0, 1, 2, 3]
    assert [request.generation_seed for request in requests] == [1000, 1001, 1002, 1003]
    assert {request.group_id for request in requests} == {"test-run:group-2"}
    assert {request.policy_version for request in requests} == {3}
    assert {request.rollout_step for request in requests} == {2}
    assert {request.top_k for request in requests} == {None}
    assert {request.tokenizer_revision for request in requests} == {"tokenizer-rev"}


def test_fallback_generation_seed_is_independent_of_run_id(tmp_path, monkeypatch):
    args = Namespace(save=str(tmp_path), n_samples_per_prompt=2)
    sample = SimpleNamespace(index=0, group_index=7)
    sampling = {"temperature": 0.8, "top_p": 1.0, "max_new_tokens": 128}

    seeds_by_run = []
    for run_id in ("strict-run", "bounded-run"):
        monkeypatch.setenv("RL_ROUTER_RUN_ID", run_id)
        monkeypatch.setenv("RL_ROUTER_STATE_DIR", str(tmp_path))
        coordinator = SlimeRLRouteCoordinator(args)
        requests = coordinator._build_group_requests(
            args, sample, sampling, [11, 22, 33], policy_version=1
        )
        seeds_by_run.append([request.generation_seed for request in requests])

    assert seeds_by_run[0] == seeds_by_run[1]


def test_runtime_records_framework_version_transition(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_ROUTER_RUN_ID", "version-test")
    monkeypatch.setenv("RL_ROUTER_STATE_DIR", str(tmp_path))
    args = Namespace(save=str(tmp_path), n_samples_per_prompt=2)
    coordinator = SlimeRLRouteCoordinator(args)
    worker_id = endpoint_to_worker_id("http://127.0.0.1:15000")
    first = RolloutWorkerState(
        worker_id=worker_id,
        endpoint="http://127.0.0.1:15000",
        lifecycle=WorkerLifecycle.READY,
        loaded_policy_version=1,
    )
    coordinator._refresh_runtime([first])
    second = RolloutWorkerState(
        worker_id=worker_id,
        endpoint=first.endpoint,
        lifecycle=WorkerLifecycle.READY,
        loaded_policy_version=2,
    )
    coordinator._refresh_runtime([second])

    assert coordinator.runtime is not None
    assert coordinator.runtime.workers[worker_id].loaded_policy_version == 2
    trace = (tmp_path / "version-test" / "trace.jsonl").read_text(encoding="utf-8")
    assert "framework_weight_version_observed" in trace


def test_worker_discovery_retries_transient_transport_failure(tmp_path, monkeypatch):
    pytest.importorskip("slime.utils.http_utils")
    import slime.utils.http_utils

    monkeypatch.setenv("RL_ROUTER_RUN_ID", "retry-test")
    monkeypatch.setenv("RL_ROUTER_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("RL_ROUTER_DISCOVERY_RETRIES", "2")
    monkeypatch.setenv("RL_ROUTER_DISCOVERY_RETRY_SECONDS", "0")
    calls = {"model_info": 0}

    async def fake_get(url):
        if url.endswith("/list_workers"):
            return {"urls": ["http://127.0.0.1:15000"]}
        if url.endswith("/model_info"):
            calls["model_info"] += 1
            if calls["model_info"] == 1:
                raise ConnectionError("transient reset")
            return {"weight_version": "3"}
        if url.endswith("/get_load"):
            return [{"num_reqs": 0, "num_waiting_reqs": 0, "num_tokens": 0}]
        raise AssertionError(url)

    monkeypatch.setattr(slime.utils.http_utils, "get", fake_get)
    args = Namespace(
        save=str(tmp_path),
        n_samples_per_prompt=2,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=3000,
    )
    workers = asyncio.run(SlimeRLRouteCoordinator(args)._discover_workers(args))
    assert len(workers) == 1
    assert workers[0].loaded_policy_version == 3
    assert calls["model_info"] == 2
