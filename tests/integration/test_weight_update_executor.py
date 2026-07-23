import asyncio

import pytest

from integrations.weight_update_executor import WeightUpdateExecutor, fixed_probe_digest
from router.rl.models import ValidationError, WorkerLifecycle
from router.rl.versioning import PolicyVersionCoordinator
from tests.helpers import workers


class FakeAdapter:
    def __init__(self, probe_text="same", fail=False):
        self.probe_text = probe_text
        self.fail = fail
        self.version = 3

    def update_weights_from_disk(
        self,
        model_path,
        policy_version,
        load_format=None,
        abort_all_requests=False,
        keep_pause=False,
        flush_cache=True,
    ):
        if self.fail:
            raise RuntimeError("injected update failure")
        self.version = policy_version
        return {"success": True, "model_info": {"weight_version": str(policy_version)}}

    def fixed_input_probe(self, prompt, max_new_tokens=1):
        return {
            "id": f"variable-{id(self)}",
            "choices": [
                {
                    "text": self.probe_text,
                    "finish_reason": "length",
                    "logprobs": {"token_logprobs": [-0.1]},
                }
            ],
        }


def test_probe_digest_ignores_request_identity():
    first = FakeAdapter().fixed_input_probe("x")
    second = {**first, "id": "another-id"}
    assert fixed_probe_digest(first) == fixed_probe_digest(second)


def test_execute_stop_the_world_update_and_probe():
    candidates = workers(version=3)
    coordinator = PolicyVersionCoordinator(candidates)
    executor = WeightUpdateExecutor(
        coordinator,
        {"w0": FakeAdapter(), "w1": FakeAdapter()},
        poll_interval_seconds=0.001,
    )
    result = asyncio.run(
        executor.execute_disk_update(
            model_path="/checkpoint/v4",
            from_version=3,
            to_version=4,
            expected_checksum="manifest-sha",
            probe_prompt="2+2=",
        )
    )
    assert result.status == "COMMITTED"
    assert set(result.workers) == {"w0", "w1"}
    assert len({item.probe_digest for item in result.workers.values()}) == 1
    assert all(worker.loaded_policy_version == 4 for worker in candidates)
    assert all(worker.lifecycle == WorkerLifecycle.READY for worker in candidates)


def test_probe_mismatch_aborts_and_isolates_updated_workers():
    candidates = workers(version=3)
    coordinator = PolicyVersionCoordinator(candidates)
    executor = WeightUpdateExecutor(
        coordinator,
        {"w0": FakeAdapter("a"), "w1": FakeAdapter("b")},
    )
    with pytest.raises(ValidationError, match="probe mismatch"):
        asyncio.run(
            executor.execute_disk_update(
                model_path="/checkpoint/v4",
                from_version=3,
                to_version=4,
                probe_prompt="probe",
            )
        )
    assert coordinator.active is not None
    assert coordinator.active.status == "ABORTED"
    assert all(worker.lifecycle == WorkerLifecycle.FAILED for worker in candidates)


def test_adapter_failure_aborts_update():
    candidates = workers(version=3)
    coordinator = PolicyVersionCoordinator(candidates)
    executor = WeightUpdateExecutor(
        coordinator,
        {"w0": FakeAdapter(fail=True), "w1": FakeAdapter()},
    )
    with pytest.raises(RuntimeError, match="injected"):
        asyncio.run(
            executor.execute_disk_update(
                model_path="/checkpoint/v4", from_version=3, to_version=4
            )
        )
    assert coordinator.active is not None
    assert coordinator.active.status == "ABORTED"
    assert candidates[0].lifecycle == WorkerLifecycle.FAILED


def test_restart_aborts_incomplete_update_and_requires_worker_recovery(tmp_path):
    state_path = tmp_path / "version_state.json"
    first_workers = workers(version=3)
    first = PolicyVersionCoordinator(first_workers, state_path=state_path)
    update = first.begin_update(3, 4)
    first.mark_drained("w0")

    restarted_workers = workers(version=3)
    restarted = PolicyVersionCoordinator(restarted_workers, state_path=state_path)

    assert restarted.active is not None
    assert restarted.active.update_id == update.update_id
    assert restarted.active.status == "ABORTED"
    assert all(worker.lifecycle == WorkerLifecycle.RECOVERING for worker in restarted_workers)
