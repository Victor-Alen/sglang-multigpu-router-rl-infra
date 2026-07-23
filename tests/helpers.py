import time

from router.rl.models import RolloutRequest, RolloutWorkerState, WorkerLifecycle


def requests(
    group_id="g1",
    group_size=4,
    version=3,
    prompt_tokens=128,
    predicted=64,
    p90=96,
):
    return [
        RolloutRequest(
            request_id=f"{group_id}:r{i}",
            prompt_id=f"{group_id}:p",
            group_id=group_id,
            sample_index=i,
            group_size=group_size,
            policy_version=version,
            rollout_step=9,
            generation_seed=100 + i,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max(256, p90),
            predicted_output_tokens=predicted,
            predicted_output_p90=p90,
            prefix_fingerprint=f"prefix-{group_id}",
            tokenizer_revision="tok-1",
            chat_template_hash="tmpl-1",
        )
        for i in range(group_size)
    ]


def workers(version=3, count=2, prefill_tps=1000.0, decode_tps=100.0):
    return [
        RolloutWorkerState(
            worker_id=f"w{i}",
            endpoint=f"http://127.0.0.1:{30000+i}",
            lifecycle=WorkerLifecycle.READY,
            loaded_policy_version=version,
            prefill_tokens_per_second_ema=prefill_tps,
            decode_tokens_per_second_ema=decode_tps,
            heartbeat_timestamp_ns=time.time_ns(),
        )
        for i in range(count)
    ]
