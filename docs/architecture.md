# RL rollout router architecture

The implementation separates three concerns:

1. `router/rl/` is framework-independent control logic. Data models, group
   placement, barriers, version transitions, buffering, metrics and replay do
   not call HTTP, Ray or SGLang.
2. `router/rl_api.py` exposes the control plane under `/rl/*` without changing
   the existing OpenAI-compatible serving routes.
3. `integrations/` translates framework metadata and performs SGLang RPCs.

The production slime data path is:

```text
slime/GRPO generate_and_rm_group
  -> integrations.slime_adapter.generate_with_rl_router
  -> discover /list_workers + probe /model_info and /get_load
  -> construct the complete virtual group once
  -> version/health eligibility filter
  -> PACK/SPLIT candidate enumeration and scoring
  -> direct /generate to the assigned SGLang worker
  -> record served weight_version and response metadata
  -> GroupTracker barrier
  -> reward/trainer only after READY
```

The `/rl/route` and `/rl/responses` HTTP endpoints expose the same runtime for
external controllers. They are not required by the in-process slime adapter,
which avoids a second proxy hop while preserving identical traces and
correctness checks.

The control path is:

```text
begin update -> DRAINING -> real SGLang update RPC -> fixed-input probe
  -> checksum/version verification -> READY -> commit
```

`POST /rl/execute-weight-update` executes this complete path. State is written
atomically; an interrupted update is aborted after Router restart and affected
workers enter `RECOVERING` until live probes re-establish their version.

Rolling update keeps old- and new-version workers in separate eligible pools.
The hard filter `request.policy_version == worker.loaded_policy_version` runs
before every policy, including fallbacks. If no matching READY worker exists,
routing fails closed.

Adaptive routing enumerates PACK-to-each-worker, EVEN SPLIT and load-proportional
SPLIT. It estimates queue, non-reused prefill, risk-adjusted output length,
makespan, duplicated prefill and worker skew. Every candidate and selected reason
is written to JSONL so the same decision can be reproduced offline.

Bounded asynchronous training is an opt-in slime entrypoint. It holds exactly
one prefetched batch and permits policy lag of at most one:

```text
generate batch N at vK -> train N while generate N+1 at vK
  -> wait for both -> update rollout weights to vK+1
  -> consume N+1 with lag 1
```

No generation overlaps a weight update. A failed prefetch, mixed engine version
or stale batch is discarded and regenerated synchronously after the update.
`bounded_async_trace.jsonl` records generated/consumed versions, lag, buffer
occupancy, fallback events and FreshTokenThroughput for acceptance.
