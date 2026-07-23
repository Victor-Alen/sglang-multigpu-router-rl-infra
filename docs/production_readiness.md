# Production readiness

## Current decision

The four-GPU operational production gate is **accepted** for controlled RL
experiments on this host. It was executed as `long-soak-prod1` on 2026-07-23
with FSDP trainer GPUs 1,2 and SGLang rollout GPUs 3,4, using Qwen3-4B.

The final gate is:

```text
results/slime_qwen3_4b_4gpu/production_gate/long-soak-prod1/
```

`production_gate.json` is accepted and `manifest.json` verifies 15 SHA-256
artifacts. The manifest records the exact environment, GPU IDs, test count and
dirty-worktree hash.

## Evidence-backed SLOs

| Invariant | Result | Limit |
|---|---:|---:|
| Automated tests | 97 passed | >=93 |
| SGLang actor-death MTTR | 56.696 s | <=120 s |
| Network-partition MTTR | 15.963 s | <=60 s |
| Private control-plane recovery | 769.011 s | <=1200 s |
| Long-soak wall time | 13109.894 s / 3.642 h | >=7200 s |
| Long-soak batches | 1200 | >=1200 |
| Terminal/initial throughput | 1.446 | >=0.5 |
| Maximum GPU temperature | 59°C | <=90°C |
| Memory growth | 0, 0, 2, 2 MiB on GPUs 1–4 | <=2048 MiB |
| Policy lag / buffer | 1 / 1 | <=1 / <=1 |
| Soak failures / fallbacks / mixed versions | 0 / 0 / 0 | all zero |

The long-soak report excludes telemetry samples taken after the final batch
while the final checkpoint is being fsynced. Those samples represent checkpoint
high-water memory, not the steady workload; the cutoff is recorded in
`long_soak_acceptance.json`.

## Fault drills

- `actor-death-prod3`: real SGLang actor process death, Ray handle replacement,
  NCCL weight-group recreation and strict fallback. Trace proves injected,
  confirmed and recovered events for the same engine.
- `network-partition-prod4`: real namespace-local kernel `iptables OUTPUT
  REJECT` for `127.0.1.1:15000` for 10 seconds; worker discovery retries and
  recovers with no mixed policy versions.
- `host-loss-prod1`: run-private Ray control plane and owned job processes are
  stopped after iter 2; a fresh Ray control plane restores model, optimizer,
  scheduler, RNG and dataset cursor, then commits iter 4 in 769.011 seconds.

Evidence and the operator procedure are documented in
`docs/chaos_recovery_runbook.md`.

## Run the gate

```bash
TRAINER_GPU_IDS=1,2 ROLLOUT_GPU_IDS=3,4 \
  ACTOR_RUN_ID=actor-death-prod3 \
  NETWORK_RUN_ID=network-partition-prod4 \
  HOST_RUN_ID=host-loss-prod1 \
  SOAK_RUN_ID=long-soak-prod1 \
  bash scripts/run_production_gate.sh
```

The launchers reserve only declared GPUs and exact run-owned process trees.
They fail closed on foreign GPU use and never issue broad `pkill`, host reboot,
or host-firewall changes.

## Remaining boundary

This is an operationally mature controlled-experiment release, not a claim of
learning quality. The short responses in this workload are truncated at 128
tokens and reward is zero; convergence, reward quality, longer contexts and
additional models need separate gates. The source worktree is intentionally
dirty during this implementation and needs a reviewed clean commit/tag before
external distribution. Physical host/rack power loss remains an isolated-node
test because this server has other tenants.
