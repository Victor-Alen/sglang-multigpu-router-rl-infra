# Chaos and recovery runbook

Status: validated on 2026-07-23 with Qwen3-4B on GPUs 1,2 (FSDP) and
3,4 (SGLang rollout).

## Safety boundary

This is a shared eight-A6000 host. Every launcher reserves only its declared
GPU set and fails closed if a target GPU or private Ray port is busy. Cleanup
matches the exact `RL_ROUTER_RUN_ID` from `/proc/<pid>/environ`; it must never
signal an unrelated process.

A physical reboot or host-wide firewall rule is prohibited on the shared
server. Host loss is therefore tested at the project failure-domain boundary:
the dedicated single-node Ray control plane and all run-owned job processes are
stopped after a durable checkpoint, then a fresh Ray control plane cold-loads
that checkpoint. This covers loss of all trainer, rollout, router and control
processes without affecting other tenants. A rack-level/physical host test still
requires an isolated node.

Network partition runs in an unprivileged user+network namespace. The test
installs a real kernel `iptables OUTPUT REJECT` rule for the selected SGLang
address and port inside that namespace. It does not modify the host firewall.

## Accepted failure drills

| Drill | Real injected failure | Recovery invariant | Accepted evidence |
|---|---|---|---|
| SGLang actor death | SGLang HTTP process tree is stopped and its Ray actor receives `SIGKILL` | Dead handle is replaced, old weight groups are destroyed, NCCL group is recreated, failed prefetch is discarded, strict fallback finishes | `actor-death-prod3`, MTTR 56.696 s |
| Network partition | Namespace-local kernel `OUTPUT REJECT` for `127.0.1.1:15000` for 10 s | Worker discovery retries until route recovery; no mixed version response | `network-partition-prod4`, MTTR 15.963 s |
| Host/control-plane loss | Run-private Ray control plane and all run-owned processes stop after iter 2 | Fresh Ray restores model, Adam, scheduler, RNG and dataset cursor; training commits iter 4 | `host-loss-prod1`, recovery 769.011 s |

Executable SLOs are in `configs/production_slo.yaml`. Current limits are actor
MTTR 120 s, network MTTR 60 s and control-plane recovery 1200 s.

## Commands

Run only after checking that the selected GPUs are reserved:

```bash
TRAINER_GPU_IDS=1,2 ROLLOUT_GPU_IDS=3,4 \
RL_ROUTER_RUN_ID=actor-death-<id> \
  bash scripts/run_actor_death_acceptance.sh

TRAINER_GPU_IDS=1,2 ROLLOUT_GPU_IDS=3,4 \
RL_ROUTER_RUN_ID=network-partition-<id> \
  bash scripts/run_network_partition_acceptance.sh

TRAINER_GPU_IDS=1,2 ROLLOUT_GPU_IDS=3,4 \
RL_ROUTER_RUN_ID=host-loss-<id> \
  bash scripts/run_host_loss_recovery_acceptance.sh
```

Destructive actor chaos is guarded by `RL_ENABLE_DESTRUCTIVE_CHAOS=1` inside
the acceptance launcher. Do not export it in routine training jobs.

## Triage

1. Preserve `launcher.log`, the bounded trace, live router trace and acceptance
   JSON before retrying.
2. Check the checkpoint tracker. A checkpoint directory without
   `latest_checkpointed_iteration.txt` pointing at it is not committed.
3. Check GPU ownership before cleanup. Use
   `scripts/cleanup_owned_gpu_processes.py --run-id <exact-id>`; never use a
   broad `pkill`.
4. After cleanup, require no project Ray/SGLang/FSDP processes and approximately
   11 MiB idle usage on each selected GPU.
5. Treat an SLO miss as a rejected release even if the job eventually recovers.

## Evidence locations

```text
results/slime_qwen3_4b_4gpu/chaos/actor-death-prod3/
results/slime_qwen3_4b_4gpu/chaos/network-partition-prod4/
results/slime_qwen3_4b_4gpu/host_loss/host-loss-prod1/
```

The production gate additionally requires the accepted multi-hour soak and
automated checks:

```bash
bash scripts/run_production_gate.sh
```
