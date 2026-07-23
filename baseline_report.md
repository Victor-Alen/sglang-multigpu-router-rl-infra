# RL Router engineering baseline report

Date: 2026-07-23, Asia/Shanghai

## Verified implementation

- Qwen3-4B executes real rollout, rule reward, FSDP training and NCCL weight
  synchronization on two Trainer plus two Rollout A6000 GPUs.
- Group-aware Adaptive/PACK/SPLIT routing, requested/served policy versions,
  Group Barrier and bounded one-step overlap are on the live slime data path.
- 81 automated tests, deterministic replay and a 3,240-run offline policy matrix
  pass. Offline results are screening data, not GPU performance claims.
- Three prompt/seed-matched strict/bounded controls measured a 25.51% mean
  steady-step reduction with a 22.62% to 28.40% Student-t 95% interval.

## Release-candidate evidence

Run `maturity-20260723-rc1` passed the executable maturity SLO on physical GPUs
1,2,3,4:

- A complete FSDP checkpoint saved model, Adam, scheduler, RNG and dataset
  cursor. A new Ray process restored it, resumed at rollout 2 and advanced global
  step from 2 to 4.
- One prefetched-result failure caused exactly one strict fallback; all four
  batches completed with lag at most one.
- An eight-batch soak accepted 4,096 tokens, observed buffer peak one and lag
  peak one, and completed 32 responses without a policy-version mismatch.
- `release_gate.json` accepted every configured SLO check. `manifest.json`
  records SHA-256 hashes for all primary evidence.
- Cleanup returned GPUs 1,2,3,4 to idle and left no private Ray process.

Evidence is under:

```text
results/slime_qwen3_4b_4gpu/maturity/maturity-20260723-rc1/
results/slime_qwen3_4b_4gpu/checkpoint_resume/maturity-20260723-rc1-checkpoint/
results/slime_qwen3_4b_4gpu/fault_injection/maturity-20260723-rc1-fault/
```

## Boundary

This is a mature controlled-experiment infrastructure, not yet a general
production training service. The worktree needs a clean reviewed release commit;
the soak is not multi-hour; process/host-loss chaos is not covered; and the short
acceptance workload had truncated responses and zero reward, so no learning or
convergence improvement is claimed. The two full checkpoints use about 93 GiB;
safe dry-run retention tooling is included.
