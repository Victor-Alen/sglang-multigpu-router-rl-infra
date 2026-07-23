# Four-A6000 deployment

## Installed stack

The reproducible default is slime v0.2.1 with its FSDP backend and Qwen3-4B
in BF16. Exact source and Hugging Face revisions are recorded in
`configs/slime_qwen3_4b_4xa6000.env`.

| Role | Physical GPUs | Ray-visible IDs | NUMA |
|---|---:|---:|---:|
| FSDP2 Trainer (validated profile) | 1,2 | 0,1 | 0 |
| SGLang Rollout engines (validated profile) | 3,4 | 2,3 | 1 |

The selected model is small enough for FSDP2 training on two 48-GiB A6000s,
while two independent one-GPU SGLang engines preserve group routing. The launch
defaults use FlashAttention 2.7.4.post1 (Ampere/SM80 path) for packed training
and disable NVLink-only NCCL behavior because this host has no NVLink fabric.

The locked slime source has two small project compatibility patches. The first makes Ring
FlashAttention optional during staged installation when context parallelism is
1, and corrects the v0.2.1 no-new-engine return value so post-training weight
synchronization receives an integer rather than a tuple. It also attaches a
monotonic version to FSDP weight transfers. The second exposes rollout-engine
versions to the bounded asynchronous driver. The final environment
includes FlashAttention 2; requests with `context_parallel_size > 1` still fail
closed with a clear error if that import is ever unavailable. The verification
script checks that both patches are applied and reversible.

## Verify

```bash
cd /home/thjiang/sglang/sglang-multigpu-router-bench
bash scripts/verify_slime_qwen3_4b.sh
```

The dependency installation is reproducible with
`scripts/install_slime_qwen3_4b_deps.sh`; it writes the resolved package set to
`configs/slime_qwen3_4b_4xa6000.pip-freeze.txt`.

## Start a bounded run

The integrated slime launcher owns all four GPUs and starts its own Ray cluster,
SGLang engines, and slime router. It does not kill existing Python or Ray
processes; it refuses to start if the selected GPUs or a user Ray cluster are
already busy.

```bash
cd /home/thjiang/sglang/sglang-multigpu-router-bench
NUM_ROLLOUTS=1 bash scripts/run_slime_qwen3_4b_4xa6000.sh
```

For a longer run, omit `NUM_ROLLOUTS=1` (the safe default is 10) or override
batch/sequence controls explicitly:

```bash
NUM_ROLLOUTS=100 \
ROLLOUT_BATCH_SIZE=2 \
SAMPLES_PER_PROMPT=2 \
MAX_RESPONSE_LEN=1024 \
bash scripts/run_slime_qwen3_4b_4xa6000.sh
```

The physical IDs remain configurable through `TRAINER_GPU_IDS` and
`ROLLOUT_GPU_IDS`. Their combined order is significant: Trainer IDs must come
first so Ray assigns its first two placement bundles to FSDP.

The launcher continuously checks ownership of all selected physical GPUs after
startup. If another user claims one, it stops only this launcher's Ray cluster;
it never signals the foreign process.

## Bounded one-step overlap acceptance

Run four or more steps so the trace proves a real lag-1 prefetched batch rather
than only initialization:

```bash
cd /home/thjiang/sglang/sglang-multigpu-router-bench
NUM_ROLLOUTS=4 bash scripts/run_bounded_async_acceptance.sh
```

The wrapper selects `integrations/slime_bounded_train.py`, enforces buffer
capacity 1 and lag at most 1, and then runs
`scripts/verify_bounded_async_trace.py --require-overlap`. A prefetch failure
falls back to strict generation on the newest policy and is visible in both the
JSONL trace and acceptance report.

The accepted paired reference is `strict-live-20260715-rep4` (Ray job
`raysubmit_uWi4hk3PGe6qYbpx`) versus `bounded-live-20260715-rep4` (Ray job
`raysubmit_c1G4i43478ThH7Lk`). The bounded side completed four batches, three
lag-1 consumptions, versions v1 through v5, and zero stale/fallback/prefetch
failures; both jobs released all target GPUs cleanly.

Use the matched strict path when the same four cards are free:

```bash
ROLLOUT_SHUFFLE=0 NUM_ROLLOUTS=4 MAX_RESPONSE_LEN=128 \
  bash scripts/run_strict_sync_acceptance.sh
```

Both acceptance wrappers call `scripts/summarize_live_run.py`, so their
end-to-end token throughput uses the same wall-time boundary including Ray/model
startup and cleanup. They default to `ROLLOUT_SHUFFLE=0`; if slime does not
provide a seed, the adapter derives one from group position and prompt identity,
not the run ID. This makes prompt and stochastic decode work replayable across
strict and bounded modes.

Compare a completed pair with `scripts/compare_strict_bounded_runs.py`. The
verifier checks prompt/decode signatures and generation seeds separately because
bounded prefetch legitimately changes `rollout_step` timing. The three-pair
aggregate is under
`results/slime_qwen3_4b_4gpu/counterbalanced_3pair_20260715/`. It observed a
25.51% mean steady-step reduction with a 22.62%–28.40% Student-t 95% interval.
Mean end-to-end throughput increased 10.14%, but its -1.06%–21.34% interval
crossed zero, so that metric is not reported as statistically established.

For reproducible repetitions, use `scripts/run_counterbalanced_acceptance.sh`;
it alternates strict-first and bounded-first order and invokes
`scripts/aggregate_paired_comparisons.py` after every verified pair.

## Production-maturity acceptance

The production gate runs 97 automated tests, dependency and patch checks, the
real actor-death, kernel network-partition and private control-plane recovery
drills, the 1200-batch/3.642-hour soak, executable SLO verification and a
SHA-256 evidence manifest:

```bash
TRAINER_GPU_IDS=1,2 ROLLOUT_GPU_IDS=3,4 \
  RL_ROUTER_RUN_ID=<unique-run-id> \
  bash scripts/run_maturity_acceptance.sh
```

The accepted reference is `maturity-20260723-rc1`. Its release gate is under
`results/slime_qwen3_4b_4gpu/maturity/maturity-20260723-rc1/`. Checkpoint save
and restart is intentionally expensive: the two full model/Adam checkpoints use
about 93 GiB. Use `scripts/prune_checkpoints.py` without `--apply` to inspect a
retention plan before any deletion.

See `docs/production_readiness.md` and `docs/chaos_recovery_runbook.md` for
SLOs, artifacts, safety boundaries and operator triage.

## Project router experiment

`scripts/start_4gpu_stack.sh` remains available for experiments with this
project's external RL router. Its worker launcher now defaults to the same locked
Python environment, patched SGLang source, and Qwen3-4B checkpoint. The
integrated slime launcher above is the validated training path; do not run both
stacks at the same time.

## Stop and rollback

The integrated launcher stops only the Ray cluster it started when it exits.
For the project's external router stack, use:

```bash
bash scripts/stop_4gpu_stack.sh
```

Rollback consists of removing the isolated paths in the lock file. The original
`/home/thjiang/sglang` checkout and base Conda environment are not modified.
