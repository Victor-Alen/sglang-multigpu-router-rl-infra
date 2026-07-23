# Experiment protocol

## Preflight

1. Capture `scripts/collect_system_info.sh` output.
2. Run `scripts/check_gpu_availability.py --gpu-ids 0,1,2,3,4,5`.
3. Freeze the environment in `docker/environment.lock.md`.
4. Run `scripts/run_correctness.sh`.
5. Record model/tokenizer revisions and chat-template hash.

Never start a formal run when the availability check reports another user's
process. This host has no reported NVLink and crosses NUMA between GPU 0?3 and
GPU 4?5, so CPU affinity and weight-sync bandwidth must be reported.

## Evaluation ladder

1. Offline replay: all policies, all matrix cells.
2. Fixed-weight rollout: RR, least queued tokens, cache-aware, PACK, SPLIT,
   load-proportional and Adaptive; three fixed seeds and at least three repeats.
3. Short strict-sync GRPO: only the strongest baseline and Adaptive.
4. Stop-the-world vs rolling update.
5. Strict sync vs bounded one-step overlap with `max_policy_lag=1`.

Save raw JSONL before aggregation. Primary metrics are group p50/p95/p99,
GroupGoodput, duplicated prefill, worker skew, update bubble and
FreshTokenThroughput. Report failures and confidence intervals; do not substitute
the composite CEGG metric for raw measurements.

The experiment matrix is `configs/experiments/rl_fixed_weight_matrix.json`.
