# RL-aware SGLang Router: production evidence report

## Outcome

The project is an operationally mature four-GPU RL rollout/training system for
Qwen3-4B. It integrates slime FSDP, two SGLang engines, group-aware routing,
policy-version barriers, bounded overlap, checkpoint recovery, live chaos
drills, long-soak telemetry and an executable production gate.

## Final long-soak result

`long-soak-prod1` consumed exactly 1200 batches in 13109.894 seconds (3.642
hours). Initial and terminal token rates were 44.557 and 64.424 token/s,
respectively (ratio 1.446). Maximum temperature was 59°C. Memory growth after
the final-batch cutoff was 0, 0, 2 and 2 MiB for GPUs 1–4. Lag and buffer stayed
at one; all 1200 batches had complete router groups, zero prefetch failures,
zero strict fallbacks and zero mixed-version responses.

The run committed full FSDP checkpoints at iter 600 and iter 1200. The tracker
points to iter 1200 and both checkpoints contain model, optimizer, scheduler,
RNG and metadata/cursor state.

## Recovery result

The actor drill killed a real SGLang actor and recovered in 56.696 seconds. The
network drill installed a real namespace-local kernel reject for 10 seconds and
recovered in 15.963 seconds. The host-loss drill stopped the project-private
Ray control plane after iter 2; a fresh control plane restored state and reached
iter 4 in 769.011 seconds.

## Reproducibility and integrity

`scripts/run_production_gate.sh` reruns the 97-test suite, dependency/patch
checks, all evidence checks and writes `manifest.json`. The manifest hashes 15
JSONL/JSON/CSV/config artifacts and records GPU topology and worktree status.
The source tree is dirty during development, so a clean reviewed commit/tag is
still required for distribution.

## Interpretation boundary

The acceptance workload uses 128-token responses and produced zero reward. The
evidence proves operational continuity and recovery, not learning quality,
convergence or general performance for other models and context lengths.
