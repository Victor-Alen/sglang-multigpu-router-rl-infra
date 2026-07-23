# Known issues and explicit boundaries

Status: 2026-07-23.

1. The source worktree is dirty. A clean reviewed commit and immutable release
   tag are required before external distribution.
2. Each complete Qwen3-4B FSDP+Adam checkpoint is about 47 GiB. Use the safe
   retention tool and monitor storage before repeated production runs.
3. The operational workload truncates responses at 128 tokens and reward is
   zero. No convergence or reward-quality claim is made.
4. Physical host/rack power-loss testing is not safe on this shared server. The
   accepted host drill stops only the project-private Ray control plane and all
   exact-run-owned processes; isolated-node testing is required for physical
   power loss.
5. Results are for Qwen3-4B and this four-GPU placement. Longer context,
   longer generation, larger models and different worker counts need capacity
   envelopes and new acceptance evidence.
6. Compatibility patches are reversible and CI-covered, but still need an
   upstream SGLang/slime review before being treated as generally supported.
7. Foreign GPU workloads exist on the server. Launchers fail closed and do not
   signal foreign processes; production scheduling still needs a reserved GPU
   window.

Resolved in this release: live FSDP restart, bounded overlap and version
barriers, deterministic fallback, real actor death, kernel network partition,
private control-plane loss, three-hour soak, SLO gate, evidence manifest and
operator runbook.
