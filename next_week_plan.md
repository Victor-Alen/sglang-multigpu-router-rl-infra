# Remaining M9 execution plan

The core four-GPU system is implemented and live-accepted. The remaining work
is statistical and release-oriented:

1. Reserve GPUs 1,3,5,7 and run matched strict-sync versus bounded-overlap jobs
   with at least three seeds/repetitions and nontrivial rewards.
2. Run online fixed-weight PACK, EVEN SPLIT, Cache-aware and Adaptive workloads
   across prompt/output buckets; report Group p50/p95/p99, duplicated prefill,
   worker skew and bootstrap 95% confidence intervals.
3. Perform one live stop-the-world disk update, one rolling update, and one
   deliberately failed worker update; retain drain/update/probe timing and prove
   that failed workers receive no traffic.
4. Save a checkpoint, stop every project service, resume, and verify monotonic
   policy versions plus fixed-input output/log-prob alignment.
5. Repeat the strongest strict and async configurations with longer responses
   and report learning metrics separately from systems metrics.
6. Split the two reversible slime patches into upstream-ready PRs, add upstream
   tests, and submit only after confirming current main-branch APIs.

No speedup, learning-quality, or six-GPU scaling claim should be made until the
corresponding repeated live experiment is complete.
