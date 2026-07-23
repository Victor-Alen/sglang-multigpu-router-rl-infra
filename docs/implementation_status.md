# Implementation status

Status: 2026-07-23. The validated profile is Qwen3-4B on physical GPUs 1,2
(FSDP) and 3,4 (two SGLang TP1 rollout engines).

| Milestone | Status | Evidence |
|---|---|---|
| Hardware and locked environment | Complete | `configs/slime_qwen3_4b_4xa6000.env`, topology and pip freeze |
| Router policies and metadata | Complete | RR/least-load/cache-aware, group barriers and version metadata |
| Real GRPO/RLVR path | Accepted | Live rollout, reward, FSDP training and NCCL weight publication |
| Checkpoint restart | Accepted | Model/Adam/scheduler/RNG/cursor restored; iter 2→4 after cold Ray restart |
| Bounded async | Accepted | Lag and buffer bound one, deterministic fallback and trace verifiers |
| SGLang actor death | Accepted | MTTR 56.696 s; replacement and weight-group reinitialization |
| Network partition | Accepted | Kernel namespace REJECT, recovery MTTR 15.963 s |
| Host/control-plane loss | Accepted | Private Ray cold recovery 769.011 s, state restored |
| Multi-hour stability | Accepted | 1200 batches, 13109.894 s, 3.642 h, no failures/fallbacks/mixed versions |
| Production evidence gate | Accepted | 97 tests, 28 checks, 15-artifact verified manifest |

The operational gate does not establish model convergence: reward remains zero
for the intentionally short smoke-style response length. Longer-response and
learning-quality experiments are separate work.
