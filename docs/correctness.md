# Correctness contract

The following invariants are enforced in code and covered by tests:

- A group has one prompt, group size, rollout step, policy version, tokenizer,
  chat template and sampling configuration.
- Sample indexes are unique and within the declared group size.
- Only `READY` workers with a fresh heartbeat and the exact requested version
  are eligible.
- `DRAINING`, `UPDATING`, `FAILED` and stale workers never receive a new group.
- A response whose served version differs from the group version is rejected.
- A sample index cannot become two training samples after retry.
- A group becomes trainable only after all expected responses complete.
- Router state is atomically persisted and recovered without trusting worker
  versions; controllers must re-heartbeat workers after restart.
- Weight updates require drain, zero in-flight work, version/checksum verification
  and explicit commit. A failed worker is isolated.
- The async buffer rejects future, stale, duplicate and over-capacity batches.

Run the complete CPU suite with `bash scripts/run_correctness.sh`. The generated
32-group Golden Trace is replayed with the same policy, and assignments must
match exactly.

GPU correctness still required before a formal RL result:

- Trainer/SGLang fixed-token log-prob comparison.
- Fixed-input logits digest after each real checkpoint update.
- Tokenizer revision, chat-template hash and special-token IDs from the selected
  model.
- Worker-crash and Router-restart tests against live SGLang processes.
