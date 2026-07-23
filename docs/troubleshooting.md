# Troubleshooting

`no READY worker for policy version ...` means the Router correctly failed
closed. Check worker heartbeat age, lifecycle and loaded version at
`GET /rl/workers`; do not bypass the version filter.

If an update cannot drain, inspect `running_requests` and `queued_requests`.
Do not mark the worker drained manually while either is non-zero.

After a checksum mismatch, the worker is `FAILED`. Reload it from a known
checkpoint, run a fixed-input probe, then re-register/heartbeat it as `READY`.

If offline replay differs, preserve both traces and compare worker snapshots,
objective weights and request length predictions. Replay depends on the recorded
state, not current live workers.

If the six-GPU preflight fails, wait for the owners' jobs to finish or obtain an
explicit allocation. Never terminate unrelated GPU processes.

The adapter schema was checked against SGLang commit `3d1b54e...`: it calls
`POST /update_weights_from_disk` with `weight_version` and verifies the result
through `GET /model_info`. A live checkpoint reload is still mandatory before a
formal run.
