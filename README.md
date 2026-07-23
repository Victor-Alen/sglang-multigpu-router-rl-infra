# SGLang RL Rollout Router Infrastructure

Production-oriented routing and control-plane infrastructure for reinforcement-learning (RL) rollout workloads built around SGLang. The project provides deterministic routing policies, worker lifecycle management, bounded asynchronous rollouts, version-aware weight updates, observability, replay, and fault-injection tooling.

## Scope

- OpenAI-compatible HTTP routing across SGLang workers.
- Round-robin, least-load, cache-aware, fixed-pack, fixed-split, and adaptive routing policies.
- RL group/sample metadata, barriers, bounded rollout buffering, and policy-version tracking.
- Coordinated weight updates with checksum validation and rolling-update states.
- JSONL traces, Prometheus-compatible metrics, deterministic offline replay, and run manifests.
- Automated correctness, integration, and fault-recovery acceptance tests.

The implementation is based on the SGLang router architecture and adds an RL rollout control plane. It can be evaluated with simulated workers or connected to live SGLang and Slime deployments.

## Repository layout

| Path | Purpose |
| --- | --- |
| `router/` | HTTP service and routing policies |
| `router/rl/` | RL scheduling, state, telemetry, replay, and versioning |
| `integrations/` | SGLang, Slime, chaos, and weight-update adapters |
| `configs/` | 4-GPU/6-GPU deployment and experiment configurations |
| `scripts/` | Reproducible launch, acceptance, verification, and reporting commands |
| `tests/` | Unit, integration, correctness, and fault-injection tests |
| `docs/` | Architecture, deployment, experiment, and recovery runbooks |
| `patches/` | Optional upstream integration patches |

## Requirements

- Python 3.9 or newer
- Linux for GPU-backed deployments
- NVIDIA driver and CUDA-compatible PyTorch/SGLang installation for live serving
- Optional: Slime for live RL training integration

Install the package and development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Quick start (CPU/simulation)

Run the deterministic correctness suite:

```bash
bash scripts/run_correctness.sh
```

Start the router with a configuration file:

```bash
python -m router.app --config configs/rl_router_4xa6000.yaml --policy adaptive --port 8088
```

The service exposes health and metrics endpoints, including `/healthz` and `/metrics`.

## 4×A6000 deployment

Review [docs/4gpu_deployment.md](docs/4gpu_deployment.md), reserve the GPU layout, and launch the stack:

```bash
bash scripts/check_gpu_availability.py --gpu-ids 0,1,2,3
MODEL_PATH=/path/to/model bash scripts/start_4gpu_stack.sh
```

Use `scripts/stop_4gpu_stack.sh` for a controlled shutdown. The 6-GPU variants remain available for larger experiments.

## Validation and fault injection

Run targeted acceptance tests:

```bash
bash scripts/run_bounded_async_acceptance.sh
bash scripts/run_checkpoint_resume_acceptance.sh
bash scripts/run_actor_death_acceptance.sh
bash scripts/run_host_loss_recovery_acceptance.sh
bash scripts/run_network_partition_acceptance.sh
bash scripts/run_long_soak_acceptance.sh
```

The fault-injection commands operate only on processes and interfaces explicitly assigned to the test run. See [docs/chaos_recovery_runbook.md](docs/chaos_recovery_runbook.md) before running them on shared infrastructure.

## Experiments and reproducibility

Experiment matrices and SLOs are versioned under `configs/`. Every run can emit a manifest containing environment metadata, artifact hashes, policy versions, and acceptance results. Use the reporting and verification scripts under `scripts/` to compare paired policies and replay traces offline.

## Development

Run the full test suite locally:

```bash
python -m pytest -q
```

GPU- or vendor-specific tests automatically skip when their optional runtime is unavailable. Install the corresponding runtime (NVIDIA tools, SGLang, or Slime) to enable those checks.

Formatting, dependency, and test checks are defined in `.github/workflows/ci.yml`.

## Documentation

- [Architecture](docs/architecture.md)
- [Production readiness](docs/production_readiness.md)
- [Experiment protocol](docs/experiment_protocol.md)
- [Correctness contract](docs/correctness.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Chaos recovery runbook](docs/chaos_recovery_runbook.md)

## License

See the repository license file for distribution terms.
