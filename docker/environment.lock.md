# Environment lock ? A6000 host

Captured on 2026-07-15 (Asia/Shanghai). Re-run `scripts/collect_system_info.sh`
before formal measurements and replace `UNLOCKED` entries before training.

| Component | Locked value |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 5.19.0-32-generic |
| GPU | 8 ? NVIDIA RTX A6000 48 GB (use six for the 4+2 layout) |
| NVIDIA driver | 550.78 |
| CUDA compatibility reported by driver | 12.4 |
| Python (router runtime) | 3.11.14, `rayllm-compat` environment |
| PyTorch | 2.7.1 |
| Ray | 2.49.2 |
| FastAPI | 0.129.0 |
| HTTPX | 0.28.1 |
| Pydantic | 2.12.5 |
| Transformers | 4.57.6 |
| SGLang source | `3d1b54e6b63aa6030ac2e30de691df6991fc4d2e` |
| Router base commit | `75bdb2515cd49b78496b122e2bfed438c5754f8d` |
| NCCL | UNLOCKED ? shared library exists; package version query was inconclusive |
| slime | NOT INSTALLED |
| verl | NOT INSTALLED |
| Training model | NOT INSTALLED ? only `/home/thjiang/models/opt-30b` was found |
| Model revision | UNLOCKED |
| Tokenizer revision | UNLOCKED |

The environment is sufficient for Router correctness and replay tests. It is not
yet a valid, reproducible GRPO training environment. A formal RL run must first
lock slime/verl, a supported 3B?4B model and tokenizer, NCCL, the chat template,
and the launch command.
