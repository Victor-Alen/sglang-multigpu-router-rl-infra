#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_ROOT="${SLIME_ROOT:-/home/thjiang/slime}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a supported 3B-4B checkpoint}"
SLIME_LAUNCH_CMD="${SLIME_LAUNCH_CMD:-}"

python3 "$ROOT/scripts/check_gpu_availability.py" --gpu-ids 0,1,2,3,4,5 --max-used-mib "${MAX_USED_MIB:-512}"
test -d "$SLIME_ROOT/.git" || {
  echo "slime is not installed at $SLIME_ROOT; lock a tested commit before training." >&2
  exit 2
}
test -e "$MODEL_PATH" || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 2; }
test -n "$SLIME_LAUNCH_CMD" || {
  echo "Set SLIME_LAUNCH_CMD to the launch command verified for the locked slime commit." >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export RL_ROUTER_URL="${RL_ROUTER_URL:-http://127.0.0.1:8088}"
export TRAINER_GPU_IDS=0,1,2,3
export ROLLOUT_GPU_IDS=4,5
export MODEL_PATH
exec bash -lc "$SLIME_LAUNCH_CMD"
