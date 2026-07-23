#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLIME_ROOT="${SLIME_ROOT:-/home/thjiang/slime}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a supported 3B-4B checkpoint}"
SLIME_LAUNCH_CMD="${SLIME_LAUNCH_CMD:-}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,3}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-5,7}"
ROLLOUT_WORKER_URLS="${ROLLOUT_WORKER_URLS:-http://127.0.0.1:30004,http://127.0.0.1:30005}"
RL_ROUTER_URL="${RL_ROUTER_URL:-http://127.0.0.1:8088}"

python3 "$ROOT/scripts/check_gpu_layout.py" \
  --trainer-gpu-ids "$TRAINER_GPU_IDS" \
  --rollout-gpu-ids "$ROLLOUT_GPU_IDS" \
  --expected-total 4 >/dev/null
# External Rollout workers are expected to be running, so only Trainer GPUs must be idle.
python3 "$ROOT/scripts/check_gpu_availability.py" \
  --gpu-ids "$TRAINER_GPU_IDS" \
  --max-used-mib "${MAX_USED_MIB:-512}"
test -d "$SLIME_ROOT/.git" || {
  echo "slime is not installed at $SLIME_ROOT; lock a tested commit before training." >&2
  exit 2
}
test -e "$MODEL_PATH" || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 2; }
test -n "$SLIME_LAUNCH_CMD" || {
  echo "Set SLIME_LAUNCH_CMD for the locked slime commit." >&2
  exit 2
}

curl -fsS "$RL_ROUTER_URL/health" >/dev/null || {
  echo "RL Router is not healthy at $RL_ROUTER_URL" >&2
  exit 2
}
IFS=, read -r -a rollout_urls <<<"$ROLLOUT_WORKER_URLS"
if [[ ${#rollout_urls[@]} -ne 2 ]]; then
  echo "The 2+2 layout requires exactly two external Rollout URLs." >&2
  exit 2
fi
for url in "${rollout_urls[@]}"; do
  curl -fsS "$url/health" >/dev/null || {
    echo "Rollout worker is not healthy at $url" >&2
    exit 2
  }
done

# With external Rollout workers, the training process sees only its two GPUs.
export CUDA_VISIBLE_DEVICES="$TRAINER_GPU_IDS"
export TRAINER_GPU_IDS
export TRAINER_WORLD_SIZE=2
export ROLLOUT_GPU_IDS
export ROLLOUT_WORKER_URLS
export ROLLOUT_EXTERNAL=1
export RL_ROUTER_URL
export MODEL_PATH
exec bash -lc "$SLIME_LAUNCH_CMD"

