#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/thjiang/miniconda3/envs/rayllm-compat/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a supported 3B-4B checkpoint}"
LOG_DIR="${LOG_DIR:-$ROOT/results/rl_state/worker_logs}"

python3 "$ROOT/scripts/check_gpu_availability.py" --gpu-ids 4,5 --max-used-mib "${MAX_USED_MIB:-512}"
test -e "$MODEL_PATH" || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 2; }
mkdir -p "$LOG_DIR"

for spec in "4:30004:rollout-0" "5:30005:rollout-1"; do
  IFS=: read -r gpu port name <<<"$spec"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port "$port" \
    >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$LOG_DIR/$name.pid"
done

echo "workers started; logs: $LOG_DIR"
