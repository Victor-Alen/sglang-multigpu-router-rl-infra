#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/thjiang/miniconda3/envs/slime-v021/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/thjiang/sglang-slime-v056}"
MODEL_PATH="${MODEL_PATH:-/home/thjiang/models/Qwen3-4B}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,3}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-5,7}"
ROLLOUT_PORTS="${ROLLOUT_PORTS:-30004,30005}"
LOG_DIR="${LOG_DIR:-$ROOT/results/rl_state_4gpu/worker_logs}"

python3 "$ROOT/scripts/check_gpu_layout.py" \
  --trainer-gpu-ids "$TRAINER_GPU_IDS" \
  --rollout-gpu-ids "$ROLLOUT_GPU_IDS" \
  --expected-total 4 >/dev/null
python3 "$ROOT/scripts/check_gpu_availability.py" \
  --gpu-ids "$ROLLOUT_GPU_IDS" \
  --max-used-mib "${MAX_USED_MIB:-512}"
test -e "$MODEL_PATH" || { echo "MODEL_PATH does not exist: $MODEL_PATH" >&2; exit 2; }
test -d "$SGLANG_ROOT/python/sglang" || { echo "SGLang source missing: $SGLANG_ROOT" >&2; exit 2; }

IFS=, read -r -a rollout_gpus <<<"$ROLLOUT_GPU_IDS"
IFS=, read -r -a rollout_ports <<<"$ROLLOUT_PORTS"
if [[ ${#rollout_gpus[@]} -ne 2 || ${#rollout_ports[@]} -ne 2 ]]; then
  echo "The 2+2 layout requires exactly two Rollout GPU ids and two ports." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
for index in 0 1; do
  gpu="${rollout_gpus[$index]}"
  port="${rollout_ports[$index]}"
  name="rollout-$index"
  PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$PYTHON_BIN" -m sglang.launch_server \
      --model-path "$MODEL_PATH" \
      --tp-size 1 \
      --host 127.0.0.1 \
      --port "$port" \
      >"$LOG_DIR/$name.log" 2>&1 </dev/null &
  echo $! >"$LOG_DIR/$name.pid"
  echo "started $name: physical GPU $gpu, port $port, pid $(cat "$LOG_DIR/$name.pid")"
done
