#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-long-soak-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,2}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-3,4}"
ALL_GPUS="$TRAINER_GPU_IDS,$ROLLOUT_GPU_IDS"
SOAK_ROLLOUTS="${SOAK_ROLLOUTS:-1200}"
MIN_SOAK_SECONDS="${MIN_SOAK_SECONDS:-7200}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/long_soak/$RUN_ID}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$LOG_DIR/checkpoints}"
TELEMETRY="$LOG_DIR/gpu_telemetry.csv"
TIMING="$LOG_DIR/timing.json"
TRACE="$STATE_DIR/$RUN_ID/bounded_async_trace.jsonl"
mkdir -p "$LOG_DIR"

monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "timestamp_ns,gpu,memory_used_mib,temperature_c,power_w" >"$TELEMETRY"
(
  while true; do
    timestamp_ns="$(date +%s%N)"
    nvidia-smi --query-gpu=index,memory.used,temperature.gpu,power.draw \
      --format=csv,noheader,nounits --id="$ALL_GPUS" |
      awk -F',' -v timestamp="$timestamp_ns" '{
        for (i=1; i<=NF; i++) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i) }
        printf "%s,%s,%s,%s,%s\n", timestamp, $1, $2, $3, $4
      }' >>"$TELEMETRY"
    sleep "${TELEMETRY_INTERVAL_SECONDS:-60}"
  done
) &
monitor_pid=$!

start_ns="$(date +%s%N)"
env \
  TRAINER_GPU_IDS="$TRAINER_GPU_IDS" \
  ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS" \
  RL_ROUTER_RUN_ID="$RUN_ID" \
  RL_ROUTER_STATE_DIR="$STATE_DIR" \
  LOG_DIR="$LOG_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  NUM_ROLLOUTS="$SOAK_ROLLOUTS" \
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}" \
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}" \
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-600}" \
  ENABLE_DUMP_DETAILS=0 \
  ROLLOUT_SHUFFLE=0 \
  bash "$ROOT/scripts/run_bounded_async_acceptance.sh"
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"
printf '{"start_ns":%s,"end_ns":%s,"wall_seconds":%s}\n' \
  "$start_ns" "$end_ns" "$wall_seconds" >"$TIMING"
cleanup
monitor_pid=""

min_samples="$(awk -v seconds="$MIN_SOAK_SECONDS" -v interval="${TELEMETRY_INTERVAL_SECONDS:-60}" \
  'BEGIN { value=int(seconds/interval*0.75); if (value < 10) value=10; print value }')"
"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_long_soak.py" \
  --acceptance "$LOG_DIR/acceptance.json" \
  --router-acceptance "$LOG_DIR/live_router_acceptance.json" \
  --timing "$TIMING" \
  --bounded-trace "$TRACE" \
  --telemetry "$TELEMETRY" \
  --expected-batches "$SOAK_ROLLOUTS" \
  --expected-gpus "$ALL_GPUS" \
  --min-wall-seconds "$MIN_SOAK_SECONDS" \
  --min-throughput-ratio "${MIN_THROUGHPUT_RATIO:-0.5}" \
  --max-temperature-c "${MAX_GPU_TEMPERATURE_C:-90}" \
  --max-steady-memory-growth-mib "${MAX_STEADY_MEMORY_GROWTH_MIB:-2048}" \
  --min-samples-per-gpu "$min_samples" \
  --output "$LOG_DIR/long_soak_acceptance.json"

echo "Multi-hour stability acceptance passed: $LOG_DIR/long_soak_acceptance.json"
