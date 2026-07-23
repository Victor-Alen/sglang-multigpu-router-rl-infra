#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-actor-death-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/chaos/$RUN_ID}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
TRACE="$STATE_DIR/$RUN_ID/bounded_async_trace.jsonl"
mkdir -p "$LOG_DIR"

env \
  TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,2}" \
  ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-3,4}" \
  RL_ROUTER_RUN_ID="$RUN_ID" \
  RL_ROUTER_STATE_DIR="$STATE_DIR" \
  LOG_DIR="$LOG_DIR" \
  NUM_ROLLOUTS="$NUM_ROLLOUTS" \
  ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}" \
  MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" \
  ENABLE_DUMP_DETAILS=0 \
  ROLLOUT_SHUFFLE=0 \
  RL_ENABLE_DESTRUCTIVE_CHAOS=1 \
  RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID="${RL_CHAOS_ENGINE_DEATH_ROLLOUT_ID:-1}" \
  RL_CHAOS_ENGINE_INDEX="${RL_CHAOS_ENGINE_INDEX:-0}" \
  ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-1}" \
  ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-2}" \
  ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-0}" \
  ALLOW_FAILED_ROUTER_GROUPS=1 \
  EXPECTED_ROUTER_GROUPS= \
  MIN_COMPLETE_ROUTER_GROUPS="$((NUM_ROLLOUTS * ROLLOUT_BATCH_SIZE))" \
  bash "$ROOT/scripts/run_bounded_async_acceptance.sh"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_bounded_async_trace.py" \
  --trace "$TRACE" \
  --expected-batches "$NUM_ROLLOUTS" \
  --max-policy-lag 1 \
  --max-buffered-batches 1 \
  --require-overlap \
  --require-actor-death-recovery \
  --output "$LOG_DIR/actor_death_acceptance.json"

echo "SGLang actor-death acceptance passed: $LOG_DIR/actor_death_acceptance.json"
