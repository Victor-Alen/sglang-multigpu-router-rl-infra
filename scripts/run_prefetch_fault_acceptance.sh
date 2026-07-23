#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-prefetch-fault-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
FAULT_ROLLOUT_ID="${FAULT_ROLLOUT_ID:-1}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/fault_injection/$RUN_ID}"
TRACE="$STATE_DIR/$RUN_ID/bounded_async_trace.jsonl"
mkdir -p "$LOG_DIR"

EXPECTED_ROUTER_GROUPS=$(((NUM_ROLLOUTS + 1) * ROLLOUT_BATCH_SIZE)) \
RL_FAULT_INJECT_PREFETCH_ROLLOUT_ID="$FAULT_ROLLOUT_ID" \
RL_ROUTER_RUN_ID="$RUN_ID" \
LOG_DIR="$LOG_DIR" \
NUM_ROLLOUTS="$NUM_ROLLOUTS" \
ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
ROLLOUT_SHUFFLE=0 \
  bash "$ROOT/scripts/run_bounded_async_acceptance.sh"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_bounded_async_trace.py" \
  --trace "$TRACE" \
  --expected-batches "$NUM_ROLLOUTS" \
  --max-policy-lag 1 \
  --max-buffered-batches 1 \
  --require-overlap \
  --expected-prefetch-failures 1 \
  --expected-strict-fallbacks 1 \
  --require-fault-injection \
  --output "$LOG_DIR/fault_acceptance.json"

echo "Prefetch fault acceptance passed: $LOG_DIR/fault_acceptance.json"
