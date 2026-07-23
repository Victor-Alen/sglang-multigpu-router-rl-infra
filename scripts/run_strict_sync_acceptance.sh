#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-strict-sync-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-0}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/strict_sync/$RUN_ID}"
TRACE="$STATE_DIR/$RUN_ID/trace.jsonl"
EXPECTED_COMPLETED_ROLLOUTS="${EXPECTED_COMPLETED_ROLLOUTS:-$NUM_ROLLOUTS}"
EXPECTED_GROUPS=$((EXPECTED_COMPLETED_ROLLOUTS * ROLLOUT_BATCH_SIZE))
MIN_POLICY_VERSIONS="${MIN_POLICY_VERSIONS:-$EXPECTED_COMPLETED_ROLLOUTS}"
mkdir -p "$LOG_DIR"

start_ns="$(date +%s%N)"
BOUNDED_ASYNC=0 \
RL_ROUTER_RUN_ID="$RUN_ID" \
RL_ROUTER_STATE_DIR="$STATE_DIR" \
NUM_ROLLOUTS="$NUM_ROLLOUTS" \
ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
ROLLOUT_SHUFFLE="$ROLLOUT_SHUFFLE" \
bash "$ROOT/scripts/run_slime_qwen3_4b_4xa6000.sh" 2>&1 | tee "$LOG_DIR/train.log"
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_live_rl_trace.py" \
  --trace "$TRACE" \
  --expected-groups "$EXPECTED_GROUPS" \
  --min-policy-versions "$MIN_POLICY_VERSIONS" \
  --output "$LOG_DIR/live_router_acceptance.json"
"$PYTHON_ENV/bin/python" "$ROOT/scripts/summarize_live_run.py" \
  --trace "$TRACE" \
  --mode strict-sync \
  --wall-seconds "$wall_seconds" \
  --output "$LOG_DIR/live_run_summary.json"

echo "Strict-sync acceptance passed: $LOG_DIR"
