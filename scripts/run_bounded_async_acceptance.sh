#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-bounded-async-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-0}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/bounded_async/$RUN_ID}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
TRACE="$STATE_DIR/$RUN_ID/bounded_async_trace.jsonl"
EXPECTED_GROUPS=$((NUM_ROLLOUTS * ROLLOUT_BATCH_SIZE))
EXPECTED_ROUTER_GROUPS="${EXPECTED_ROUTER_GROUPS-$EXPECTED_GROUPS}"
EXPECTED_COMPLETE_ROUTER_GROUPS="${EXPECTED_COMPLETE_ROUTER_GROUPS:-}"
MIN_COMPLETE_ROUTER_GROUPS="${MIN_COMPLETE_ROUTER_GROUPS:-}"
ALLOW_FAILED_ROUTER_GROUPS="${ALLOW_FAILED_ROUTER_GROUPS:-0}"
MIN_POLICY_VERSIONS="${MIN_POLICY_VERSIONS:-$((NUM_ROLLOUTS > 1 ? NUM_ROLLOUTS - 1 : 1))}"
mkdir -p "$LOG_DIR"

start_ns="$(date +%s%N)"
BOUNDED_ASYNC=1 \
RL_MAX_PREFETCHED_BATCHES=1 \
RL_MAX_POLICY_LAG=1 \
RL_ASYNC_STRICT_FALLBACK=1 \
RL_ROUTER_RUN_ID="$RUN_ID" \
RL_ROUTER_STATE_DIR="$STATE_DIR" \
NUM_ROLLOUTS="$NUM_ROLLOUTS" \
ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
ROLLOUT_SHUFFLE="$ROLLOUT_SHUFFLE" \
bash "$ROOT/scripts/run_slime_qwen3_4b_4xa6000.sh" 2>&1 | tee "$LOG_DIR/train.log"
end_ns="$(date +%s%N)"
wall_seconds="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"

"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_bounded_async_trace.py" \
  --trace "$TRACE" \
  --expected-batches "$NUM_ROLLOUTS" \
  --max-policy-lag 1 \
  --max-buffered-batches 1 \
  --require-overlap \
  --output "$LOG_DIR/acceptance.json"
live_verify_args=()
if [[ "$ALLOW_FAILED_ROUTER_GROUPS" == "1" ]]; then
  live_verify_args+=(--allow-failed-groups)
fi
if [[ -n "$EXPECTED_COMPLETE_ROUTER_GROUPS" ]]; then
  live_verify_args+=(--expected-complete-groups "$EXPECTED_COMPLETE_ROUTER_GROUPS")
fi
if [[ -n "$MIN_COMPLETE_ROUTER_GROUPS" ]]; then
  live_verify_args+=(--min-complete-groups "$MIN_COMPLETE_ROUTER_GROUPS")
fi
if [[ -n "$EXPECTED_ROUTER_GROUPS" ]]; then
  live_verify_args+=(--expected-groups "$EXPECTED_ROUTER_GROUPS")
fi
"$PYTHON_ENV/bin/python" "$ROOT/scripts/verify_live_rl_trace.py" \
  --trace "$STATE_DIR/$RUN_ID/trace.jsonl" \
  --min-policy-versions "$MIN_POLICY_VERSIONS" \
  "${live_verify_args[@]}" \
  --output "$LOG_DIR/live_router_acceptance.json"
"$PYTHON_ENV/bin/python" "$ROOT/scripts/summarize_live_run.py" \
  --trace "$STATE_DIR/$RUN_ID/trace.jsonl" \
  --mode bounded-async \
  --wall-seconds "$wall_seconds" \
  --output "$LOG_DIR/live_run_summary.json"

echo "Bounded-async acceptance passed: $LOG_DIR/acceptance.json"
