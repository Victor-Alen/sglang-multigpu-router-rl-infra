#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${RL_CHAOS_NETNS:-0}" != "1" ]]; then
  exec unshare --user --map-root-user --net \
    env RL_CHAOS_NETNS=1 RAY_NODE_IP=127.0.0.1 \
    bash "$0" "$@"
fi
ip link set lo up
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

RUN_ID="${RL_ROUTER_RUN_ID:-network-partition-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
NAMESPACE_IPV4="$(getent hosts "$(hostname)" | awk '$1 ~ /^[0-9.]+$/ { print $1; exit }')"
test -n "$NAMESPACE_IPV4" || { echo "could not resolve namespace IPv4 address" >&2; exit 2; }
NETWORK_HOST="${RL_CHAOS_NETWORK_HOST:-$NAMESPACE_IPV4}"
NETWORK_PORT="${RL_CHAOS_NETWORK_PORT:-15000}"
NETWORK_SECONDS="${RL_CHAOS_NETWORK_SECONDS:-10}"
LOG_DIR="${LOG_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/chaos/$RUN_ID}"
STATE_DIR="${RL_ROUTER_STATE_DIR:-$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state}"
TRACE="$STATE_DIR/$RUN_ID/bounded_async_trace.jsonl"
LABEL="sglang-chaos-$(printf '%s' "$RUN_ID" | sha256sum | cut -c1-16)"
UID_VALUE="$(id -u)"
mkdir -p "$LOG_DIR"

cleanup_rule() {
  iptables -D OUTPUT -p tcp -d "$NETWORK_HOST" --dport "$NETWORK_PORT" \
    -m owner --uid-owner "$UID_VALUE" -m comment --comment "$LABEL" \
    -j REJECT --reject-with tcp-reset >/dev/null 2>&1 || true
}
trap cleanup_rule EXIT INT TERM

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
  RL_CHAOS_NETWORK_ROLLOUT_ID="${RL_CHAOS_NETWORK_ROLLOUT_ID:-1}" \
  RL_CHAOS_NETWORK_HOST="$NETWORK_HOST" \
  RL_CHAOS_NETWORK_PORT="$NETWORK_PORT" \
  RL_CHAOS_NETWORK_SECONDS="$NETWORK_SECONDS" \
  ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}" \
  ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}" \
  ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-300}" \
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
  --require-network-partition-recovery \
  --output "$LOG_DIR/network_partition_acceptance.json"

if iptables-save | grep -F -- "$LABEL" >/dev/null; then
  echo "network partition rule leaked after recovery: $LABEL" >&2
  exit 1
fi

echo "Kernel network-partition acceptance passed: $LOG_DIR/network_partition_acceptance.json"
