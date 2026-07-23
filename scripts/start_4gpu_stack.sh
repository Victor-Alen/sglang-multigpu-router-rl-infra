#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/results/rl_state_4gpu/worker_logs}"
ROUTER_LOG_DIR="${ROUTER_LOG_DIR:-$ROOT/results/rl_state_4gpu}"
ROLLOUT_PORTS="${ROLLOUT_PORTS:-30004,30005}"
ROUTER_PORT="${ROUTER_PORT:-8088}"

for port in ${ROLLOUT_PORTS//,/ } "$ROUTER_PORT"; do
  if ss -ltn | awk '{print $4}' | grep -Eq ":${port}$"; then
    echo "TCP port $port is already in use; refusing to start a duplicate service." >&2
    exit 2
  fi
done

mkdir -p "$ROUTER_LOG_DIR"
bash "$ROOT/scripts/launch_rollout_workers_4xa6000.sh"

cleanup_on_error() {
  bash "$ROOT/scripts/stop_4gpu_stack.sh" || true
}
trap cleanup_on_error ERR

IFS=, read -r -a ports <<<"$ROLLOUT_PORTS"
for port in "${ports[@]}"; do
  ready=0
  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ $ready -eq 1 ]] || { echo "Rollout worker on port $port did not become healthy." >&2; exit 2; }
done

nohup bash "$ROOT/scripts/launch_rl_router_4xa6000.sh" \
  >"$ROUTER_LOG_DIR/router.log" 2>&1 </dev/null &
echo $! >"$ROUTER_LOG_DIR/router.pid"

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null 2>&1; then
    trap - ERR
    echo "4-GPU Rollout stack is healthy."
    echo "Run training separately with: bash scripts/launch_training_4xa6000.sh"
    exit 0
  fi
  sleep 1
done
echo "Router did not become healthy; see $ROUTER_LOG_DIR/router.log" >&2
exit 2

