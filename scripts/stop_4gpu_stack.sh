#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER_LOG_DIR="${LOG_DIR:-$ROOT/results/rl_state_4gpu/worker_logs}"
ROUTER_LOG_DIR="${ROUTER_LOG_DIR:-$ROOT/results/rl_state_4gpu}"

stop_from_pidfile() {
  local pidfile="$1"
  local expected="$2"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(tr -dc '0-9' <"$pidfile")"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  local command_line
  command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$command_line" != *"$expected"* ]]; then
    echo "Refusing to stop pid $pid: command does not contain $expected" >&2
    return 1
  fi
  kill "$pid"
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || { echo "stopped pid $pid ($expected)"; return 0; }
    sleep 0.25
  done
  echo "pid $pid did not stop after SIGTERM; manual inspection required" >&2
  return 1
}

stop_from_pidfile "$ROUTER_LOG_DIR/router.pid" "router/app.py"
stop_from_pidfile "$WORKER_LOG_DIR/rollout-0.pid" "sglang.launch_server"
stop_from_pidfile "$WORKER_LOG_DIR/rollout-1.pid" "sglang.launch_server"

