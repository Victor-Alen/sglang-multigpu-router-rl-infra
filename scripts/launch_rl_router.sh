#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/thjiang/miniconda3/envs/rayllm-compat/bin/python}"
exec "$PYTHON_BIN" "$ROOT/router/app.py" \
  --config "$ROOT/configs/rl_router_6xa6000.yaml" \
  --policy cache-aware \
  --rl-policy "${RL_POLICY:-adaptive-group}" \
  --rl-initial-policy-version "${POLICY_VERSION:-0}" \
  --rl-state-dir "${RL_STATE_DIR:-$ROOT/results/rl_state}" \
  --host "${ROUTER_HOST:-127.0.0.1}" \
  --port "${ROUTER_PORT:-8088}"
