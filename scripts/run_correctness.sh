#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$ROOT"
"$PYTHON_BIN" -m compileall -q router integrations rewards scripts tests
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/generate_golden_trace.py --output tests/fixtures/golden_trace.jsonl
"$PYTHON_BIN" scripts/run_offline_replay.py \
  --trace tests/fixtures/golden_trace.jsonl \
  --policy adaptive-group \
  --output results/rl_state/golden_replay_summary.json
