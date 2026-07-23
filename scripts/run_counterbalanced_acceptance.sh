#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "${LOCK_FILE:-$ROOT/configs/slime_qwen3_4b_4xa6000.env}"

START_REP="${START_REP:-1}"
NUM_PAIRS="${NUM_PAIRS:-3}"
RUN_LABEL="${RUN_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-1,3}"
ROLLOUT_GPU_IDS="${ROLLOUT_GPU_IDS:-5,7}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
SUMMARY_DIR="$ROOT/results/slime_qwen3_4b_4gpu/counterbalanced_${RUN_LABEL}"
comparisons=()

run_one() {
  local mode="$1"
  local run_id="$2"
  TRAINER_GPU_IDS="$TRAINER_GPU_IDS" \
  ROLLOUT_GPU_IDS="$ROLLOUT_GPU_IDS" \
  NUM_ROLLOUTS="$NUM_ROLLOUTS" \
  MAX_RESPONSE_LEN="$MAX_RESPONSE_LEN" \
  MAX_TOKENS_PER_GPU="$MAX_TOKENS_PER_GPU" \
  ROLLOUT_SHUFFLE=0 \
  RL_ROUTER_RUN_ID="$run_id" \
    bash "$ROOT/scripts/run_${mode}_acceptance.sh"
}

for ((offset=0; offset<NUM_PAIRS; offset++)); do
  rep=$((START_REP + offset))
  strict_id="strict-${RUN_LABEL}-rep${rep}"
  bounded_id="bounded-${RUN_LABEL}-rep${rep}"
  if ((rep % 2 == 1)); then
    run_one bounded_async "$bounded_id"
    run_one strict_sync "$strict_id"
  else
    run_one strict_sync "$strict_id"
    run_one bounded_async "$bounded_id"
  fi
  pair_dir="$SUMMARY_DIR/pair${rep}"
  "$PYTHON_ENV/bin/python" "$ROOT/scripts/compare_strict_bounded_runs.py" \
    --strict-dir "$ROOT/results/slime_qwen3_4b_4gpu/strict_sync/$strict_id" \
    --bounded-dir "$ROOT/results/slime_qwen3_4b_4gpu/bounded_async/$bounded_id" \
    --strict-trace "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$strict_id/trace.jsonl" \
    --bounded-trace "$ROOT/results/slime_qwen3_4b_4gpu/rl_router_state/$bounded_id/trace.jsonl" \
    --output-dir "$pair_dir"
  comparisons+=(--comparison "$pair_dir/comparison.json")
done

"$PYTHON_ENV/bin/python" "$ROOT/scripts/aggregate_paired_comparisons.py" \
  "${comparisons[@]}" --output-dir "$SUMMARY_DIR"
echo "Counterbalanced acceptance passed: $SUMMARY_DIR"
